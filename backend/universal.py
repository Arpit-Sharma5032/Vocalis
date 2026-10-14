"""Universal speech handler for Vocalis.

Enables contrastive analysis on ANY arbitrary audio recording (including long
10-30 minute speeches) by:
1. Auto-transcribing audio via WhisperX ASR and caching VAD segment boundaries.
2. Matching one of the 4 archival benchmark excerpts ONLY if both word overlap
   AND recording duration/word-count match the excerpt.
3. Aligning long recordings in 25-second VAD/silence-bounded segments so wav2vec2
   never compresses words into the start of the timeline.
4. Synthesizing a syllable-timed, uniform-expressive-pitch, level-normalized Ideal
   Reference Baseline (`_custom_ref`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

import backend.alignment as align_mod
import backend.pipeline as pipe_mod

SAMPLE_RATE = 16000
DEVICE = getattr(align_mod, "DEVICE", "cpu")
CUSTOM_ID = "_custom_ref"

_ASR_MODEL = None
_ALIGN_CACHE: dict[str, tuple] = {}
_LAST_ASR_SEGMENTS: dict[str, list[dict]] = {}


def _norm_word(word: str) -> str:
    """Strip punctuation and lowercase a word token."""
    return re.sub(r"[^a-z0-9']+", "", str(word).lower()).strip("'")


def _count_syllables(word: str) -> int:
    """Heuristic syllable counter for English words."""
    clean = re.sub(r"[^a-z]", "", str(word).lower())
    if not clean or len(clean) <= 3:
        return 1
    clean = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", clean)
    clean = re.sub(r"^y", "", clean)
    vowels = re.findall(r"[aeiouy]{1,2}", clean)
    return max(1, len(vowels))


def _fit_exact_duration(chunk: np.ndarray, target_dur_s: float, allow_psola: bool = True) -> np.ndarray:
    """Stretch or compress `chunk` so its duration equals `target_dur_s`."""
    target_len = max(int(0.06 * SAMPLE_RATE), int(round(target_dur_s * SAMPLE_RATE)))
    if len(chunk) < int(0.03 * SAMPLE_RATE):
        return np.zeros(target_len, dtype=np.float32)

    actual_dur = len(chunk) / SAMPLE_RATE
    rate = float(np.clip(actual_dur / max(0.06, target_dur_s), 0.45, 2.2))

    out = chunk
    if allow_psola and abs(rate - 1.0) > 0.08 and len(chunk) >= int(0.08 * SAMPLE_RATE):
        try:
            from dataset_builder.perturb import _psola_stretch
            out = _psola_stretch(chunk, rate=rate)
        except Exception:
            try:
                out = librosa.effects.time_stretch(chunk.astype(np.float32), rate=rate)
            except Exception:
                out = chunk
    elif not allow_psola and abs(rate - 1.0) > 0.08:
        try:
            out = librosa.effects.time_stretch(chunk.astype(np.float32), rate=rate)
        except Exception:
            out = chunk

    if len(out) != target_len:
        x_old = np.linspace(0.0, 1.0, num=max(1, len(out)), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        out = np.interp(x_new, x_old, out).astype(np.float32)
    return out


def _trim_swallowed_silences(audio: np.ndarray, words: list[dict]) -> list[dict]:
    """Self-contained silence trimmer so WhisperX swallowed pauses don't corrupt _custom_ref."""
    hop = int(0.01 * SAMPLE_RATE)
    frame_len = int(0.025 * SAMPLE_RATE)
    if len(audio) < frame_len:
        return words

    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop, center=True)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-7))
    p5 = float(np.percentile(db, 5))
    p90 = float(np.percentile(db, 90))
    sil_thresh = min(max(p5 + 10.0, p90 - 30.0), p90 - 15.0)
    is_silent = db < sil_thresh

    refined: list[dict] = []
    min_sil_frames = 20  # 0.20s

    for w in words:
        w_copy = dict(w)
        s = w_copy.get("start")
        e = w_copy.get("end")
        if s is None or e is None or float(e) <= float(s):
            refined.append(w_copy)
            continue

        f_start = max(0, int(round(float(s) / 0.01)))
        f_end = min(len(is_silent), int(round(float(e) / 0.01)))
        if f_end - f_start < min_sil_frames:
            refined.append(w_copy)
            continue

        window_sil = is_silent[f_start:f_end]
        max_sil_run = 0
        cur_sil = 0
        for flag in window_sil:
            if flag:
                cur_sil += 1
                if cur_sil > max_sil_run:
                    max_sil_run = cur_sil
            else:
                cur_sil = 0

        if max_sil_run >= min_sil_frames:
            best_v_start, best_v_end = 0, 0
            cur_v_start = None
            for idx, flag in enumerate(window_sil):
                if not flag:
                    if cur_v_start is None:
                        cur_v_start = idx
                    if (idx + 1 - cur_v_start) > (best_v_end - best_v_start):
                        best_v_start, best_v_end = cur_v_start, idx + 1
                else:
                    cur_v_start = None
            if best_v_end - best_v_start >= 4:
                w_copy["start"] = round((f_start + best_v_start) * 0.01, 3)
                w_copy["end"] = round((f_start + best_v_end) * 0.01, 3)

        refined.append(w_copy)
    return refined


def _chunk_long_transcript_into_segments(audio: np.ndarray, transcript: str, chunk_s: float = 25.0) -> list[dict]:
    """Split a long transcript into ~25s segments snapped to low-energy pause points."""
    duration = len(audio) / SAMPLE_RATE
    words = transcript.strip().split()
    if duration <= 65.0 or len(words) <= 40:
        return [{"text": transcript.strip(), "start": 0.0, "end": duration}]

    hop = int(0.01 * SAMPLE_RATE)
    rms = librosa.feature.rms(y=audio, frame_length=int(0.025 * SAMPLE_RATE), hop_length=hop, center=True)[0]

    n_chunks = max(2, int(np.ceil(duration / chunk_s)))
    boundaries = [0.0]
    for k in range(1, n_chunks):
        target_t = k * (duration / n_chunks)
        # Search +/- 2.5s around target_t for the quietest frame
        lo_f = max(0, int((target_t - 2.5) / 0.01))
        hi_f = min(len(rms) - 1, int((target_t + 2.5) / 0.01))
        best_f = lo_f + int(np.argmin(rms[lo_f : hi_f + 1])) if hi_f > lo_f else int(target_t / 0.01)
        boundaries.append(round(best_f * 0.01, 2))
    boundaries.append(round(duration, 2))

    segments = []
    n_words = len(words)
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        w0 = int(round((t0 / duration) * n_words))
        w1 = int(round((t1 / duration) * n_words)) if i < len(boundaries) - 2 else n_words
        seg_text = " ".join(words[w0:w1]).strip()
        if seg_text and t1 > t0:
            segments.append({"text": seg_text, "start": t0, "end": t1})
    return segments or [{"text": transcript.strip(), "start": 0.0, "end": duration}]


def smart_align_transcript(
    audio_path: str | Path,
    transcript: str,
    align_fn=None,
    language: str = "en",
) -> list[dict]:
    """Align transcript across the full duration of short or long (10-30 min) audio."""
    if align_fn is not None:
        return align_fn(str(audio_path), transcript)

    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    duration = len(audio) / SAMPLE_RATE

    # Use cached WhisperX ASR segments if available for this audio file and matching word count
    cached_segs = _LAST_ASR_SEGMENTS.get(str(Path(audio_path).resolve()))
    if cached_segs:
        cached_text = " ".join(s.get("text", "").strip() for s in cached_segs).strip()
        if abs(len(cached_text.split()) - len(transcript.split())) <= max(5, int(0.1 * len(transcript.split()))):
            segments = cached_segs
        else:
            segments = _chunk_long_transcript_into_segments(audio, transcript)
    elif duration > 65.0:
        segments = _chunk_long_transcript_into_segments(audio, transcript)
    else:
        # Short clip: use existing alignment function
        fn = getattr(pipe_mod, "align_transcript", getattr(align_mod, "align_transcript"))
        return fn(str(audio_path), transcript)

    if language not in _ALIGN_CACHE:
        _ALIGN_CACHE[language] = whisperx.load_align_model(language_code=language, device=DEVICE)
    model, metadata = _ALIGN_CACHE[language]

    result = whisperx.align(
        segments, model, metadata, audio, DEVICE, return_char_alignments=False
    )
    return [
        {"word": w["word"], "start": w.get("start"), "end": w.get("end")}
        for w in result.get("word_segments", [])
        if w.get("word")
    ]


def transcribe_audio(audio_path: str | Path, language: str = "en") -> str:
    """Transcribe arbitrary audio using WhisperX ASR and cache timestamped VAD segments."""
    import whisperx

    global _ASR_MODEL
    if _ASR_MODEL is None:
        _ASR_MODEL = whisperx.load_model(
            "base.en" if language == "en" else "base",
            device=DEVICE,
            compute_type="int8",
        )
    audio = whisperx.load_audio(str(audio_path))
    result = _ASR_MODEL.transcribe(audio, batch_size=8)
    segments = [
        {
            "text": seg.get("text", "").strip(),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
        }
        for seg in result.get("segments", [])
        if seg.get("text", "").strip()
    ]
    if not segments:
        raise ValueError("Could not detect any spoken words in the uploaded audio.")

    _LAST_ASR_SEGMENTS[str(Path(audio_path).resolve())] = segments
    return " ".join(s["text"] for s in segments).strip()


def _match_benchmark_speech(
    transcript: str,
    raw_dir: Path,
    transcript_dir: Path,
    audio_duration_s: float | None = None,
) -> str | None:
    """Return benchmark speech_id ONLY if ordered phrases (bigrams & sequence) match."""
    import difflib

    user_tokens = [_norm_word(w) for w in transcript.split() if _norm_word(w)]
    if len(user_tokens) < 8 or not raw_dir.exists():
        return None
    user_bigrams = set(zip(user_tokens, user_tokens[1:]))
    if not user_bigrams:
        return None

    best_id, best_score = None, 0.0
    for wav_path in sorted(raw_dir.glob("*.wav")):
        sid = wav_path.stem
        if sid.startswith("_"):
            continue
        t_path = transcript_dir / f"{sid}.txt"
        if not t_path.exists():
            continue

        # 1. Check audio duration ratio so long speeches don't match 90s excerpts
        if audio_duration_s is not None and audio_duration_s > 0:
            try:
                ref_dur = float(librosa.get_duration(path=str(wav_path)))
                dur_ratio = min(audio_duration_s, ref_dur) / max(1.0, max(audio_duration_s, ref_dur))
                if dur_ratio < 0.55:
                    continue
            except Exception:
                pass

        ref_tokens = [_norm_word(w) for w in t_path.read_text(encoding="utf-8").split() if _norm_word(w)]
        if len(ref_tokens) < 8:
            continue

        # 2. Check word count ratio
        len_ratio = min(len(user_tokens), len(ref_tokens)) / max(1, max(len(user_tokens), len(ref_tokens)))
        if len_ratio < 0.60:
            continue

        # 3. Require both 2-word phrase (bigram) overlap AND ordered sequence similarity
        ref_bigrams = set(zip(ref_tokens, ref_tokens[1:]))
        bigram_overlap = len(user_bigrams & ref_bigrams) / max(1, max(len(user_bigrams), len(ref_bigrams)))
        seq_ratio = difflib.SequenceMatcher(None, user_tokens, ref_tokens).ratio()

        if bigram_overlap >= 0.35 and seq_ratio >= 0.55:
            score = 0.5 * bigram_overlap + 0.5 * seq_ratio
            if score > best_score:
                best_score = score
                best_id = sid

    return best_id


def _impose_uniform_expressive_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Impose a consistent ~3.6 ST P10-P90 prosodic pitch contour on the ideal reference."""
    dur = len(audio) / sr
    if dur < 0.5 or dur > 360.0:
        # Skip heavy Praat resynthesis on very long (>6 min) files to keep latency fast
        return audio
    try:
        call("random_initializeWithSeedUnsafelyButPredictably", 42)
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
        pitch = snd.to_pitch(time_step=0.04, pitch_floor=75.0, pitch_ceiling=500.0)
        f0 = pitch.selected_array["frequency"].astype(float)
        voiced = f0[f0 > 0]
        if len(voiced) < 20:
            return audio
        med = float(np.median(voiced))

        manip = call(snd, "To Manipulation", 0.02, 75.0, 500.0)
        pt = call(manip, "Extract pitch tier")
        call(pt, "Remove points between", 0.0, snd.xmax)

        xs = pitch.xs()
        step = max(1, int(round(0.08 / 0.04)))
        for idx in range(0, len(xs), step):
            t = float(xs[idx])
            f = float(f0[idx]) if f0[idx] > 0 else med
            orig_st = 12.0 * np.log2(max(f, 50.0) / med)
            prosodic_wave = 2.1 * np.sin(2.0 * np.pi * t / 1.8) + 0.9 * np.cos(2.0 * np.pi * t / 0.75)
            new_st = np.clip(0.25 * orig_st + prosodic_wave, -6.5, 6.5)
            new_f = float(np.clip(med * (2.0 ** (new_st / 12.0)), 75.0, 500.0))
            call(pt, "Add point", t, new_f)

        call([pt, manip], "Replace pitch tier")
        resynth = call(manip, "Get resynthesis (overlap-add)")
        out = resynth.values[0].astype(np.float32)
        if len(out) >= len(audio):
            return out[: len(audio)]
        return np.pad(out, (0, len(audio) - len(out)))
    except Exception:
        return audio


def build_ideal_reference_from_participant(
    audio_path: str | Path,
    part_words: list[dict],
    raw_dir: Path,
    align_dir: Path,
    transcript_dir: Path,
) -> tuple[str, str]:
    """Synthesize an Ideal Reference WAV + Alignment from participant audio."""
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)
    is_long_audio = (len(audio) / SAMPLE_RATE) > 180.0

    refined_words = _trim_swallowed_silences(audio, part_words)

    clean_items: list[dict] = []
    for w in refined_words:
        if w.get("start") is None or w.get("end") is None:
            continue
        nw = _norm_word(w["word"])
        if clean_items and nw and nw == _norm_word(clean_items[-1]["word"]):
            continue
        clean_items.append(w)

    if not clean_items:
        raise ValueError("No valid aligned words found in audio.")

    unit_durs = []
    word_rms_list = []
    for w in clean_items:
        dur = max(0.06, float(w["end"]) - float(w["start"]))
        units = _count_syllables(w["word"]) + 0.35
        unit_durs.append(dur / units)
        s_idx = max(0, int(float(w["start"]) * SAMPLE_RATE))
        e_idx = min(len(audio), int(float(w["end"]) * SAMPLE_RATE))
        if e_idx > s_idx:
            rms = float(np.sqrt(np.mean(audio[s_idx:e_idx] ** 2) + 1e-12))
            if rms > 1e-4:
                word_rms_list.append(rms)

    target_unit_s = float(np.clip(np.median(unit_durs), 0.14, 0.20))
    target_rms = float(np.percentile(word_rms_list, 85)) if word_rms_list else 0.08

    ref_chunks: list[np.ndarray] = [np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)]
    cursor_s = 0.25
    ref_words: list[dict] = []

    for i, w in enumerate(clean_items):
        s_idx = max(0, int(float(w["start"]) * SAMPLE_RATE))
        e_idx = min(len(audio), int(float(w["end"]) * SAMPLE_RATE))
        chunk = audio[s_idx:e_idx].copy()

        c_rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        if c_rms > 1e-4:
            gain = np.clip(target_rms / c_rms, 0.25, 8.0)
            chunk = np.clip(chunk * gain, -0.95, 0.95)

        ideal_dur = (_count_syllables(w["word"]) + 0.35) * target_unit_s
        chunk = _fit_exact_duration(chunk, ideal_dur, allow_psola=not is_long_audio)

        fade = min(int(0.005 * SAMPLE_RATE), len(chunk) // 4)
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            chunk[:fade] *= ramp
            chunk[-fade:] *= ramp[::-1]

        w_dur = len(chunk) / SAMPLE_RATE
        ref_words.append({
            "word": w["word"],
            "start": round(cursor_s, 3),
            "end": round(cursor_s + w_dur, 3),
        })
        ref_chunks.append(chunk)
        cursor_s += w_dur

        token = w["word"].strip()
        if i < len(clean_items) - 1:
            if token.endswith((".", "?", "!")):
                gap_s = 0.40
            elif token.endswith((",", ";", ":", "—", "-")):
                gap_s = 0.18
            else:
                gap_s = 0.04
            ref_chunks.append(np.zeros(int(gap_s * SAMPLE_RATE), dtype=np.float32))
            cursor_s += gap_s

    ref_chunks.append(np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32))
    ref_audio = np.concatenate(ref_chunks)
    ref_audio = _impose_uniform_expressive_pitch(ref_audio, SAMPLE_RATE)

    raw_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    ref_wav_path = raw_dir / f"{CUSTOM_ID}.wav"
    ref_txt_path = transcript_dir / f"{CUSTOM_ID}.txt"
    ref_align_path = align_dir / f"{CUSTOM_ID}_alignment.json"

    ref_transcript = " ".join(w["word"] for w in ref_words)
    sf.write(ref_wav_path, ref_audio, SAMPLE_RATE)
    ref_txt_path.write_text(ref_transcript, encoding="utf-8")
    ref_align_path.write_text(json.dumps(ref_words, indent=2), encoding="utf-8")

    part_transcript = " ".join(w["word"] for w in part_words if w.get("word"))
    return CUSTOM_ID, part_transcript


def prepare_reference_and_transcript(
    audio_path: str | Path,
    speech_id: str | None = "auto",
    transcript: str | None = None,
    reference_audio_path: str | Path | None = None,
    raw_dir: Path = Path("data/raw_audio"),
    align_dir: Path = Path("data/processed_metrics"),
    transcript_dir: Path = Path("data/transcripts"),
    align_fn=None,
) -> tuple[str, str | None]:
    """Resolve (effective_speech_id, effective_transcript) for any audio input."""
    sid = (speech_id or "auto").strip()
    clean_transcript = (transcript or "").strip()

    if reference_audio_path is not None:
        if not clean_transcript:
            clean_transcript = transcribe_audio(reference_audio_path)
        ref_audio, _ = librosa.load(str(reference_audio_path), sr=SAMPLE_RATE, mono=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        align_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)

        ref_wav_path = raw_dir / f"{CUSTOM_ID}.wav"
        sf.write(ref_wav_path, ref_audio, SAMPLE_RATE)
        (transcript_dir / f"{CUSTOM_ID}.txt").write_text(clean_transcript, encoding="utf-8")
        ref_words = smart_align_transcript(ref_wav_path, clean_transcript, align_fn=align_fn)
        (align_dir / f"{CUSTOM_ID}_alignment.json").write_text(
            json.dumps(ref_words, indent=2), encoding="utf-8"
        )
        return CUSTOM_ID, clean_transcript

    if sid not in ("auto", "custom", ""):
        return sid, (clean_transcript if clean_transcript else None)

    user_supplied_transcript = bool(clean_transcript)
    if not clean_transcript:
        clean_transcript = transcribe_audio(audio_path)

    try:
        audio_dur_s = float(librosa.get_duration(path=str(audio_path)))
    except Exception:
        audio_dur_s = None

    matched = _match_benchmark_speech(clean_transcript, raw_dir, transcript_dir, audio_duration_s=audio_dur_s)
    if matched is not None and (raw_dir / f"{matched}.wav").exists():
        return matched, (clean_transcript if user_supplied_transcript else None)

    part_words = smart_align_transcript(audio_path, clean_transcript, align_fn=align_fn)
    return build_ideal_reference_from_participant(
        audio_path, part_words, raw_dir, align_dir, transcript_dir
    )