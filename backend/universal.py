"""Universal speech handler for Vocalis.

Enables contrastive analysis on ANY arbitrary audio recording without depending
on internal helper names from other modules.
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


def _psola_stretch_safe(audio: np.ndarray, rate: float) -> np.ndarray:
    """Stretch audio by `rate` using Praat PSOLA or librosa fallback."""
    try:
        from dataset_builder.perturb import _psola_stretch
        return _psola_stretch(audio, rate=rate)
    except Exception:
        return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)


def _run_align(audio_path: str | Path, transcript: str, align_fn=None) -> list[dict]:
    """Run word alignment using provided align_fn, pipeline.align_transcript, or alignment.py."""
    if align_fn is not None:
        return align_fn(str(audio_path), transcript)
    fn = getattr(pipe_mod, "align_transcript", getattr(align_mod, "align_transcript"))
    return fn(str(audio_path), transcript)


def transcribe_audio(audio_path: str | Path, language: str = "en") -> str:
    """Transcribe arbitrary audio using WhisperX ASR when no transcript is provided."""
    import whisperx

    global _ASR_MODEL
    if _ASR_MODEL is None:
        _ASR_MODEL = whisperx.load_model(
            "base.en" if language == "en" else "base",
            device=DEVICE,
            compute_type="int8",
        )
    audio = whisperx.load_audio(str(audio_path))
    result = _ASR_MODEL.transcribe(audio, batch_size=4)
    segments = result.get("segments", [])
    text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
    if not text:
        raise ValueError("Could not detect any spoken words in the uploaded audio.")
    return text


def _match_benchmark_speech(transcript: str, raw_dir: Path, transcript_dir: Path) -> str | None:
    """Return benchmark speech_id if transcript has >= 45% word overlap with it."""
    user_tokens = [_norm_word(w) for w in transcript.split() if _norm_word(w)]
    if len(user_tokens) < 5 or not raw_dir.exists():
        return None
    user_set = set(user_tokens)

    best_id, best_score = None, 0.0
    for wav_path in sorted(raw_dir.glob("*.wav")):
        sid = wav_path.stem
        if sid.startswith("_"):
            continue
        t_path = transcript_dir / f"{sid}.txt"
        if not t_path.exists():
            continue
        ref_tokens = [_norm_word(w) for w in t_path.read_text(encoding="utf-8").split() if _norm_word(w)]
        if not ref_tokens:
            continue
        overlap = len(user_set & set(ref_tokens)) / max(1, min(len(user_set), len(set(ref_tokens))))
        if overlap > best_score:
            best_score = overlap
            best_id = sid

    return best_id if best_score >= 0.45 else None


def _ensure_expressive_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Ensure the synthesized ideal reference has healthy (~3.0 ST) pitch variation."""
    if len(audio) < sr // 2:
        return audio
    try:
        call("random_initializeWithSeedUnsafelyButPredictably", 42)
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
        f0 = pitch.selected_array["frequency"].astype(float)
        voiced = f0[f0 > 0]
        if len(voiced) < 20:
            return audio
        med = float(np.median(voiced))
        st = 12.0 * np.log2(voiced / med)
        spread = float(np.percentile(st, 90) - np.percentile(st, 10))
        if spread >= 2.8:
            return audio

        manip = call(snd, "To Manipulation", 0.01, 75.0, 500.0)
        pt = call(manip, "Extract pitch tier")
        n_pts = call(pt, "Get number of points")
        if n_pts == 0:
            return audio
        times = [call(pt, "Get time from index", i) for i in range(1, n_pts + 1)]
        freqs = [call(pt, "Get value at index", i) for i in range(1, n_pts + 1)]
        call(pt, "Remove points between", 0.0, snd.xmax)

        gain = min(3.5, 3.2 / max(0.6, spread))
        for t, f in zip(times, freqs):
            dev_st = 12.0 * np.log2(max(f, 50.0) / med)
            arch_st = 1.2 * np.sin(2.0 * np.pi * t / 4.0) if spread < 1.0 else 0.0
            new_st = np.clip(dev_st * gain + arch_st, -7.0, 7.0)
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

    clean_items: list[dict] = []
    for w in part_words:
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

    target_unit_s = float(np.clip(np.median(unit_durs), 0.14, 0.21))
    target_rms = float(np.percentile(word_rms_list, 80)) if word_rms_list else 0.08

    ref_chunks: list[np.ndarray] = [np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)]
    cursor_s = 0.25
    ref_words: list[dict] = []

    for i, w in enumerate(clean_items):
        s_idx = max(0, int(float(w["start"]) * SAMPLE_RATE))
        e_idx = min(len(audio), int(float(w["end"]) * SAMPLE_RATE))
        chunk = audio[s_idx:e_idx].copy()
        if len(chunk) < int(0.04 * SAMPLE_RATE):
            chunk = np.zeros(int(0.08 * SAMPLE_RATE), dtype=np.float32)

        c_rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        if c_rms > 1e-4:
            gain = np.clip(target_rms / c_rms, 0.35, 3.5)
            chunk = np.clip(chunk * gain, -0.95, 0.95)

        actual_dur = len(chunk) / SAMPLE_RATE
        ideal_dur = (_count_syllables(w["word"]) + 0.35) * target_unit_s
        rate = float(np.clip(actual_dur / max(0.08, ideal_dur), 0.55, 1.85))
        if abs(rate - 1.0) > 0.12 and len(chunk) >= int(0.12 * SAMPLE_RATE):
            try:
                chunk = _psola_stretch_safe(chunk, rate=rate)
            except Exception:
                pass

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
    ref_audio = _ensure_expressive_pitch(ref_audio, SAMPLE_RATE)

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
        ref_words = _run_align(ref_wav_path, clean_transcript, align_fn=align_fn)
        (align_dir / f"{CUSTOM_ID}_alignment.json").write_text(
            json.dumps(ref_words, indent=2), encoding="utf-8"
        )
        return CUSTOM_ID, clean_transcript

    if sid not in ("auto", "custom", ""):
        return sid, (clean_transcript if clean_transcript else None)

    # Universal "auto" / "custom" mode
    user_supplied_transcript = bool(clean_transcript)
    if not clean_transcript:
        clean_transcript = transcribe_audio(audio_path)

    matched = _match_benchmark_speech(clean_transcript, raw_dir, transcript_dir)
    if matched is not None and (raw_dir / f"{matched}.wav").exists():
        # If the user did not supply a custom transcript (e.g. they uploaded moderate_recording.m4a),
        # return None so pipeline.py uses the exact canonical transcript for 1-to-1 alignment!
        return matched, (clean_transcript if user_supplied_transcript else None)

    part_words = _run_align(audio_path, clean_transcript, align_fn=align_fn)
    return build_ideal_reference_from_participant(
        audio_path, part_words, raw_dir, align_dir, transcript_dir
    )