"""Acoustic feature extraction: F0, energy, MFCC, pauses and per-word statistics.

All frame-level features share one time grid (10 ms hop), and all are
normalized per speaker (see normalization.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import pandas as pd
import parselmouth

from backend.normalization import (
    cmvn,
    drop_short_runs,
    fix_octave_jumps,
    hz_to_semitones,
    relative_level_db,
)

SR = 16000
HOP = 160  # samples -> 10 ms
FRAME = 400  # samples -> 25 ms
HOP_S = HOP / SR
N_MFCC = 13


@dataclass
class FrameFeatures:
    """Frame-level features on a common 10 ms grid (all arrays share length)."""

    times: np.ndarray  # frame centre times (s)
    f0_hz: np.ndarray  # NaN where unvoiced
    f0_st: np.ndarray  # semitones re speaker median, octave errors repaired
    energy_db: np.ndarray  # absolute RMS level (dBFS)
    energy_rel_db: np.ndarray  # dB relative to the speaker's peak speech level
    mfcc: np.ndarray  # (frames, 13), CMVN-normalized
    median_f0_hz: float
    level_db: float  # reference speech level (90th percentile)
    noise_floor_db: float  # 5th percentile level


def load_audio(path: str) -> np.ndarray:
    """Load an audio file as 16 kHz mono float32."""
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio


def extract_f0_hz(audio: np.ndarray, grid_times: np.ndarray) -> np.ndarray:
    """Two-pass F0 tracking, resampled onto the grid (NaN = unvoiced).

    Pass 1 uses a wide range to learn the speaker's pitch quartiles; pass 2
    narrows the search range to 0.75*Q1 .. 2.5*Q3, which sharply reduces octave
    errors (Hirst & De Looze's recommended two-pass method).
    """
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SR)
    first = snd.to_pitch(time_step=HOP_S, pitch_floor=60.0, pitch_ceiling=600.0)
    voiced = first.selected_array["frequency"]
    voiced = voiced[voiced > 0]
    if len(voiced) < 20:
        return np.full(len(grid_times), np.nan)
    q1, q3 = np.percentile(voiced, [25, 75])
    pitch = snd.to_pitch(
        time_step=HOP_S, pitch_floor=max(50.0, 0.75 * q1), pitch_ceiling=min(800.0, 2.5 * q3)
    )
    xs, values = pitch.xs(), pitch.selected_array["frequency"].astype(float)
    idx = np.rint((grid_times - xs[0]) / HOP_S).astype(int)
    ok = (idx >= 0) & (idx < len(xs))
    f0 = np.full(len(grid_times), np.nan)
    f0[ok] = values[idx[ok]]
    f0[f0 <= 0] = np.nan
    return f0


def extract_features(audio: np.ndarray) -> FrameFeatures:
    """Compute all frame-level features for one recording."""
    rms = librosa.feature.rms(y=audio, frame_length=FRAME, hop_length=HOP, center=True)[0]
    n = len(rms)
    times = np.arange(n) * HOP_S
    energy_db = 20.0 * np.log10(np.maximum(rms, 1e-5))
    energy_rel, level = relative_level_db(energy_db)

    f0_hz = extract_f0_hz(audio, times)
    median_f0 = float(np.nanmedian(f0_hz)) if np.isfinite(f0_hz).any() else float("nan")
    f0_st = (
        drop_short_runs(fix_octave_jumps(hz_to_semitones(f0_hz, median_f0)))
        if np.isfinite(median_f0)
        else f0_hz
    )

    mfcc = librosa.feature.mfcc(
        y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=FRAME, hop_length=HOP, n_mels=40, center=True
    ).T[:n]
    return FrameFeatures(
        times=times, f0_hz=f0_hz, f0_st=f0_st, energy_db=energy_db, energy_rel_db=energy_rel,
        mfcc=cmvn(mfcc), median_f0_hz=median_f0, level_db=level,
        noise_floor_db=float(np.percentile(energy_db, 5)),
    )


def find_pauses(feats: FrameFeatures, min_len_s: float = 0.25) -> list[tuple[float, float]]:
    """Detect silent stretches from the energy contour.

    A frame is silent if it is below max(noise floor + 10 dB, speech level - 30 dB).
    Runs of silent frames at least `min_len_s` long are returned as (start, end).
    """
    threshold = max(feats.noise_floor_db + 10.0, feats.level_db - 30.0)
    silent = feats.energy_db < threshold
    pauses, start = [], None
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * HOP_S >= min_len_s:
                pauses.append((float(feats.times[start]), float(feats.times[i])))
            start = None
    if start is not None and (len(silent) - start) * HOP_S >= min_len_s:
        pauses.append((float(feats.times[start]), float(feats.times[-1] + HOP_S)))
    return pauses


def word_features(feats: FrameFeatures, words: list[dict]) -> pd.DataFrame:
    """Summarize the frame features inside each aligned word.

    Columns: word, start, end, dur, gap_before, f0_mean_st, f0_std_st,
    f0_range_st (P90-P10), energy_mean_db, energy_max_db, voiced_frac.
    """
    rows, prev_end = [], None
    for w in words:
        mask = (feats.times >= w["start"]) & (feats.times < w["end"])
        st = feats.f0_st[mask]
        st = st[np.isfinite(st)]
        en = feats.energy_rel_db[mask]
        enough = len(st) >= 3
        rows.append({
            "word": w["word"], "start": w["start"], "end": w["end"],
            "dur": w["end"] - w["start"],
            "gap_before": 0.0 if prev_end is None else max(0.0, w["start"] - prev_end),
            "f0_mean_st": float(st.mean()) if enough else np.nan,
            "f0_std_st": float(st.std()) if enough else np.nan,
            "f0_range_st": float(np.percentile(st, 90) - np.percentile(st, 10)) if enough else np.nan,
            "energy_mean_db": float(en.mean()) if len(en) else np.nan,
            "energy_max_db": float(en.max()) if len(en) else np.nan,
            "voiced_frac": float(mask.sum() and np.isfinite(feats.f0_st[mask]).mean()),
        })
        prev_end = w["end"]
    return pd.DataFrame(rows)