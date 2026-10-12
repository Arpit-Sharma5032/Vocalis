"""Speaker-agnostic normalization helpers.

Every acoustic feature is expressed *relative to the speaker's own typical
value*, so a deep voice and a high voice, or a loud and a quiet recording,
become directly comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hz_to_semitones(f0_hz: np.ndarray, ref_hz: float | None = None) -> np.ndarray:
    """Convert Hz to semitones relative to a reference (default: speaker's median F0).

    12 semitones = one octave. Using the speaker's median as the reference
    removes the speaker's natural pitch height, leaving only pitch *movement*.
    """
    ref = np.nanmedian(f0_hz) if ref_hz is None else ref_hz
    with np.errstate(divide="ignore", invalid="ignore"):
        return 12.0 * np.log2(f0_hz / ref)


def _repair_pass(st: np.ndarray, window: int, jump: float) -> np.ndarray:
    """One repair pass against the local median (see fix_octave_jumps)."""
    local = pd.Series(st).rolling(window, center=True, min_periods=3).median().to_numpy()
    out = st.copy()
    out[np.isfinite(st) & np.isnan(local)] = np.nan  # isolated voiced frames are tracker glitches
    for i in np.flatnonzero(np.abs(st - local) > jump):
        for shift in (12.0, -12.0):
            if abs(st[i] - shift - local[i]) < 3.0:
                out[i] = st[i] - shift
                break
        else:
            out[i] = np.nan
    return out


def fix_octave_jumps(
    st: np.ndarray, window: int = 11, jump: float = 7.0, max_passes: int = 4
) -> np.ndarray:
    """Repair or drop pitch-tracker octave errors in a semitone contour.

    A frame that sits more than `jump` semitones from its local median is
    shifted by +/-12 if that lands it near its neighbours (an octave error),
    otherwise it is discarded (set to NaN). Voiced frames with fewer than 3
    voiced neighbours in the window are also discarded as isolated glitches.
    Runs of consecutive errors skew the local median, so the repair is repeated
    until nothing changes (or `max_passes`).
    """
    current = st
    for _ in range(max_passes):
        repaired = _repair_pass(current, window, jump)
        if np.array_equal(repaired, current, equal_nan=True):
            break
        current = repaired
    return current


def drop_short_runs(st: np.ndarray, min_len: int = 5) -> np.ndarray:
    """Discard voiced runs shorter than `min_len` frames (50 ms by default).

    Real voiced speech segments last longer than this; shorter bursts are
    almost always tracker artefacts.
    """
    out = st.copy()
    voiced = np.isfinite(st)
    i, n = 0, len(st)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            if j - i < min_len:
                out[i:j] = np.nan
            i = j
        else:
            i += 1
    return out


def relative_level_db(db: np.ndarray, percentile: float = 90.0) -> tuple[np.ndarray, float]:
    """Express frame energy in dB relative to the speaker's peak speech level.

    The reference is a high percentile of the whole recording, so a recording
    made at a different gain gives the same relative values. Returns
    (relative_db, reference_db).
    """
    ref = float(np.percentile(db, percentile))
    return db - ref, ref


def cmvn(mfcc: np.ndarray) -> np.ndarray:
    """Cepstral mean and variance normalization over time (frames x coefficients).

    Removes constant channel and speaker coloration from the spectrum.
    """
    return (mfcc - mfcc.mean(axis=0)) / (mfcc.std(axis=0) + 1e-8)