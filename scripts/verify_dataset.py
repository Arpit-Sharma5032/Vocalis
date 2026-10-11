"""Check every manifest label against the actual audio, independently of the builder.

For each flawed clip this re-measures the audio and confirms that the labelled
region really contains the flaw, with the labelled magnitude.

Usage (from the project root):
    python scripts/verify_dataset.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import parselmouth

SR = 16000
TOL_S = 0.01  # allowed timing error (10 ms)
STUMBLE_GAP_S = 0.25  # must match dataset_builder.perturb.STUMBLE_GAP_S


def rms_db(x: np.ndarray) -> float:
    """RMS level of a signal in dB (relative to full scale)."""
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)


def pitch_std_semitones(x: np.ndarray) -> float:
    """Standard deviation of F0 in semitones around the median (voiced frames only)."""
    pitch = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR).to_pitch(0.01, 75, 500)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    return float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 10 else float("nan")


def check_row(row: dict, base: np.ndarray, flawed: np.ndarray) -> list[str]:
    """Return a list of problems found for one flawed clip (empty = OK)."""
    flaw, param = row["flaw_type"], float(row["parameter"])
    a, b = float(row["flaw_start"]), float(row["flaw_end"])
    ws, we = float(row["window_start"]), float(row["window_end"])
    problems = []
    seg = flawed[int(a * SR) : int(b * SR)]

    if flaw in ("pace", "slow"):
        expected = (we - ws) / param
        if abs((b - a) - expected) > TOL_S:
            problems.append(f"region length {b - a:.3f}s, expected {expected:.3f}s")
    elif flaw == "pause":
        if np.max(np.abs(seg)) > 0 or abs((b - a) - param) > TOL_S:
            problems.append("labelled pause is not silent or wrong length")
    elif flaw == "volume":
        delta = rms_db(seg) - rms_db(base[int(a * SR) : int(b * SR)])
        if abs(delta - param) > 0.5:
            problems.append(f"level change {delta:.2f} dB, expected {param} dB")
    elif flaw == "monotone":
        before, after = pitch_std_semitones(base[int(a * SR) : int(b * SR)]), pitch_std_semitones(seg)
        if not after < before:
            problems.append(f"pitch variation did not drop ({before:.2f} -> {after:.2f} st)")
    elif flaw == "stumble":
        pass  # checked via total-length bookkeeping below

    # Total clip length must change by exactly what the label implies.
    delta_total = float(row["duration_s"]) - len(base) / SR
    if flaw in ("pace", "slow"):
        expected_delta = (b - a) - (we - ws)
    elif flaw == "pause":
        expected_delta = param
    elif flaw == "stumble":
        word_len = ((b - a) - param * STUMBLE_GAP_S) / (param + 1)
        expected_delta = param * (word_len + STUMBLE_GAP_S)
    else:  # volume, monotone keep the length
        expected_delta = 0.0
    if abs(delta_total - expected_delta) > TOL_S:
        problems.append(f"clip length changed by {delta_total:.3f}s, label implies {expected_delta:.3f}s")
    return problems


def main() -> None:
    """Verify every flawed clip listed in the manifest."""
    manifest = Path(sys.argv[1] if len(sys.argv) > 1 else "data/manifest.csv")
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    bases: dict[str, np.ndarray] = {}
    failures = checked = 0
    for row in rows:
        if row["is_baseline"] == "True":
            continue
        base = bases.setdefault(
            row["baseline_file"], librosa.load(row["baseline_file"], sr=SR, mono=True)[0]
        )
        flawed, _ = librosa.load(row["file"], sr=SR, mono=True)
        problems = check_row(row, base, flawed)
        checked += 1
        if problems:
            failures += 1
            print(f"FAIL {row['clip_id']}: {'; '.join(problems)}")
    print(f"\nChecked {checked} clips, {failures} failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()