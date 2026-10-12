"""Plot baseline vs flawed features for one dataset clip, with the flaw region shaded.

Usage (from the project root):
    python -m scripts.plot_features kennedy_inaugural__monotone_severe
Output: docs/figures/<clip_id>.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backend.features import extract_features, load_audio  # noqa: E402


def plot_clip(row: dict, out_dir: Path = Path("docs/figures")) -> Path:
    """Draw F0 (semitones) and energy (dB) for baseline and flawed audio side by side."""
    base = extract_features(load_audio(row["baseline_file"]))
    flawed = extract_features(load_audio(row["file"]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharey="row")
    panels = [
        (axes[0][0], base, "f0_st", "Baseline: pitch", "semitones re median"),
        (axes[0][1], flawed, "f0_st", "Participant: pitch", ""),
        (axes[1][0], base, "energy_rel_db", "Baseline: energy", "dB re peak level"),
        (axes[1][1], flawed, "energy_rel_db", "Participant: energy", ""),
    ]
    for ax, feats, key, title, ylabel in panels:
        ax.plot(feats.times, getattr(feats, key), lw=0.8, color="#1f4e79")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
    fs, fe = float(row["flaw_start"]), float(row["flaw_end"])
    ws, we = float(row["window_start"]), float(row["window_end"])
    for ax in (axes[0][1], axes[1][1]):
        ax.axvspan(fs, max(fe, fs + 0.05), color="#d62728", alpha=0.25, label="labelled flaw")
    for ax in (axes[0][0], axes[1][0]):
        ax.axvspan(ws, we, color="#7f7f7f", alpha=0.15, label="source window")
    axes[0][1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{row['clip_id']}  ({row['flaw_type']}, {row['severity']}, parameter {row['parameter']})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{row['clip_id']}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    """Plot the clip named on the command line."""
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    manifest = Path(sys.argv[2] if len(sys.argv) > 2 else "data/manifest.csv")
    rows = {r["clip_id"]: r for r in csv.DictReader(manifest.open(encoding="utf-8"))}
    if sys.argv[1] not in rows or rows[sys.argv[1]]["is_baseline"] == "True":
        sys.exit(f"'{sys.argv[1]}' is not a flawed clip in {manifest}")
    print(f"Saved {plot_clip(rows[sys.argv[1]])}")


if __name__ == "__main__":
    main()