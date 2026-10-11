"""Generate the full contrastive dataset from the baseline speeches.

For every baseline speech, one clip is produced per (flaw type x severity),
each containing exactly ONE injected flaw. Every clip gets a row in a manifest
CSV with the exact ground-truth start/end of the flaw.

Run from the project root:
    python -m dataset_builder.build_dataset
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import librosa
import soundfile as sf

from dataset_builder.perturb import SEVERITY, SR, inject
from dataset_builder.windows import (
    Window,
    candidate_windows,
    load_words,
    pick_pause_point,
    pick_stumble_word,
    select_windows,
)

DEFAULT_SPEECHES = [
    "kennedy_inaugural",
    "eisenhower_farewell",
    "fdr_fireside_01",
    "lincoln_gettysburg",
]
FLAW_ORDER = ["pace", "slow", "pause", "volume", "monotone", "stumble"]
SEVERITIES = ("subtle", "moderate", "severe")
DEFAULT_SEED = 42

MANIFEST_COLUMNS = [
    "clip_id", "speech_id", "file", "is_baseline", "flaw_type", "severity", "parameter",
    "flaw_start", "flaw_end", "window_start", "window_end", "anchor_word",
    "window_text", "duration_s", "baseline_file",
]


def plan_flaws(
    words: list[dict], speech_id: str, seed: int, min_s: float, max_s: float
) -> list[dict]:
    """Decide where each flaw type goes in this speech.

    Windows are spread evenly across the speech; the assignment of flaw types
    to windows is shuffled with a seed so no flaw always sits in the same place.
    Returns one plan dict per flaw type (in FLAW_ORDER).
    """
    try:
        windows = select_windows(candidate_windows(words, min_s, max_s), len(FLAW_ORDER))
    except ValueError as err:
        raise ValueError(f"{speech_id}: {err} Try --min-window 3 or --max-window 12.") from err
    order = FLAW_ORDER[:]
    random.Random(f"{seed}:{speech_id}").shuffle(order)
    assigned: dict[str, Window] = dict(zip(order, windows))

    plans = []
    for flaw in FLAW_ORDER:
        win = assigned[flaw]
        anchor = ""
        if flaw == "pause":
            start, anchor = pick_pause_point(win)
            end = start  # pause is a point insertion; inject() ignores `end`
        elif flaw == "stumble":
            word = pick_stumble_word(win)
            start, end, anchor = word["start"], word["end"], word["word"]
        else:
            start, end = win.start, win.end
        plans.append({"flaw": flaw, "window": win, "start": start, "end": end, "anchor": anchor})
    return plans


def build_speech(
    speech_id: str, raw_dir: Path, align_dir: Path, out_dir: Path,
    seed: int, min_s: float, max_s: float,
) -> list[dict]:
    """Write all flawed clips for one speech and return its manifest rows."""
    baseline_path = raw_dir / f"{speech_id}.wav"
    words = load_words(align_dir / f"{speech_id}_alignment.json")
    audio, _ = librosa.load(baseline_path, sr=SR, mono=True)

    rows = [{
        "clip_id": speech_id, "speech_id": speech_id, "file": baseline_path.as_posix(),
        "is_baseline": True, "flaw_type": "none", "severity": "none", "parameter": "",
        "flaw_start": "", "flaw_end": "", "window_start": "", "window_end": "",
        "anchor_word": "", "window_text": "", "duration_s": round(len(audio) / SR, 3),
        "baseline_file": baseline_path.as_posix(),
    }]

    speech_dir = out_dir / speech_id
    speech_dir.mkdir(parents=True, exist_ok=True)
    for plan in plan_flaws(words, speech_id, seed, min_s, max_s):
        for severity in SEVERITIES:
            try:
                flawed, label = inject(audio, plan["flaw"], severity, plan["start"], plan["end"])
            except ValueError as err:
                raise RuntimeError(
                    f"{speech_id} {plan['flaw']} {severity} at "
                    f"{plan['start']:.2f}-{plan['end']:.2f}s failed: {err}"
                ) from err
            clip_id = f"{speech_id}__{plan['flaw']}_{severity}"
            out_path = speech_dir / f"{clip_id}.wav"
            sf.write(out_path, flawed, SR, subtype="PCM_16")
            win = plan["window"]
            rows.append({
                "clip_id": clip_id, "speech_id": speech_id, "file": out_path.as_posix(),
                "is_baseline": False, "flaw_type": plan["flaw"], "severity": severity,
                "parameter": label.parameter, "flaw_start": label.start, "flaw_end": label.end,
                "window_start": round(win.start, 3), "window_end": round(win.end, 3),
                "anchor_word": plan["anchor"], "window_text": win.text,
                "duration_s": round(len(flawed) / SR, 3), "baseline_file": baseline_path.as_posix(),
            })
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    """Write manifest rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    speeches: list[str], raw_dir: Path, align_dir: Path, out_dir: Path, manifest_path: Path,
    seed: int = DEFAULT_SEED, min_s: float = 4.0, max_s: float = 9.0,
) -> list[dict]:
    """Build every flawed clip for every speech and write the manifest."""
    rows: list[dict] = []
    for speech_id in speeches:
        speech_rows = build_speech(speech_id, raw_dir, align_dir, out_dir, seed, min_s, max_s)
        print(f"{speech_id}: {len(speech_rows) - 1} flawed clips")
        rows.extend(speech_rows)
    write_manifest(rows, manifest_path)
    return rows


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build the contrastive speech dataset.")
    parser.add_argument("--speeches", nargs="+", default=DEFAULT_SPEECHES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_audio"))
    parser.add_argument("--align-dir", type=Path, default=Path("data/processed_metrics"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/perturbed"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--min-window", type=float, default=4.0, help="shortest flaw window (s)")
    parser.add_argument("--max-window", type=float, default=9.0, help="longest flaw window (s)")
    args = parser.parse_args()

    rows = build_dataset(
        args.speeches, args.raw_dir, args.align_dir, args.out_dir, args.manifest,
        args.seed, args.min_window, args.max_window,
    )
    flawed = sum(1 for r in rows if not r["is_baseline"])
    print(f"\nDone: {flawed} flawed clips + {len(rows) - flawed} baselines -> {args.manifest}")
    print(f"Flaw types: {', '.join(SEVERITY)}")


if __name__ == "__main__":
    main()