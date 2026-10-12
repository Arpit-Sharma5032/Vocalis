"""Calibrate detection thresholds with cross-speech validation.

Thresholds are tuned on some speeches and scored on speeches the tuning never
saw, then the roles swap (2-fold, by speech). The reported numbers are the
HELD-OUT ones. The thresholds finally saved are tuned on all speeches, so
their own in-sample score is optimistic and is not reported as a result.

Each threshold only affects its own flaw types, so each is tuned separately by
maximising F1 over a grid:
    pace_log        -> pace, slow
    pitch_log       -> monotone
    energy_db       -> volume
    pause_excess_s  -> pause

Usage (from the project root):
    python -m evaluation.calibrate --alignment whisperx
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from backend.detection import Deviations, Thresholds, compute_deviations, regions_from_deviations
from dataset_builder.oracle import oracle_alignment
from dataset_builder.windows import load_words
from evaluation.evaluate import IOU_HIT, features_for, iou, overlaps

GRIDS = {
    "pace_log": [math.log(x) for x in (1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50)],
    "pitch_log": [math.log(x) for x in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)],
    "energy_db": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "pause_excess_s": [0.15, 0.20, 0.30, 0.40, 0.50],
}
RELEVANT = {
    "pace_log": {"pace", "slow"},
    "pitch_log": {"monotone"},
    "energy_db": {"volume"},
    "pause_excess_s": {"pause"},
}


@dataclass
class Item:
    """One flawed clip with its precomputed deviations and ground truth."""

    speech: str
    flaw_type: str
    start: float
    end: float
    dev: Deviations


def f1_score(tp: int, fp: int, fn: int) -> float:
    """F1 = 2TP / (2TP + FP + FN); 0 when there is nothing to score."""
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def pick_best(scores: dict[float, float], param: str) -> float:
    """Value with the highest F1; ties go to the more conservative threshold."""
    sign = -1.0 if param == "pitch_log" else 1.0  # more negative pitch_log = stricter
    return max(scores, key=lambda v: (round(scores[v], 6), sign * v))


def counts(items: list[Item], param: str, value: float, base: Thresholds) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) for one threshold value."""
    th = replace(base, **{param: value})
    kinds = RELEVANT[param]
    tp = fp = n_gt = 0
    for it in items:
        regions = [r for r in regions_from_deviations(it.dev, th) if r.flaw_type in kinds]
        if it.flaw_type in kinds:
            n_gt += 1
            if any(r.flaw_type == it.flaw_type and iou((r.start, r.end), (it.start, it.end)) >= IOU_HIT
                   for r in regions):
                tp += 1
        fp += sum(1 for r in regions if not overlaps((r.start, r.end), (it.start, it.end)))
    return tp, fp, n_gt - tp


def tune(items: list[Item], base: Thresholds) -> dict[str, float]:
    """Best value for every threshold on these items."""
    best = {}
    for param, grid in GRIDS.items():
        scores = {v: f1_score(*counts(items, param, v, base)) for v in grid}
        best[param] = pick_best(scores, param)
    return best


def load_items(args: argparse.Namespace, window_words: int) -> list[Item]:
    """Compute deviations for every synthetic flawed clip."""
    items = []
    for row in csv.DictReader(args.manifest.open(encoding="utf-8")):
        if row["is_baseline"] == "True":
            continue
        base_words = load_words(args.align_dir / f"{row['speech_id']}_alignment.json")
        if args.alignment == "oracle":
            part_words = oracle_alignment(base_words, row)
        else:
            path = args.clip_align_dir / f"{row['clip_id']}_alignment.json"
            try:
                part_words = load_words(path)
            except (FileNotFoundError, ValueError):
                continue
        dev = compute_deviations(features_for(row["baseline_file"]), base_words,
                                 features_for(row["file"]), part_words, window_words)
        items.append(Item(row["speech_id"], row["flaw_type"], float(row["flaw_start"]),
                          float(row["flaw_end"]), dev))
    return items


def main() -> None:
    """Run 2-fold cross-speech calibration and save the final thresholds."""
    p = argparse.ArgumentParser(description="Calibrate detection thresholds.")
    p.add_argument("--alignment", choices=["whisperx", "oracle"], default="whisperx")
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    p.add_argument("--align-dir", type=Path, default=Path("data/processed_metrics"))
    p.add_argument("--clip-align-dir", type=Path, default=Path("data/processed_metrics/clips"))
    p.add_argument("--out", type=Path, default=Path("evaluation/results"))
    args = p.parse_args()

    base = Thresholds()
    items = load_items(args, base.window_words)
    speeches = sorted({it.speech for it in items})
    if len(speeches) < 2:
        raise SystemExit("Need at least two speeches for cross-speech validation.")
    half = len(speeches) // 2
    folds = [(speeches[:half], speeches[half:]), (speeches[half:], speeches[:half])]

    lines = [f"# Threshold calibration ({args.alignment} alignment)\n",
             "Held-out F1 per threshold: tuned on one group of speeches, scored on the other.\n",
             "| threshold | default | calibrated (per fold) | held-out F1 default | held-out F1 calibrated |",
             "|---|---|---|---|---|"]
    default = asdict(base)
    for param in GRIDS:
        chosen, f_def, f_cal = [], [], []
        for dev_sp, test_sp in folds:
            dev_items = [i for i in items if i.speech in dev_sp]
            test_items = [i for i in items if i.speech in test_sp]
            value = tune(dev_items, base)[param]
            chosen.append(value)
            f_def.append(f1_score(*counts(test_items, param, default[param], base)))
            f_cal.append(f1_score(*counts(test_items, param, value, base)))
        show = (lambda v: f"{math.exp(v):.2f}x" if param.endswith("_log") else f"{v:g}")
        lines.append(f"| {param} | {show(default[param])} | {', '.join(show(v) for v in chosen)} | "
                     f"{sum(f_def) / 2:.2f} | {sum(f_cal) / 2:.2f} |")

    final = tune(items, base)
    lines.append("\nFinal thresholds (tuned on all speeches; their in-sample score is optimistic):\n")
    lines.append("```json\n" + json.dumps({**default, **final}, indent=2) + "\n```")
    report = "\n".join(lines)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"calibration_{args.alignment}.md").write_text(report, encoding="utf-8")
    (args.out / f"thresholds_{args.alignment}.json").write_text(json.dumps({**default, **final}, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()