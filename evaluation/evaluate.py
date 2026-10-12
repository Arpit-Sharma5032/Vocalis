"""Evaluate flaw detection against ground truth.

Two test beds:
  * synthetic clips (data/manifest.csv): every flaw's exact start/end is known;
  * our own real recordings (data/own_recordings/labels.csv): performed flaws,
    read by a different speaker than the baseline (a real speaker-agnostic test).

A flaw counts as DETECTED if a predicted region of the same type overlaps it with
IoU >= 0.3. A predicted region counts as a FALSE POSITIVE if it overlaps no
labelled flaw at all.

Usage (from the project root):
    python -m evaluation.evaluate --alignment whisperx   # real aligner output
    python -m evaluation.evaluate --alignment oracle     # exact word times (upper bound)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

from backend.detection import Region, Thresholds, analyze
from backend.features import FrameFeatures, extract_features, load_audio
from dataset_builder.oracle import oracle_alignment
from dataset_builder.windows import load_words

IOU_HIT = 0.3
SEVERITY_ORDER = ["subtle", "moderate", "severe"]
_feature_cache: dict[str, FrameFeatures] = {}


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection over union of two time intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if two intervals share any time."""
    return min(a[1], b[1]) > max(a[0], b[0])


def features_for(path: str) -> FrameFeatures:
    """Extract (and cache) features for an audio file."""
    if path not in _feature_cache:
        _feature_cache[path] = extract_features(load_audio(path))
    return _feature_cache[path]


def score_clip(
    regions: list[Region], truths: list[tuple[str, float, float, str]], meta: dict
) -> tuple[list[dict], int, int]:
    """Score predicted regions against labelled flaws for one clip.

    truths: (flaw_type, start, end, severity). Returns (per-flaw rows, n_false_positives, n_predicted).
    """
    rows = []
    for ftype, start, end, severity in truths:
        same = [r for r in regions if r.flaw_type == ftype]
        best = max((iou((r.start, r.end), (start, end)) for r in same), default=0.0)
        confused = sorted({r.flaw_type for r in regions
                           if r.flaw_type != ftype and overlaps((r.start, r.end), (start, end))})
        rows.append({**meta, "flaw_type": ftype, "severity": severity, "gt_start": start, "gt_end": end,
                     "detected": best >= IOU_HIT, "best_iou": best, "confused_with": ",".join(confused)})
    fps = sum(1 for r in regions if not any(overlaps((r.start, r.end), (s, e)) for _, s, e, _ in truths))
    return rows, fps, len(regions)


def evaluate_synthetic(args: argparse.Namespace, th: Thresholds) -> tuple[list[dict], list[dict]]:
    """Run detection on every synthetic flawed clip in the manifest."""
    results, clip_stats = [], []
    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    for row in manifest:
        if row["is_baseline"] == "True":
            continue
        speech = row["speech_id"]
        base_words = load_words(args.align_dir / f"{speech}_alignment.json")
        if args.alignment == "oracle":
            part_words = oracle_alignment(base_words, row)
        else:
            path = args.clip_align_dir / f"{row['clip_id']}_alignment.json"
            if not path.exists():
                print(f"  missing alignment: {path.name}")
                continue
            try:
                part_words = load_words(path)
            except ValueError as err:
                print(f"  alignment failed for {row['clip_id']}: {err}")
                continue
        regions = analyze(features_for(row["baseline_file"]), base_words, features_for(row["file"]), part_words, th)
        truth = [(row["flaw_type"], float(row["flaw_start"]), float(row["flaw_end"]), row["severity"])]
        rows, fps, n_pred = score_clip(regions, truth, {"source": "synthetic", "clip_id": row["clip_id"], "speech_id": speech})
        results.extend(rows)
        clip_stats.append({"source": "synthetic", "clip_id": row["clip_id"], "false_positives": fps, "predicted": n_pred})
    return results, clip_stats


def evaluate_own(args: argparse.Namespace, th: Thresholds) -> tuple[list[dict], list[dict]]:
    """Run detection on our own real recordings (a different speaker from the baseline)."""
    labels_path = args.own_dir / "labels.csv"
    if not labels_path.exists():
        return [], []
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(labels_path.open(encoding="utf-8")):
        by_clip[row["clip_id"]].append(row)
    base_path = Path(args.own_baseline)
    base_words = load_words(args.align_dir / f"{base_path.stem}_alignment.json")
    base_feats = features_for(base_path.as_posix())
    results, clip_stats = [], []
    for clip_id, rows in by_clip.items():
        align = args.align_dir / f"{clip_id}_alignment.json"
        if not align.exists():
            continue
        regions = analyze(base_feats, base_words, features_for(rows[0]["file"]), load_words(align), th)
        truths = [(r["flaw_type"], float(r["flaw_start"]), float(r["flaw_end"]), r["severity"])
                  for r in rows if r["flaw_type"] != "none"]
        flaw_rows, fps, n_pred = score_clip(regions, truths, {"source": "own", "clip_id": clip_id, "speech_id": base_path.stem})
        results.extend(flaw_rows)
        clip_stats.append({"source": "own", "clip_id": clip_id, "false_positives": fps, "predicted": n_pred})
    return results, clip_stats


def summarize(results: list[dict], clip_stats: list[dict]) -> str:
    """Build a Markdown summary of the evaluation."""
    out = []
    df = pd.DataFrame(results)
    cs = pd.DataFrame(clip_stats)
    for source in ("synthetic", "own"):
        sub = df[df["source"] == source] if not df.empty else df
        if sub.empty:
            continue
        out.append(f"\n### {source.capitalize()} clips: detection rate (IoU >= {IOU_HIT})\n")
        recall = sub.pivot_table(index="flaw_type", columns="severity", values="detected", aggfunc="mean")
        recall = recall.reindex(columns=[c for c in SEVERITY_ORDER if c in recall.columns])
        out.append((recall * 100).round(0).astype("Int64").astype(str).add("%").to_markdown())
        out.append(f"\n### {source.capitalize()} clips: mean IoU of detected flaws\n")
        hit = sub[sub["detected"]]
        if not hit.empty:
            iou_t = hit.pivot_table(index="flaw_type", columns="severity", values="best_iou", aggfunc="mean")
            iou_t = iou_t.reindex(columns=[c for c in SEVERITY_ORDER if c in iou_t.columns])
            out.append(iou_t.round(2).to_markdown())
        c = cs[cs["source"] == source]
        out.append(
            f"\nOverall detection rate {sub['detected'].mean() * 100:.0f}% of {len(sub)} flaws; "
            f"{int(c['false_positives'].sum())} false-positive regions over {len(c)} clips "
            f"({c['false_positives'].mean():.2f} per clip)."
        )
    return "\n".join(out)


def main() -> None:
    """Run the evaluation and write results."""
    p = argparse.ArgumentParser(description="Evaluate flaw detection.")
    p.add_argument("--alignment", choices=["whisperx", "oracle"], default="whisperx")
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    p.add_argument("--align-dir", type=Path, default=Path("data/processed_metrics"))
    p.add_argument("--clip-align-dir", type=Path, default=Path("data/processed_metrics/clips"))
    p.add_argument("--own-dir", type=Path, default=Path("data/own_recordings"))
    p.add_argument("--own-baseline", default="data/raw_audio/lincoln_gettysburg.wav")
    p.add_argument("--out", type=Path, default=Path("evaluation/results"))
    args = p.parse_args()

    th = Thresholds()
    results, clip_stats = evaluate_synthetic(args, th)
    own_results, own_stats = evaluate_own(args, th)
    results += own_results
    clip_stats += own_stats
    if not results:
        raise SystemExit("No results: check that alignments exist for the clips.")

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out / f"detection_results_{args.alignment}.csv", index=False)
    summary = summarize(results, clip_stats)
    (args.out / f"summary_{args.alignment}.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()