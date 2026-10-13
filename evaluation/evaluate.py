"""Evaluate flaw detection against ground truth.

Test beds:
  * synthetic clips (data/manifest.csv): every flaw's exact start/end is known;
  * sets of our own real recordings (--own-set DIR=BASELINE_WAV, each with a
    labels.csv): performed flaws, read by a different speaker than the baseline
    (a real speaker-agnostic test).

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
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backend.detection import Region, Thresholds, analyze, load_thresholds
from backend.features import HOP_S, FrameFeatures, extract_features, load_audio
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
) -> tuple[list[dict], list[Region], int]:
    """Score predicted regions against labelled flaws for one clip.

    truths: (flaw_type, start, end, severity).
    Returns (per-flaw rows, false-positive regions, number of predicted regions).
    """
    rows = []
    for ftype, start, end, severity in truths:
        same = [r for r in regions if r.flaw_type == ftype]
        best = max((iou((r.start, r.end), (start, end)) for r in same), default=0.0)
        confused = sorted({r.flaw_type for r in regions
                           if r.flaw_type != ftype and overlaps((r.start, r.end), (start, end))})
        rows.append({**meta, "flaw_type": ftype, "severity": severity, "gt_start": start, "gt_end": end,
                     "detected": best >= IOU_HIT, "best_iou": best, "confused_with": ",".join(confused)})
    fps = [r for r in regions if not any(overlaps((r.start, r.end), (s, e)) for _, s, e, _ in truths)]
    return rows, fps, len(regions)


def _record(source: str, clip_id: str, feats: FrameFeatures, fps: list[Region], n_pred: int,
            clip_stats: list[dict], fp_rows: list[dict]) -> None:
    """Store per-clip false-positive statistics."""
    clip_stats.append({"source": source, "clip_id": clip_id, "false_positives": len(fps), "predicted": n_pred,
                       "minutes": len(feats.times) * HOP_S / 60})
    fp_rows.extend({"source": source, "clip_id": clip_id, "predicted_type": r.flaw_type,
                    "start": round(r.start, 3), "end": round(r.end, 3), "score": round(r.score, 3)} for r in fps)


def evaluate_synthetic(args: argparse.Namespace, th: Thresholds) -> tuple[list[dict], list[dict], list[dict]]:
    """Run detection on every synthetic flawed clip in the manifest."""
    results, clip_stats, fp_rows = [], [], []
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
        part_feats = features_for(row["file"])
        regions = analyze(features_for(row["baseline_file"]), base_words, part_feats, part_words, th)
        truth = [(row["flaw_type"], float(row["flaw_start"]), float(row["flaw_end"]), row["severity"])]
        rows, fps, n_pred = score_clip(regions, truth, {"source": "synthetic", "clip_id": row["clip_id"], "speech_id": speech})
        results.extend(rows)
        _record("synthetic", row["clip_id"], part_feats, fps, n_pred, clip_stats, fp_rows)
    return results, clip_stats, fp_rows


def evaluate_own(args: argparse.Namespace, th: Thresholds, own_dir: Path, own_baseline: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Run detection on one set of our own real recordings (a different speaker from the baseline)."""
    source = f"own:{own_dir.name}"
    labels_path = own_dir / "labels.csv"
    if not labels_path.exists():
        return [], [], []
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(labels_path.open(encoding="utf-8")):
        by_clip[row["clip_id"]].append(row)
    base_path = Path(own_baseline)
    base_words = load_words(args.align_dir / f"{base_path.stem}_alignment.json")
    base_feats = features_for(base_path.as_posix())
    results, clip_stats, fp_rows = [], [], []
    for clip_id, rows in by_clip.items():
        align = args.align_dir / f"{clip_id}_alignment.json"
        if not align.exists():
            continue
        part_feats = features_for(rows[0]["file"])
        regions = analyze(base_feats, base_words, part_feats, load_words(align), th)
        truths = [(r["flaw_type"], float(r["flaw_start"]), float(r["flaw_end"]), r["severity"])
                  for r in rows if r["flaw_type"] != "none"]
        flaw_rows, fps, n_pred = score_clip(regions, truths, {"source": source, "clip_id": clip_id, "speech_id": base_path.stem})
        results.extend(flaw_rows)
        _record(source, clip_id, part_feats, fps, n_pred, clip_stats, fp_rows)
    return results, clip_stats, fp_rows


def summarize(results: list[dict], clip_stats: list[dict], fp_rows: list[dict]) -> str:
    """Build a Markdown summary of the evaluation."""
    out = []
    df = pd.DataFrame(results)
    cs = pd.DataFrame(clip_stats)
    sources = sorted(df["source"].unique(), key=lambda x: (x != "synthetic", x)) if not df.empty else []
    for source in sources:
        sub = df[df["source"] == source]
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
        minutes = c["minutes"].sum()
        out.append(
            f"\nOverall detection rate {sub['detected'].mean() * 100:.0f}% of {len(sub)} flaws; "
            f"{int(c['false_positives'].sum())} false-positive regions over {len(c)} clips "
            f"({c['false_positives'].mean():.2f} per clip, {c['false_positives'].sum() / minutes:.2f} per minute of audio)."
        )
        fp = pd.DataFrame(fp_rows)
        fp = fp[fp["source"] == source] if not fp.empty else fp
        if not fp.empty:
            by_type = fp.groupby("predicted_type").size().rename("false positives").to_frame()
            by_type["per minute"] = (by_type["false positives"] / minutes).round(2)
            out.append(f"\n### {source.capitalize()} clips: false positives by predicted type\n")
            out.append(by_type.to_markdown())
            if source.startswith("own"):
                per_clip = fp.groupby("clip_id").size().rename("false positives").to_frame().to_markdown()
                out.append("\nBy recording:\n\n" + per_clip)
    return "\n".join(out)


def main() -> None:
    """Run the evaluation and write results."""
    p = argparse.ArgumentParser(description="Evaluate flaw detection.")
    p.add_argument("--alignment", choices=["whisperx", "oracle"], default="whisperx")
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    p.add_argument("--align-dir", type=Path, default=Path("data/processed_metrics"))
    p.add_argument("--clip-align-dir", type=Path, default=Path("data/processed_metrics/clips"))
    p.add_argument("--own-set", action="append", default=None, metavar="DIR=BASELINE_WAV",
                   help="a set of own recordings and the baseline it reads (repeatable). "
                        "Default: data/own_recordings=data/raw_audio/lincoln_gettysburg.wav")
    p.add_argument("--out", type=Path, default=Path("evaluation/results"))
    p.add_argument("--thresholds", type=Path, default=None,
                   help="JSON from evaluation.calibrate; default = untuned built-in thresholds")
    p.add_argument("--tag", default="", help="suffix for output file names")
    p.add_argument("--no-refine", action="store_true", help="skip silence-aware word boundary refinement")
    p.add_argument("--noise-k", type=float, default=None, help="override the noise-adaptive threshold factor")
    args = p.parse_args()

    th = load_thresholds(str(args.thresholds)) if args.thresholds else Thresholds()
    if args.no_refine:
        th = replace(th, refine_boundaries=False)
    if args.noise_k is not None:
        th = replace(th, noise_k=args.noise_k)
    
    results, clip_stats, fp_rows = evaluate_synthetic(args, th)
    for spec in args.own_set or ["data/own_recordings=data/raw_audio/lincoln_gettysburg.wav"]:
        own_dir, own_baseline = spec.split("=", 1)
        own_results, own_stats, own_fps = evaluate_own(args, th, Path(own_dir), own_baseline)
        results += own_results
        clip_stats += own_stats
        fp_rows += own_fps
    if not results:
        raise SystemExit("No results: check that alignments exist for the clips.")

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out / f"detection_results_{args.alignment}{args.tag}.csv", index=False)
    pd.DataFrame(fp_rows, columns=["source", "clip_id", "predicted_type", "start", "end", "score"]).to_csv(
        args.out / f"false_positives_{args.alignment}{args.tag}.csv", index=False)
    summary = summarize(results, clip_stats, fp_rows)
    (args.out / f"summary_{args.alignment}{args.tag}.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()