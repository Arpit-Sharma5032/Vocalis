"""Align every flawed clip in the manifest with WhisperX (model loaded once).

Resumable: clips that already have an alignment file are skipped.

Usage (from the project root):
    python -m scripts.align_clips
    python -m scripts.align_clips --force     # redo everything
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from backend.alignment import align_with_model, load_aligner
from dataset_builder.oracle import clip_transcript
from dataset_builder.windows import load_words


def main() -> None:
    """Align all flawed clips and write one JSON file per clip."""
    parser = argparse.ArgumentParser(description="Batch-align flawed clips.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--align-dir", type=Path, default=Path("data/processed_metrics"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed_metrics/clips"))
    parser.add_argument("--force", action="store_true", help="re-align clips that already have output")
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(args.manifest.open(encoding="utf-8")) if r["is_baseline"] != "True"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if args.force or not (args.out_dir / f"{r['clip_id']}_alignment.json").exists()]
    print(f"{len(rows)} flawed clips, {len(todo)} to align")
    if not todo:
        return

    model, metadata = load_aligner()
    for n, row in enumerate(todo, 1):
        base_words = load_words(args.align_dir / f"{row['speech_id']}_alignment.json")
        transcript = clip_transcript(base_words, row)
        started = time.time()
        words = align_with_model(row["file"], transcript, model, metadata)
        out = args.out_dir / f"{row['clip_id']}_alignment.json"
        out.write_text(json.dumps(words, indent=2), encoding="utf-8")
        missing = sum(1 for w in words if w["start"] is None or w["end"] is None)
        print(f"[{n}/{len(todo)}] {row['clip_id']}: {len(words)} words, {missing} missing, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()