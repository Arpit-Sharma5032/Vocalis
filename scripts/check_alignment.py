"""Sanity-check alignment JSON files: missing timestamps, bad order, large gaps.

Usage:
    python scripts/check_alignment.py            # checks every alignment file
    python scripts/check_alignment.py <file>     # checks one file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GAP_WARN_S = 1.5  # a silence longer than this between words is worth a look


def check(path: Path) -> bool:
    """Print a summary for one alignment file. Return True if it looks healthy."""
    words = json.loads(path.read_text(encoding="utf-8"))
    timed = [w for w in words if w["start"] is not None and w["end"] is not None]
    missing = [w["word"] for w in words if w not in timed]
    backwards = [w["word"] for w in timed if w["end"] < w["start"]]
    gaps = [
        (b["start"] - a["end"], a["word"], b["word"])
        for a, b in zip(timed, timed[1:])
        if b["start"] - a["end"] > GAP_WARN_S
    ]

    print(f"\n{path.name}")
    print(f"  words: {len(words)}  missing timestamps: {len(missing)}  backwards: {len(backwards)}")
    if timed:
        print(f"  spans {timed[0]['start']:.2f}s -> {timed[-1]['end']:.2f}s")
    for word in missing:
        print(f"  MISSING: {word}")
    for gap, before, after in gaps:
        print(f"  long gap {gap:.2f}s between '{before}' and '{after}'")
    return not missing and not backwards


def main() -> None:
    """Check the given file, or all files in data/processed_metrics."""
    if len(sys.argv) > 1:
        files = [Path(sys.argv[1])]
    else:
        files = sorted(Path("data/processed_metrics").glob("*_alignment.json"))
    results = [check(f) for f in files]
    print("\nAll files look healthy." if all(results) else "\nSome files need attention.")


if __name__ == "__main__":
    main()