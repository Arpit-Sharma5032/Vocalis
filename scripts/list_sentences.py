"""Print sentences with start/end times from an alignment JSON.

Usage:
    python scripts/list_sentences.py data/processed_metrics/kennedy_inaugural_alignment.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SENTENCE_END = (".", "?", "!")


def sentences(words: list[dict]) -> list[tuple[float, float, str]]:
    """Group words into sentences ending at . ? ! and return (start, end, text)."""
    out, current = [], []
    for w in words:
        current.append(w)
        if w["word"].endswith(SENTENCE_END):
            out.append((current[0]["start"], current[-1]["end"], " ".join(x["word"] for x in current)))
            current = []
    if current:
        out.append((current[0]["start"], current[-1]["end"], " ".join(x["word"] for x in current)))
    return out


def main() -> None:
    """Load the JSON given on the command line and print each sentence."""
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    words = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for i, (start, end, text) in enumerate(sentences(words)):
        print(f"[{i:2d}] {start:6.2f} -> {end:6.2f} ({end - start:4.1f}s)  {text[:80]}")


if __name__ == "__main__":
    main()