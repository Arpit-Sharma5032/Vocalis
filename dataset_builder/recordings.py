"""Plan, guide and label our own real (non-synthetic) validation recordings.

We read the same transcript three times: once well ("good") and twice with
flaws performed on purpose at planned sentences (moderate, then subtle).
Flaw regions are labelled from forced alignment of each take.

Run from the project root:
    python -m dataset_builder.recordings plan     # print the cue sheet
    python -m dataset_builder.recordings label    # build labels.csv after alignment
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from dataset_builder.windows import ends_phrase, load_words

SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")
FLAWS = ("pace", "monotone", "pause", "volume")


def takes(prefix: str = "own") -> tuple[str, str, str]:
    """File stems of the three takes for a recording set, e.g. own_good, own_flawed_moderate, ..."""
    return (f"{prefix}_good", f"{prefix}_flawed_moderate", f"{prefix}_flawed_subtle")


LABEL_COLUMNS = [
    "clip_id", "file", "flaw_type", "severity", "flaw_start", "flaw_end",
    "sentence_index", "sentence_text",
]

INSTRUCTIONS = {
    "pace": {
        "moderate": "RUSH it: read about 1.5x faster than your normal pace.",
        "subtle": "Slightly hurried: about 1.15x faster than normal.",
    },
    "monotone": {
        "moderate": "Read it FLAT, like a robot: as little pitch movement as you can manage.",
        "subtle": "Dull it down: about half your usual pitch movement.",
    },
    "pause": {
        "moderate": "Stop for about 1.2 s in the MIDDLE of the sentence, where there is no punctuation "
                    "(count 'one-thousand-one').",
        "subtle": "A hesitation of about 0.6 s mid-sentence, where there is no punctuation.",
    },
    "volume": {
        "moderate": "Speak noticeably QUIETER (about half as loud), same distance from the mic.",
        "subtle": "Slightly quieter than normal (just a little).",
    },
}


def split_sentences(text: str) -> list[str]:
    """Split a transcript into sentences at . ? ! followed by whitespace."""
    return [s for s in SENTENCE_SPLIT.split(" ".join(text.split())) if s]


def token_ranges(sentences: list[str]) -> list[tuple[int, int]]:
    """Inclusive (first, last) whitespace-token index of each sentence."""
    ranges, pos = [], 0
    for s in sentences:
        n = len(s.split())
        ranges.append((pos, pos + n - 1))
        pos += n
    return ranges


def assign_flaws(sentences: list[str], min_words: int = 8) -> dict[int, str]:
    """Choose one sentence per flaw, spread evenly through the speech.

    Preference order: inner sentences with >= min_words words; then any sentence
    with >= min_words words (short texts); then any sentence with >= 5 words.
    """
    n = len(FLAWS)
    attempts = [(range(1, len(sentences) - 1), min_words), (range(len(sentences)), min_words),
                (range(len(sentences)), 5)]
    for candidates, words_needed in attempts:
        eligible = [i for i in candidates if len(sentences[i].split()) >= words_needed]
        if len(eligible) >= n:
            break
    else:
        raise ValueError(f"Need {n} sentences of >= 5 words in the transcript, found {len(eligible)}.")
    picks = [eligible[round(k * (len(eligible) - 1) / (n - 1))] for k in range(n)]
    return dict(zip(picks, FLAWS))


def make_plan(transcript_path: Path) -> dict:
    """Build the recording plan from a transcript file."""
    sentences = split_sentences(transcript_path.read_text(encoding="utf-8"))
    assignments = assign_flaws(sentences)
    return {
        "transcript": transcript_path.as_posix(),
        "sentences": sentences,
        "token_ranges": token_ranges(sentences),
        "assignments": {str(i): flaw for i, flaw in assignments.items()},
    }


def print_cue_sheet(plan: dict) -> None:
    """Print the sentences and, for each take, what to do on which sentence."""
    print("\nREAD THE WHOLE TRANSCRIPT, word for word, in every take. Never skip or add words.\n")
    for i, sentence in enumerate(plan["sentences"]):
        flaw = plan["assignments"].get(str(i))
        mark = f"   <-- FLAW SENTENCE ({flaw})" if flaw else ""
        print(f"[{i}] {sentence}{mark}")
    for severity in ("moderate", "subtle"):
        print(f"\n=== TAKE own_flawed_{severity} ===")
        for i, flaw in sorted((int(k), v) for k, v in plan["assignments"].items()):
            print(f"  sentence [{i}] ({flaw}): {INSTRUCTIONS[flaw][severity]}")
    print("\n=== TAKE own_good ===\n  Read everything clearly and naturally, with no deliberate flaws.")


def find_pause_gap(words: list[dict], first: int, last: int) -> tuple[float, float]:
    """Largest gap between consecutive words inside a sentence, ignoring punctuation breaks.

    Returns (gap_start, gap_end) in seconds. Falls back to any gap if every
    word pair is separated by punctuation.
    """
    pairs = [(words[i], words[i + 1]) for i in range(first, last)]
    plain = [(a, b) for a, b in pairs if not ends_phrase(a["word"])]
    a, b = max(plain or pairs, key=lambda p: p[1]["start"] - p[0]["end"])
    return a["end"], b["start"]


def label_take(take_id: str, plan: dict, words: list[dict], audio_dir: Path) -> list[dict]:
    """Make label rows for one aligned take."""
    total_tokens = plan["token_ranges"][-1][1] + 1
    if len(words) != total_tokens:
        raise ValueError(
            f"{take_id}: alignment has {len(words)} words but the transcript has {total_tokens}. "
            "The take does not match the transcript word for word."
        )
    file = (audio_dir / f"{take_id}.wav").as_posix()
    severity = "none" if take_id.endswith("_good") else take_id.rsplit("_", 1)[-1]
    if severity == "none":
        return [{"clip_id": take_id, "file": file, "flaw_type": "none", "severity": "none",
                 "flaw_start": "", "flaw_end": "", "sentence_index": "", "sentence_text": ""}]
    rows = []
    for key, flaw in plan["assignments"].items():
        i = int(key)
        first, last = plan["token_ranges"][i]
        if flaw == "pause":
            start, end = find_pause_gap(words, first, last)
        else:
            start, end = words[first]["start"], words[last]["end"]
        rows.append({"clip_id": take_id, "file": file, "flaw_type": flaw, "severity": severity,
                     "flaw_start": round(start, 3), "flaw_end": round(end, 3),
                     "sentence_index": i, "sentence_text": plan["sentences"][i]})
    return rows


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Own-recording planner and labeller.")
    parser.add_argument("command", choices=["plan", "label"])
    parser.add_argument("--transcript", type=Path, default=Path("data/transcripts/lincoln_gettysburg.txt"))
    parser.add_argument("--dir", type=Path, default=Path("data/own_recordings"))
    parser.add_argument("--prefix", default="own", help="file-name prefix of the takes (own, ownb, ...)")
    parser.add_argument("--align-dir", type=Path, default=Path("data/processed_metrics"))
    args = parser.parse_args()
    plan_path = args.dir / "plan.json"

    if args.command == "plan":
        args.dir.mkdir(parents=True, exist_ok=True)
        plan = make_plan(args.transcript)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print_cue_sheet(plan)
        print(f"\nPlan saved to {plan_path}")
        return

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for take in takes(args.prefix):
        align = args.align_dir / f"{take}_alignment.json"
        if not align.exists():
            print(f"skipping {take}: no alignment yet ({align})")
            continue
        rows.extend(label_take(take, plan, load_words(align), args.dir))
        print(f"labelled {take}")
    out = args.dir / "labels.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()