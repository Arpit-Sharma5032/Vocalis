"""Exact word timings for a perturbed clip, derived from the baseline and the known edit.

Because we created every flaw ourselves, we know exactly how each baseline word
moved. This "oracle" alignment is used (a) for offline tests that need no
aligner and (b) as an upper bound that separates aligner error from detector
error in evaluation.
"""
from __future__ import annotations

from dataset_builder.perturb import STUMBLE_GAP_S

TOL = 2e-3  # manifest times are rounded to 1 ms


def _shifted(word: dict, delta: float) -> dict:
    return dict(word, start=word["start"] + delta, end=word["end"] + delta)


def stumble_index(words: list[dict], row: dict) -> int:
    """Index of the baseline word that was repeated (matched by its start time)."""
    start = float(row["flaw_start"])
    for i, w in enumerate(words):
        if abs(w["start"] - start) < TOL:
            return i
    raise ValueError(f"No word starts at {start}s for clip {row.get('clip_id')}")


def oracle_alignment(words: list[dict], row: dict) -> list[dict]:
    """Return the word timings of the flawed clip described by a manifest row."""
    flaw, param = row["flaw_type"], float(row["parameter"])
    if flaw in ("volume", "monotone"):
        return [dict(w) for w in words]

    if flaw in ("pace", "slow"):
        a, b = float(row["window_start"]), float(row["window_end"])
        delta = (b - a) / param - (b - a)
        out = []
        for w in words:
            if w["start"] >= a - TOL and w["end"] <= b + TOL:
                out.append(dict(w, start=a + (w["start"] - a) / param, end=a + (w["end"] - a) / param))
            elif w["start"] >= b - TOL:
                out.append(_shifted(w, delta))
            else:
                out.append(dict(w))
        return out

    if flaw == "pause":
        at = float(row["flaw_start"])
        return [_shifted(w, param) if w["start"] >= at - TOL else dict(w) for w in words]

    if flaw == "stumble":
        k, repeats = stumble_index(words, row), int(param)
        word = words[k]
        dur = word["end"] - word["start"]
        step = STUMBLE_GAP_S + dur
        out = [dict(w) for w in words[: k + 1]]
        for i in range(repeats):
            start = word["end"] + i * step + STUMBLE_GAP_S
            out.append(dict(word, start=start, end=start + dur))
        out.extend(_shifted(w, repeats * step) for w in words[k + 1 :])
        return out

    raise ValueError(f"Unknown flaw type: {flaw}")


def clip_transcript(words: list[dict], row: dict) -> str:
    """Transcript to align a flawed clip against.

    Identical to the baseline text, except that a stumble clip must include the
    repeated word(s), exactly as a verbatim transcript of what was said would.
    """
    tokens = [w["word"] for w in words]
    if row["flaw_type"] == "stumble":
        k = stumble_index(words, row)
        tokens[k + 1 : k + 1] = [words[k]["word"]] * int(float(row["parameter"]))
    return " ".join(tokens)