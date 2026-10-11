"""Choose word-aligned flaw locations from a forced-alignment JSON.

All selection is deterministic: the same alignment always yields the same
windows, so the dataset can be rebuilt exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PHRASE_END = (",", ";", ":", ".", "?", "!", "—", "–")
_TRAILING_QUOTES = "\"'”’)]"


@dataclass(frozen=True)
class Window:
    """A stretch of speech made of whole phrases, with exact word timings."""

    start: float
    end: float
    words: tuple[dict, ...]

    @property
    def text(self) -> str:
        """The words in the window as one string."""
        return " ".join(w["word"] for w in self.words)

    @property
    def midpoint(self) -> float:
        """Time at the centre of the window (seconds)."""
        return (self.start + self.end) / 2.0


def load_words(path: str | Path) -> list[dict]:
    """Load an alignment JSON and make sure every word has timestamps."""
    words = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [w["word"] for w in words if w["start"] is None or w["end"] is None]
    if missing:
        raise ValueError(f"{path}: words without timestamps: {missing[:5]}")
    return words


def ends_phrase(word: str) -> bool:
    """True if the word ends with punctuation that marks a natural pause."""
    return word.rstrip(_TRAILING_QUOTES).endswith(PHRASE_END)


def split_phrases(words: list[dict]) -> list[list[dict]]:
    """Split the word list into phrases after every punctuation-ended word."""
    phrases: list[list[dict]] = []
    current: list[dict] = []
    for w in words:
        current.append(w)
        if ends_phrase(w["word"]):
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    return phrases


def split_long_phrase(phrase: list[dict], max_s: float) -> list[list[dict]]:
    """Split a phrase longer than max_s at its largest natural gap near the middle.

    Speakers like Kennedy deliver 15-20 s sentences with almost no punctuation,
    so long phrases must be cut at the most natural pause available. Cuts are
    only considered in the middle 40% so both halves stay reasonably sized.
    """
    duration = phrase[-1]["end"] - phrase[0]["start"]
    if duration <= max_s or len(phrase) < 2:
        return [phrase]
    lo, hi = phrase[0]["start"] + 0.3 * duration, phrase[0]["start"] + 0.7 * duration
    cuts = [i for i in range(1, len(phrase)) if lo <= phrase[i]["start"] <= hi]
    pool = cuts or list(range(1, len(phrase)))
    k = max(pool, key=lambda i: phrase[i]["start"] - phrase[i - 1]["end"])
    return split_long_phrase(phrase[:k], max_s) + split_long_phrase(phrase[k:], max_s)


def candidate_windows(words: list[dict], min_s: float = 4.0, max_s: float = 9.0) -> list[Window]:
    """Return non-overlapping windows lasting min_s..max_s seconds.

    Windows start and end on word boundaries at natural pauses: punctuation
    where available, otherwise the largest gap inside an over-long phrase.
    """
    phrases = [p for ph in split_phrases(words) for p in split_long_phrase(ph, max_s)]
    windows: list[Window] = []
    i = 0
    while i < len(phrases):
        j = i
        while j < len(phrases) and phrases[j][-1]["end"] - phrases[i][0]["start"] < min_s:
            j += 1
        if j >= len(phrases):
            break
        start, end = phrases[i][0]["start"], phrases[j][-1]["end"]
        if end - start <= max_s:
            members = tuple(w for p in phrases[i : j + 1] for w in p)
            windows.append(Window(start, end, members))
            i = j + 1
        else:
            i += 1  # this phrase run is too long to use as one window
    return windows


def select_windows(windows: list[Window], n: int) -> list[Window]:
    """Pick n windows spread evenly across the speech (in time order)."""
    if len(windows) < n:
        raise ValueError(f"Need {n} windows but only found {len(windows)}; lower min_s.")
    if n == 1:
        return [windows[len(windows) // 2]]
    indices = [round(i * (len(windows) - 1) / (n - 1)) for i in range(n)]
    return [windows[k] for k in indices]


def pick_pause_point(window: Window, max_gap_s: float = 0.12) -> tuple[float, str]:
    """Find a mid-phrase spot where a speaker would not normally pause.

    Returns (insertion_time, word_before_the_pause). Prefers word pairs with a
    tiny natural gap, closest to the window's centre.
    """
    pairs = [
        (a, b)
        for a, b in zip(window.words, window.words[1:])
        if not ends_phrase(a["word"])
    ]
    tight = [(a, b) for a, b in pairs if b["start"] - a["end"] < max_gap_s]
    pool = tight or pairs
    if not pool:
        raise ValueError(f"No mid-phrase word boundary in window {window.start:.2f}-{window.end:.2f}s")
    a, b = min(pool, key=lambda p: abs(p[0]["end"] - window.midpoint))
    at = (a["end"] + b["start"]) / 2.0 if b["start"] > a["end"] else a["end"]
    return at, a["word"]


def pick_stumble_word(window: Window, min_dur: float = 0.2, max_dur: float = 0.7) -> dict:
    """Pick a short, clean word near the window's centre to repeat.

    Avoids the first and last word and any word carrying punctuation.
    """
    inner = window.words[1:-1]
    clean = [w for w in inner if not ends_phrase(w["word"])]
    good = [w for w in clean if min_dur <= w["end"] - w["start"] <= max_dur]
    if good:
        return min(good, key=lambda w: abs((w["start"] + w["end"]) / 2 - window.midpoint))
    short = [w for w in clean if w["end"] - w["start"] <= 0.9]
    if not short:
        raise ValueError(f"No suitable word to repeat in window {window.start:.2f}-{window.end:.2f}s")
    return max(short, key=lambda w: w["end"] - w["start"])