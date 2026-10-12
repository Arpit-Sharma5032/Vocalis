"""Small text helpers shared by the dataset builder and the detector."""
from __future__ import annotations

import re

PHRASE_END = (",", ";", ":", ".", "?", "!", "—", "–")
_TRAILING_QUOTES = "\"'”’)]"


def ends_phrase(word: str) -> bool:
    """True if the word ends with punctuation that marks a natural pause."""
    return word.rstrip(_TRAILING_QUOTES).endswith(PHRASE_END)


def normalize_word(word: str) -> str:
    """Lower-case a word and strip punctuation, for matching across transcripts."""
    return re.sub(r"[^a-z0-9']", "", word.lower())