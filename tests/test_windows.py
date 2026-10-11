"""Tests for deterministic, word-aligned flaw location selection."""
import pytest

from dataset_builder.windows import (
    candidate_windows,
    ends_phrase,
    pick_pause_point,
    pick_stumble_word,
    select_windows,
)


def _words(n: int = 60, dur: float = 0.4, comma_every: int = 5) -> list[dict]:
    """Synthetic alignment: n words, a comma every few words, tight natural gaps."""
    words, t = [], 0.0
    for i in range(n):
        text = f"w{i}" + ("," if (i + 1) % comma_every == 0 else "")
        words.append({"word": text, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + (0.3 if text.endswith(",") else 0.03)
    return words


def test_ends_phrase_handles_punctuation_and_quotes():
    assert ends_phrase("revolution.") and ends_phrase("hand,") and ends_phrase('said."')
    assert not ends_phrase("the")


def test_windows_respect_duration_and_do_not_overlap():
    windows = candidate_windows(_words(), min_s=4.0, max_s=9.0)
    assert len(windows) >= 3
    for w in windows:
        assert 4.0 <= w.end - w.start <= 9.0
        assert w.words[-1]["word"].endswith(",")  # ends on a phrase boundary
    for a, b in zip(windows, windows[1:]):
        assert a.end <= b.start


def test_select_windows_is_even_and_deterministic():
    windows = candidate_windows(_words(), min_s=2.0, max_s=9.0)
    assert select_windows(windows, 4) == select_windows(windows, 4)
    assert len(select_windows(windows, 4)) == 4
    with pytest.raises(ValueError):
        select_windows(windows[:2], 5)


def test_pause_point_is_mid_phrase_inside_window():
    win = candidate_windows(_words(), min_s=4.0, max_s=9.0)[0]
    at, word_before = pick_pause_point(win)
    assert win.start < at < win.end
    assert not ends_phrase(word_before)


def test_stumble_word_is_clean_inner_word():
    win = candidate_windows(_words(), min_s=4.0, max_s=9.0)[0]
    word = pick_stumble_word(win)
    assert word not in (win.words[0], win.words[-1])
    assert not ends_phrase(word["word"])


def test_long_unpunctuated_phrase_is_split_into_usable_windows():
    """Kennedy-style delivery: 20 s of speech with a single comma at the end."""
    words, t = [], 0.0
    for i in range(50):
        gap = 0.25 if i == 24 else 0.03  # one small natural pause in the middle
        words.append({"word": f"w{i}", "start": round(t, 3), "end": round(t + 0.4, 3)})
        t += 0.4 + gap
    words[-1]["word"] = "w49."
    windows = candidate_windows(words, min_s=4.0, max_s=9.0)
    assert len(windows) >= 2
    assert all(4.0 <= w.end - w.start <= 9.0 for w in windows)