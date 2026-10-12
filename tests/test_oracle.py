"""Oracle word timings must agree with what the perturbation engine really did."""
import numpy as np
import pytest

from dataset_builder.oracle import clip_transcript, oracle_alignment
from dataset_builder.perturb import SR, inject
from dataset_builder.windows import candidate_windows, pick_pause_point, pick_stumble_word


def _words(n=140):
    words, t = [], 0.5
    for i in range(n):
        text = f"w{i}" + ("," if (i + 1) % 5 == 0 else "")
        words.append({"word": text, "start": round(t, 3), "end": round(t + 0.4, 3)})
        t += 0.4 + (0.3 if text.endswith(",") else 0.03)
    return words


def _voiced(seconds):
    t = np.arange(int(seconds * SR)) / SR
    f0 = 150.0 * (1.0 + 0.17 * np.sin(2 * np.pi * 0.8 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (0.3 * sig / np.max(np.abs(sig))).astype(np.float32)


def _row(flaw, severity, words):
    """Inject a flaw into voiced audio; return (manifest row, baseline length, flawed length)."""
    audio = _voiced(words[-1]["end"] + 1.0)
    window = candidate_windows(words, 4.0, 9.0)[3]
    if flaw == "pause":
        start, _ = pick_pause_point(window)
        end = start
    elif flaw == "stumble":
        w = pick_stumble_word(window)
        start, end = w["start"], w["end"]
    else:
        start, end = window.start, window.end
    flawed, label = inject(audio, flaw, severity, start, end)
    row = {"clip_id": f"x__{flaw}", "flaw_type": flaw, "parameter": label.parameter,
           "flaw_start": label.start, "flaw_end": label.end,
           "window_start": window.start, "window_end": window.end}
    return row, len(audio) / SR, len(flawed) / SR


@pytest.mark.parametrize("flaw", ["pace", "slow", "pause", "stumble"])
def test_last_word_moves_by_exactly_the_length_change(flaw):
    words = _words()
    row, base_len, flawed_len = _row(flaw, "severe", words)
    oracle = oracle_alignment(words, row)
    assert oracle[-1]["end"] - words[-1]["end"] == pytest.approx(flawed_len - base_len, abs=0.02)


def test_length_preserving_flaws_keep_word_times():
    words = _words()
    row, _, _ = _row("volume", "severe", words)
    assert oracle_alignment(words, row) == words


def test_stumble_transcript_contains_the_repeated_words_and_matches_oracle_length():
    words = _words()
    row, _, _ = _row("stumble", "severe", words)
    tokens = clip_transcript(words, row).split()
    assert len(tokens) == len(words) + int(row["parameter"])
    assert len(oracle_alignment(words, row)) == len(tokens)