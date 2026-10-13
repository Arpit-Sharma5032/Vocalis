"""Tests for planning and labelling our own recordings."""
import pytest

from dataset_builder.recordings import (
    FLAWS,
    assign_flaws,
    find_pause_gap,
    label_take,
    make_plan,
    split_sentences,
    takes,
    token_ranges,
)

TEXT = (
    "First short one. "
    "This second sentence is long enough to host a flaw easily. "
    "Here is another long sentence with plenty of words inside it. "
    "A third long sentence follows right after the one before it. "
    "And the fourth long sentence is also quite wordy, as you see. "
    "Short end."
)


def _words(text: str, dur: float = 0.3):
    out, t = [], 0.0
    for tok in text.split():
        out.append({"word": tok, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + 0.05
    return out


def test_split_sentences_and_token_ranges_cover_every_word():
    sentences = split_sentences(TEXT)
    assert len(sentences) == 6
    ranges = token_ranges(sentences)
    assert ranges[0][0] == 0 and ranges[-1][1] == len(TEXT.split()) - 1
    assert all(b[0] == a[1] + 1 for a, b in zip(ranges, ranges[1:]))


def test_assign_flaws_picks_distinct_inner_long_sentences():
    sentences = split_sentences(TEXT)
    chosen = assign_flaws(sentences, min_words=8)
    assert set(chosen.values()) == set(FLAWS)
    assert 0 not in chosen and len(sentences) - 1 not in chosen
    assert all(len(sentences[i].split()) >= 8 for i in chosen)


def test_assign_flaws_errors_when_text_too_short():
    with pytest.raises(ValueError):
        assign_flaws(["One.", "Two two.", "Three."])


def test_pause_gap_is_largest_non_punctuation_gap():
    words = _words("one two three four, five six seven")
    words[2]["end"] -= 0.0
    for w in words[3:]:  # open a big gap after 'three' (index 2 -> 3)
        w["start"] += 1.0
        w["end"] += 1.0
    start, end = find_pause_gap(words, 0, len(words) - 1)
    assert (start, end) == (words[2]["end"], words[3]["start"])


def test_label_take_uses_sentence_boundaries_and_rejects_mismatch(tmp_path):
    (tmp_path / "t.txt").write_text(TEXT, encoding="utf-8")
    plan = make_plan(tmp_path / "t.txt")
    words = _words(TEXT)
    rows = label_take("own_flawed_moderate", plan, words, tmp_path)
    assert len(rows) == len(FLAWS) and all(r["severity"] == "moderate" for r in rows)
    pace = next(r for r in rows if r["flaw_type"] == "pace")
    first, last = plan["token_ranges"][int(pace["sentence_index"])]
    assert pace["flaw_start"] == words[first]["start"] and pace["flaw_end"] == words[last]["end"]
    assert label_take("own_good", plan, words, tmp_path)[0]["flaw_type"] == "none"
    with pytest.raises(ValueError):
        label_take("own_flawed_moderate", plan, words[:-1], tmp_path)


def test_takes_use_the_given_prefix_and_good_take_is_detected_by_suffix(tmp_path):
    assert takes("ownb") == ("ownb_good", "ownb_flawed_moderate", "ownb_flawed_subtle")
    (tmp_path / "t.txt").write_text(TEXT, encoding="utf-8")
    plan = make_plan(tmp_path / "t.txt")
    words = _words(TEXT)
    assert label_take("ownb_good", plan, words, tmp_path)[0]["flaw_type"] == "none"
    assert all(r["severity"] == "subtle" for r in label_take("ownb_flawed_subtle", plan, words, tmp_path))


def test_short_text_relaxes_the_rules_instead_of_failing():
    """Kennedy-style excerpt: only 3 long inner sentences, but 4 long sentences overall."""
    sentences = [
        "A short opening line here.",
        "This second sentence has plenty of words in it for sure.",
        "And the third sentence is also long enough to qualify here.",
        "The fourth sentence is long enough as well to be picked.",
        "The very last sentence is long too and could host a flaw.",
    ]
    chosen = assign_flaws(sentences)
    assert set(chosen.values()) == set(FLAWS) and len(chosen) == 4