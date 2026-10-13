"""End-to-end pipeline and API tests on synthetic data with a stubbed aligner."""
import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from backend.pipeline import Paths, analyze_reading, available_speeches
from dataset_builder.oracle import oracle_alignment
from dataset_builder.perturb import SR, inject
from dataset_builder.windows import candidate_windows, pick_pause_point

SPEECH = "kennedy_inaugural"


def _voiced(seconds=62.0):
    t = np.arange(int(seconds * SR)) / SR
    f0 = 150.0 * (1.0 + 0.17 * np.sin(2 * np.pi * 0.8 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (0.3 * sig / np.max(np.abs(sig))).astype(np.float32)


def _words(n=140):
    words, t = [], 0.5
    for i in range(n):
        text = f"w{i}" + ("," if (i + 1) % 5 == 0 else "")
        words.append({"word": text, "start": round(t, 3), "end": round(t + 0.4, 3)})
        t += 0.4 + (0.3 if text.endswith(",") else 0.03)
    return words


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    paths = Paths(root / "raw", root / "align", root / "tx")
    for d in (paths.raw_dir, paths.align_dir, paths.transcript_dir):
        d.mkdir()
    audio, words = _voiced(), _words()
    sf.write(paths.raw_dir / f"{SPEECH}.wav", audio, SR)
    (paths.align_dir / f"{SPEECH}_alignment.json").write_text(json.dumps(words))
    (paths.transcript_dir / f"{SPEECH}.txt").write_text(" ".join(w["word"] for w in words))

    window = candidate_windows(words, 4.0, 9.0)[3]
    at, _ = pick_pause_point(window)
    flawed, label = inject(audio, "pause", "severe", at, at)
    row = {"flaw_type": "pause", "parameter": label.parameter, "flaw_start": label.start, "flaw_end": label.end,
           "window_start": window.start, "window_end": window.end}
    flawed_wav = root / "flawed.wav"
    sf.write(flawed_wav, flawed, SR)
    return {"paths": paths, "audio": audio, "words": words, "flawed_wav": flawed_wav,
            "flawed_words": oracle_alignment(words, row), "label": label, "clean_wav": paths.raw_dir / f"{SPEECH}.wav"}


def test_available_speeches_lists_complete_references(world):
    speeches = available_speeches(world["paths"])
    assert [s["id"] for s in speeches] == [SPEECH] and speeches[0]["words"] == 140


def test_flawed_reading_gets_a_lower_score_and_an_explained_region(world):
    clean = analyze_reading(str(world["clean_wav"]), SPEECH, None, lambda w, t: world["words"], world["paths"])
    flawed = analyze_reading(str(world["flawed_wav"]), SPEECH, None, lambda w, t: world["flawed_words"], world["paths"])
    assert clean["overall"] == 100.0 and clean["regions"] == []
    assert flawed["overall"] < clean["overall"] and flawed["dimensions"]["fluency"]["score"] < 100.0
    region = next(r for r in flawed["regions"] if r["flaw_type"] == "pause")
    assert abs(region["start"] - world["label"].start) < 0.2 and region["summary"] and region["advice"]
    json.dumps(flawed, allow_nan=False)  # report must be strict-JSON safe
    assert len(flawed["series"]["participant"]["t"]) == len(flawed["series"]["participant"]["pitch_st"])


def test_words_without_timestamps_are_ignored_with_a_warning(world):
    words = [dict(w) for w in world["words"]]
    words[3]["start"] = words[3]["end"] = None
    report = analyze_reading(str(world["clean_wav"]), SPEECH, None, lambda w, t: words, world["paths"])
    assert report["warnings"] and "1 words" in report["warnings"][0]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_api_end_to_end(world):
    from fastapi.testclient import TestClient

    from backend.api import app

    app.state.paths = world["paths"]
    app.state.align_fn = lambda wav, text: world["flawed_words"]
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/speeches").json()[0]["id"] == SPEECH
    assert client.get(f"/api/speeches/{SPEECH}/audio").status_code == 200

    with open(world["flawed_wav"], "rb") as f:
        ok = client.post("/api/analyze", data={"speech_id": SPEECH}, files={"audio": ("r.wav", f, "audio/wav")})
    assert ok.status_code == 200 and ok.json()["overall"] < 100.0

    assert client.post("/api/analyze", data={"speech_id": "nope"},
                       files={"audio": ("r.wav", b"x", "audio/wav")}).status_code == 404
    assert client.post("/api/analyze", data={"speech_id": SPEECH},
                       files={"audio": ("r.wav", b"not audio at all", "audio/wav")}).status_code == 400


def test_report_words_are_cut_back_around_swallowed_silences(world):
    audio, words = world["audio"], world["words"]
    label = world["label"]
    swallowed = [dict(w, start=label.start - 0.03) if abs(w["start"] - label.end) < 0.06 else dict(w)
                 for w in world["flawed_words"]]
    report = analyze_reading(str(world["flawed_wav"]), SPEECH, None, lambda w, t: swallowed, world["paths"])
    after = [w for w in report["words"]["participant"] if w["s"] >= label.end - 0.1 and w["s"] <= label.end + 0.2]
    assert after and all(w["s"] >= label.end - 0.1 for w in after)
    assert not any(w["s"] < label.end - 0.2 and w["e"] > label.end for w in report["words"]["participant"])