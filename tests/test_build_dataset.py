"""Smoke test: build a tiny dataset from synthetic audio and check the manifest."""
import csv
import json

import numpy as np
import soundfile as sf

from dataset_builder.build_dataset import FLAW_ORDER, SEVERITIES, build_dataset
from dataset_builder.perturb import SR


def _make_inputs(tmp_path):
    t = np.arange(int(40 * SR)) / SR
    f0 = 150.0 + 30.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    audio = (0.3 * sig / np.max(np.abs(sig))).astype(np.float32)
    raw, align = tmp_path / "raw", tmp_path / "align"
    raw.mkdir(), align.mkdir()
    sf.write(raw / "demo.wav", audio, SR)
    words, tt = [], 0.5
    for i in range(70):
        text = f"w{i}" + ("," if (i + 1) % 5 == 0 else "")
        words.append({"word": text, "start": round(tt, 3), "end": round(tt + 0.4, 3)})
        tt += 0.4 + (0.3 if text.endswith(",") else 0.03)
    (align / "demo_alignment.json").write_text(json.dumps(words))
    return raw, align


def test_manifest_has_every_flaw_and_severity_with_valid_regions(tmp_path):
    raw, align = _make_inputs(tmp_path)
    manifest = tmp_path / "manifest.csv"
    build_dataset(["demo"], raw, align, tmp_path / "out", manifest, min_s=3.0, max_s=8.0)
    rows = list(csv.DictReader(manifest.open()))
    flawed = [r for r in rows if r["is_baseline"] == "False"]
    assert len(flawed) == len(FLAW_ORDER) * len(SEVERITIES)
    assert {r["flaw_type"] for r in flawed} == set(FLAW_ORDER)
    for r in flawed:
        assert 0 <= float(r["flaw_start"]) < float(r["flaw_end"]) <= float(r["duration_s"])


def test_rebuild_gives_identical_manifest_and_audio(tmp_path):
    raw, align = _make_inputs(tmp_path)
    for run in ("a", "b"):
        build_dataset(["demo"], raw, align, tmp_path / run, tmp_path / f"{run}.csv", min_s=3.0, max_s=8.0)
    a = (tmp_path / "a.csv").read_text().replace("/a/", "/x/")
    b = (tmp_path / "b.csv").read_text().replace("/b/", "/x/")
    assert a == b
    for f in (tmp_path / "a" / "demo").glob("*.wav"):
        x, _ = sf.read(f)
        y, _ = sf.read(tmp_path / "b" / "demo" / f.name)
        assert np.array_equal(x, y), f.name