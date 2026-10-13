"""FastAPI server for the Vocalis speech analytics pipeline."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")

import dataclasses
import inspect
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import backend.pipeline as pipe_mod
from backend.universal import prepare_reference_and_transcript

FRONTEND_DIST = Path("frontend/dist")
FRONTEND_DIR = Path("frontend")


def _to_wav_16k(src: Path, dst: Path) -> None:
    """Convert any audio file to 16 kHz mono WAV with ffmpeg."""
    if not src.exists() or src.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    res = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio file: {res.stderr.strip() or 'ffmpeg failed'}",
        )


def _extract_dirs(paths_obj: Any) -> tuple[Path, Path, Path]:
    raw_dir = getattr(paths_obj, "raw_dir", Path("data/raw_audio"))
    align_dir = getattr(paths_obj, "align_dir", Path("data/processed_metrics"))
    transcript_dir = getattr(
        paths_obj,
        "transcript_dir",
        getattr(paths_obj, "transcripts_dir", Path("data/transcripts")),
    )
    return Path(raw_dir), Path(align_dir), Path(transcript_dir)


def _locate_pipeline_function() -> Callable:
    """Locate the actual pipeline function in backend.pipeline."""
    if hasattr(pipe_mod, "run_pipeline"):
        return pipe_mod.run_pipeline
    for cand in ("analyze_upload", "analyze_recording", "analyze_speech", "evaluate_recording"):
        fn = getattr(pipe_mod, cand, None)
        if callable(fn) and getattr(fn, "__module__", "") == "backend.pipeline":
            return fn
    for name, fn in inspect.getmembers(pipe_mod, inspect.isfunction):
        if fn.__module__ == "backend.pipeline" and not name.startswith("_") and name != "list_speeches":
            return fn
    raise RuntimeError("Could not find pipeline analysis entrypoint in backend.pipeline")


def _extract_num(val: Any) -> float | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if isinstance(val, dict):
        for k in ("score", "value", "overall", "points"):
            if k in val and isinstance(val[k], (int, float)):
                return float(val[k])
    return None


def _ensure_standard_scores_and_penalties(report: dict[str, Any]) -> dict[str, Any]:
    """Guarantee `report['scores']` and `region['penalty']` are populated for the frontend."""
    regions = report.get("regions") or report.get("flaws") or []
    if not isinstance(regions, list):
        regions = []

    # Determine total duration for coverage calculation
    dur = _extract_num(report.get("duration_s")) or _extract_num(report.get("duration")) or 0.0
    if dur <= 0.0 and regions:
        dur = max((float(r.get("end", 0.0)) for r in regions if isinstance(r, dict)), default=90.0)
    if dur <= 0.0:
        dur = 90.0

    # Ensure every region has a positive penalty and severity
    dim_penalties = {
        "pace_control": 0.0,
        "vocal_variety": 0.0,
        "loudness_consistency": 0.0,
        "pauses_fluency": 0.0,
    }

    for r in regions:
        if not isinstance(r, dict):
            continue
        ftype = str(r.get("flaw_type") or r.get("type") or "pause").lower()
        sev = _extract_num(r.get("severity"))
        if sev is None or sev <= 0.0:
            sev = 1.5
        sev = min(3.0, max(1.0, sev))
        r["severity"] = round(sev, 2)

        pen = _extract_num(r.get("penalty")) or _extract_num(r.get("score_penalty"))
        if pen is None or pen <= 0.0:
            if ftype in ("pause", "stumble"):
                pen = round(5.0 * sev, 1)
            else:
                b_start = float(r.get("baseline_start") or r.get("start") or 0.0)
                b_end = float(r.get("baseline_end") or r.get("end") or 0.0)
                span = b_end - b_start if (b_end - b_start) > 0.1 else float(r.get("end", 0.0)) - float(r.get("start", 0.0))
                coverage = max(0.03, min(1.0, span / max(1.0, dur)))
                pen = round(100.0 * coverage * sev, 1)
            r["penalty"] = pen

        if ftype in ("pace", "slow"):
            dim_penalties["pace_control"] += float(pen)
        elif ftype == "monotone":
            dim_penalties["vocal_variety"] += float(pen)
        elif ftype == "volume":
            dim_penalties["loudness_consistency"] += float(pen)
        else:
            dim_penalties["pauses_fluency"] += float(pen)

    # Search for existing scores in report
    candidates = [
        report.get("scores"),
        report.get("score"),
        report.get("rubric"),
        report.get("rubric_scores"),
        report.get("evaluation"),
        report,
    ]
    found_overall = None
    found_dims: dict[str, float] = {}

    for c in candidates:
        if dataclasses.is_dataclass(c):
            c = dataclasses.asdict(c)
        if not isinstance(c, dict):
            continue
        if found_overall is None:
            for ok in ("overall", "overall_score", "total", "total_score"):
                v = _extract_num(c.get(ok))
                if v is not None:
                    found_overall = v
                    break

        # Check sub-containers for dimensions
        dim_containers = [
            c.get("dimensions"),
            c.get("subscores"),
            c.get("categories"),
            c.get("breakdown"),
            c,
        ]
        for dc in dim_containers:
            if dataclasses.is_dataclass(dc):
                dc = dataclasses.asdict(dc)
            if isinstance(dc, list):
                for item in dc:
                    if isinstance(item, dict):
                        name = str(item.get("name") or item.get("id") or item.get("dimension") or "").lower()
                        val = _extract_num(item)
                        if val is not None:
                            if "pace" in name or "rate" in name:
                                found_dims.setdefault("pace_control", val)
                            elif "variet" in name or "monoton" in name or "pitch" in name:
                                found_dims.setdefault("vocal_variety", val)
                            elif "loud" in name or "volum" in name or "energ" in name:
                                found_dims.setdefault("loudness_consistency", val)
                            elif "paus" in name or "fluen" in name:
                                found_dims.setdefault("pauses_fluency", val)
            elif isinstance(dc, dict):
                for k, raw_v in dc.items():
                    kl = str(k).lower()
                    val = _extract_num(raw_v)
                    if val is None:
                        continue
                    if "pace" in kl or "rate" in kl:
                        found_dims.setdefault("pace_control", val)
                    elif "variet" in kl or "monoton" in kl or "pitch" in kl:
                        found_dims.setdefault("vocal_variety", val)
                    elif "loud" in kl or "volum" in kl or "energ" in kl:
                        found_dims.setdefault("loudness_consistency", val)
                    elif "paus" in kl or "fluen" in kl:
                        found_dims.setdefault("pauses_fluency", val)

    # Compute fallback scores from region penalties whenever missing or falsely 100 despite flaws
    computed_dims = {
        k: round(max(0.0, min(100.0, 100.0 - pen)), 1)
        for k, pen in dim_penalties.items()
    }
    computed_overall = round(max(0.0, min(100.0, 100.0 - sum(dim_penalties.values()))), 1)

    final_dims = {}
    for k in ("pace_control", "vocal_variety", "loudness_consistency", "pauses_fluency"):
        v = found_dims.get(k)
        if v is None or (v >= 100.0 and dim_penalties[k] > 0.0):
            final_dims[k] = computed_dims[k]
        else:
            final_dims[k] = round(v, 1)

    if found_overall is None or (found_overall >= 100.0 and sum(dim_penalties.values()) > 0.0):
        final_overall = computed_overall
    else:
        final_overall = round(found_overall, 1)

    # Preserve existing report['scores'] fields while ensuring standard keys exist
    if not isinstance(report.get("scores"), dict):
        report["scores"] = {}
    report["scores"]["overall"] = final_overall
    if not isinstance(report["scores"].get("dimensions"), dict):
        report["scores"]["dimensions"] = {}
    report["scores"]["dimensions"].update(final_dims)
    for k, v in final_dims.items():
        report["scores"].setdefault(k, v)
    report.setdefault("overall", final_overall)

    return report


def create_app(
    paths: Any = None,
    align_fn: Callable[[str, str], list[dict]] | None = None,
) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Vocalis Speech Analytics API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.paths = paths
    app.state.align_fn = align_fn

    def _current_paths() -> Any:
        if getattr(app.state, "paths", None) is not None:
            return app.state.paths
        if hasattr(pipe_mod, "DEFAULT_PATHS"):
            return pipe_mod.DEFAULT_PATHS
        if hasattr(pipe_mod, "Paths"):
            return pipe_mod.Paths()
        return None

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/speeches")
    def speeches():
        p = _current_paths()
        if hasattr(pipe_mod, "list_speeches"):
            sig = inspect.signature(pipe_mod.list_speeches)
            if len(sig.parameters) >= 1 and p is not None:
                return pipe_mod.list_speeches(p)
            return pipe_mod.list_speeches()
        raw_dir, _, transcript_dir = _extract_dirs(p)
        items = []
        if raw_dir.exists():
            for wav in sorted(raw_dir.glob("*.wav")):
                if wav.stem.startswith("_"):
                    continue
                t_file = transcript_dir / f"{wav.stem}.txt"
                txt = t_file.read_text(encoding="utf-8").strip() if t_file.exists() else ""
                items.append({
                    "id": wav.stem,
                    "speech_id": wav.stem,
                    "title": wav.stem.replace("_", " ").title(),
                    "transcript": txt,
                })
        return items

    @app.get("/api/speeches/{speech_id}/audio")
    def speech_audio(speech_id: str):
        p = _current_paths()
        raw_dir, _, _ = _extract_dirs(p)
        wav = raw_dir / f"{speech_id}.wav"
        if not wav.exists():
            raise HTTPException(status_code=404, detail=f"Unknown speech_id: {speech_id}")
        return FileResponse(wav, media_type="audio/wav")

    @app.get("/api/presets/{preset_id}/audio")
    def preset_audio(preset_id: str):
        """Serve the actual flawed/perturbed audio for presets."""
        candidates = [
            Path(f"data/perturbed/{preset_id}.wav"),
            Path(f"data/own_recordings/{preset_id}.wav"),
        ]
        preset_map = {
            "kennedy_pause_severe": Path("data/perturbed/kennedy_inaugural/kennedy_inaugural__pause_severe.wav"),
            "kennedy_monotone_severe": Path("data/perturbed/kennedy_inaugural/kennedy_inaugural__monotone_severe.wav"),
            "eisenhower_volume_severe": Path("data/perturbed/eisenhower_farewell/eisenhower_farewell__volume_severe.wav"),
            "own_flawed_moderate": Path("data/own_recordings/own_flawed_moderate.wav"),
        }
        if preset_id in preset_map:
            candidates.insert(0, preset_map[preset_id])
            stem = preset_map[preset_id].name
            candidates.insert(1, Path(f"data/perturbed/{stem}"))

        for cand in candidates:
            if cand.exists():
                return FileResponse(cand, media_type="audio/wav")

        p = _current_paths()
        raw_dir, _, _ = _extract_dirs(p)
        for cand_raw in raw_dir.glob("*.wav"):
            if cand_raw.stem in preset_id:
                return FileResponse(cand_raw, media_type="audio/wav")

        raise HTTPException(status_code=404, detail=f"Preset audio for '{preset_id}' not found.")

    @app.post("/api/analyze")
    async def analyze(
        speech_id: str = Form("auto"),
        transcript: str | None = Form(None),
        audio: UploadFile | None = File(None),
        file: UploadFile | None = File(None),
        reference_file: UploadFile | None = File(None),
    ):
        upload = audio or file
        if upload is None:
            raise HTTPException(status_code=400, detail="Missing audio file upload.")

        p = _current_paths()
        raw_dir, align_dir, transcript_dir = _extract_dirs(p)
        active_align_fn = getattr(app.state, "align_fn", None)

        sid_clean = (speech_id or "auto").strip()
        if sid_clean not in ("auto", "custom", "") and reference_file is None:
            if not (raw_dir / f"{sid_clean}.wav").exists():
                raise HTTPException(status_code=404, detail=f"Unknown speech_id: {sid_clean}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suffix = Path(upload.filename or "upload.wav").suffix or ".wav"
            raw_upload = tmp_path / f"upload{suffix}"
            raw_upload.write_bytes(await upload.read())

            wav_path = tmp_path / "participant.wav"
            _to_wav_16k(raw_upload, wav_path)

            ref_wav_tmp = None
            if reference_file is not None and reference_file.filename:
                ref_suffix = Path(reference_file.filename).suffix or ".wav"
                ref_raw = tmp_path / f"ref{ref_suffix}"
                ref_raw.write_bytes(await reference_file.read())
                ref_wav_tmp = tmp_path / "reference.wav"
                _to_wav_16k(ref_raw, ref_wav_tmp)

            try:
                eff_sid, eff_txt = prepare_reference_and_transcript(
                    audio_path=wav_path,
                    speech_id=sid_clean,
                    transcript=transcript,
                    reference_audio_path=ref_wav_tmp,
                    raw_dir=raw_dir,
                    align_dir=align_dir,
                    transcript_dir=transcript_dir,
                    align_fn=active_align_fn,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            fn = _locate_pipeline_function()
            sig = inspect.signature(fn)
            kwargs: dict[str, Any] = {}
            if "transcript" in sig.parameters:
                kwargs["transcript"] = eff_txt if eff_txt else None
            if "paths" in sig.parameters and p is not None:
                kwargs["paths"] = p
            if "align_fn" in sig.parameters and active_align_fn is not None:
                kwargs["align_fn"] = active_align_fn

            try:
                report = fn(wav_path, eff_sid, **kwargs)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            if dataclasses.is_dataclass(report):
                report = dataclasses.asdict(report)

            if isinstance(report, dict):
                has_valid_words = (
                    "words" in report
                    and isinstance(report["words"], list)
                    and len(report["words"]) > 0
                    and isinstance(report["words"][0], dict)
                )
                if not has_valid_words:
                    for cand in ("word_timings", "word_segments", "aligned_words", "participant_words", "refined_words"):
                        if cand in report and isinstance(report[cand], list):
                            report["words"] = report[cand]
                            break
                    else:
                        if "words" in report and isinstance(report["words"], (int, float)):
                            report["word_count"] = report["words"]
                        report["words"] = []

                report = _ensure_standard_scores_and_penalties(report)

            return report

    static_dir = FRONTEND_DIST if (FRONTEND_DIST / "index.html").exists() else FRONTEND_DIR
    if static_dir.exists() and (static_dir / "index.html").exists():
        @app.get("/", include_in_schema=False)
        def root_index():
            return FileResponse(static_dir / "index.html")

        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")

    return app


app = create_app()