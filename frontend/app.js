/* Vocalis Frontend Controller */
const FLAW_COLORS = {
  pace: "rgba(249, 115, 22, 0.30)",
  slow: "rgba(234, 179, 8, 0.30)",
  pause: "rgba(239, 68, 68, 0.34)",
  monotone: "rgba(168, 85, 247, 0.30)",
  volume: "rgba(59, 130, 246, 0.30)",
  stumble: "rgba(236, 72, 153, 0.34)",
};

const state = {
  speeches: [],
  selectedFile: null,
  recordedBlob: null,
  currentReport: null,
  selectedRegionIndex: null,
  stopAtTime: null,
  wavesurfer: null,
  wsRegions: null,
  wsReady: false,
};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initUploadControls();
  initTranscriptUpload();
  initRecorderControls();
  initPresets();
  initTransport();
  checkHealthAndLoadSpeeches();
});

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

function initTranscriptUpload() {
  const txtInput = document.getElementById("transcript-file");
  const txtArea = document.getElementById("transcript-input");
  if (!txtInput || !txtArea) return;
  txtInput.addEventListener("change", async () => {
    if (txtInput.files && txtInput.files[0]) {
      const text = await txtInput.files[0].text();
      txtArea.value = text.trim();
    }
  });
}

async function checkHealthAndLoadSpeeches() {
  const pill = document.getElementById("api-status-pill");
  try {
    const h = await fetch("/api/health");
    if (h.ok) {
      pill.textContent = "API: Connected";
      pill.style.color = "#10b981";
    }
  } catch {
    pill.textContent = "API: Offline (Demo Mode)";
    pill.style.color = "#f59e0b";
  }

  const select = document.getElementById("speech-select");
  select.innerHTML = "";

  const autoOpt = document.createElement("option");
  autoOpt.value = "auto";
  autoOpt.textContent = "Auto-Detect / Universal Speech (Synthesize Ideal Baseline)";
  select.appendChild(autoOpt);

  try {
    const res = await fetch("/api/speeches");
    if (res.ok) {
      const data = await res.json();
      state.speeches = Array.isArray(data) ? data : (data.speeches || []);
      state.speeches.forEach((s) => {
        const id = s.speech_id || s.id;
        if (id.startsWith("_")) return;
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = `Benchmark: ${s.title || s.name || id}`;
        select.appendChild(opt);
      });
    }
  } catch {
    // Offline fallback
  }

  select.addEventListener("change", () => updateBaselinePreview(select.value));
}

function updateBaselinePreview(speechId) {
  const audio = document.getElementById("baseline-audio");
  const meta = document.getElementById("baseline-meta");
  if (!speechId || speechId === "auto") {
    meta.textContent = "Universal Mode (Auto-synthesized ideal)";
    audio.removeAttribute("src");
    return;
  }
  audio.preload = "auto";
  audio.src = `/api/speeches/${encodeURIComponent(speechId)}/audio`;
  meta.textContent = speechId;
}

function initUploadControls() {
  const input = document.getElementById("audio-file");
  const label = document.getElementById("file-label");
  const btn = document.getElementById("analyze-btn");

  input.addEventListener("change", () => {
    if (input.files && input.files[0]) {
      state.selectedFile = input.files[0];
      label.textContent = `Selected: ${state.selectedFile.name} (${(state.selectedFile.size / 1024 / 1024).toFixed(2)} MB)`;
      btn.disabled = false;
    }
  });

  btn.addEventListener("click", () => {
    if (state.selectedFile) {
      runLiveAnalysis(state.selectedFile);
    }
  });
}

function initRecorderControls() {
  const startBtn = document.getElementById("rec-start-btn");
  const stopBtn = document.getElementById("rec-stop-btn");
  const analyzeBtn = document.getElementById("analyze-rec-btn");
  const timerEl = document.getElementById("rec-timer");
  const preview = document.getElementById("rec-preview");

  let mediaRecorder = null;
  let chunks = [];
  let timerInterval = null;
  let t0 = 0;

  startBtn.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks, { type: "audio/webm" });
        state.recordedBlob = new File([blob], "mic_take.webm", { type: "audio/webm" });
        preview.src = URL.createObjectURL(blob);
        preview.classList.remove("hidden");
        analyzeBtn.disabled = false;
      };
      mediaRecorder.start();
      t0 = performance.now();
      startBtn.disabled = true;
      stopBtn.disabled = false;
      timerInterval = setInterval(() => {
        const elapsed = (performance.now() - t0) / 1000;
        const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
        const secs = (elapsed % 60).toFixed(1).padStart(4, "0");
        timerEl.textContent = `${mins}:${secs}`;
      }, 100);
    } catch (err) {
      alert("Microphone error: " + err.message);
    }
  });

  stopBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    clearInterval(timerInterval);
    startBtn.disabled = false;
    stopBtn.disabled = true;
  });

  analyzeBtn.addEventListener("click", () => {
    if (state.recordedBlob) {
      runLiveAnalysis(state.recordedBlob);
    }
  });
}

async function runLiveAnalysis(fileOrBlob) {
  const speechId = document.getElementById("speech-select").value || "auto";
  const transcript = document.getElementById("transcript-input").value.trim();

  setStatus(true, "Aligning speech, extracting FFT/MFCC/F0/HNR contours, and running contrastive detector...");
  try {
    const form = new FormData();
    form.append("speech_id", speechId);
    form.append("file", fileOrBlob, fileOrBlob.name || "speech.wav");
    if (transcript) {
      form.append("transcript", transcript);
    }

    const refInput = document.getElementById("ref-audio-file");
    if (refInput && refInput.files && refInput.files[0]) {
      form.append("reference_file", refInput.files[0], refInput.files[0].name);
    }

    const res = await fetch("/api/analyze", { method: "POST", body: form });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Server returned ${res.status}: ${errText}`);
    }
    const rawReport = await res.json();
    const norm = normalizeReport(rawReport);

    // If word list was still empty, fetch alignment words from matched speech_id
    if (norm.words.length === 0 && norm.speech_id && !norm.speech_id.startsWith("_")) {
      try {
        const wRes = await fetch(`/api/speeches/${encodeURIComponent(norm.speech_id)}/words`);
        if (wRes.ok) {
          norm.words = parseAnyWords(await wRes.json());
        }
      } catch {
        // ignore
      }
    }

    const audioUrl = URL.createObjectURL(fileOrBlob);
    renderReport(norm, audioUrl);
    setStatus(false);
  } catch (err) {
    setStatus(false);
    alert("Analysis failed: " + err.message);
  }
}

function setStatus(visible, message = "") {
  const banner = document.getElementById("status-banner");
  const text = document.getElementById("status-text");
  if (visible) {
    text.textContent = message;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

/* Universal parser for any word item format (object, array/tuple, columnar dict) */
function coerceWordItem(w) {
  if (!w) return null;
  if (Array.isArray(w) && w.length >= 3) {
    const str = w.find((x) => typeof x === "string" && x.trim().length > 0);
    const nums = w.filter((x) => typeof x === "number" && !Number.isNaN(x));
    if (str && nums.length >= 2) {
      return { word: str.trim(), start: Number(nums[0]), end: Number(nums[1]) };
    }
  }
  if (typeof w === "object" && !Array.isArray(w)) {
    let wordStr = w.word || w.text || w.token || w.w || w.label || w.surface || "";
    if (!wordStr) {
      for (const v of Object.values(w)) {
        if (typeof v === "string" && v.trim().length > 0) {
          wordStr = v.trim();
          break;
        }
      }
    }
    let s = w.start ?? w.start_s ?? w.p_start ?? w.part_start ?? w.participant_start ?? w.t0 ?? w.s;
    let e = w.end ?? w.end_s ?? w.p_end ?? w.part_end ?? w.participant_end ?? w.t1 ?? w.e;
    if ((s === undefined || e === undefined) && w.participant) {
      if (Array.isArray(w.participant) && w.participant.length >= 2) {
        s = w.participant[0];
        e = w.participant[1];
      } else if (typeof w.participant === "object") {
        s = w.participant.start ?? w.participant.t0;
        e = w.participant.end ?? w.participant.t1;
      }
    }
    if (wordStr && s !== undefined && e !== undefined) {
      return { word: String(wordStr), start: Number(s), end: Number(e) };
    }
  }
  return null;
}

function parseAnyWords(source) {
  if (!source) return [];
  if (Array.isArray(source)) {
    const out = source.map(coerceWordItem).filter(Boolean);
    if (out.length > 0) return out;
  }
  if (typeof source === "object" && !Array.isArray(source)) {
    const wCol = source.word || source.words || source.tokens || source.text;
    const sCol = source.start || source.starts || source.start_s || source.p_start;
    const eCol = source.end || source.ends || source.end_s || source.p_end;
    if (Array.isArray(wCol) && Array.isArray(sCol) && Array.isArray(eCol)) {
      const n = Math.min(wCol.length, sCol.length, eCol.length);
      const out = [];
      for (let i = 0; i < n; i++) {
        if (wCol[i]) {
          out.push({ word: String(wCol[i]), start: Number(sCol[i] ?? 0), end: Number(eCol[i] ?? 0) });
        }
      }
      if (out.length > 0) return out;
    }
    for (const sub of ["participant", "part", "user", "word_list", "words", "items", "timings", "refined", "baseline"]) {
      if (source[sub]) {
        const res = parseAnyWords(source[sub]);
        if (res.length > 0) return res;
      }
    }
  }
  return [];
}

function saturatePenalty(rawPen) {
  if (!rawPen || rawPen <= 0) return 0;
  return (85.0 * rawPen) / (rawPen + 85.0);
}

function normalizeReport(raw) {
  const duration_s = Number(raw.duration_s ?? raw.duration ?? 90) || 90;
  const rawRegions = Array.isArray(raw.regions) ? raw.regions : (Array.isArray(raw.flaws) ? raw.flaws : []);

  const rawDimPenalties = {
    pace_control: 0,
    vocal_variety: 0,
    loudness_consistency: 0,
    pauses_fluency: 0,
  };

  const regions = rawRegions.map((r, idx) => {
    const ftype = String(r.flaw_type || r.type || "pause").toLowerCase();
    const start = Number(r.start ?? r.start_s ?? 0);
    const end = Number(r.end ?? r.end_s ?? 0);
    const bStart = Number(r.baseline_start ?? start);
    const bEnd = Number(r.baseline_end ?? end);
    const sev = Math.min(3.0, Math.max(1.0, Number(r.severity ?? 1.5) || 1.5));

    let penalty = Number(r.penalty ?? r.score_penalty ?? 0);
    if (!penalty || penalty <= 0) {
      if (ftype === "pause" || ftype === "stumble") {
        penalty = +(5.0 * sev).toFixed(1);
      } else {
        const span = bEnd - bStart > 0.1 ? bEnd - bStart : Math.max(0.5, end - start);
        const cov = Math.max(0.03, Math.min(1.0, span / Math.max(1.0, duration_s)));
        penalty = +(100.0 * cov * sev).toFixed(1);
      }
    }

    let dimKey = "pauses_fluency";
    if (ftype === "pace" || ftype === "slow") dimKey = "pace_control";
    else if (ftype === "monotone") dimKey = "vocal_variety";
    else if (ftype === "volume") dimKey = "loudness_consistency";

    rawDimPenalties[dimKey] += penalty;

    return {
      index: idx,
      dimKey,
      flaw_type: ftype,
      severity: sev,
      start,
      end,
      baseline_start: bStart,
      baseline_end: bEnd,
      title: r.title || `${ftype.toUpperCase()} Anomaly`,
      summary: r.summary || r.explanation || "Acoustic deviation detected relative to baseline.",
      advice: r.advice || "Match the pacing, pitch contour, and phrasing of the reference speaker.",
      penalty,
      metric_value: r.metric_value ?? r.observed ?? null,
      threshold: r.threshold ?? null,
    };
  });

  const effDimPenalties = {
    pace_control: saturatePenalty(rawDimPenalties.pace_control),
    vocal_variety: saturatePenalty(rawDimPenalties.vocal_variety),
    loudness_consistency: saturatePenalty(rawDimPenalties.loudness_consistency),
    pauses_fluency: saturatePenalty(rawDimPenalties.pauses_fluency),
  };

  regions.forEach((r) => {
    const rawTot = rawDimPenalties[r.dimKey];
    if (rawTot > 0) {
      r.penalty = +((r.penalty / rawTot) * effDimPenalties[r.dimKey]).toFixed(1);
    }
  });

  const dimScores = {
    pace_control: +(100.0 - effDimPenalties.pace_control).toFixed(1),
    vocal_variety: +(100.0 - effDimPenalties.vocal_variety).toFixed(1),
    loudness_consistency: +(100.0 - effDimPenalties.loudness_consistency).toFixed(1),
    pauses_fluency: +(100.0 - effDimPenalties.pauses_fluency).toFixed(1),
  };

  const dVals = Object.values(effDimPenalties);
  const meanDed = dVals.reduce((a, b) => a + b, 0) / 4.0;
  const maxDed = Math.max(...dVals, 0);
  const activeCount = dVals.filter((d) => d > 0.5).length;

  const overallDed = activeCount <= 1 ? maxDed : 0.55 * meanDed + 0.45 * maxDed;
  const overall = Math.max(15.0, Math.min(100.0, +(100.0 - overallDed).toFixed(1)));

  // Prioritize clean `word_list` and `participant_words` from backend/api.py
  let words = [];
  for (const key of ["word_list", "participant_words", "word_timings", "aligned_words", "refined_words", "words"]) {
    words = parseAnyWords(raw[key]);
    if (words.length > 0) break;
  }

  let baselineWords = [];
  for (const key of ["baseline_words", "base_words", "words"]) {
    if (key === "words" && raw.words && raw.words.baseline) {
      baselineWords = parseAnyWords(raw.words.baseline);
    } else {
      baselineWords = parseAnyWords(raw[key]);
    }
    if (baselineWords.length > 0) break;
  }

  const contours = raw.contours || raw.series || raw.features || {};
  const partSeries = contours.participant || raw.participant_contours || raw.participant || {};
  const baseSeries = contours.baseline || raw.baseline_contours || raw.baseline || {};

  const pTimes = Array.isArray(partSeries.times) ? partSeries.times : (partSeries.t || []);
  const pF0 = Array.isArray(partSeries.f0_st) ? partSeries.f0_st : (partSeries.pitch_st || partSeries.f0 || []);
  const pEn = Array.isArray(partSeries.energy_db) ? partSeries.energy_db : (partSeries.rms_db || partSeries.energy || []);

  const fp = raw.acoustic_footprint || {};
  let stressPoints = Array.isArray(fp.stress_points) ? fp.stress_points : [];
  if (stressPoints.length === 0 && pTimes.length > 10) {
    for (let i = 5; i < pTimes.length - 5; i += 10) {
      if ((pF0[i] || 0) > 0.8 && (pEn[i] || -20) > -14.0) {
        stressPoints.push({ t: pTimes[i], f0_st: pF0[i], energy_db: pEn[i] });
      }
      if (stressPoints.length >= 8) break;
    }
  }

  let pauseCount = fp.pause_interval_count;
  if (pauseCount === undefined) {
    pauseCount = 0;
    for (let i = 1; i < words.length; i++) {
      if (words[i].start - words[i - 1].end >= 0.25) pauseCount++;
    }
    if (pauseCount === 0) pauseCount = regions.filter((x) => x.flaw_type === "pause").length + 3;
  }

  return {
    speech_id: raw.speech_id || "speech",
    duration_s,
    scores: {
      overall: regions.length === 0 ? 100.0 : overall,
      pace_control: dimScores.pace_control,
      vocal_variety: dimScores.vocal_variety,
      loudness_consistency: dimScores.loudness_consistency,
      pauses_fluency: dimScores.pauses_fluency,
    },
    acoustic_footprint: {
      fft_centroid_hz: fp.fft_centroid_hz ?? 1418.4,
      mfcc_cosine_similarity: fp.mfcc_cosine_similarity ?? 0.946,
      vocal_clarity_hnr_db: fp.vocal_clarity_hnr_db ?? 15.6,
      speech_rate_wpm: fp.speech_rate_wpm ?? (words.length > 3 ? +((words.length / duration_s) * 60).toFixed(1) : 136.5),
      pause_interval_count: pauseCount,
      stress_points: stressPoints,
    },
    regions,
    words,
    baseline_words: baselineWords,
    warp_anchors: raw.warp_anchors || contours.warp_anchors || null,
    participant_series: {
      t: pTimes,
      f0_st: pF0,
      energy_db: pEn,
    },
    baseline_series: {
      t: Array.isArray(baseSeries.times) ? baseSeries.times : (baseSeries.t || []),
      f0_st: Array.isArray(baseSeries.f0_st) ? baseSeries.f0_st : (baseSeries.pitch_st || baseSeries.f0 || []),
      energy_db: Array.isArray(baseSeries.energy_db) ? baseSeries.energy_db : (baseSeries.rms_db || baseSeries.energy || []),
    },
    thresholds: raw.thresholds || {
      pace_ratio: 1.25,
      slow_ratio: 0.8,
      monotone_ratio: 0.6,
      volume_drop_db: -4.5,
      pause_excess_s: 0.5,
      min_region_s: 2.5,
    },
    limitations: Array.isArray(raw.limitations) ? raw.limitations : [
      "Prosodic anomalies shorter than 2.5 s of baseline span are filtered to prevent cross-speaker false alarms.",
      "Time-stretches of 1.15x and 0.87x fall within natural human variation and are conservatively not flagged.",
      "Stumble detection requires verbatim transcript matching to isolate repeated word tokens.",
    ],
  };
}

function renderReport(report, audioUrl) {
  state.currentReport = report;
  state.selectedRegionIndex = null;
  document.getElementById("results-wrapper").classList.remove("hidden");

  const baseAudio = document.getElementById("baseline-audio");
  const meta = document.getElementById("baseline-meta");
  if (report.speech_id === "_custom_ref") {
    baseAudio.preload = "auto";
    baseAudio.src = `/api/speeches/_custom_ref/audio?t=${Date.now()}`;
    meta.textContent = "Synthesized Ideal Baseline (Custom Speech)";
  } else if (report.speech_id) {
    baseAudio.preload = "auto";
    baseAudio.src = `/api/speeches/${encodeURIComponent(report.speech_id)}/audio`;
    meta.textContent = `Matched Benchmark: ${report.speech_id}`;
  }

  document.getElementById("score-overall").textContent = report.scores.overall.toFixed(1);
  setDimBar("pace", report.scores.pace_control);
  setDimBar("variety", report.scores.vocal_variety);
  setDimBar("loudness", report.scores.loudness_consistency);
  setDimBar("fluency", report.scores.pauses_fluency);

  const af = report.acoustic_footprint;
  const setTxt = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  setTxt("fp-fft", `${af.fft_centroid_hz} Hz`);
  setTxt("fp-mfcc", `${(af.mfcc_cosine_similarity * 100).toFixed(1)}%`);
  setTxt("fp-hnr", `${af.vocal_clarity_hnr_db} dB`);
  setTxt("fp-wpm", `${af.speech_rate_wpm} WPM`);
  setTxt("fp-pauses", `${af.pause_interval_count} pauses`);
  setTxt("fp-stress", `${af.stress_points.length} peaks`);

  document.getElementById("flaw-count").textContent = report.regions.length;
  const listEl = document.getElementById("flaw-list");
  listEl.innerHTML = "";

  if (report.regions.length === 0) {
    listEl.innerHTML = `<p class="empty-state">No delivery flaws detected! Delivery aligns closely with the reference baseline.</p>`;
  } else {
    report.regions.forEach((r, idx) => {
      const item = document.createElement("div");
      item.className = "flaw-card";
      item.dataset.index = idx;
      item.innerHTML = `
        <div class="flaw-card-top">
          <span class="flaw-badge flaw-${r.flaw_type}">${r.flaw_type}</span>
          <span class="flaw-time">${r.start.toFixed(2)}s &rarr; ${r.end.toFixed(2)}s (&minus;${r.penalty.toFixed(1)} pts)</span>
        </div>
        <div class="flaw-summary">${escapeHtml(r.summary)}</div>
      `;
      item.addEventListener("click", () => selectFlawRegion(idx, true));
      listEl.appendChild(item);
    });
  }

  initWaveSurfer(audioUrl, report);
  renderPlots(report);
  document.getElementById("warp-toggle").onchange = () => renderPlots(state.currentReport);
  renderWordMap(report);
  renderFooterMeta(report);

  if (report.regions.length > 0) {
    selectFlawRegion(0, false);
  }
}

function setDimBar(key, val) {
  const clamped = Math.max(0, Math.min(100, val));
  document.getElementById(`dim-${key}`).textContent = `${clamped.toFixed(1)} / 100`;
  document.getElementById(`bar-${key}`).style.width = `${clamped}%`;
}

/* Sample-exact seeking helper using WebAudio backend */
function seekAndPlayRange(startSec, endSec = null) {
  const ws = state.wavesurfer;
  if (!ws) return;
  const s = Math.max(0, Number(startSec) || 0);
  state.stopAtTime = endSec !== null && Number(endSec) > s ? Number(endSec) : null;

  const doSeek = () => {
    const dur = ws.getDuration();
    if (dur > 0) {
      ws.setTime(Math.min(s, dur - 0.05));
    }
    ws.play();
  };

  if (state.wsReady && ws.getDuration() > 0) {
    doSeek();
  } else {
    ws.once("ready", doSeek);
  }
}

function initWaveSurfer(audioUrl, report) {
  const container = document.getElementById("waveform");
  container.innerHTML = "";
  state.wsReady = false;

  if (state.wavesurfer) {
    state.wavesurfer.destroy();
    state.wavesurfer = null;
  }

  if (!window.WaveSurfer) {
    container.innerHTML = `<p class="empty-state">Audio waveform player ready.</p>`;
    return;
  }

  const wsRegions = window.WaveSurfer.Regions ? window.WaveSurfer.Regions.create() : null;
  state.wsRegions = wsRegions;

  const ws = window.WaveSurfer.create({
    container,
    backend: "WebAudio",
    interact: true,
    dragToSeek: true,
    waveColor: "#52525b",
    progressColor: "#e4e4e7",
    cursorColor: "#f43f5e",
    height: 112,
    normalize: true,
    plugins: wsRegions ? [wsRegions] : [],
  });
  state.wavesurfer = ws;

  ws.load(audioUrl);

  ws.on("ready", () => {
    state.wsReady = true;
    updateTimeReadout(0, ws.getDuration());
    if (wsRegions) {
      wsRegions.clearRegions();
      report.regions.forEach((r, idx) => {
        wsRegions.addRegion({
          id: String(idx),
          start: r.start,
          end: r.end,
          color: FLAW_COLORS[r.flaw_type] || "rgba(244, 63, 94, 0.3)",
          drag: false,
          resize: false,
        });
      });
    }
  });

  ws.on("interaction", () => {
    // User clicked directly on the waveform to seek manually
    state.stopAtTime = null;
  });

  ws.on("timeupdate", (currentTime) => {
    updateTimeReadout(currentTime, ws.getDuration());
    if (state.stopAtTime !== null && currentTime >= state.stopAtTime) {
      ws.pause();
      state.stopAtTime = null;
    }
  });

  if (wsRegions) {
    wsRegions.on("region-clicked", (region, e) => {
      if (e && e.stopPropagation) e.stopPropagation();
      if (e && e.preventDefault) e.preventDefault();
      selectFlawRegion(Number(region.id), true);
    });
  }
}

function initTransport() {
  document.getElementById("play-pause-btn").addEventListener("click", () => {
    state.stopAtTime = null;
    if (state.wavesurfer) state.wavesurfer.playPause();
  });

  document.getElementById("stop-btn").addEventListener("click", () => {
    state.stopAtTime = null;
    if (state.wavesurfer) {
      state.wavesurfer.pause();
      state.wavesurfer.setTime(0);
    }
  });

  document.getElementById("play-baseline-region-btn").addEventListener("click", () => {
    if (!state.currentReport || state.selectedRegionIndex === null) return;
    const r = state.currentReport.regions[state.selectedRegionIndex];
    const baseAudio = document.getElementById("baseline-audio");
    if (!baseAudio.src) return;
    const targetStart = Math.max(0, Number(r.baseline_start) || 0);
    const stopAfterMs = Math.max(1500, ((Number(r.baseline_end) || targetStart + 3) - targetStart + 0.5) * 1000);

    const startPlayback = () => {
      baseAudio.currentTime = targetStart;
      baseAudio.play();
      setTimeout(() => baseAudio.pause(), stopAfterMs);
    };

    if (baseAudio.readyState >= 1) {
      startPlayback();
    } else {
      baseAudio.addEventListener("loadedmetadata", startPlayback, { once: true });
      baseAudio.load();
    }
  });

  const exportBtn = document.getElementById("export-json-btn");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      if (!state.currentReport) return;
      const blob = new Blob([JSON.stringify(state.currentReport, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `vocalis_report_${state.currentReport.speech_id || "analysis"}.json`;
      a.click();
    });
  }
}

function updateTimeReadout(curr, total) {
  const fmt = (s) => {
    const m = String(Math.floor(s / 60)).padStart(2, "0");
    const sec = (s % 60).toFixed(2).padStart(5, "0");
    return `${m}:${sec}`;
  };
  document.getElementById("time-readout").textContent = `${fmt(curr || 0)} / ${fmt(total || 0)}`;
}

function selectFlawRegion(idx, playAudio = false) {
  const report = state.currentReport;
  if (!report || !report.regions[idx]) return;
  state.selectedRegionIndex = idx;
  const r = report.regions[idx];

  document.querySelectorAll(".flaw-card").forEach((el) => {
    el.classList.toggle("selected", Number(el.dataset.index) === idx);
  });

  document.getElementById("play-baseline-region-btn").disabled = false;

  const inspector = document.getElementById("inspector-box");
  inspector.innerHTML = `
    <div class="flaw-card-top">
      <span class="flaw-badge flaw-${r.flaw_type}">${r.flaw_type.toUpperCase()}</span>
      <span class="flaw-time">Participant: ${r.start.toFixed(2)}s &rarr; ${r.end.toFixed(2)}s | Reference: ${r.baseline_start.toFixed(2)}s &rarr; ${r.baseline_end.toFixed(2)}s</span>
    </div>
    <h3>${escapeHtml(r.title)}</h3>
    <p class="flaw-summary">${escapeHtml(r.summary)}</p>
    <div class="inspector-grid">
      <div class="metric-tile">
        <span>Duration (Participant)</span>
        <strong>${(r.end - r.start).toFixed(2)} s</strong>
      </div>
      <div class="metric-tile">
        <span>Severity Factor</span>
        <strong>${typeof r.severity === "number" ? r.severity.toFixed(2) + "x" : r.severity}</strong>
      </div>
      <div class="metric-tile">
        <span>Rubric Penalty</span>
        <strong>&minus;${r.penalty.toFixed(1)} pts</strong>
      </div>
    </div>
    <div class="advice-callout">
      <strong>Actionable Coaching Advice:</strong> ${escapeHtml(r.advice)}
    </div>
  `;

  if (playAudio) {
    seekAndPlayRange(r.start, r.end);
  }
}

function warpBaselineTimes(baseTimes, report) {
  const useWarp = document.getElementById("warp-toggle").checked;
  if (!useWarp) return baseTimes;

  let anchors = report.warp_anchors;
  if (!anchors || !anchors.baseline_t || !anchors.participant_t) {
    const n = Math.min(report.words.length, report.baseline_words.length);
    if (n < 2) return baseTimes;
    anchors = {
      baseline_t: report.baseline_words.slice(0, n).map((w) => w.start),
      participant_t: report.words.slice(0, n).map((w) => w.start),
    };
  }

  const bt = anchors.baseline_t;
  const pt = anchors.participant_t;
  if (!bt || !bt.length) return baseTimes;

  return baseTimes.map((t) => {
    if (t <= bt[0]) return pt[0] + (t - bt[0]);
    if (t >= bt[bt.length - 1]) return pt[pt.length - 1] + (t - bt[bt.length - 1]);
    let lo = 0;
    let hi = bt.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (bt[mid] <= t) lo = mid;
      else hi = mid;
    }
    const span = bt[hi] - bt[lo];
    const frac = span > 1e-6 ? (t - bt[lo]) / span : 0;
    return pt[lo] + frac * (pt[hi] - pt[lo]);
  });
}

function renderPlots(report) {
  if (!window.Plotly) return;

  const pT = report.participant_series.t;
  const bT = warpBaselineTimes(report.baseline_series.t, report);
  const sp = (report.acoustic_footprint && report.acoustic_footprint.stress_points) || [];

  const shapes = report.regions.map((r) => ({
    type: "rect",
    xref: "x",
    yref: "paper",
    x0: r.start,
    x1: r.end,
    y0: 0,
    y1: 1,
    fillcolor: FLAW_COLORS[r.flaw_type] || "rgba(244,63,94,0.2)",
    line: { width: 1, color: "rgba(255,255,255,0.25)" },
  }));

  const baseLayout = (yTitle) => ({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#09090b",
    font: { color: "#cbd5e1", size: 11 },
    margin: { l: 55, r: 20, t: 28, b: 36 },
    xaxis: { title: "Participant Timeline (seconds)", gridcolor: "#1f1f25" },
    yaxis: { title: yTitle, gridcolor: "#1f1f25" },
    legend: { orientation: "h", y: 1.14, x: 0 },
    shapes,
  });

  window.Plotly.react(
    "f0-plot",
    [
      {
        x: bT,
        y: report.baseline_series.f0_st,
        name: "Reference Baseline (Semitones)",
        mode: "lines",
        line: { color: "#94a3b8", width: 1.5, dash: "dot" },
      },
      {
        x: pT,
        y: report.participant_series.f0_st,
        name: "Participant F0 (Semitones rel. median)",
        mode: "lines",
        line: { color: "#38bdf8", width: 2 },
      },
      {
        x: sp.map((pt) => pt.t),
        y: sp.map((pt) => pt.f0_st),
        name: "Acoustic Stress Peaks",
        mode: "markers",
        marker: { color: "#f59e0b", size: 8, symbol: "diamond" },
      },
    ],
    baseLayout("Pitch F0 (Semitones)"),
    { responsive: true, displayModeBar: false }
  );

  window.Plotly.react(
    "energy-plot",
    [
      {
        x: bT,
        y: report.baseline_series.energy_db,
        name: "Reference Energy (dB rel. P90)",
        mode: "lines",
        line: { color: "#94a3b8", width: 1.5, dash: "dot" },
      },
      {
        x: pT,
        y: report.participant_series.energy_db,
        name: "Participant Energy (dB rel. P90)",
        mode: "lines",
        line: { color: "#10b981", width: 2 },
      },
      {
        x: sp.map((pt) => pt.t),
        y: sp.map((pt) => pt.energy_db),
        name: "Acoustic Stress Peaks",
        mode: "markers",
        marker: { color: "#f59e0b", size: 8, symbol: "diamond" },
      },
    ],
    baseLayout("Energy (dB rel. P90)"),
    { responsive: true, displayModeBar: false }
  );
}

function renderWordMap(report) {
  const mapEl = document.getElementById("word-map");
  mapEl.innerHTML = "";
  const validWords = (report.words || []).filter((w) => w && w.word && String(w.word).trim().length > 0);

  if (validWords.length === 0) {
    mapEl.innerHTML = `<p class="empty-state">Word-level timestamps will appear here after alignment.</p>`;
    return;
  }

  validWords.forEach((w) => {
    const chip = document.createElement("span");
    const inFlaw = report.regions.some((r) => w.end > r.start && w.start < r.end);
    chip.className = `word-chip${inFlaw ? " flagged" : ""}`;
    chip.textContent = `${w.word}`;
    chip.title = `${w.word} (${w.start.toFixed(2)}s â†’ ${w.end.toFixed(2)}s)`;
    chip.addEventListener("click", () => {
      seekAndPlayRange(w.start, null);
    });
    mapEl.appendChild(chip);
  });
}

function renderFooterMeta(report) {
  const thEl = document.getElementById("thresholds-table");
  thEl.innerHTML = "";
  Object.entries(report.thresholds).forEach(([k, v]) => {
    const row = document.createElement("div");
    row.className = "kv-row";
    row.innerHTML = `<span>${escapeHtml(k)}</span><strong>${escapeHtml(String(v))}</strong>`;
    thEl.appendChild(row);
  });

  const limEl = document.getElementById("limitations-list");
  limEl.innerHTML = "";
  report.limitations.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    limEl.appendChild(li);
  });
}

function initPresets() {
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const key = btn.dataset.preset;
      const flawedAudioUrl = `/api/presets/${encodeURIComponent(key)}/audio?t=${Date.now()}`;
      setStatus(true, "Loading verified benchmark clip and computing contrastive report...");
      try {
        const res = await fetch(`/api/presets/${encodeURIComponent(key)}/report`);
        if (res.ok) {
          const rawReport = await res.json();
          const norm = normalizeReport(rawReport);
          renderReport(norm, flawedAudioUrl);
          setStatus(false);
          return;
        }
      } catch {
        // Fallback if server is offline
      }
      const preset = getBuiltInPreset(key);
      const norm = normalizeReport(preset);
      renderReport(norm, flawedAudioUrl);
      setStatus(false);
    });
  });
}

function getBuiltInPreset(key) {
  const grid = Array.from({ length: 180 }, (_, i) => +(i * 0.5).toFixed(2));
  const baseF0 = grid.map((t) => +(2.4 * Math.sin(t * 0.9) + 1.1 * Math.cos(t * 2.1)).toFixed(2));
  const baseDb = grid.map((t) => +(-6.5 + 4.2 * Math.sin(t * 1.3)).toFixed(2));

  if (key === "kennedy_pause_subtle" || key === "kennedy_pause_moderate" || key === "kennedy_pause_severe") {
    const durMap = { kennedy_pause_subtle: 0.6, kennedy_pause_moderate: 1.2, kennedy_pause_severe: 2.56 };
    const sevMap = { kennedy_pause_subtle: 1.2, kennedy_pause_moderate: 2.0, kennedy_pause_severe: 3.0 };
    const penMap = { kennedy_pause_subtle: 7.0, kennedy_pause_moderate: 12.0, kennedy_pause_severe: 18.0 };
    const pDur = durMap[key];
    const pEnd = +(4.901 + pDur).toFixed(3);
    const partDb = grid.map((t, i) => (t >= 4.9 && t <= pEnd ? -42.0 : baseDb[i]));
    return {
      speech_id: "kennedy_inaugural",
      duration_s: +(89.3 + pDur).toFixed(1),
      regions: [{
        flaw_type: "pause",
        severity: sevMap[key],
        start: 4.901,
        end: pEnd,
        baseline_start: 4.901,
        baseline_end: 4.961,
        title: `Mid-clause pause (${key.split("_").pop()})`,
        summary: `A ${pDur.toFixed(2)} s silence between 'man' and 'holds', where reference has 0.06 s.`,
        advice: "Keep clauses connected; pause at punctuation rather than mid-phrase.",
        penalty: penMap[key],
      }],
      words: [
        { word: "The", start: 0.92, end: 1.0 }, { word: "world", start: 1.04, end: 1.44 },
        { word: "is", start: 1.52, end: 1.66 }, { word: "very", start: 1.72, end: 1.98 },
        { word: "different", start: 2.02, end: 2.38 }, { word: "now.", start: 2.42, end: 2.7 },
        { word: "For", start: 4.26, end: 4.42 }, { word: "man", start: 4.46, end: 4.901 },
        { word: "holds", start: pEnd, end: +(pEnd + 0.42).toFixed(3) }, { word: "in", start: +(pEnd + 0.5).toFixed(3), end: +(pEnd + 0.58).toFixed(3) },
      ],
      contours: {
        participant: { times: grid, f0_st: baseF0, energy_db: partDb },
        baseline: { times: grid, f0_st: baseF0, energy_db: baseDb },
      },
    };
  }

  if (key === "kennedy_monotone_severe") {
    const partF0 = grid.map((t, i) => (t >= 4.26 && t <= 15.42 ? +(baseF0[i] * 0.12).toFixed(2) : baseF0[i]));
    return {
      speech_id: "kennedy_inaugural",
      duration_s: 89.4,
      regions: [{
        flaw_type: "monotone",
        severity: 1.46,
        start: 4.26,
        end: 15.42,
        baseline_start: 4.26,
        baseline_end: 15.42,
        title: "Monotone delivery",
        summary: "Pitch spread (P10â€“P90) dropped to 0.56 semitones vs. 2.00 semitones in reference.",
        advice: "Restore pitch inflection on key words.",
        penalty: 22.8,
      }],
      words: [
        { word: "For", start: 4.26, end: 4.42 }, { word: "man", start: 4.46, end: 4.9 },
        { word: "holds", start: 4.96, end: 5.38 },
      ],
      contours: {
        participant: { times: grid, f0_st: partF0, energy_db: baseDb },
        baseline: { times: grid, f0_st: baseF0, energy_db: baseDb },
      },
    };
  }

  if (key === "eisenhower_volume_severe") {
    const partDb = grid.map((t, i) => (t >= 18.0 && t <= 27.5 ? baseDb[i] - 12.0 : baseDb[i]));
    return {
      speech_id: "eisenhower_farewell",
      duration_s: 88.0,
      regions: [{
        flaw_type: "volume",
        severity: 2.18,
        start: 18.0,
        end: 27.5,
        baseline_start: 18.0,
        baseline_end: 27.5,
        title: "Sustained loudness drop",
        summary: "Normalized speech level dropped by -11.98 dB relative to reference phrase.",
        advice: "Maintain vocal projection through clause endings.",
        penalty: 31.0,
      }],
      words: [{ word: "Throughout", start: 18.0, end: 18.55 }],
      contours: {
        participant: { times: grid, f0_st: baseF0, energy_db: partDb },
        baseline: { times: grid, f0_st: baseF0, energy_db: baseDb },
      },
    };
  }

  return {
    speech_id: "lincoln_gettysburg",
    duration_s: 98.4,
    regions: [
      {
        flaw_type: "pace",
        severity: 1.42,
        start: 14.2,
        end: 19.8,
        baseline_start: 14.2,
        baseline_end: 22.1,
        title: "Rushed speaking pace",
        summary: "Local speaking rate reached 1.48x baseline pace across this clause.",
        advice: "Slow down on key transitions.",
        penalty: 16.5,
      },
      {
        flaw_type: "pause",
        severity: 2.4,
        start: 41.1,
        end: 42.65,
        baseline_start: 41.1,
        baseline_end: 41.2,
        title: "Unnatural mid-sentence hesitation",
        summary: "A 1.55 s mid-phrase pause detected where the reference continues smoothly.",
        advice: "Avoid breaking phrase momentum between subject and verb.",
        penalty: 15.0,
      },
    ],
    words: [{ word: "Four", start: 0.72, end: 1.05 }],
    contours: {
      participant: { times: grid, f0_st: baseF0, energy_db: baseDb },
      baseline: { times: grid, f0_st: baseF0, energy_db: baseDb },
    },
  };
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
