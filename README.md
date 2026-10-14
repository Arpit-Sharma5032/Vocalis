# Vocalis: Contrastive Speech Analytics & Temporal Flaw Grounding

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![WhisperX](https://img.shields.io/badge/Alignment-WhisperX_wav2vec2-FF6F00)
![Praat Parselmouth](https://img.shields.io/badge/Acoustics-Praat_Parselmouth_%26_Librosa-8A2BE2)
![Tests](https://img.shields.io/badge/Tests-77_Passing-10B981)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

**Vocalis** is an end-to-end contrastive speech evaluation system built for **Track C**. Given a reference ("ideal") speech and a participant's delivery of the same text, Vocalis performs silence-refined forced alignment, extracts speaker-normalized prosodic contours, localizes delivery flaws to exact word timestamps, generates mathematically grounded causal explanations, and computes a deterministic rubric score.

---

## 1. System Architecture

```
+-----------------------------------------------------------------------------------+
|                        VOCALIS CONTRASTIVE SPEECH PIPELINE                        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Reference Audio + Transcript]          [Participant Audio (Upload / Mic)]       |
|                |                                         |                        |
|                +-------------------+---------------------+                        |
|                                    v                                              |
|                     1. WhisperX Forced Alignment (16 kHz)                         |
|                        + Silence-Aware Boundary Refinement                        |
|                                    |                                              |
|                                    v                                              |
|                     2. Speaker-Agnostic Feature Extraction                        |
|                        - Two-Pass Praat F0 -> Semitones rel. speaker median       |
|                        - 25 ms RMS Energy -> dB rel. speaker P90 speech level     |
|                        - 13 MFCCs (CMVN) + Adaptive Noise-Floor Pause Detector    |
|                                    |                                              |
|                                    v                                              |
|                     3. Word-Level Contrastive Deviation Engine                    |
|                        - 1-to-1 Word Twin Matching & 5-Word Neighbourhoods        |
|                        - Global Speaker Median Offset Subtraction                 |
|                        - Punctuation-Aware Excess Pause & Stumble Rules           |
|                        - 2.5s Baseline-Span Persistence Filter                    |
|                                    |                                              |
|                                    v                                              |
|                     4. Causal Explainability & Rubric Scoring                     |
|                        - Quantitative rationale + actionable coaching advice      |
|                        - Reproducible 0-100 Rubric across 4 delivery dimensions   |
|                                    |                                              |
|                                    v                                              |
|                     5. Interactive Studio Dashboard (WaveSurfer + Plotly)         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Custom Contrastive Dataset & Bad-Mirror Gradient

G Drive link - https://drive.google.com/file/d/1B1Pf7y0hASzi5KbZ-r2Kxu4yFILNRmLC/view?usp=sharing

Rather than relying on unverified synthetic TTS pairs or manual timestamp guessing, Vocalis constructs a **hybrid contrastive corpus** combining **72 programmatic perturbations** (with exact analytical ground-truth timestamps on the output timeline) and **real human-performed cross-speaker recordings**:

1. **Four Public-Domain Reference Baselines (`data/raw_audio/`)**:
   - `kennedy_inaugural` — John F. Kennedy, Inaugural Address (1961), 90 s, ~105 WPM (deliberate rhetorical oratory).
   - `eisenhower_farewell` — Dwight D. Eisenhower, Farewell Address (1961), 88 s, ~130 WPM (measured studio address).
   - `fdr_fireside_01` — FDR Fireside Chat 1 (1933 text, LibriVox public-domain reading), 89 s, ~176 WPM (brisk conversational reading).
   - `lincoln_gettysburg` — Gettysburg Address (1863 text, LibriVox public-domain reading), 104 s, ~155 WPM (interpretive reading).
2. **Six Flaw Types × Three Severity Levels (`dataset_builder/perturb.py`)**:
   - **Pace (Rushed)**: Praat PSOLA pitch-preserving time-stretch at `1.15x` (subtle), `1.50x` (moderate), `2.00x` (severe).
   - **Slow (Dragged)**: Praat PSOLA time-stretch at `0.87x` (subtle), `0.67x` (moderate), `0.50x` (severe).
   - **Pause (Hesitation)**: Mid-clause silence insertion of `0.6s` (subtle), `1.2s` (moderate), `2.5s` (severe).
   - **Volume (Trail-off)**: Gain attenuation of `-3.0 dB` (subtle), `-6.0 dB` (moderate), `-12.0 dB` (severe).
   - **Monotone (Flat Pitch)**: Praat pitch-tier compression toward the window median $F_0$ by `50%` (subtle), `75%` (moderate), `98%` (severe).
   - **Stumble (Disfluency)**: Verbatim word repetition (1, 2, or 3 repeats) with `0.25s` hesitation gaps and `10 ms` cosine fades.
3. **Real Human-Performed Cross-Speaker Validation (`data/own_recordings/` & `data/own_recordings_b/`)**:
   - Three takes of `lincoln_gettysburg` (`own_good`, `own_flawed_moderate`, `own_flawed_subtle`) performed by a second human speaker with planned flaws on specific sentences, plus an untouched held-out clean take (`ownb_good`) on `kennedy_inaugural`.

---

## 3. Speaker-Agnostic Signal Processing (`backend/features.py`, `backend/normalization.py`)

To compare speakers with different vocal tracts and recording gains without false alarms:
- **Two-Pass $F_0$ & Semitone Normalization**: Pass 1 extracts raw pitch over $[75, 500]\text{ Hz}$ to estimate speaker quartiles $Q_1, Q_3$. Pass 2 restricts search to $[0.75 Q_1, 2.5 Q_3]$, repairs octave jumps iteratively, removes isolated single-frame glitches and voiced bursts $<50\text{ ms}$, and converts to semitones relative to the speaker's median pitch:
  $$F_0^{\text{ST}}(t) = 12 \log_2 \left(\frac{F_0(t)}{\text{median}(F_0)}\right)$$
- **P90-Normalized Energy Contour**: Frame RMS energy (25 ms window, 10 ms hop) is converted to dB and normalized against the speaker's 90th-percentile voiced speech level ($L_{90}$), eliminating microphone gain differences.
- **Silence-Aware Word Boundary Refinement**: Forced aligners (wav2vec2/WhisperX) frequently absorb mid-phrase silences into an adjacent word's duration (e.g., stretching a 0.4 s word across a 2.5 s pause). Vocalis inspects the energy contour inside every aligned word and trims internal silent stretches $\ge 0.20\text{ s}$ back to the longest voiced span.

---

## 4. Quantitative Evaluation Results (`evaluation/`)

### 4.1 Synthetic Corpus (72 Flawed Clips across 4 Speeches, WhisperX Alignment)

| Flaw Type | Subtle Recall | Moderate Recall | Severe Recall | Mean IoU (Detected) |
| :--- | :---: | :---: | :---: | :---: |
| **Pace (Fast)** | 0% | 100% | 100% | 0.61 |
| **Slow (Dragged)** | 0% | 100% | 100% | 0.89 |
| **Pause (Mid-Clause)** | **100%** | **100%** | **100%** | **0.95** |
| **Volume Drop** | **100%** | **100%** | **100%** | **0.76** |
| **Monotone** | 25% | 75% | **100%** | **0.76** |
| **Stumble** | **100%** | **100%** | **100%** | **0.99** |

- **Overall Synthetic Detection Rate**: **85%** across all severities (**98%** on moderate + severe), with **only 4 false-positive regions across 72 clips (0.04 FP/min)** after silence-aware boundary refinement and the 2.5 s persistence rule.
- **Ablation (Silence Refinement)**: Without silence-aware word boundary refinement, WhisperX absorbed inserted pauses into adjacent words, causing 148 false-positive regions (1.32 FP/min). Refinement reduced synthetic false positives by **97.3%**.

### 4.2 Real Human-Performed Cross-Speaker Validation

- **Lincoln Validation Set (`own_recordings`)**: **75% recall** (100% on pause, volume, and monotone; subtle pace below threshold) with **2.3 false-positive regions/min** across a completely different human voice and microphone.
- **Untouched Held-Out Clean Test (`own_recordings_b` on Kennedy)**: **1 false positive in 75 seconds (0.80 FP/min; 95% Poisson CI: 0.02–4.5/min)** on a transcript never used during threshold development.

### 4.3 Rubric Score Monotonicity across Severity Gradient

| Flaw Type | Clean Baseline | Subtle | Moderate | Severe |
| :--- | :---: | :---: | :---: | :---: |
| **Pace** | 100.0 | 100.0 | 86.0 | 75.8 |
| **Slow** | 100.0 | 100.0 | 86.5 | 66.7 |
| **Pause** | 100.0 | 94.0 | 88.0 | 85.0 |
| **Volume** | 100.0 | 89.7 | 78.2 | 74.3 |
| **Monotone** | 100.0 | 100.0 | 87.4 | 81.8 |
| **Stumble** | 100.0 | 95.0 | 90.0 | 85.0 |

---

## 5. Quickstart Guide

### Prerequisites
- **Python 3.11** (via Conda or `venv`)
- **FFmpeg** installed and on `PATH` (`winget install Gyan.FFmpeg` or `conda install -c conda-forge ffmpeg`)

### Installation & Setup
```bash
# 1. Clone the repository
git clone [https://github.com/Arpit-Sharma5032/Vocalis.git](https://github.com/Arpit-Sharma5032/Vocalis.git)
cd Vocalis

# 2. Create and activate environment
conda create -n vocalis python=3.11 -y
conda activate vocalis

# 3. Install pinned dependencies
pip install -r requirements.txt
```

### Rebuild the Contrastive Dataset & Run Verification
```bash
# Deterministically generate all 72 perturbed clips + manifest.csv (seeded Praat PSOLA)
python -m dataset_builder.build_dataset

# Independently verify all 72 audio files against their ground-truth labels
python scripts/verify_dataset.py

# Run the 77-test suite
python -m pytest
```

### Start the Interactive Studio Dashboard
```bash
uvicorn backend.api:app --reload
```
Open **`http://127.0.0.1:8000`** in your browser to use the interactive dashboard, or **`http://127.0.0.1:8000/docs`** for the interactive OpenAPI Swagger specification.

---

## 6. Documented Limitations & Negative Results

1. **Persistence Cutoff (`min_region_s = 2.5s`)**: To prevent natural cross-speaker prosodic variance from triggering false alarms, pace, slow, monotone, and volume anomalies must span at least 2.5 s of baseline speech. Brief 1–2 word prosodic dips are intentionally ignored.
2. **Subtle Pace/Slow Threshold Floor**: Time-stretches of `1.15x` and `0.87x` fall within natural human clause-to-clause rate variability and are not flagged under our conservative default thresholds (`1.25x` / `0.80x`).
3. **Cross-Speech Threshold Calibration & `noise_k` (Negative Results)**: Two-fold cross-speech grid search (`evaluation/calibrate.py`) achieved 0.79 held-out F1 vs. 0.89 for our domain-reasoned defaults due to speaker style variance; we therefore retain the untuned defaults and report both honestly.
4. **Verbatim Transcript Requirement for Stumbles**: Forced alignment requires the repeated token in the transcript to align a word stumble; when aligning against a canonical transcript without ASR transcript diffing, word repeats are absorbed.

---

## 7. Attributions & Licenses

- **Baseline Speeches**: Public-domain US federal government works (JFK 1961 Inaugural, Eisenhower 1961 Farewell via Miller Center / Presidential Libraries) and public-domain LibriVox volunteer readings (FDR Fireside Chat 1, Lincoln Gettysburg Address). Full details in [`data/README.md`](data/README.md).
- **Codebase**: Released under the [MIT License](LICENSE). Own validation recordings in `data/own_recordings/` are released under CC BY 4.0.