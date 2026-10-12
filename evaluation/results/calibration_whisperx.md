# Threshold calibration (whisperx alignment)

Held-out F1 per threshold: tuned on one group of speeches, scored on the other.

| threshold | default | calibrated (per fold) | held-out F1 default | held-out F1 calibrated |
|---|---|---|---|---|
| pace_log | 1.25x | 1.30x, 1.10x | 0.80 | 0.62 |
| pitch_log | 0.60x | 0.70x, 0.60x | 0.73 | 0.73 |
| energy_db | 2.5 | 2.5, 2.5 | 1.00 | 1.00 |
| pause_excess_s | 0.3 | 0.5, 0.5 | 1.00 | 1.00 |

## Overall held-out performance (all flaw types together)

| configuration | precision | recall | F1 | false positives per clip |
|---|---|---|---|---|
| untuned defaults | 0.94 | 0.85 | 0.89 | 0.06 |
| calibrated thresholds | 0.74 | 0.90 | 0.79 | 0.49 |
| calibrated + noise-adaptive (noise_k) | 0.74 | 0.89 | 0.78 | 0.47 |

noise_k chosen per fold: [5.0, 5.0]

Final thresholds (tuned on all speeches; their in-sample score is optimistic):

```json
{
  "window_words": 5,
  "pace_log": 0.26236426446749106,
  "pitch_log": -0.35667494393873245,
  "energy_db": 2.5,
  "pause_excess_s": 0.5,
  "pause_punct_excess_s": 1.0,
  "min_region_words": 3,
  "merge_gap_words": 1,
  "refine_boundaries": true,
  "noise_k": 5.0
}
```