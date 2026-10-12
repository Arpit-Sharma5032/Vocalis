# Threshold calibration (whisperx alignment)

Held-out F1 per threshold: tuned on one group of speeches, scored on the other.

| threshold | default | calibrated (per fold) | held-out F1 default | held-out F1 calibrated |
|---|---|---|---|---|
| pace_log | 1.25x | 1.10x, 1.10x | 0.27 | 0.38 |
| pitch_log | 0.60x | 0.70x, 0.60x | 0.73 | 0.73 |
| energy_db | 2.5 | 2.5, 2.5 | 1.00 | 1.00 |
| pause_excess_s | 0.3 | 0.5, 0.5 | 0.19 | 0.19 |

Final thresholds (tuned on all speeches; their in-sample score is optimistic):

```json
{
  "window_words": 5,
  "pace_log": 0.09531017980432493,
  "pitch_log": -0.35667494393873245,
  "energy_db": 2.5,
  "pause_excess_s": 0.5,
  "min_region_words": 3,
  "merge_gap_words": 1
}
```