
### Synthetic clips: detection rate (IoU >= 0.3)

| flaw_type   | subtle   | moderate   | severe   |
|:------------|:---------|:-----------|:---------|
| monotone    | 25%      | 75%        | 100%     |
| pace        | 0%       | 100%       | 100%     |
| pause       | 100%     | 100%       | 100%     |
| slow        | 0%       | 100%       | 100%     |
| stumble     | 100%     | 100%       | 100%     |
| volume      | 100%     | 100%       | 100%     |

### Synthetic clips: mean IoU of detected flaws

| flaw_type   |   subtle |   moderate |   severe |
|:------------|---------:|-----------:|---------:|
| monotone    |     0.81 |       0.82 |     0.84 |
| pace        |   nan    |       0.85 |     0.92 |
| pause       |     0.91 |       0.95 |     0.98 |
| slow        |   nan    |       0.88 |     0.87 |
| stumble     |     0.99 |       1    |     0.99 |
| volume      |     0.77 |       0.75 |     0.75 |

Overall detection rate 83% of 72 flaws; 0 false-positive regions over 72 clips (0.00 per clip, 0.00 per minute of audio).

### Own:own_recordings clips: detection rate (IoU >= 0.3)

| flaw_type   | subtle   | moderate   |
|:------------|:---------|:-----------|
| monotone    | 100%     | 100%       |
| pace        | 0%       | 100%       |
| pause       | 100%     | 100%       |
| volume      | 100%     | 100%       |

### Own:own_recordings clips: mean IoU of detected flaws

| flaw_type   |   subtle |   moderate |
|:------------|---------:|-----------:|
| monotone    |     0.72 |       0.76 |
| pace        |   nan    |       0.39 |
| pause       |     1    |       1    |
| volume      |     0.36 |       0.75 |

Overall detection rate 88% of 8 flaws; 12 false-positive regions over 3 clips (4.00 per clip, 2.32 per minute of audio).

### Own:own_recordings clips: false positives by predicted type

| predicted_type   |   false positives |   per minute |
|:-----------------|------------------:|-------------:|
| monotone         |                 4 |         0.77 |
| pause            |                 3 |         0.58 |
| volume           |                 5 |         0.97 |

By recording:

| clip_id             |   false positives |
|:--------------------|------------------:|
| own_flawed_moderate |                 5 |
| own_flawed_subtle   |                 2 |
| own_good            |                 5 |

### Own:own_recordings_b clips (no labelled flaws)

1 false-positive regions over 1 clips (0.80 per minute of audio).

| predicted_type   |   false positives |
|:-----------------|------------------:|
| volume           |                 1 |