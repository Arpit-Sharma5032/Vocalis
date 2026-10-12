
### Synthetic clips: detection rate (IoU >= 0.3)

| flaw_type   | subtle   | moderate   | severe   |
|:------------|:---------|:-----------|:---------|
| monotone    | 50%      | 75%        | 100%     |
| pace        | 0%       | 100%       | 100%     |
| pause       | 100%     | 75%        | 75%      |
| slow        | 0%       | 100%       | 100%     |
| stumble     | 100%     | 100%       | 100%     |
| volume      | 100%     | 100%       | 100%     |

### Synthetic clips: mean IoU of detected flaws

| flaw_type   |   subtle |   moderate |   severe |
|:------------|---------:|-----------:|---------:|
| monotone    |     0.58 |       0.82 |     0.84 |
| pace        |   nan    |       0.85 |     0.92 |
| pause       |     0.91 |       0.95 |     0.98 |
| slow        |   nan    |       0.91 |     0.89 |
| stumble     |     0.99 |       1    |     0.99 |
| volume      |     0.77 |       0.75 |     0.75 |

Overall detection rate 82% of 72 flaws; 148 false-positive regions over 72 clips (2.06 per clip, 1.32 per minute of audio).

### Synthetic clips: false positives by predicted type

| predicted_type   |   false positives |   per minute |
|:-----------------|------------------:|-------------:|
| monotone         |                 4 |         0.04 |
| pace             |                78 |         0.69 |
| pause            |                66 |         0.59 |

### Own clips: detection rate (IoU >= 0.3)

| flaw_type   | subtle   | moderate   |
|:------------|:---------|:-----------|
| monotone    | 100%     | 100%       |
| pace        | 0%       | 100%       |
| pause       | 100%     | 100%       |
| volume      | 100%     | 100%       |

### Own clips: mean IoU of detected flaws

| flaw_type   |   subtle |   moderate |
|:------------|---------:|-----------:|
| monotone    |     0.72 |       0.76 |
| pace        |   nan    |       0.39 |
| pause       |     1    |       1    |
| volume      |     0.43 |       0.75 |

Overall detection rate 88% of 8 flaws; 41 false-positive regions over 3 clips (13.67 per clip, 7.94 per minute of audio).

### Own clips: false positives by predicted type

| predicted_type   |   false positives |   per minute |
|:-----------------|------------------:|-------------:|
| monotone         |                11 |         2.13 |
| pause            |                 6 |         1.16 |
| volume           |                24 |         4.65 |

By recording:

| clip_id             |   false positives |
|:--------------------|------------------:|
| own_flawed_moderate |                13 |
| own_flawed_subtle   |                10 |
| own_good            |                18 |