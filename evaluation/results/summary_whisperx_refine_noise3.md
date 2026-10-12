
### Synthetic clips: detection rate (IoU >= 0.3)

| flaw_type   | subtle   | moderate   | severe   |
|:------------|:---------|:-----------|:---------|
| monotone    | 50%      | 75%        | 100%     |
| pace        | 0%       | 100%       | 100%     |
| pause       | 100%     | 100%       | 100%     |
| slow        | 0%       | 100%       | 100%     |
| stumble     | 100%     | 100%       | 100%     |
| volume      | 100%     | 100%       | 100%     |

### Synthetic clips: mean IoU of detected flaws

| flaw_type   |   subtle |   moderate |   severe |
|:------------|---------:|-----------:|---------:|
| monotone    |     0.59 |       0.82 |     0.84 |
| pace        |   nan    |       0.85 |     0.92 |
| pause       |     0.91 |       0.95 |     0.98 |
| slow        |   nan    |       0.88 |     0.87 |
| stumble     |     0.99 |       1    |     0.99 |
| volume      |     0.77 |       0.75 |     0.75 |

Overall detection rate 85% of 72 flaws; 4 false-positive regions over 72 clips (0.06 per clip, 0.04 per minute of audio).

### Synthetic clips: false positives by predicted type

| predicted_type   |   false positives |   per minute |
|:-----------------|------------------:|-------------:|
| monotone         |                 4 |         0.04 |

### Own clips: detection rate (IoU >= 0.3)

| flaw_type   | subtle   | moderate   |
|:------------|:---------|:-----------|
| monotone    | 0%       | 0%         |
| pace        | 0%       | 0%         |
| pause       | 100%     | 100%       |
| volume      | 0%       | 0%         |

### Own clips: mean IoU of detected flaws

| flaw_type   |   subtle |   moderate |
|:------------|---------:|-----------:|
| pause       |        1 |          1 |

Overall detection rate 25% of 8 flaws; 6 false-positive regions over 3 clips (2.00 per clip, 1.16 per minute of audio).

### Own clips: false positives by predicted type

| predicted_type   |   false positives |   per minute |
|:-----------------|------------------:|-------------:|
| pause            |                 6 |         1.16 |

By recording:

| clip_id             |   false positives |
|:--------------------|------------------:|
| own_flawed_moderate |                 2 |
| own_flawed_subtle   |                 2 |
| own_good            |                 2 |