
### Synthetic clips: detection rate (IoU >= 0.3)

| flaw_type   | subtle   | moderate   | severe   |
|:------------|:---------|:-----------|:---------|
| monotone    | 100%     | 75%        | 100%     |
| pace        | 100%     | 100%       | 100%     |
| pause       | 100%     | 75%        | 75%      |
| slow        | 100%     | 100%       | 100%     |
| stumble     | 100%     | 100%       | 100%     |
| volume      | 100%     | 100%       | 100%     |

### Synthetic clips: mean IoU of detected flaws

| flaw_type   |   subtle |   moderate |   severe |
|:------------|---------:|-----------:|---------:|
| monotone    |     0.67 |       0.82 |     0.79 |
| pace        |     0.72 |       0.64 |     0.5  |
| pause       |     0.91 |       0.95 |     0.98 |
| slow        |     0.85 |       0.9  |     0.88 |
| stumble     |     0.99 |       1    |     0.99 |
| volume      |     0.77 |       0.75 |     0.75 |

Overall detection rate 96% of 72 flaws; 168 false-positive regions over 72 clips (2.33 per clip).

### Own clips: detection rate (IoU >= 0.3)

| flaw_type   | subtle   | moderate   |
|:------------|:---------|:-----------|
| monotone    | 100%     | 100%       |
| pace        | 100%     | 100%       |
| pause       | 100%     | 100%       |
| volume      | 100%     | 100%       |

### Own clips: mean IoU of detected flaws

| flaw_type   |   subtle |   moderate |
|:------------|---------:|-----------:|
| monotone    |     0.78 |       0.76 |
| pace        |     0.34 |       0.78 |
| pause       |     1    |       1    |
| volume      |     0.43 |       0.75 |

Overall detection rate 100% of 8 flaws; 68 false-positive regions over 3 clips (22.67 per clip).