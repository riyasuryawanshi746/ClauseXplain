# Span head vs classifier scan (fold 2, 500 windows)

| Method | forward passes / window | ms / window | token-F1 | exact match | IoU |
|---|---|---|---|---|---|
| span head | 1 | 54 | 0.764 | 0.540 | 0.713 |
| classifier scan | 51 | 1340 | 0.662 | 0.106 | 0.592 |

Candidate ceiling (best any candidate could score): token-F1 0.963.
Span head minus scan, token-F1: +0.102 (95% bootstrap interval +0.066 to +0.139).

One fold, a random sample of test windows, windowed task. Speed is wall-clock on this hardware.