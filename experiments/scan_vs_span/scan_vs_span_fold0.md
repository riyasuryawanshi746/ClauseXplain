# Span head vs classifier scan (fold 0, 500 windows)

| Method | forward passes / window | ms / window | token-F1 | exact match | IoU |
|---|---|---|---|---|---|
| span head | 1 | 54 | 0.748 | 0.494 | 0.698 |
| classifier scan | 50 | 1303 | 0.615 | 0.134 | 0.556 |

Candidate ceiling (best any candidate could score): token-F1 0.972.
Span head minus scan, token-F1: +0.132 (95% bootstrap interval +0.092 to +0.170).

One fold, a random sample of test windows, windowed task. Speed is wall-clock on this hardware.