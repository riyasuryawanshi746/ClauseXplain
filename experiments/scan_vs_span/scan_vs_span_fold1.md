# Span head vs classifier scan (fold 1, 500 windows)

| Method | forward passes / window | ms / window | token-F1 | exact match | IoU |
|---|---|---|---|---|---|
| span head | 1 | 54 | 0.755 | 0.510 | 0.703 |
| classifier scan | 50 | 1300 | 0.613 | 0.122 | 0.552 |

Candidate ceiling (best any candidate could score): token-F1 0.959.
Span head minus scan, token-F1: +0.142 (95% bootstrap interval +0.105 to +0.180).

One fold, a random sample of test windows, windowed task. Speed is wall-clock on this hardware.