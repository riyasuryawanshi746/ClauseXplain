# Span head vs classifier scan (folds 0 to 2)

Question: is the span head a better and cheaper way to find a clause inside a long window than scanning the window with the category classifier?

**Protocol** (`phase3_model/scan_vs_span.py`, notebook: `notebooks/scan_vs_span.ipynb`, Tesla T4)
- Same windows, same gold spans, same scoring as training. 500 random test windows per fold (answer guaranteed inside a 3000-char window).
- **Span head:** the joint model (`ckpt_joint_cls_span_foldK`), one forward pass per window.
- **Classifier scan:** the window is cut into sentence-aligned candidate spans (about 50 per window), each candidate goes through the category classifier (`ckpt_cls_only_foldK`), and the candidate with the highest probability for the target category wins.
- 95% interval: paired bootstrap over windows (2000 resamples).

| Fold | Span head token-F1 | Scan token-F1 | Difference (95% CI) | Exact match (head / scan) | ms per window (head / scan) |
|---|---|---|---|---|---|
| 0 | 0.748 | 0.615 | +0.132 (+0.092, +0.170) | 0.494 / 0.134 | 54 / 1303 |
| 1 | 0.755 | 0.613 | +0.142 (+0.105, +0.180) | 0.510 / 0.122 | 54 / 1300 |
| 2 | 0.764 | 0.662 | +0.102 (+0.066, +0.139) | 0.540 / 0.106 | 54 / 1340 |
| **Mean** | **0.756** | **0.630** | **+0.125** | | **54 / 1314 (about 24x faster)** |

The scan needs about 50 forward passes per window against 1 for the span head. The best any candidate could score (candidate ceiling) is 0.96 to 0.97, so the scan is limited by the classifier's choices, not by the candidate set.

**Read with care**
- The two methods use different checkpoints (joint vs cls_only), by design: each is the natural model for its job.
- Per-fold intervals are over windows within one fold. Three folds is too few for a cross-fold interval, so call it "consistent across 3 folds", not "significant".
- Windowed task, not comparable to full-contract CUAD numbers. Latency is wall-clock on a T4.
- A 20-window trial run (+0.212) was overwritten by the 500-window run. Do not cite it.
- LIME is deliberately not used: it explains words, it does not localise a clause, so it is not a like-for-like baseline.
