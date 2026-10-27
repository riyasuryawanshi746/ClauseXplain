# Experiments index

**Only `stageB_3fold/` and `scan_vs_span/` are the numbers to cite.** Everything in `archive/` is kept for the record and is superseded.

| Folder | Run | Setup | Status |
|---|---|---|---|
| `stageB_3fold/` | 2026-10-10, T4 | cls_only, span_only, joint_cls_span; folds 0-2 of 5 (contract-grouped); 3000-char windows; 2 epochs | **Canonical.** `results.json`, `comparisons.md` (with TF-IDF and span baselines) and `partial/` (per-fold runs, `splits.json`, `config.json`). Settings: 2 epochs, 50% of span items per epoch, so span results are probably undertrained. |
| `scan_vs_span/` | folds 0-2 | span head vs classifier scan, per-fold md/json | **Canonical** for the speed/accuracy comparison. |
| `archive/2026-10-09_single_fold_3000char/` | 1 fold | same setup as stageB, one fold | Superseded by stageB_3fold. |
| `archive/2026-10-08_timing_run_6000char/` | 1 fold, 1 epoch | cls_only + span_only, 6000-char windows | Used only to estimate GPU time. |
| `archive/2026-10-07_first_run_6000char/` | 1 fold | first end-to-end run, 6000-char windows | Superseded; different window size, not comparable. |

Notes: fold 0 of a 5-fold run equals a 1-fold run with the same seed. Span scores moved between reruns of the same config (GPU nondeterminism, e.g. fold-0 span_only token-F1 0.746 vs 0.658), so don't compare single runs; use the paired 3-fold numbers.

Notebooks are in `../notebooks/`. Checkpoints are not stored in the repo.
