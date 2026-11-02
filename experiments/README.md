# Experiments index

**Cite only `stageB_3fold/`, `scan_vs_span/`, `indian_eval_v4/` and `legalbert_baseline_v4/`.** Everything in `archive/` is kept for the record and is superseded.

| Folder | Run | Setup | Status |
|---|---|---|---|
| `stageB_3fold/` | 2026-10-10, T4 | cls_only, span_only, joint_cls_span; folds 0-2 of 5 (contract-grouped); 3000-char windows; 2 epochs | **Canonical.** `results.json`, `comparisons.md` (with TF-IDF and span baselines) and `partial/` (per-fold runs, `splits.json`, `config.json`). Settings: 2 epochs, 50% of span items per epoch, so span results are probably undertrained. |
| `scan_vs_span/` | folds 0-2 | span head vs classifier scan, per-fold md/json | **Canonical** for the speed/accuracy comparison. |
| `indian_eval_v4/` | 2026-11-02, T4 | Longformer joint folds 0-2 on frozen `built_v4` (334 adjudicated clauses) + judgment slice | **Canonical** Indian results (README also tabulates LegalBERT and TF-IDF). |
| `legalbert_baseline_v4/` | 2026-11-02, T4 | LegalBERT cls_only, same CUAD folds; scored on `built_v4` | **Canonical** LegalBERT baseline. |
| `indian_eval_v3/`, `legalbert_baseline/` | 2026-11-02 | same, pre-adjudication labels | Superseded by the v4 folders. |
| `indian_eval/` | 2026-10-28 | Longformer on the 228-clause v2 set | Superseded. |
| `archive/2026-10-09_single_fold_3000char/` | 1 fold | same setup as stageB, one fold | Superseded by stageB_3fold. |
| `archive/2026-10-08_timing_run_6000char/` | 1 fold, 1 epoch | cls_only + span_only, 6000-char windows | Used only to estimate GPU time. |
| `archive/2026-10-07_first_run_6000char/` | 1 fold | first end-to-end run, 6000-char windows | Superseded; different window size, not comparable. |

Notes: fold 0 of a 5-fold run equals a 1-fold run with the same seed. Span scores moved between reruns of the same config (GPU nondeterminism, e.g. fold-0 span_only token-F1 0.746 vs 0.658), so don't compare single runs; use the paired 3-fold numbers.

Notebooks are in `../notebooks/`. Checkpoints are not stored in the repo.
