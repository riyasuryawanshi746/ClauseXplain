# Experiments index

**Cite only `stageB_3fold/`, `scan_vs_span/`, `indian_eval_v4/`, `legalbert_baseline_v4/`, `indian_finetune/`, `indian_ledgar/`, `indian_inlegalbert/` and `indian_v5/`.** Everything in `archive/` is kept for the record and is superseded.

| Folder | Run | Setup | Status |
|---|---|---|---|
| `stageB_3fold/` | 2026-10-10, T4 | cls_only, span_only, joint_cls_span; folds 0-2 of 5 (contract-grouped); 3000-char windows; 2 epochs | **Canonical.** `results.json`, `comparisons.md` (with TF-IDF and span baselines) and `partial/` (per-fold runs, `splits.json`, `config.json`). Settings: 2 epochs, 50% of span items per epoch, so span results are probably undertrained. |
| `scan_vs_span/` | folds 0-2 | span head vs classifier scan, per-fold md/json | **Canonical** for the speed/accuracy comparison. |
| `indian_eval_v4/` | 2026-11-02, T4 | Longformer joint folds 0-2 on frozen `built_v4` (334 adjudicated clauses) + judgment slice | **Canonical** Indian results (README also tabulates LegalBERT and TF-IDF). |
| `legalbert_baseline_v4/` | 2026-11-02, T4 | LegalBERT cls_only, same CUAD folds; scored on `built_v4` | **Canonical** LegalBERT baseline. |
| `indian_finetune/` | 2026-11-03, T4 | Document-held-out training on `built_v4` (leave-one-group-out, 12 folds): TF-IDF, LegalBERT, Longformer, with/without CUAD init and class weights/smoothing/LLRD | **Canonical** adapted results (best: LegalBERT CUAD->Indian 0.566 acc / 0.528 macro-F1). |
| `indian_ledgar/` | 2026-11-03 | Step 5: same protocol + up to 100 LEDGAR provisions per category (`ledgar_extra.py`) | **Canonical.** TF-IDF in `local_cpu/`, LegalBERT 3 seeds in `kaggle/` (CUAD->Indian 0.565 acc / 0.524 mF1; + LEDGAR 0.600 / 0.512). |
| `indian_inlegalbert/` | 2026-11-03, T4 | Step 6: InLegalBERT encoder, same protocol, 3 seeds + 1 LEDGAR run | **Canonical.** 0.537 acc / 0.450 mF1, below LegalBERT. |
| `indian_v6/` | 2026-11-05 | v5 + batch 4 = 905 clauses, 29 groups | LegalBERT s13 on the same 334: 0.629 / 0.569 (0.604 without the single Non-Solicitation clause); + LEDGAR pending. |
| `indian_v5/` | 2026-11-03 | v4 + batch 3 = 788 clauses, 23 groups | **Canonical.** LegalBERT seed 13 on the same 334 test clauses: 0.611 / 0.609 (+ LEDGAR 0.644 / 0.636) vs 0.565 / 0.524 without batch 3. |
| `archive/2026-11-02_indian_eval_v3/`, `archive/2026-11-02_legalbert_baseline_v3/` | 2026-11-02 | same, pre-adjudication labels | Superseded by the v4 folders. |
| `archive/2026-10-28_indian_eval_v2/` | 2026-10-28 | Longformer on the 228-clause v2 set | Superseded. |
| `archive/2026-10-09_single_fold_3000char/` | 1 fold | same setup as stageB, one fold | Superseded by stageB_3fold. |
| `archive/2026-10-08_timing_run_6000char/` | 1 fold, 1 epoch | cls_only + span_only, 6000-char windows | Used only to estimate GPU time. |
| `archive/2026-10-07_first_run_6000char/` | 1 fold | first end-to-end run, 6000-char windows | Superseded; different window size, not comparable. |

Notes: fold 0 of a 5-fold run equals a 1-fold run with the same seed. Span scores moved between reruns of the same config (GPU nondeterminism, e.g. fold-0 span_only token-F1 0.746 vs 0.658), so don't compare single runs; use the paired 3-fold numbers.

Notebooks are in `../notebooks/` (superseded ones in `../notebooks/archive/`). Checkpoints are not stored in the repo.
