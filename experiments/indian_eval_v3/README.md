# Indian-set evaluation v3 (334 clauses + judgment slice, 3 CUAD-trained checkpoints)

Checkpoint: `joint_cls_span` folds 0-2 (CUAD only; never saw Indian text). Run 2026-11-02 on Kaggle with
`notebooks/indian_eval_v3_kaggle.ipynb` (executed copy: `notebooks/indian_eval_v3_kaggle_run.ipynb`).
Test data: `indian_set/built_v3` (228 v2 clauses + 106 batch-2 clauses from CPWD GCC 2023 and the SBM
waste-to-compost draft) and `indian_set/built_judgments` (11 clauses quoted in judgments, scored separately).
192 of the 334 clauses are in categories with CUAD training data (incl. Other); 142 are in the 6 unseen ones.
Per-fold files: `main/foldK/` and `judgments/foldK/` (`indian_eval.md/json`, `indian_predictions.csv`, rule flags for main).

Reproducibility check: the "original 228" slice gives 0.364 / 0.364 / 0.403, identical to the v2 run.

## Main set, seen categories (n = 192)

| | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---|---|---|---|
| Longformer accuracy | 0.328 | 0.333 | 0.370 | 0.344 |
| TF-IDF accuracy | 0.307 | 0.365 | 0.339 | 0.337 |
| Majority accuracy | 0.047 | 0.047 | 0.047 | 0.047 |
| Longformer macro-F1 | 0.349 | 0.333 | 0.348 | 0.343 |
| TF-IDF macro-F1 | 0.327 | 0.355 | 0.353 | 0.345 |
| Span head token-F1 | 0.322 | 0.322 | 0.325 | 0.323 |
| Lexical-centroid span baseline | 0.306 | 0.307 | 0.309 | 0.307 |
| Whole-window span baseline | 0.198 | 0.198 | 0.195 | 0.197 |
| Random span | 0.097 | 0.108 | 0.109 | 0.105 |

## Longformer accuracy by source (seen categories)

| Slice | n | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---|---|---|---|---|
| Original 228 (v2) | 129 | 0.364 | 0.364 | 0.403 | 0.377 |
| Batch 2 only | 63 | 0.254 | 0.270 | 0.302 | 0.275 |
| All 334 | 192 | 0.328 | 0.333 | 0.370 | 0.344 |
| Judgments (separate) | 11 | 0.455 | 0.727 | 0.636 | 0.606 |

TF-IDF was not scored per slice (only on the whole set). Judgment slice: TF-IDF accuracy 0.667, macro-F1 0.421
(Longformer 0.367); its span numbers are meaningless (each clause is its own document, whole-window = 1.0).

## Unseen categories: what the model predicts instead (fold 0)

Confidentiality -> Other 34/47; Payment -> Commercial Terms 21/44; Indemnification -> Liability 15/19;
Force Majeure -> Liability 10/19; Privacy -> Other 3/7; Modification -> Assignment 2/6.

## Reading it honestly

- Longformer is no better than TF-IDF out of domain: accuracy 0.344 vs 0.337, macro-F1 0.343 vs 0.345. All intervals overlap.
- Batch 2 is harder (0.275 vs 0.377), but n = 63 and the intervals overlap (fold 0: 0.163-0.373 vs 0.286-0.450).
  Possible reasons, not tested: keyword-picked units, no rows marked not-a-clause (definitions and garbled OCR rows are kept), PDF noise in GCC.
- Dispute Resolution is still ~0 (0.00 / 0.11 / 0.11, n = 18); "Other" is mostly misread (0.11-0.17, n = 47).
- Span head beats lexical-centroid by only ~0.016 token-F1 on Indian text (0.323 vs 0.307), versus 0.760 vs 0.604 on CUAD.
- Same clauses in every fold, so folds are not independent. The 32 second-annotator disagreements are not adjudicated yet,
  so labels in the original 228 may still change. Do not report per-document accuracy (n = 1-41).
