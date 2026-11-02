# Indian-set evaluation v4: FINAL (adjudicated labels, 334 clauses + judgment slice)

**Canonical Indian results; these are the numbers in the paper.** Run 2026-11-02 on Kaggle T4.
- Longformer `joint_cls_span` folds 0-2 (CUAD only): `notebooks/indian_eval_v4_kaggle.ipynb` -> `main/`, `judgments/` here
  (executed copy `notebooks/indian_eval_v4_kaggle_run.ipynb`).
- LegalBERT `cls_only`, retrained on the same CUAD folds: `experiments/legalbert_baseline_v4/`.
- Test data: `indian_set/built_v4` (frozen; 32 second-annotator disagreements adjudicated blind, 11 labels changed)
  and `indian_set/built_judgments`. 192 of 334 clauses are in categories with CUAD training data; 142 are not.

## Main set, seen categories (n = 192)

| | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---|---|---|---|
| Longformer accuracy | 0.323 | 0.344 | 0.375 | 0.347 |
| LegalBERT accuracy | 0.359 | 0.349 | 0.359 | 0.356 |
| TF-IDF accuracy | 0.302 | 0.365 | 0.333 | 0.333 |
| Majority accuracy | 0.047 | 0.047 | 0.047 | 0.047 |
| Longformer macro-F1 | 0.316 | 0.343 | 0.355 | 0.338 |
| LegalBERT macro-F1 | 0.354 | 0.324 | 0.349 | 0.342 |
| TF-IDF macro-F1 | 0.325 | 0.356 | 0.350 | 0.344 |
| Span head token-F1 (Longformer) | 0.317 | 0.317 | 0.331 | 0.322 |
| Lexical-centroid span | 0.306 | 0.297 | 0.301 | 0.302 |
| Whole-window span | 0.197 | 0.196 | 0.195 | 0.196 |
| Random span | 0.097 | 0.097 | 0.100 | 0.098 |
| Span exact match, head / lexical | 0.123 / 0.011 | 0.084 / 0.011 | 0.121 / 0.011 | 0.109 / 0.011 |

## Accuracy by source (seen categories)

| Slice | n | Longformer (F0 / F1 / F2, mean) | LegalBERT (F0 / F1 / F2, mean) |
|---|---|---|---|
| Original pool | 129 | 0.357 / 0.380 / 0.411, 0.382 | 0.411 / 0.380 / 0.388, 0.393 |
| Batch 2 | 63 | 0.254 / 0.270 / 0.302, 0.275 | 0.254 / 0.286 / 0.302, 0.280 |
| Judgment slice (separate) | 11 | 0.455 / 0.727 / 0.636, 0.606 | 0.545 in every fold |

Judgment slice: TF-IDF accuracy 0.667, macro-F1 0.421 (Longformer 0.367, LegalBERT 0.229). Its span numbers are meaningless.
Wilson 95% intervals, fold 0, Longformer: all 0.261-0.392; original pool 0.279-0.442; batch 2 0.163-0.373.

## Per category and unseen categories (fold 0)

Longformer F1: Penalties 0.56, Termination 0.53, Governing Law 0.38, IP 0.38, Liability 0.32, Assignment 0.31,
Commercial Terms 0.23, Other 0.22 (n = 45), Warranty 0.20, Dispute Resolution 0.00 (n = 18).
LegalBERT F1: Penalties 0.81, Termination 0.55, Governing Law 0.52, Other 0.13, Dispute Resolution 0.00.

Unseen categories (Longformer / LegalBERT): Confidentiality -> Other 33 / 30 of 46; Payment -> Commercial Terms 20 / 18 of 43;
Indemnification -> Liability 15 / 15 of 19; Force Majeure -> Liability 10 / 13 of 19; Privacy -> Other 4 / 7 of 8; Modification -> Assignment 2 / 2 of 7.

## Change from v3 (pre-adjudication)

Longformer accuracy 0.344 -> 0.347, macro-F1 0.343 -> 0.338; original pool 0.377 -> 0.382; batch 2 unchanged (0.275, its labels did not change).
LegalBERT was retrained, so its change mixes the new labels with GPU run-to-run noise (CUAD macro-F1 0.870 -> 0.866 on identical data).

## Reading it honestly

- All three classifiers are the same out of domain (accuracy 0.333-0.356, macro-F1 0.338-0.344); every interval overlaps.
- The span head beats the lexical baseline by only 0.020 token-F1 on Indian text (0.156 on CUAD), though its exact match is far higher.
- Batch 2 scores lower than the original pool for both models, but n = 63 and intervals overlap.
- Same clauses in every fold (folds not independent); single seed; per-document n is 1-41, so no per-document claims.
