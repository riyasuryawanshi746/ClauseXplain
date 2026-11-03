# LegalBERT classification baseline (CUAD folds 0-2 + Indian v3)

> Superseded by `experiments/legalbert_baseline_v4/` (adjudicated labels). Kept for traceability; do not quote these numbers.

`nlpaueb/legal-bert-base-uncased` (110M parameters, 512 tokens), classifier only, run 2026-11-02 on a Kaggle T4 with
`notebooks/archive/legalbert_baseline_kaggle.ipynb` (executed copy: `notebooks/archive/legalbert_baseline_kaggle_run.ipynb`).
Same CUAD folds as the Longformer runs (`splits.json` reused; the log confirms "splits loaded ... identical to the
original run"), same settings: 2 epochs, lr 2e-5 (head 1e-4), seed 13, batch 16, max 512 tokens. No span head.
Indian labels are pre-adjudication (same as `experiments/archive/2026-11-02_indian_eval_v3/`).

## CUAD (12 categories scored, about 2,030 test clauses per fold)

| Model | Fold 0 | Fold 1 | Fold 2 | Mean macro-F1 | Mean micro-F1 |
|---|---|---|---|---|---|
| LegalBERT cls_only | 0.861 | 0.880 | 0.870 | 0.870 +- 0.009 | 0.891 |
| Longformer cls_only | n/a | n/a | n/a | 0.839 +- 0.017 | 0.875 |
| TF-IDF + LR | n/a | n/a | n/a | 0.802 +- 0.016 | 0.853 |

Longformer and TF-IDF per-fold values are in `stageB_3fold/partial/` (gitignored, not in the repo), so no paired
per-fold difference is computed here. Training time: about 3.4 min per fold (5.0 GB peak GPU memory).

## Indian main set, seen categories (n = 192)

| | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---|---|---|---|
| LegalBERT accuracy | 0.375 | 0.323 | 0.349 | 0.349 |
| Longformer accuracy (v3) | 0.328 | 0.333 | 0.370 | 0.344 |
| TF-IDF accuracy | 0.307 | 0.365 | 0.339 | 0.337 |
| LegalBERT macro-F1 | 0.388 | 0.301 | 0.335 | 0.341 |
| Longformer macro-F1 (v3) | 0.349 | 0.333 | 0.348 | 0.343 |
| TF-IDF macro-F1 | 0.327 | 0.355 | 0.353 | 0.345 |

LegalBERT by source: original 228 pool 0.419 / 0.341 / 0.364 (mean 0.375), batch 2 0.286 / 0.286 / 0.317 (mean 0.296).
Judgment slice (n = 11): accuracy 0.545 and macro-F1 0.229 in every fold (Longformer 0.606, TF-IDF 0.667).

Unseen categories (fold 0) go to the same neighbours as with Longformer: Confidentiality -> Other 31/47,
Payment -> Commercial Terms 23/44, Indemnification -> Liability 17/19, Force Majeure -> Liability 14/19.
Per category (fold 0): Penalties F1 0.78, Termination 0.57, Governing Law 0.50; Dispute Resolution 0.00, Warranty 0.13, Other 0.17.

## Reading it honestly

- On CUAD, LegalBERT is the strongest classifier we have: +0.031 macro-F1 over the Longformer classifier, and its
  lowest fold (0.861) is above the Longformer mean. Clause classification does not need Longformer's long context;
  the reason to use Longformer here is the span head.
- On Indian text all three are the same: 0.349 / 0.344 / 0.337 accuracy, 0.341 / 0.343 / 0.345 macro-F1. The domain-shift
  drop is not specific to Longformer. LegalBERT is also no better on unseen categories or on Dispute Resolution.
- Single seed, three folds, the same Indian clauses in every fold; intervals are wide (fold 0: 0.31-0.45).
