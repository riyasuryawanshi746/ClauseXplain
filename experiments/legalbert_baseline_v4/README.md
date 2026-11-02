# LegalBERT classification baseline v4 (FINAL; adjudicated Indian labels)

Same setup as `experiments/legalbert_baseline/` (nlpaueb/legal-bert-base-uncased, cls_only, CUAD folds 0-2 with the
reused `splits.json`, 2 epochs, lr 2e-5 / head 1e-4, seed 13), retrained 2026-11-02 with
`notebooks/legalbert_baseline_v4_kaggle.ipynb` (executed copy `notebooks/legalbert_baseline_v4_kaggle_run.ipynb`),
scored on `indian_set/built_v4` and the judgment slice.

CUAD macro-F1: 0.854 / 0.877 / 0.866, mean 0.866 +- 0.011; micro-F1 0.889; about 3.3 min per fold.
The first run on identical CUAD data gave 0.870 +- 0.009: same seed, so the 0.004 gap is GPU non-determinism.

Indian results (accuracy 0.356, macro-F1 0.342 on 192 seen-category clauses) are tabulated next to Longformer and
TF-IDF in `experiments/indian_eval_v4/README.md`.
