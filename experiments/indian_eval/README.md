# Indian-set evaluation (228 clauses, 3 CUAD-trained checkpoints)

Checkpoint: `joint_cls_span` folds 0-2 (CUAD only; never saw Indian text). Scored on the
129 clauses whose category has CUAD training data (incl. "Other"); 99 clauses are in the
6 categories with no training data and are reported separately (what the model predicts instead).
Run on Kaggle with `notebooks/indian_eval_v2_kaggle.ipynb`. Per-fold files: `indian_eval.md/json`,
`indian_predictions.csv` (per clause), `indian_rule_flags.csv` (unreviewed by a lawyer).

| | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---|---|---|---|
| Longformer accuracy | 0.364 | 0.364 | 0.403 | 0.377 |
| TF-IDF accuracy | 0.333 | 0.372 | 0.341 | 0.349 |
| Longformer macro-F1 | 0.387 | 0.367 | 0.392 | 0.382 |
| TF-IDF macro-F1 | 0.349 | 0.371 | 0.360 | 0.360 |
| Span head token-F1 | 0.323 | 0.361 | 0.357 | 0.347 |
| Lexical-centroid span baseline | 0.299 | 0.316 | 0.320 | 0.312 |
| Random span | 0.120 | 0.105 | 0.109 | 0.111 |

Caveats: same 129 clauses in every fold (folds are not independent); all accuracy intervals
overlap TF-IDF; single annotator (no valid inter-annotator check yet); Noida rows are OCR-noisy;
Dispute Resolution F1 = 0 (CUAD only has "Covenant Not To Sue" for it); CUAD macro-F1 is ~0.84,
so the domain-shift drop is the main finding. Per-document n is 1-41: do not report per-document accuracy as a result.
