# Indian set: document-held-out training (leave-one-document-group-out, 12 folds)

Every clause is predicted by a model that never saw its document. Zero-shot CUAD models (no Indian training)
scored 0.347 acc / 0.338 macro-F1 (Longformer) on the 192 seen-category clauses only.

| Run | all acc (95% CI) | all macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| v6_legalbert_cuad_indian_ledgar_s13 | 0.650 (0.618-0.680) | 0.643 (0.602-0.676) | 0.589 / 0.615 | 0.755 / 0.838 | 0.671 / 0.491 | 0.636 |

## v4 clauses only (n=334) and the added batches

| run | v4 acc | v4 macro-F1 | added acc | added macro-F1 |
|---|---|---|---|---|
| v6_legalbert_cuad_indian_ledgar_s13 | 0.638 (n=334) | 0.574 | 0.657 (n=571) | 0.625 |
