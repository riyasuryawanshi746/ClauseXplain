# Indian set: document-held-out training (leave-one-document-group-out, 12 folds)

Every clause is predicted by a model that never saw its document. Zero-shot CUAD models (no Indian training)
scored 0.347 acc / 0.338 macro-F1 (Longformer) on the 192 seen-category clauses only.

| Run | all acc (95% CI) | all macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| v6_legalbert_cuad_indian_s13 | 0.631 (0.599-0.662) | 0.628 (0.591-0.657) | 0.571 / 0.594 | 0.734 / 0.819 | 0.653 / 0.462 | 0.545 |

## v4 clauses only (n=334) and the added batches

| run | v4 acc | v4 macro-F1 | added acc | added macro-F1 |
|---|---|---|---|---|
| v6_legalbert_cuad_indian_s13 | 0.629 (n=334) | 0.569 | 0.632 (n=571) | 0.616 |
