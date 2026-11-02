# Indian set: document-held-out training (leave-one-document-group-out, 12 folds)

Every clause is predicted by a model that never saw its document. Zero-shot CUAD models (no Indian training)
scored 0.347 acc / 0.338 macro-F1 (Longformer) on the 192 seen-category clauses only.

| Run | all acc (95% CI) | all macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| tfidf_cuad_indian_ledgar_balanced | 0.470 (0.417-0.524) | 0.489 (0.413-0.529) | 0.500 / 0.570 | 0.430 / 0.490 | 0.513 / 0.377 | 0.545 |
| tfidf_indian_ledgar_balanced | 0.557 (0.503-0.609) | 0.457 (0.409-0.520) | 0.464 / 0.464 | 0.683 / 0.619 | 0.575 / 0.519 | 0.364 |
| tfidf_indian_ledgar_missing_balanced | 0.470 (0.417-0.524) | 0.333 (0.292-0.385) | 0.260 / 0.268 | 0.754 / 0.642 | 0.469 / 0.472 | 0.273 |
