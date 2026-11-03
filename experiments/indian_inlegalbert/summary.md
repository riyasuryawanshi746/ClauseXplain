# Indian set: document-held-out training (leave-one-document-group-out, 12 folds)

Every clause is predicted by a model that never saw its document. Zero-shot CUAD models (no Indian training)
scored 0.347 acc / 0.338 macro-F1 (Longformer) on the 192 seen-category clauses only.

| Run | all acc (95% CI) | all macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| inlegalbert_cuad_indian_ledgar_s13 | 0.578 (0.524-0.630) | 0.504 (0.452-0.569) | 0.500 / 0.511 | 0.683 / 0.632 | 0.623 / 0.481 | 0.636 |
| inlegalbert_cuad_indian_s13 | 0.527 (0.473-0.580) | 0.442 (0.396-0.500) | 0.505 / 0.491 | 0.556 / 0.478 | 0.566 / 0.443 | 0.727 |
| inlegalbert_cuad_indian_s14 | 0.542 (0.488-0.595) | 0.455 (0.408-0.513) | 0.516 / 0.502 | 0.577 / 0.496 | 0.583 / 0.453 | 0.727 |
| inlegalbert_cuad_indian_s15 | 0.542 (0.488-0.595) | 0.452 (0.405-0.510) | 0.516 / 0.502 | 0.577 / 0.486 | 0.579 / 0.462 | 0.727 |
