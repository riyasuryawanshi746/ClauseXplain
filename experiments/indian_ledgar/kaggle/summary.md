# Indian set: document-held-out training (leave-one-document-group-out, 12 folds)

Every clause is predicted by a model that never saw its document. Zero-shot CUAD models (no Indian training)
scored 0.347 acc / 0.338 macro-F1 (Longformer) on the 192 seen-category clauses only.

| Run | all acc (95% CI) | all macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| legalbert_cuad_indian_ledgar_s13 | 0.590 (0.536-0.641) | 0.504 (0.454-0.571) | 0.505 / 0.512 | 0.704 / 0.630 | 0.645 / 0.472 | 0.636 |
| legalbert_cuad_indian_ledgar_s14 | 0.620 (0.567-0.670) | 0.532 (0.481-0.602) | 0.536 / 0.542 | 0.732 / 0.659 | 0.680 / 0.491 | 0.636 |
| legalbert_cuad_indian_ledgar_s15 | 0.590 (0.536-0.641) | 0.501 (0.451-0.566) | 0.505 / 0.508 | 0.704 / 0.619 | 0.645 / 0.472 | 0.636 |
| legalbert_cuad_indian_s13 | 0.569 (0.515-0.621) | 0.527 (0.454-0.562) | 0.536 / 0.591 | 0.613 / 0.521 | 0.618 / 0.462 | 0.727 |
| legalbert_cuad_indian_s14 | 0.569 (0.515-0.621) | 0.528 (0.457-0.562) | 0.542 / 0.595 | 0.606 / 0.513 | 0.614 / 0.472 | 0.636 |
| legalbert_cuad_indian_s15 | 0.557 (0.503-0.609) | 0.517 (0.443-0.552) | 0.536 / 0.593 | 0.585 / 0.500 | 0.605 / 0.453 | 0.636 |
