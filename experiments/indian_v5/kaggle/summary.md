# Indian set: document-held-out training (leave-one-document-group-out, 12 folds)

Every clause is predicted by a model that never saw its document. Zero-shot CUAD models (no Indian training)
scored 0.347 acc / 0.338 macro-F1 (Longformer) on the 192 seen-category clauses only.

| Run | all acc (95% CI) | all macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| v5_legalbert_cuad_indian_ledgar_s13 | 0.655 (0.621-0.687) | 0.648 (0.604-0.681) | 0.596 / 0.634 | 0.762 / 0.802 | 0.680 / 0.491 | 0.636 |
| v5_legalbert_cuad_indian_s13 | 0.613 (0.578-0.646) | 0.609 (0.568-0.644) | 0.560 / 0.600 | 0.708 / 0.750 | 0.639 / 0.443 | 0.727 |

## v4 clauses only (n=334) and batch 3 only

| run | v4 acc | v4 macro-F1 | batch3 acc | batch3 macro-F1 |
|---|---|---|---|---|
| v5_legalbert_cuad_indian_s13 | 0.614 (n=345) | 0.579 | 0.615 (n=454) | 0.581 |
| v5_legalbert_cuad_indian_ledgar_s13 | 0.643 (n=345) | 0.600 | 0.663 (n=454) | 0.627 |
