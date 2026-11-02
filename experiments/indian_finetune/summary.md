# Indian set: document-held-out training (leave-one-document-group-out, 12 folds)

Every clause is predicted by a model that never saw its document. Zero-shot CUAD models (no Indian training)
scored 0.347 acc / 0.338 macro-F1 (Longformer) on the 192 seen-category clauses only.

| Run | all acc (95% CI) | all macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| legalbert_cuad_indian | 0.566 (0.512-0.618) | 0.528 (0.457-0.562) | 0.536 / 0.597 | 0.606 / 0.516 | 0.623 / 0.443 | 0.727 |
| legalbert_cuad_indian_tricks | 0.512 (0.459-0.565) | 0.466 (0.420-0.528) | 0.495 / 0.501 | 0.535 / 0.510 | 0.566 / 0.396 | 0.455 |
| legalbert_indian | 0.467 (0.414-0.521) | 0.296 (0.257-0.345) | 0.312 / 0.231 | 0.676 / 0.508 | 0.518 / 0.358 | 0.364 |
| legalbert_indian_tricks | 0.249 (0.205-0.298) | 0.272 (0.219-0.327) | 0.182 / 0.221 | 0.338 / 0.448 | 0.316 / 0.104 | 0.182 |
| longformer_cuad_indian | 0.521 (0.467-0.574) | 0.469 (0.413-0.517) | 0.490 / 0.560 | 0.563 / 0.494 | 0.544 / 0.472 | 0.364 |
| longformer_cuad_indian_tricks | 0.488 (0.435-0.541) | 0.472 (0.420-0.530) | 0.479 / 0.503 | 0.500 / 0.574 | 0.509 / 0.443 | 0.364 |
| tfidf_cuad_indian | 0.284 (0.239-0.335) | 0.292 (0.215-0.322) | 0.432 / 0.472 | 0.085 / 0.095 | 0.285 / 0.283 | 0.818 |
| tfidf_indian | 0.422 (0.370-0.476) | 0.297 (0.261-0.345) | 0.328 / 0.269 | 0.549 / 0.439 | 0.417 / 0.434 | 0.273 |
| tfidf_indian_balanced | 0.482 (0.429-0.536) | 0.351 (0.310-0.401) | 0.396 / 0.336 | 0.599 / 0.485 | 0.474 / 0.500 | 0.182 |
