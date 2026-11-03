# Indian-context evaluation

11 clauses: 11 in categories seen in training, 0 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 11 | 0.455 | 0.213 to 0.720 | 0.190 |
| majority (trained on CUAD fold 0) | 11 | 0.000 |  | 0.000 |
| tfidf_logreg (trained on CUAD fold 0) | 11 | 0.636 | 0.354 to 0.848 | 0.310 |

## Span localisation (windowed)

| Method | n | token-F1 | exact match | IoU |
|---|---|---|---|---|
| span head (best-match gold) | 11 | 0.662 | 0.000 | 0.542 |
| span head (own label) | 11 | 0.662 | 0.000 | 0.542 |
| random span | 11 | 0.633 | 0.091 | 0.504 |
| whole_window_baseline | 11 | 1.000 | 1.000 | 1.000 |
| lexical_centroid_baseline | 11 | 0.746 | 0.182 | 0.637 |

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.839 | 0.875 | n/a |
| span_only | n/a | n/a | 0.736 |
| joint_cls_span | 0.837 | 0.872 | 0.760 |