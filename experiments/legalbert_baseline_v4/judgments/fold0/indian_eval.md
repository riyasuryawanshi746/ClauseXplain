# Indian-context evaluation

11 clauses: 11 in categories seen in training, 0 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 11 | 0.545 | 0.280 to 0.787 | 0.229 |
| majority (trained on CUAD fold 0) | 11 | 0.000 |  | 0.000 |
| tfidf_logreg (trained on CUAD fold 0) | 11 | 0.636 | 0.354 to 0.848 | 0.310 |

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.866 | 0.889 | n/a |