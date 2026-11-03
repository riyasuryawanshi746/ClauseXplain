# Indian-context evaluation

334 clauses: 192 in categories seen in training, 142 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 192 | 0.323 | 0.261 to 0.392 | 0.301 |
| majority (trained on CUAD fold 1) | 192 | 0.047 |  | 0.008 |
| tfidf_logreg (trained on CUAD fold 1) | 192 | 0.365 | 0.300 to 0.435 | 0.355 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x29, Intellectual Property x12, Assignment x2, Termination x2, Non-Compete x1, Liability x1 |
| Force Majeure | Liability x19 |
| Indemnification | Liability x16, Termination x1, Other x1, Intellectual Property x1 |
| Modification | Assignment x2, Termination x2, Warranty x1, Other x1 |
| Payment | Commercial Terms x22, Liability x12, Penalties x4, Assignment x2, Non-Compete x1, Intellectual Property x1, Warranty x1, Termination x1 |
| Privacy / Data Protection | Other x3, Intellectual Property x2, Termination x1, Commercial Terms x1 |

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.870 | 0.891 | n/a |