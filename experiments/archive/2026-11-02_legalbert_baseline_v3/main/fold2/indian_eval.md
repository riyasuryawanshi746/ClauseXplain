# Indian-context evaluation

334 clauses: 192 in categories seen in training, 142 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 192 | 0.349 | 0.285 to 0.419 | 0.335 |
| majority (trained on CUAD fold 2) | 192 | 0.047 |  | 0.008 |
| tfidf_logreg (trained on CUAD fold 2) | 192 | 0.339 | 0.275 to 0.408 | 0.353 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x40, Termination x3, Intellectual Property x3, Liability x1 |
| Force Majeure | Liability x16, Termination x3 |
| Indemnification | Liability x16, Termination x1, Other x1, Intellectual Property x1 |
| Modification | Assignment x2, Termination x2, Warranty x1, Other x1 |
| Payment | Commercial Terms x22, Liability x10, Penalties x4, Termination x2, Non-Compete x2, Assignment x2, Warranty x1, Other x1 |
| Privacy / Data Protection | Other x4, Termination x2, Warranty x1 |

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.870 | 0.891 | n/a |