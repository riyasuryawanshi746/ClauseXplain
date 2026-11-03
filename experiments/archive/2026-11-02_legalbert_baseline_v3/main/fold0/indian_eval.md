# Indian-context evaluation

334 clauses: 192 in categories seen in training, 142 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 192 | 0.375 | 0.310 to 0.445 | 0.388 |
| majority (trained on CUAD fold 0) | 192 | 0.047 |  | 0.008 |
| tfidf_logreg (trained on CUAD fold 0) | 192 | 0.307 | 0.246 to 0.376 | 0.327 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x31, Intellectual Property x13, Termination x2, Liability x1 |
| Force Majeure | Liability x14, Termination x5 |
| Indemnification | Liability x17, Termination x1, Intellectual Property x1 |
| Modification | Assignment x2, Termination x2, Commercial Terms x1, Other x1 |
| Payment | Commercial Terms x23, Liability x11, Penalties x3, Warranty x2, Non-Compete x2, Assignment x2, Termination x1 |
| Privacy / Data Protection | Other x6, Intellectual Property x1 |

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.870 | 0.891 | n/a |