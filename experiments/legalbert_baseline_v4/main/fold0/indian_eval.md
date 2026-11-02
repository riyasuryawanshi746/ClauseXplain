# Indian-context evaluation

334 clauses: 192 in categories seen in training, 142 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 192 | 0.359 | 0.295 to 0.429 | 0.354 |
| majority (trained on CUAD fold 0) | 192 | 0.047 |  | 0.008 |
| tfidf_logreg (trained on CUAD fold 0) | 192 | 0.302 | 0.242 to 0.370 | 0.325 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x30, Intellectual Property x12, Termination x3, Liability x1 |
| Force Majeure | Liability x13, Termination x6 |
| Indemnification | Liability x15, Termination x1, Other x1, Dispute Resolution x1, Intellectual Property x1 |
| Modification | Assignment x2, Termination x2, Liability x1, Warranty x1, Other x1 |
| Payment | Commercial Terms x18, Liability x13, Termination x3, Penalties x3, Other x2, Assignment x2, Non-Compete x1, Warranty x1 |
| Privacy / Data Protection | Other x7, Intellectual Property x1 |

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.866 | 0.889 | n/a |