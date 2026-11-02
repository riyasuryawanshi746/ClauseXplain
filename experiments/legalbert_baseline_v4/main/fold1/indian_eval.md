# Indian-context evaluation

334 clauses: 192 in categories seen in training, 142 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 192 | 0.349 | 0.285 to 0.419 | 0.324 |
| majority (trained on CUAD fold 1) | 192 | 0.047 |  | 0.008 |
| tfidf_logreg (trained on CUAD fold 1) | 192 | 0.365 | 0.300 to 0.435 | 0.356 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x26, Intellectual Property x15, Non-Compete x2, Termination x2, Liability x1 |
| Force Majeure | Liability x17, Termination x2 |
| Indemnification | Liability x17, Termination x1, Other x1 |
| Modification | Assignment x2, Termination x2, Liability x1, Warranty x1, Other x1 |
| Payment | Commercial Terms x22, Liability x11, Penalties x4, Non-Compete x1, Dispute Resolution x1, Assignment x1, Warranty x1, Termination x1, Other x1 |
| Privacy / Data Protection | Other x5, Termination x1, Intellectual Property x1, Commercial Terms x1 |

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.866 | 0.889 | n/a |