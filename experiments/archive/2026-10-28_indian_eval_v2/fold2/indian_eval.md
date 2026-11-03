# Indian-context evaluation

228 clauses: 129 in categories seen in training, 99 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 129 | 0.403 | 0.322 to 0.489 | 0.392 |
| majority (trained on CUAD fold 2) | 129 | 0.070 |  | 0.012 |
| tfidf_logreg (trained on CUAD fold 2) | 129 | 0.341 | 0.265 to 0.426 | 0.360 |

## Span localisation (windowed)

| Method | n | token-F1 | exact match | IoU |
|---|---|---|---|---|
| span head (best-match gold) | 129 | 0.357 | 0.116 | 0.327 |
| span head (own label) | 129 | 0.285 | 0.085 | 0.259 |
| random span | 129 | 0.109 | 0.000 | 0.079 |
| whole_window_baseline | 129 | 0.214 | 0.000 | 0.124 |
| lexical_centroid_baseline | 129 | 0.320 | 0.000 | 0.267 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x31, Intellectual Property x2, Termination x2, Assignment x2, Non-Compete x1, Liability x1 |
| Force Majeure | Liability x9, Termination x2, Dispute Resolution x1, Warranty x1 |
| Indemnification | Liability x12, Other x1, Dispute Resolution x1 |
| Modification | Assignment x2, Termination x2, Other x1 |
| Payment | Commercial Terms x10, Termination x4, Non-Compete x2, Penalties x2, Other x1, Warranty x1, Governing Law / Jurisdiction x1, Liability x1 |
| Privacy / Data Protection | Other x5, Termination x1 |

## Rule-layer findings (see indian_rule_flags.csv)

- DPDPA_S08_01:gap: 3
- DPDPA_S08_02:gap: 6
- ICA_S124_01:none: 9
- ICA_S124_01:possible: 5
- ICA_S27_02:possible: 1
- ICA_S28_01:none: 10
- ICA_S39_01:context: 5
- ICA_S73_01:context: 2
- ICA_S74_01:flag: 6
- ICA_S74_01:none: 9
- ICA_S75_01:context: 5
- MSMED_S15_01:needs_fact: 1
- MSMED_S15_01:none: 21
- MSMED_S16_01:none: 22

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.839 | 0.875 | n/a |
| span_only | n/a | n/a | 0.736 |
| joint_cls_span | 0.837 | 0.872 | 0.760 |