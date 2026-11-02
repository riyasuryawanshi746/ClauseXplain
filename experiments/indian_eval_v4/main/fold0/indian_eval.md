# Indian-context evaluation

334 clauses: 192 in categories seen in training, 142 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 192 | 0.323 | 0.261 to 0.392 | 0.316 |
| majority (trained on CUAD fold 0) | 192 | 0.047 |  | 0.008 |
| tfidf_logreg (trained on CUAD fold 0) | 192 | 0.302 | 0.242 to 0.370 | 0.325 |

## Span localisation (windowed)

| Method | n | token-F1 | exact match | IoU |
|---|---|---|---|---|
| span head (best-match gold) | 187 | 0.317 | 0.123 | 0.287 |
| span head (own label) | 187 | 0.287 | 0.112 | 0.260 |
| random span | 187 | 0.097 | 0.000 | 0.068 |
| whole_window_baseline | 187 | 0.197 | 0.000 | 0.113 |
| lexical_centroid_baseline | 187 | 0.306 | 0.011 | 0.258 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x33, Intellectual Property x6, Termination x4, Non-Compete x2, Liability x1 |
| Force Majeure | Liability x10, Termination x3, Assignment x2, Intellectual Property x2, Commercial Terms x1, Non-Compete x1 |
| Indemnification | Liability x15, Intellectual Property x2, Other x1, Non-Compete x1 |
| Modification | Assignment x2, Termination x2, Other x2, Liability x1 |
| Payment | Commercial Terms x20, Liability x13, Other x3, Termination x2, Assignment x2, Penalties x1, Non-Compete x1, Warranty x1 |
| Privacy / Data Protection | Other x4, Intellectual Property x3, Termination x1 |

## Rule-layer findings (see indian_rule_flags.csv)

- DPDPA_S08_01:gap: 5
- DPDPA_S08_02:gap: 8
- ICA_S124_01:none: 13
- ICA_S124_01:possible: 6
- ICA_S27_02:possible: 1
- ICA_S28_01:none: 18
- ICA_S39_01:context: 7
- ICA_S73_01:context: 4
- ICA_S74_01:flag: 6
- ICA_S74_01:none: 14
- ICA_S75_01:context: 6
- MSMED_S15_01:needs_fact: 1
- MSMED_S15_01:none: 42
- MSMED_S16_01:needs_fact: 1
- MSMED_S16_01:none: 42

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.839 | 0.875 | n/a |
| span_only | n/a | n/a | 0.736 |
| joint_cls_span | 0.837 | 0.872 | 0.760 |