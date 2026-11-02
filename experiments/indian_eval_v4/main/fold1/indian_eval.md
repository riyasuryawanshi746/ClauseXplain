# Indian-context evaluation

334 clauses: 192 in categories seen in training, 142 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 192 | 0.344 | 0.280 to 0.413 | 0.343 |
| majority (trained on CUAD fold 1) | 192 | 0.047 |  | 0.008 |
| tfidf_logreg (trained on CUAD fold 1) | 192 | 0.365 | 0.300 to 0.435 | 0.356 |

## Span localisation (windowed)

| Method | n | token-F1 | exact match | IoU |
|---|---|---|---|---|
| span head (best-match gold) | 190 | 0.317 | 0.084 | 0.274 |
| span head (own label) | 190 | 0.279 | 0.068 | 0.240 |
| random span | 190 | 0.097 | 0.000 | 0.067 |
| whole_window_baseline | 190 | 0.196 | 0.000 | 0.113 |
| lexical_centroid_baseline | 190 | 0.297 | 0.011 | 0.250 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x34, Intellectual Property x5, Termination x2, Non-Compete x2, Non-Solicitation x1, Assignment x1, Liability x1 |
| Force Majeure | Liability x9, Termination x3, Assignment x3, Non-Compete x2, Commercial Terms x2 |
| Indemnification | Liability x16, Other x1, Intellectual Property x1, Non-Solicitation x1 |
| Modification | Termination x3, Assignment x2, Liability x1, Other x1 |
| Payment | Commercial Terms x22, Liability x7, Penalties x5, Warranty x3, Assignment x3, Non-Compete x1, Governing Law / Jurisdiction x1, Other x1 |
| Privacy / Data Protection | Other x7, Termination x1 |

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