# Indian-context evaluation

334 clauses: 192 in categories seen in training, 142 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 192 | 0.370 | 0.305 to 0.440 | 0.348 |
| majority (trained on CUAD fold 2) | 192 | 0.047 |  | 0.008 |
| tfidf_logreg (trained on CUAD fold 2) | 192 | 0.339 | 0.275 to 0.408 | 0.353 |

## Span localisation (windowed)

| Method | n | token-F1 | exact match | IoU |
|---|---|---|---|---|
| span head (best-match gold) | 190 | 0.325 | 0.121 | 0.296 |
| span head (own label) | 190 | 0.275 | 0.100 | 0.250 |
| random span | 190 | 0.109 | 0.000 | 0.079 |
| whole_window_baseline | 190 | 0.195 | 0.000 | 0.112 |
| lexical_centroid_baseline | 190 | 0.309 | 0.011 | 0.263 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x38, Intellectual Property x3, Termination x2, Assignment x2, Non-Compete x1, Liability x1 |
| Force Majeure | Liability x12, Termination x3, Warranty x2, Dispute Resolution x1, Non-Compete x1 |
| Indemnification | Liability x16, Other x1, Dispute Resolution x1, Non-Solicitation x1 |
| Modification | Assignment x2, Termination x2, Other x2 |
| Payment | Commercial Terms x13, Liability x10, Penalties x7, Termination x4, Non-Compete x3, Other x2, Warranty x2, Assignment x2, Governing Law / Jurisdiction x1 |
| Privacy / Data Protection | Other x6, Termination x1 |

## Rule-layer findings (see indian_rule_flags.csv)

- DPDPA_S08_01:gap: 4
- DPDPA_S08_02:gap: 7
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
- MSMED_S15_01:none: 43
- MSMED_S16_01:needs_fact: 1
- MSMED_S16_01:none: 43

## Reference: CUAD cross-validated test numbers (mean over folds)

| Model | macro-F1 | micro-F1 | span token-F1 |
|---|---|---|---|
| cls_only | 0.839 | 0.875 | n/a |
| span_only | n/a | n/a | 0.736 |
| joint_cls_span | 0.837 | 0.872 | 0.760 |