# Indian-context evaluation

32 clauses: 12 in categories seen in training, 20 in unseen categories. Small sample: read gaps, not decimals.

## Category classification (seen categories only)

| Model | n | accuracy | 95% interval | macro-F1 |
|---|---|---|---|---|
| ClauseXplain checkpoint | 12 | 0.500 | 0.254 to 0.746 | 0.436 |
| majority (trained on CUAD fold 0) | 12 | 0.000 |  | 0.000 |
| tfidf_logreg (trained on CUAD fold 0) | 12 | 0.417 | 0.193 to 0.680 | 0.422 |

## Span localisation (windowed)

| Method | n | token-F1 | exact match | IoU |
|---|---|---|---|---|
| span head (best-match gold) | 12 | 0.583 | 0.000 | 0.502 |
| span head (own label) | 12 | 0.583 | 0.000 | 0.502 |
| random span | 12 | 0.177 | 0.000 | 0.117 |
| whole_window_baseline | 12 | 0.325 | 0.000 | 0.204 |
| lexical_centroid_baseline | 12 | 0.484 | 0.000 | 0.407 |

## Categories the model never saw: what it predicted instead

| True category | predictions |
|---|---|
| Confidentiality | Other x7, Termination x1, Intellectual Property x1 |
| Force Majeure | Liability x4, Termination x1 |
| Indemnification | Liability x4, Intellectual Property x1, Other x1 |

## Rule-layer findings (see indian_rule_flags.csv)

- ICA_S124_01:none: 3
- ICA_S124_01:possible: 3
- ICA_S27_02:possible: 1
- ICA_S74_01:flag: 3
- ICA_S74_01:none: 6
