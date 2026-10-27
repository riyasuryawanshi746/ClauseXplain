# ClauseXplain Phase 3 results

*Run: 2026-10-07 23:26:37  |  device: cuda  |  encoder: allenai/longformer-base-4096  |  folds: 1*

## Read this first
- Span task is WINDOWED localisation (answer guaranteed in a 6000-char window); not comparable to full-contract CUAD numbers.
- Category labels derive from the 41->18 CUAD mapping; compare against the TF-IDF baseline before claiming anything.
- 6 categories (Confidentiality, Force Majeure, Indemnification, Modification, Payment, Privacy / Data Protection) have no data unless --weak_labels_jsonl is supplied; they are n/a, not 0.
- 'Joint' = shared encoder trained on alternating clause/span batches; each batch only trains heads it has labels for.
- Span 'best-match gold' scores against the closest of all labelled spans of that category inside the window (a window can hold several valid answers); 'own label only' is the stricter version.
- Risk labels: none used.
- Folds: 1; seed 13. With one fold there is no variance estimate.

## Category classification (mean over folds; support = total over folds)

| Category | support | majority F1 | tfidf_logreg F1 | cls_only F1 |
|---|---|---|---|---|
| Assignment | 27 | 0.000 | 0.807 | 0.071 |
| Commercial Terms | 34 | 0.000 | 0.811 | 0.608 |
| Dispute Resolution | 1 | 0.000 | 1.000 | 0.000 |
| Governing Law / Jurisdiction | 12 | 0.000 | 1.000 | 0.000 |
| Intellectual Property | 63 | 0.360 | 0.820 | 0.551 |
| Liability | 48 | 0.000 | 0.917 | 0.413 |
| Non-Compete | 19 | 0.000 | 0.444 | 0.000 |
| Non-Solicitation | 4 | 0.000 | 0.857 | 0.000 |
| Other | 30 | 0.000 | 0.833 | 0.200 |
| Penalties | 0 | n/a (no support) | n/a (no support) | n/a (no support) |
| Termination | 45 | 0.000 | 0.848 | 0.774 |
| Warranty | 4 | 0.000 | 0.000 | 0.000 |
| Confidentiality | 0 | n/a (no support) | n/a (no support) | n/a (no support) |
| Force Majeure | 0 | n/a (no support) | n/a (no support) | n/a (no support) |
| Indemnification | 0 | n/a (no support) | n/a (no support) | n/a (no support) |
| Modification | 0 | n/a (no support) | n/a (no support) | n/a (no support) |
| Payment | 0 | n/a (no support) | n/a (no support) | n/a (no support) |
| Privacy / Data Protection | 0 | n/a (no support) | n/a (no support) | n/a (no support) |
| **Micro-F1** | | **0.220** | **0.819** | **0.495** |
| **Macro-F1** (supported classes) | | **0.033** | **0.758** | **0.238** |

Per-fold std of macro-F1: cls_only=0.000

## Span localisation (windowed; answer guaranteed inside window)

| Model | n | Token-F1 | Exact Match | IoU |
|---|---|---|---|---|
| span_only (best-match gold) | 286 | 0.296 | 0.094 | 0.244 |
| span_only (own label only) | 286 | 0.159 | 0.052 | 0.129 |
| random-span baseline | 286 | 0.067 | 0.000 | 0.047 |

## Inference latency, ms per item (batch eval, this hardware)

| experiment | clause view | span view |
|---|---|---|
| cls_only | 65.6 | nan |
| span_only | nan | 221.1 |
