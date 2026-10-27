# ClauseXplain Phase 3 results

*Run: 2026-10-09 10:24:32  |  device: cuda  |  encoder: allenai/longformer-base-4096  |  folds: 1*

## Read this first
- Span task is WINDOWED localisation (answer guaranteed in a 3000-char window); not comparable to full-contract CUAD numbers.
- Category labels derive from the 41->18 CUAD mapping; compare against the TF-IDF baseline before claiming anything.
- 6 categories (Confidentiality, Force Majeure, Indemnification, Modification, Payment, Privacy / Data Protection) have no data unless --weak_labels_jsonl is supplied; they are n/a, not 0.
- 'Joint' = shared encoder trained on alternating clause/span batches; each batch only trains heads it has labels for.
- Span 'best-match gold' scores against the closest of all labelled spans of that category inside the window (a window can hold several valid answers); 'own label only' is the stricter version.
- Risk labels: none used.
- Folds: 1; seed 13. With one fold there is no variance estimate.
- Span training used a random 50% of span items (compute saver).

## Category classification (mean over folds; support = total over folds)

| Category | support | majority F1 | tfidf_logreg F1 | cls_only F1 | joint_cls_span F1 |
|---|---|---|---|---|---|
| Assignment | 192 | 0.000 | 0.836 | 0.881 | 0.871 |
| Commercial Terms | 224 | 0.000 | 0.831 | 0.863 | 0.876 |
| Dispute Resolution | 37 | 0.000 | 0.700 | 0.703 | 0.730 |
| Governing Law / Jurisdiction | 79 | 0.000 | 0.987 | 1.000 | 0.994 |
| Intellectual Property | 451 | 0.371 | 0.860 | 0.858 | 0.866 |
| Liability | 270 | 0.000 | 0.929 | 0.930 | 0.934 |
| Non-Compete | 145 | 0.000 | 0.607 | 0.692 | 0.706 |
| Non-Solicitation | 27 | 0.000 | 0.773 | 0.815 | 0.902 |
| Other | 211 | 0.000 | 0.852 | 0.852 | 0.844 |
| Penalties | 38 | 0.000 | 0.500 | 0.667 | 0.500 |
| Termination | 291 | 0.000 | 0.812 | 0.835 | 0.834 |
| Warranty | 18 | 0.000 | 0.722 | 0.722 | 0.718 |
| Confidentiality | 0 | n/a (no support) | n/a (no support) | n/a (no support) | n/a (no support) |
| Force Majeure | 0 | n/a (no support) | n/a (no support) | n/a (no support) | n/a (no support) |
| Indemnification | 0 | n/a (no support) | n/a (no support) | n/a (no support) | n/a (no support) |
| Modification | 0 | n/a (no support) | n/a (no support) | n/a (no support) | n/a (no support) |
| Payment | 0 | n/a (no support) | n/a (no support) | n/a (no support) | n/a (no support) |
| Privacy / Data Protection | 0 | n/a (no support) | n/a (no support) | n/a (no support) | n/a (no support) |
| **Micro-F1** | | **0.227** | **0.834** | **0.852** | **0.854** |
| **Macro-F1** (supported classes) | | **0.031** | **0.784** | **0.818** | **0.815** |

Only one fold: no variance estimate, so differences of a few points are not evidence.

## Span localisation (windowed; answer guaranteed inside window)

| Model | n | Token-F1 | Exact Match | IoU |
|---|---|---|---|---|
| span_only (best-match gold) | 1971 | 0.746 | 0.542 | 0.706 |
| span_only (own label only) | 1968 | 0.501 | 0.355 | 0.469 |
| random-span baseline | 1971 | 0.113 | 0.000 | 0.082 |
| joint_cls_span (best-match gold) | 1971 | 0.712 | 0.447 | 0.656 |
| joint_cls_span (own label only) | 1968 | 0.496 | 0.307 | 0.452 |

## Inference latency, ms per item (batch eval, this hardware)

| experiment | clause view | span view |
|---|---|---|
| cls_only | 26.2 | nan |
| span_only | nan | 57.8 |
| joint_cls_span | 26.2 | 57.8 |
