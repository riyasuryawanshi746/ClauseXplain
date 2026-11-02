# ClauseXplain Phase 3 results

*Run: 2026-11-02 14:21:12  |  device: cuda  |  encoder: nlpaueb/legal-bert-base-uncased  |  folds: 3*

## Read this first
- Span task is WINDOWED localisation (answer guaranteed in a 6000-char window); not comparable to full-contract CUAD numbers.
- Category labels derive from the 41->18 CUAD mapping; compare against the TF-IDF baseline before claiming anything.
- 6 categories (Confidentiality, Force Majeure, Indemnification, Modification, Payment, Privacy / Data Protection) have no data unless --weak_labels_jsonl is supplied; they are n/a, not 0.
- 'Joint' = shared encoder trained on alternating clause/span batches; each batch only trains heads it has labels for.
- Span 'best-match gold' scores against the closest of all labelled spans of that category inside the window (a window can hold several valid answers); 'own label only' is the stricter version.
- Risk labels: none used.
- Folds run: 3 of 5; seed 13. With one fold there is no variance estimate.

## Category classification (mean over folds; support = total over folds)

| Category | support | cls_only F1 |
|---|---|---|
| Assignment | 556 | 0.906 |
| Commercial Terms | 634 | 0.898 |
| Dispute Resolution | 112 | 0.882 |
| Governing Law / Jurisdiction | 274 | 0.988 |
| Intellectual Property | 1338 | 0.880 |
| Liability | 848 | 0.956 |
| Non-Compete | 481 | 0.737 |
| Non-Solicitation | 74 | 0.830 |
| Other | 669 | 0.893 |
| Penalties | 71 | 0.711 |
| Termination | 928 | 0.900 |
| Warranty | 111 | 0.863 |
| Confidentiality | 0 | n/a (no support) |
| Force Majeure | 0 | n/a (no support) |
| Indemnification | 0 | n/a (no support) |
| Modification | 0 | n/a (no support) |
| Payment | 0 | n/a (no support) |
| Privacy / Data Protection | 0 | n/a (no support) |
| **Micro-F1** | | **0.891** |
| **Macro-F1** (supported classes) | | **0.870** |

Std of macro-F1 across folds: cls_only=0.009

## Inference latency, ms per item (batch eval, this hardware)

| experiment | clause view | span view |
|---|---|---|
| cls_only | 3.8 | nan |
