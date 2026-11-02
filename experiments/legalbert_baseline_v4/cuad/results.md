# ClauseXplain Phase 3 results

*Run: 2026-11-02 14:58:28  |  device: cuda  |  encoder: nlpaueb/legal-bert-base-uncased  |  folds: 3*

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
| Assignment | 556 | 0.905 |
| Commercial Terms | 634 | 0.892 |
| Dispute Resolution | 112 | 0.868 |
| Governing Law / Jurisdiction | 274 | 0.986 |
| Intellectual Property | 1338 | 0.882 |
| Liability | 848 | 0.953 |
| Non-Compete | 481 | 0.744 |
| Non-Solicitation | 74 | 0.817 |
| Other | 669 | 0.892 |
| Penalties | 71 | 0.701 |
| Termination | 928 | 0.894 |
| Warranty | 111 | 0.854 |
| Confidentiality | 0 | n/a (no support) |
| Force Majeure | 0 | n/a (no support) |
| Indemnification | 0 | n/a (no support) |
| Modification | 0 | n/a (no support) |
| Payment | 0 | n/a (no support) |
| Privacy / Data Protection | 0 | n/a (no support) |
| **Micro-F1** | | **0.889** |
| **Macro-F1** (supported classes) | | **0.866** |

Std of macro-F1 across folds: cls_only=0.011

## Inference latency, ms per item (batch eval, this hardware)

| experiment | clause view | span view |
|---|---|---|
| cls_only | 3.7 | nan |
