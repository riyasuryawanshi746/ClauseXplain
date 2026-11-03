# ClauseXplain Phase 3 results

*Run: 2026-11-02 20:35:44  |  device: cuda  |  encoder: law-ai/InLegalBERT  |  folds: 1*

## Read this first
- Span task is WINDOWED localisation (answer guaranteed in a 6000-char window); not comparable to full-contract CUAD numbers.
- Category labels derive from the 41->18 CUAD mapping; compare against the TF-IDF baseline before claiming anything.
- 6 categories (Confidentiality, Force Majeure, Indemnification, Modification, Payment, Privacy / Data Protection) have no data unless --weak_labels_jsonl is supplied; they are n/a, not 0.
- 'Joint' = shared encoder trained on alternating clause/span batches; each batch only trains heads it has labels for.
- Span 'best-match gold' scores against the closest of all labelled spans of that category inside the window (a window can hold several valid answers); 'own label only' is the stricter version.
- Risk labels: none used.
- Folds run: 1 of 5; seed 13. With one fold there is no variance estimate.

## Category classification (mean over folds; support = total over folds)

| Category | support | cls_only F1 |
|---|---|---|
| Assignment | 192 | 0.873 |
| Commercial Terms | 224 | 0.871 |
| Dispute Resolution | 37 | 0.800 |
| Governing Law / Jurisdiction | 79 | 1.000 |
| Intellectual Property | 451 | 0.855 |
| Liability | 270 | 0.941 |
| Non-Compete | 145 | 0.689 |
| Non-Solicitation | 27 | 0.840 |
| Other | 211 | 0.864 |
| Penalties | 38 | 0.714 |
| Termination | 291 | 0.864 |
| Warranty | 18 | 0.800 |
| Confidentiality | 0 | n/a (no support) |
| Force Majeure | 0 | n/a (no support) |
| Indemnification | 0 | n/a (no support) |
| Modification | 0 | n/a (no support) |
| Payment | 0 | n/a (no support) |
| Privacy / Data Protection | 0 | n/a (no support) |
| **Micro-F1** | | **0.863** |
| **Macro-F1** (supported classes) | | **0.843** |

Only one fold: no variance estimate, so differences of a few points are not evidence.

## Inference latency, ms per item (batch eval, this hardware)

| experiment | clause view | span view |
|---|---|---|
| cls_only | 3.1 | nan |
