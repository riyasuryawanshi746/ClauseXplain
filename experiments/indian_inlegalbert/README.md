# Step 6: InLegalBERT encoder (document-held-out Indian test)

Run 2026-11-03 on Kaggle T4 with `notebooks/indian_inlegalbert_kaggle.ipynb`. `law-ai/InLegalBERT` (Paul et al. 2023:
LEGAL-BERT further pretrained on Indian court judgments; same tokenizer and size) in place of LEGAL-BERT, same protocol
and settings: trained on CUAD fold 0 (CUAD fold-0 macro-F1 0.843, vs 0.864 for LEGAL-BERT in the same setup), then on
Indian clauses with documents held out (12 folds, 8 epochs, last-epoch model).

| Run | acc | macro-F1 | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| InLegalBERT CUAD -> Indian (s13-15, mean +- s.d.) | 0.537 +- 0.009 | 0.450 +- 0.007 | 0.512 / 0.498 | 0.570 / 0.487 | 0.576 / 0.453 | 0.727 |
| InLegalBERT CUAD -> Indian + LEDGAR (s13) | 0.578 (0.524-0.630) | 0.504 | 0.500 / 0.511 | 0.683 / 0.632 | 0.623 / 0.481 | 0.636 |
| LEGAL-BERT CUAD -> Indian (reference, `../indian_ledgar/kaggle/`) | 0.565 +- 0.007 | 0.524 +- 0.006 | 0.538 / 0.593 | 0.601 / 0.511 | 0.613 / 0.462 | 0.667 |

Paired bootstrap for InLegalBERT minus LEGAL-BERT accuracy (same seed): s13 (-0.078, -0.006), s14 (-0.060, 0.006),
s15 (-0.048, 0.021). InLegalBERT is lower on every seed and on CUAD; pretraining on judgments does not carry over to
contract clauses in this setup. Per-seed values in `summary.md`.
