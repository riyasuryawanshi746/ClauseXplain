# v5: 788 Indian clauses (v4 + batch 3)

**Data.** `indian_set/clauses_v5.csv` -> `built_v5` (local only: batch 3 clauses name private individuals). v4's 334
clauses plus 454 batch 3 clauses (385 random units, 69 keyword top-up units; 69 units marked not-a-clause) from 15
private-sector documents. Batch 3 labels: two annotators, guide v1 round 1 (kappa 0.57), re-check of the 200
disagreements under guide v2 (kappa 0.73 over all 523 units), then 127 remaining disagreements adjudicated blind.
The adjudicated label matched annotator 1 in 111, annotator 2 in 7 and neither in 9. The adjudicator is annotator 1,
so even with blinding, batch 3 labels lean towards annotator 1: only 16 of 523 labels differ from annotator 1's
round 2. Report this in the paper.

**Protocol.** Same as `indian_finetune/`: leave-one-document-group-out, now 23 groups (batch 3 template groups azure,
leave_license, dec2022 held out together, `TEMPLATE_GROUPS` in `finetune_indian.py`).

## TF-IDF (local CPU, 2026-11-03, `--tricks`, Indian clauses only)

| Train / test | test clauses | accuracy | macro-F1 |
|---|---|---|---|
| v4 (12 folds) | v4, 334 | 0.482 | 0.351 |
| v5 (23 folds) | **same v4 334** | 0.536 | 0.469 |
| v5 (23 folds) | batch 3, 454 | 0.568 | 0.505 |
| v5 (23 folds) | all 788 | 0.555 | 0.503 |

On the same 334 test clauses, the extra batch 3 training data raises TF-IDF accuracy by 0.054 and macro-F1 by 0.118.
This is one deterministic run, without a significance test yet.

## LegalBERT (Kaggle)

Not yet run: `notebooks/make_indian_v5.py` -> `notebooks/indian_v5_kaggle.ipynb` (gitignored, embeds batch 3 text;
keep the Kaggle notebook private). Two runs (CUAD->Indian, + LEDGAR, seed 13), about 1.5 h each. The summary scores
the v4 clauses alone so the result can be compared with `indian_ledgar/kaggle` (0.565 / 0.524 over 3 seeds).
