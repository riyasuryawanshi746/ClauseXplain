# v5: 788 Indian clauses (v4 + batch 3)

**Data.** `indian_set/clauses_v5.csv` -> `built_v5` (788 clauses, local only). Public version without the two
leave-and-license deeds, identifiers masked: `clauses_v5_public.csv` -> `built_v5_public` (769 clauses). v4's 334
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

## LegalBERT (Kaggle, 2026-11-03, seed 13, `kaggle/`)

Same settings as `indian_ledgar/kaggle`. Scored on the same 334 v4 test clauses (`compare_v5.py`; predictions here have
no clause text). v4 = the earlier runs without batch 3 (3 seeds).

| Run | acc (334) | macro-F1 (334) | seen mF1 | unseen mF1 | batch 3 acc (454) |
|---|---|---|---|---|---|
| v4 CUAD->Indian (mean of 3 seeds) | 0.565 | 0.524 | 0.593 | 0.511 | -- |
| **v5 CUAD->Indian** | 0.611 | 0.609 | 0.614 | 0.732 | 0.615 |
| v4 + LEDGAR (mean of 3 seeds) | 0.600 | 0.512 | 0.521 | 0.636 | -- |
| **v5 + LEDGAR** | 0.644 | 0.636 | 0.658 | 0.727 | 0.663 |

Paired bootstrap (2000 resamples, 95% interval), v5 seed 13 against each v4 seed:

- CUAD->Indian: macro-F1 +0.08 to +0.09, all three intervals exclude 0; accuracy +0.04 to +0.05, one of three excludes 0.
- + LEDGAR: macro-F1 +0.10 to +0.14, all three exclude 0; accuracy +0.02 to +0.05, two of three exclude 0.
- v5 + LEDGAR vs v5: accuracy +0.033 [0.000, 0.063], macro-F1 +0.027 [-0.007, 0.065].

The gain is mostly in categories CUAD lacks (seed 13: Modification 0.00 -> 0.33, Privacy 0.00 -> 0.50,
Confidentiality 0.75 -> 0.87); seen-category accuracy is unchanged (0.538 -> 0.536). Only one seed for v5.
Rerun the comparison from the repo root: `python experiments/indian_v5/compare_v5.py . experiments/indian_v5/kaggle`.
