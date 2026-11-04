# v6: 905 Indian clauses (v5 + batch 4)

**Data.** `indian_set/clauses_v6.csv` -> `built_v6` (905 clauses, local only because of batch 3); public version
`clauses_v6_public.csv` -> `built_v6_public` (886; no leave-and-license deeds). Batch 4: 117 clauses from 8 documents
(6 groups), guide v2 from the start, round 1 kappa 0.40, re-check of the 80 disagreements kappa 0.79, 28 adjudicated
blind (final = annotator 1 in 2, annotator 2 in 24, neither in 2). See `indian_set/labelling/BATCH4_INSTRUCTIONS.md`.

**Protocol.** Same as v5, 29 groups (batch 4 groups dhoot and ssa_jan2025 held out together). Test clauses for the
comparison: the same 334 v4 clauses.

## TF-IDF (local CPU, `--tricks`)

| Training data | accuracy (334) | macro-F1 (334) |
|---|---|---|
| v4 | 0.482 | 0.351 |
| v5 (+ batch 3) | 0.536 | 0.469 |
| v6 (+ batch 3 + batch 4) | 0.554 | 0.503 |

## LegalBERT (Kaggle, 2026-11-05, seed 13, `kaggle/`)

Same settings as v5; one run per notebook (`notebooks/make_indian_v6.py`). Scored on the same 334 v4 test clauses
(`compare_v6.py`; predictions here have no clause text).

| Run | acc (334) | macro-F1 (334) | mF1 without Non-Solicitation (16 cat.) | seen mF1 | unseen mF1 | added acc (571) |
|---|---|---|---|---|---|---|
| v4 CUAD->Indian (3 seeds) | 0.565 | 0.524 | 0.494 | 0.593 | 0.511 | -- |
| v5 CUAD->Indian (s13) | 0.611 | 0.609 | 0.584 | 0.614 | 0.732 | -- |
| **v6 CUAD->Indian (s13)** | 0.629 | 0.569 | 0.604 | 0.544 | 0.745 | 0.632 |
| v6 + LEDGAR | pending | | | | | |

- v5 -> v6: accuracy +0.018 [-0.006, +0.042]; macro-F1 -0.040, all of it from the single Non-Solicitation test clause
  (predicted correctly by every v4/v5 run, as Confidentiality by v6: F1 1.00 -> 0.00, which moves 17-category
  macro-F1 by about 0.06). Without it: +0.020 [-0.007, +0.050]. So batch 4 adds a small gain that is within noise.
- v4 (each of 3 seeds) -> v6: accuracy +0.06 to +0.07, all intervals exclude 0; macro-F1 +0.04 to +0.05 (one of three
  excludes 0; without Non-Solicitation, 0.494 -> 0.604).
- Per category (v5 -> v6, seed 13): Payment 0.57 -> 0.63, Termination 0.62 -> 0.67, Commercial Terms 0.42 -> 0.48,
  Assignment 0.47 -> 0.53, Liability 0.24 -> 0.33; others within 0.03.
- Seen-category macro-F1 drops (0.614 -> 0.544), again mostly Non-Solicitation, which is a seen category.

Rerun: `python experiments/indian_v6/compare_v6.py . experiments/indian_v6/kaggle`.
