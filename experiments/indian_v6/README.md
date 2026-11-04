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
| v4 + LEDGAR (3 seeds) | 0.600 | 0.512 | 0.544 | 0.521 | 0.636 | -- |
| v5 + LEDGAR (s13) | 0.644 | 0.636 | 0.613 | 0.658 | 0.727 | -- |
| **v6 + LEDGAR (s13)** | 0.638 | 0.574 | 0.609 | 0.559 | 0.726 | 0.657 |

- v5 -> v6: accuracy +0.018 [-0.006, +0.042]; macro-F1 -0.040, all of it from the single Non-Solicitation test clause
  (predicted correctly by every v4/v5 run, as Confidentiality by v6: F1 1.00 -> 0.00, which moves 17-category
  macro-F1 by about 0.06). Without it: +0.020 [-0.007, +0.050]. So batch 4 adds a small gain that is within noise.
- v4 (each of 3 seeds) -> v6: accuracy +0.06 to +0.07, all intervals exclude 0; macro-F1 +0.04 to +0.05 (one of three
  excludes 0; without Non-Solicitation, 0.494 -> 0.604).
- Per category (v5 -> v6, seed 13): Payment 0.57 -> 0.63, Termination 0.62 -> 0.67, Commercial Terms 0.42 -> 0.48,
  Assignment 0.47 -> 0.53, Liability 0.24 -> 0.33; others within 0.03.
- Seen-category macro-F1 drops (0.614 -> 0.544), again mostly Non-Solicitation, which is a seen category.

**With LEDGAR** (`kaggle/ledgar_summary.md`):
- v5 + LEDGAR -> v6 + LEDGAR: accuracy -0.006 [-0.039, +0.027], macro-F1 -0.062 [-0.093, +0.024]. Same
  Non-Solicitation clause (v5 + LEDGAR got it, v6 + LEDGAR says Confidentiality, as all three v4 + LEDGAR seeds did);
  without it -0.004 [-0.039, +0.033]. No change.
- v6 -> v6 + LEDGAR: accuracy +0.009 [-0.021, +0.042], macro-F1 +0.005 [-0.031, +0.044]. With 905 Indian clauses,
  LEDGAR no longer adds anything measurable (on v4 it added +0.035 accuracy; on v5 +0.033).
- v4 + LEDGAR (each seed) -> v6 + LEDGAR: accuracy +0.048 / +0.018 / +0.048 (two of three exclude 0); macro-F1
  +0.069 / +0.042 / +0.073 (two of three exclude 0).
- Per category (v6 -> v6 + LEDGAR), LEDGAR helps Commercial Terms (0.48 -> 0.59) and Governing Law (0.56 -> 0.62)
  and costs Intellectual Property (0.67 -> 0.57); small numbers either way (n = 9-24).
- All clauses (905 incl. added, document-held-out): 0.650 acc / 0.643 macro-F1 for v6 + LEDGAR; added-clause
  accuracy 0.657 (n = 571), vs 0.632 for plain v6.

**Reading.** Best on the fixed 334 is still v5 + LEDGAR (0.644 / 0.636); v6 runs are level with it on accuracy and
lower on macro-F1 only because of one clause. Batch 4 did not move the fixed test set beyond noise; with one seed per
run, differences under about 0.03 accuracy should not be read as real.

Note: the "12 folds" in the header of `kaggle/*_summary.md` is stale text from the generator; the run used 29 groups
(see the fold lines in `notebooks/ran/indian_v6_*_kaggle_ran.ipynb`).

Rerun: `python experiments/indian_v6/compare_v6.py . experiments/indian_v6/kaggle`.
