# Step 5: LEDGAR clauses as extra training data (document-held-out Indian test)

Same protocol as `../indian_finetune/` (leave-one-document-group-out, 12 folds, n = 334, settings fixed in code).
Extra data: LEDGAR (LexGLUE train split, 60k SEC provisions, 100 labels) mapped by `phase3_model/ledgar_extra.py`
using the category definitions in `indian_set/labelling/LABELLING_GUIDE.md` (mapping fixed 2026-11-03 before any
result). 24,737 provisions map to 11 categories; at most 100 per category (sampling seed 13) are added to every
training fold. LEDGAR has **no** Force Majeure, Privacy, Liability, Penalties, Non-Compete, Non-Solicitation or
Commercial Terms label. It is dominated by credit and employment agreements, so it is a different domain from the
(mostly government) Indian documents. LEDGAR data is downloaded by the script, not stored in the repo.

## TF-IDF (local CPU, `local_cpu/`, balanced class weights)

| Run | acc (95% CI) | macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| Indian only (from `../indian_finetune/`) | 0.482 (0.429-0.536) | 0.351 (0.310-0.401) | 0.396 / 0.336 | 0.599 / 0.485 | 0.474 / 0.500 | 0.182 |
| Indian + LEDGAR (11 categories) | **0.557** (0.503-0.609) | 0.457 (0.409-0.520) | 0.464 / 0.464 | 0.683 / 0.619 | 0.575 / 0.519 | 0.364 |
| Indian + LEDGAR (5 missing/weak categories only) | 0.470 (0.417-0.524) | 0.333 (0.292-0.385) | 0.260 / 0.268 | 0.754 / 0.642 | 0.469 / 0.472 | 0.273 |
| CUAD + Indian (from `../indian_finetune/local_cpu/`) | 0.416 (0.365-0.470) | 0.427 (0.352-0.462) | 0.484 / 0.540 | 0.324 / 0.351 | 0.439 / 0.368 | 0.636 |
| CUAD + Indian + LEDGAR (11 categories) | 0.470 (0.417-0.524) | **0.489** (0.413-0.529) | 0.500 / 0.570 | 0.430 / 0.490 | 0.513 / 0.377 | 0.545 |

Reading: LEDGAR adds about +0.06 to +0.11 macro-F1 for TF-IDF. Adding it only for the five missing/weak categories
(Confidentiality, Indemnification, Modification, Payment, Dispute Resolution) raises those but shifts predictions
toward them and hurts the seen categories, so the 11-category version is the main configuration.
(A first attempt at the 5-category run passed `--extra_categories` twice, so it used Dispute Resolution only; it was
discarded and rerun.)

## Neural (Kaggle T4, `kaggle/`, run 2026-11-03 with `notebooks/indian_ledgar_kaggle.ipynb`)

LegalBERT trained on CUAD fold 0 (retrained in this notebook: CUAD fold-0 macro-F1 0.864), then on Indian clauses,
with and without LEDGAR (cap 100 per category). Three seeds each; mean +- s.d. over seeds.

| Run | acc | macro-F1 | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| LegalBERT CUAD -> Indian (s13-15) | 0.565 +- 0.007 | 0.524 +- 0.006 | 0.538 / 0.593 | 0.601 / 0.511 | 0.613 / 0.462 | 0.667 |
| + LEDGAR (s13-15) | **0.600 +- 0.017** | 0.512 +- 0.017 | 0.516 / 0.521 | 0.714 / **0.636** | 0.656 / 0.478 | 0.636 |

Per-seed values in `kaggle/summary.md`. Paired bootstrap (1000 resamples over clauses, same seed) for the accuracy
gain from LEDGAR: s13 (-0.018, 0.063), s14 (0.009, 0.096), s15 (-0.006, 0.075); only s14 excludes zero.

Reading: the baseline replicates the first run (0.566 / 0.528) closely. LEDGAR raises accuracy by about 0.035 by
improving the categories CUAD lacks (unseen macro-F1 +0.125) while the seen categories lose about 0.07 macro-F1, so
overall macro-F1 does not move. Suggestive, not conclusive. InLegalBERT results: `../indian_inlegalbert/`.
