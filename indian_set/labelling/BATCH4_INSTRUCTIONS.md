# Batch 4: more private-sector Indian contracts (label sheet, 2026-11-04)

Texts are in git (`indian_set/documents_batch4/`): PAN, Aadhaar, mobile, e-mail and account numbers are masked
(`redact_batch3.py` patterns) before the label sheet was built, so labels and offsets point into the public text.

## Documents (8 files, 6 groups, all Indian companies / Indian law, none in CUAD)

| Group | Documents | Type |
|---|---|---|
| dhoot | dhoot_business_transfer_2020, dhoot_employment_2025 | business transfer (with deed of assignment, amendment); employment (offer-document material contracts) |
| ssa_jan2025 | ssa_spa_jan2025_part1, ssa_spa_jan2025_part2 | two OCR'd parts of one share subscription and purchase agreement (no shared text) |
| (own group) | hero_yamaha_jv_agreement | joint venture agreement |
| (own group) | indusface_data_processing_agreement | data processing agreement |
| (own group) | mrn_canepower_master_services | master services agreement and amendment |
| (own group) | selling_distribution_gujarat_2025 | selling and distribution agreement |

Left out of the 11 PDFs received: Infosys CEO employment agreement 2017 (California law; one copy has no text
layer), the 164-page scanned copy of the January 2025 SSA (no text layer, needs OCR), and one exact duplicate.

## The sheet (148 units)

120 random units (at most 20 per group) + 28 keyword top-up units for thin categories (Penalties, Non-Solicitation,
Non-Compete, Intellectual Property, Liability, Governing Law / Jurisdiction, Privacy / Data Protection, Force
Majeure). Rows are shuffled; `batch4_sheet_key_DO_NOT_SHARE.csv` (not in git) says which pool a unit came from.

## How to label (about 30 minutes each)

1. Read `LABELLING_GUIDE.md` (v2). Same rules as batch 3 round 2.
2. Annotator 1 fills `label_sheet_batch4_annotator1.csv`, annotator 2 fills `label_sheet_batch4_annotator2.csv`,
   independently, all 148 rows: `your_category` (exact name) or `not_a_clause (y)` = y. `context_before` /
   `context_after` are the neighbouring units, only to help read list items; label `clause_text`.
3. Then: agreement + disagreements (`make_label_sheet.py convert --second`), blind adjudication (`adjudicate.py`),
   merge into v6 = v5 + batch 4 (training only; the test set stays the 334 v4 clauses).

Rebuild the sheet (same seed, same sheet):
`python make_label_sheet.py build --docs_dir ../indian_set/documents_batch4 --random_per_group 20 --dedupe --group dhoot=dhoot_business_transfer_2020,dhoot_employment_2025 ssa_jan2025=ssa_spa_jan2025_part1,ssa_spa_jan2025_part2 --topup_per_category 4 --topup_categories Penalties Non-Solicitation Non-Compete "Intellectual Property" Liability "Governing Law / Jurisdiction" "Privacy / Data Protection" "Force Majeure" --out ../indian_set/labelling/batch4/label_sheet_batch4_full.csv`
(the annotator sheets add context columns and shuffle rows with seed 4).

## Status (2026-11-04)

Round 1 done (guide v2, both annotators, all 148 units): raw agreement 0.46, kappa 0.40, 80 disagreements. Main
pairs: not-a-clause vs Other (18; party introductions such as "This Agreement is made ... by and between:", which
guide v2 lists as party blocks), seller/company representations (guide section 4: Warranty) vs their topic, Other vs
Payment. Same path as batch 3: each annotator re-checks only the 80 disagreements under guide v2, seeing only their
own round-1 label (`recheck_annotator1.csv` to annotator 1, `recheck_annotator2.csv` to annotator 2), then
`recheck.py merge` and blind adjudication. Filled sheets stay out of git until the re-check is merged.

**Round 2 (re-check, 2026-11-04):** raw agreement 0.81, kappa 0.79 over all 148 units (the 68 round-1 agreements were
not re-checked, so this is somewhat optimistic). Annotator 1 changed 30 of 80 labels, annotator 2 changed 38.
28 disagreements remained -> blind adjudication (`round2/adjudication_sheet_batch4_filled.csv`): final label =
annotator 1 in 2, annotator 2 in 24, neither in 2. Result: 117 clauses, 31 not-a-clause
(`round2/label_sheet_batch4_adjudicated.csv`). v6 = v5 + batch 4: `clauses_v6.csv` -> `built_v6` (905 clauses,
local only because of batch 3) and `clauses_v6_public.csv` -> `built_v6_public` (886 clauses, in git).
