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
