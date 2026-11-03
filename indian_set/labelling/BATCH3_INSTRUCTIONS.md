# Batch 3: private-sector Indian contracts (label sheet, 2026-11-03)

Texts and sheets are **local only** (gitignored: `indian_set/documents_batch3/`, `indian_set/labelling/batch3/`)
because several documents name private individuals (leave-and-license parties, employees, salaries).
Only this file and the tooling are in git.

## Documents (15, all governed by Indian law, text layer, none in CUAD)

| Group | Documents | Type |
|---|---|---|
| azure | azure_employment_2022, azure_employment_2023, azure_executive_employment_2022 | employment (SEC filings; 62-89% shared text) |
| leave_license | leave_license_mh_2024a, leave_license_mh_2024b | Maharashtra leave and license (60% shared) |
| dec2022 | dec2022_share_subscription_purchase, dec2022_shareholders_agreement | same transaction (21% shared) |
| (own group) | share_subscription_purchase_executed, kaynes_share_purchase_2022, concord_sha_amendment_2021 | share deals |
| (own group) | cyient_master_services_2022, majesco_master_services_2016, infosys_springboard_mou_2023 | services / MoU |
| (own group) | hinduja_business_transfer_2021, mmfsl_trust_deed_amendment_2023 | business transfer, trust deed |

Left out of the 28 received: 7 scanned PDFs without text (need OCR), 1 duplicate, 2 court documents, Zoomcar
employment (Delaware law), Freshworks DPA (German law; web page). Template groups must be held out together in
document-held-out training (add them to `group_of` in `finetune_indian.py` when batch 3 is used).

## The sheet (523 units)

- 446 **random** units: at most 45 per group, no keyword selection, so categories appear in their natural mix.
- 77 **top-up** units, keyword-assisted, for weak categories (Modification 10, Liability 10, Assignment 10,
  Non-Compete 10, Warranty 10, Indemnification 10, Privacy 9, Non-Solicitation 6, Force Majeure 2). Report them
  separately in the paper.
- Near-duplicate units (mostly repeated text across the Azure agreements) are skipped. Rows are shuffled, and the
  sheets do not say which pool or keyword a unit came from (`batch3_sheet_key_DO_NOT_SHARE.csv` has that).

## Status (2026-11-03)

Round 1 is done (guide v1, both annotators, all 523 units): raw agreement 0.62, kappa 0.57, 200 disagreements,
mostly Commercial Terms vs Other, not-a-clause (party blocks), and Payment. Guide v2 is now final
(`LABELLING_GUIDE.md`). **Round 2 = re-check**: each annotator re-labels only the 200 disagreement units under v2,
seeing their own round-1 label but not the other's:

1. Give `recheck_annotator1.csv` to annotator 1 and `recheck_annotator2.csv` to annotator 2 (never the other's).
   Fill `your_category` or `not_a_clause (y)` on every row; `context_before` / `context_after` show the
   neighbouring text for list items. Read guide v2 first.
2. `python recheck.py merge --sheet1 ../indian_set/labelling/batch3/label_sheet_batch3_annotator1_filled.csv --sheet2 ../indian_set/labelling/batch3/label_sheet_batch3_annotator2_filled.csv --recheck1 ../indian_set/labelling/batch3/recheck_annotator1_filled.csv --recheck2 ../indian_set/labelling/batch3/recheck_annotator2_filled.csv --out_dir ../indian_set/labelling/batch3`
   -> round-2 sheets and `recheck_report.md` (round 1 and round 2 kappa, who changed what).
3. Convert the round-2 sheets (`make_label_sheet.py convert --sheet round2_annotator1.csv --second round2_annotator2.csv`)
   and adjudicate what is still split, blind (`adjudicate.py sheet`).

**Round 2 result (2026-11-03):** raw agreement 0.76, kappa 0.73 over all 523 units (the 323 round-1 agreements
were not re-checked, so this is somewhat optimistic compared with a fresh double annotation). Each annotator changed
about 140 of the 200 re-checked labels. 127 disagreements remain ->
`batch3/round2/adjudication_sheet_round2.csv` (blind; key in `adjudication_key_DO_NOT_OPEN.csv`). A first
round-2 submission had identical labels in both sheets and was discarded (`batch3/discarded_round2_attempt1/`).

**Adjudication (2026-11-03):** all 127 adjudicated blind (`round2/adjudication_sheet_round2_filled.csv`). Final label =
annotator 1 in 111, annotator 2 in 7, neither in 9; 16 of 523 labels differ from annotator 1's round 2 ->
`round2/label_sheet_batch3_adjudicated.csv`. 454 clauses, 69 not-a-clause. Merged with v4 into
`indian_set/clauses_v5.csv` -> `built_v5` (788 clauses, local only). Results: `experiments/indian_v5/`.

## How to label (round 1, kept for the record)

1. **First fix the guideline** (`LABELLING_GUIDE.md`, phase A of the plan): clearer rules for Other vs Commercial
   Terms, Warranty vs Liability, and "not a clause". Label batch 3 with that version.
2. Annotator 1 fills `label_sheet_batch3_annotator1.csv`, annotator 2 fills `label_sheet_batch3_annotator2.csv`,
   **independently, all 523 units**. Per row: `your_category` (exact category name) or `not_a_clause` = y
   (addresses, signature blocks, stamp text, definitions, recitals). `notes` for doubts.
3. Agreement and disagreements:
   `python make_label_sheet.py convert --sheet ../indian_set/labelling/batch3/label_sheet_batch3_annotator1.csv --second ../indian_set/labelling/batch3/label_sheet_batch3_annotator2.csv --out ../indian_set/labelling/batch3/clauses_batch3.csv`
4. Adjudicate the disagreements blind (`adjudicate.py`), then build the new frozen set (v5).

Rebuild the sheet (same seed gives the same sheet):
`python make_label_sheet.py build --docs_dir ../indian_set/documents_batch3 --random_per_group 45 --dedupe --group azure=azure_employment_2022,azure_employment_2023,azure_executive_employment_2022 leave_license=leave_license_mh_2024a,leave_license_mh_2024b dec2022=dec2022_share_subscription_purchase,dec2022_shareholders_agreement --topup_per_category 10 --topup_categories Modification "Privacy / Data Protection" Liability Assignment Non-Compete Non-Solicitation "Force Majeure" Warranty Indemnification --out ../indian_set/labelling/batch3/label_sheet_batch3_full.csv`
