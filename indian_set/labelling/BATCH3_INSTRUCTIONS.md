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

## How to label

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
