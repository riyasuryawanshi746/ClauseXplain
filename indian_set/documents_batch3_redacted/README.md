# Batch 3 documents (redacted public copy)

13 private-sector Indian agreements used in batch 3 (v5 set). All are governed by Indian law and were obtained from
public sources: SEC exhibits (Azure Power, Majesco), stock-exchange filings and offer-document material contracts
(Mahindra Finance, Hinduja Global Solutions, Kaynes, Concord Enviro, Cyient, the share subscription and dec2022
documents), and a published university MoU (Infosys Springboard).

- Made with `python indian_set/redact_batch3.py` from the local originals. PAN, Aadhaar, mobile numbers, e-mail
  addresses, bank account and demat numbers are replaced with `[REDACTED]`. Names of company officers and parties
  that appear in the public filings are left in.
- The two Maharashtra leave-and-license deeds are **not included**: they are between private individuals.
- Labels: `../clauses_v5_public.csv` -> `../built_v5_public/` (769 clauses, masked the same way, offsets into these
  files). The reported v5 runs used the full local set of 788 (`clauses_v5.csv`, with the leave-and-license clauses);
  see `../labelling/BATCH3_INSTRUCTIONS.md`.
- Text was extracted from PDFs; the e-stamp pages of scanned documents contain OCR noise.
