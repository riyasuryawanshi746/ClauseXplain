# Guide: finding Indian contracts to grow the clause set

Goal: ~15-20 NEW clauses per category from NEW documents, kept as a separate test set (never mixed into the 228 used so far).

## Priority categories (current counts in the 228-clause set)
| Category | Have | Where to look |
|---|---|---|
| Dispute Resolution | 10 (model scores F1 0) | Model Concession Agreements (Dispute Resolution article), CPWD GCC (arbitration clause), consultancy RFPs |
| Warranty | 16 | CPWD/EPC GCC (defects liability), goods supply contracts, GeM-style terms |
| Indemnification | 15 | Concession agreements (Indemnity article), EPC GCC, IT service templates |
| Governing Law | 7 | Concession agreements ("Governing Law and Jurisdiction"), any contract's boilerplate |
| Liability | 5 | GCC "limitation of liability", consultancy contracts |
| Assignment | 5 | Concession agreements, GCC ("assignment and subcontracting") |
| Modification | 5 | Boilerplate ("amendments") in concession and service agreements |
| Privacy / Data Protection | 6 | IT/cloud service templates, data-hosting or empanelment agreements |
| Non-Solicitation / Non-Compete | 1 / 0 | Hard in government documents; need private-sector agreements (see below) |
| Force Majeure, Payment, Termination, Penalties | 13 / 22 / 12 / 15 | Almost any works or services contract |

## Where to look (found by search; check each link still works and the terms of use)
- PPP in India model concession agreements: https://www.pppinindia.gov.in/model_concession_agreement
- NITI Aayog model concession agreement (electric buses; the notification page is only a cover note, get the full MCA from NITI): https://www.niti.gov.in
- Draft Model Concession Agreement, Waste to Compost: https://sbmurban.org/storage/app/media/Draft_Model_Concession_Agreement-Waste_to_Compost.pdf
- Central Public Procurement Portal standard bidding documents (consultancy, works, services): https://eprocure.gov.in/cppp/sites/default/files/standard_biddingdocs/Procurement_Consultancy_Services.pdf
- CPWD General Conditions of Contract 2023 (copies on institute sites, e.g. https://iitk.ac.in/iwd/file/2025/12-AC-EE-2025/GCC2023.pdf)
- Model RFP for consulting services: https://smartnet.niua.org/sites/default/files/resources/Model_RFP_Consulting_Services_26-Dec-2011_25112.pdf
- Private-sector style clauses (non-solicit, non-compete): court judgments that quote the contract (search indiankanoon.org for "non-solicitation clause" / "restraint of trade Section 27"), and agreements filed publicly by companies. Use judgments only where the clause is quoted in full; note that the source is a judgment.

## Pick documents like this
- Text-based PDFs you can select/copy text from. Avoid scans (OCR errors hurt, as with the Noida airport document).
- Real agreements or full model contracts. Avoid short notices, circulars and forms that are mostly blanks.
- Mix of document types; no single document should give more than ~10-15 clauses (one PMC agreement already gives 58).
- Government or official sources. Do not use Scribd/SlideShare copies; get the original from the official site.
- Note the URL and date accessed for each file (needed for the paper's data statement).

## What to send me
1. The PDF(s) (or .txt). Name them like `indian_<issuer>_<type>_<year>.pdf`.
2. A line per file: URL, date accessed.
Then I run `prepare_documents.py` (clean text) and `--browse` (candidate clauses per category) and send you a sheet to label. You pick the category for each candidate; I do not assign labels for you.

## Rules while selecting clauses
- Copy clause text exactly; do not edit. It must appear verbatim in the document (the builder checks and rejects anything that does not).
- One clause = one obligation or right, 100-900 characters. No headings, no partial sentences.
- Label with `LABELLING_GUIDE.md`. When unsure between a category and Other, follow the tie-break rules there.
- Do not look at model predictions while labelling.
