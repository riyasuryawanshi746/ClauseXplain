CLAUSEXPLAIN -- DATASET CONSTRUCTION README (Phase 1: CUAD -> 18-category taxonomy)
====================================================================================

1. DATASET SOURCE
-----------------
Contract Understanding Atticus Dataset (CUAD) v1, Hendrycks et al., NeurIPS 2021
Datasets and Benchmarks track. 510 commercial contracts sourced from SEC EDGAR
filings, 13,823 expert-annotated clause spans across 41 categories. License:
CC BY 4.0.

2. CUAD SCHEMA
--------------
CUAD is distributed as a SQuAD 2.0-style extractive QA file: each contract is
one "context" (the full contract text), paired with 41 fixed "questions" (one
per category). Each question has is_impossible=True if that category is not
present in the contract, or one or more answer spans (character offsets into
the contract text) if it is present. There is no pre-segmented "clause" object
in the raw file -- a clause, for this project's purposes, IS a CUAD answer
span, referred to throughout this project as a "CUAD-highlighted span" rather
than a "clause," since CUAD's expert annotators highlighted the legally
relevant portion of text, not necessarily a complete, self-contained clause.

3. CUAD LIMITATIONS FOR THIS PROJECT
-------------------------------------
- Jurisdiction: overwhelmingly US. Checking all answered "Governing Law"
  spans, only 5 of 510 contracts (~1%) even mention India, and none are
  governed by Indian law.
- Contract type: skews toward complex SEC-filed M&A/corporate agreements
  (Strategic Alliance, Sponsorship, Distributor, Joint Venture), not the
  simpler freelancer/SME service contracts this project targets.
- Category coverage: 6 of ClauseXplain's 18 target categories receive zero
  CUAD-derived training spans (see missing_six_categories.txt).
- No risk, legal-reference, or Indian-law information of any kind is present
  in CUAD. It provides contract text and a legally-informed category scheme,
  nothing more.

4. CLAUSEXPLAIN 18-CATEGORY TAXONOMY
--------------------------------------
Termination, Payment, Liability, Indemnification, Intellectual Property,
Confidentiality, Privacy / Data Protection, Non-Compete, Non-Solicitation,
Penalties, Dispute Resolution, Governing Law / Jurisdiction, Assignment,
Modification, Warranty, Force Majeure, Other, Commercial Terms.

"Commercial Terms" is the 18th category, split off from "Payment" to keep
payment-timing/invoicing provisions (relevant to MSMED Act, 2006 Sections
15-16) legally and semantically separate from deal-structure/economic
provisions (revenue sharing, minimum commitments, pricing power).

5. 41 -> 18 MAPPING METHODOLOGY
----------------------------------
Every one of the 41 CUAD categories was individually reviewed and assigned:
  - a target clausexplain_category (or left blank if metadata/excluded)
  - a category_status: METADATA (structural, not risk-bearing), EXCLUDED
    (out of scope for v1), or MAPPED (assigned to one of the 18)
  - a mapping_reason explaining the decision
  - a legal_basis_or_note, using cautious language ("may raise issues
    under Section X, depending on scope and context") rather than
    definitive legal claims
  - a needs_review flag for mappings that are judgment calls rather than
    direct/obvious matches (22.5% of all spans are flagged this way)

Full detail and reasoning for every one of the 41 categories: category_mapping.csv

6. THE "COMMERCIAL TERMS" DECISION
-------------------------------------
Five CUAD categories (Minimum Commitment, Revenue/Profit Sharing, Volume
Restriction, Most Favored Nation, Price Restrictions -- 1,078 spans total)
were routed to a new 18th category rather than force-fit into "Payment."
This was a deliberate decision: these clauses concern deal-structure/
commercial economics, not payment timing or invoicing, and forcing them
into Payment would have (a) diluted Payment's label with semantically
unrelated content and (b) obscured that Payment strictly-defined has ZERO
CUAD-derived training signal (see item 7).

7. SIX MISSING CATEGORIES -- IMPORTANT CORRECTION
----------------------------------------------------
The project's original framing assumed the sixth zero-coverage category was
"Dispute Resolution." Verified directly against the actual mapping, this is
incorrect: Dispute Resolution has narrow but non-zero coverage (173 spans,
via "Covenant Not To Sue"). The verified sixth zero-coverage category is
PAYMENT -- a direct consequence of the Commercial Terms split in item 6.
The corrected list of six zero-coverage categories:
  Indemnification, Confidentiality, Privacy / Data Protection, Modification,
  Force Majeure, Payment.
Full detail for each: missing_six_categories.txt

8. PROVENANCE
--------------
Every record in clauses_mapped.jsonl carries: source="CUAD_v1", the original
cuad_category (never overwritten), and span_type="CUAD-highlighted span"
(never called a "clause" outright, to avoid implying more structural
completeness than CUAD's spans actually have).

9. WHAT HAS AND HAS NOT BEEN ANNOTATED
-----------------------------------------
HAS been done: category mapping only (41 CUAD categories -> 18 ClauseXplain
categories), with documented reasoning and cautious legal language.

HAS NOT been done, and must not be assumed present: risk levels, risk
scores, unfairness/ambiguity/enforceability scores, Indian legal-provision
citations tied to specific clauses, explanations, recommendations,
contradiction labels. All of these require a separate future annotation
phase (see annotations.jsonl, currently empty, reserved for that purpose).

10. REPRODUCTION INSTRUCTIONS
--------------------------------
From a clean checkout with only CUAD_v1.json in data/raw/:
  1. python inspect_and_convert_cuad.py    -> contracts.csv, clauses.jsonl
  2. python apply_category_mapping.py      -> clauses_mapped.jsonl
     (reads category_mapping.csv; fails loudly if any category is unmapped)
  3. python generate_statistics.py         -> dataset_statistics.txt
  4. python generate_review_sample.py      -> mapping_review_sample.csv
  5. python validate.py                    -> run all validation checks

All random sampling uses a fixed seed (42). Re-running the full pipeline on
an unmodified CUAD_v1.json will reproduce identical output files byte-for-
byte (aside from any changes made to category_mapping.csv, which is a
human-reviewed artifact and the intended point of iteration).

11. KNOWN LIMITATIONS
------------------------
- 22.5% of mapped spans are flagged needs_review=True; these should not be
  used for risk-model training until a human resolves the flag.
- The "Payment"/"Commercial Terms" split, the Non-Compete/Exclusivity
  grouping, and several individual mappings (see needs_review=True rows)
  are design judgments, not settled legal fact -- state them as such in
  the paper.
- "Competitive Restriction Exception" is grouped with Non-Compete for
  mapping purposes but is a risk-REDUCING carve-out; any future risk-
  scoring logic must not treat it as risk-increasing merely because it
  shares a category.
- Contract-type and jurisdiction mismatch (CUAD is US/SEC-filed corporate
  agreements; target use case is Indian freelancer/SME contracts) applies
  to the entire dataset, not just the six zero-coverage categories.
- A bug in the original slugify() function (120-character truncation
  causing two distinct contract titles to collide) was found during this
  regeneration's validation step and fixed by appending a content hash to
  the contract_id. This is disclosed here for reproducibility transparency
  -- anyone re-running an EARLIER version of inspect_and_convert_cuad.py
  would reproduce the bug, not this corrected dataset.

12. RECOMMENDED NEXT STEP
----------------------------
Manually review mapping_review_sample.csv (100 rows, 60 needs_review=True +
40 confident, fixed seed). This is Level 1/2 validation (schema and
annotation-consistency review) before any risk annotation begins. Only
after this review should annotations.jsonl and a risk-labeling protocol be
designed -- that is explicitly future work, not part of this regeneration.
