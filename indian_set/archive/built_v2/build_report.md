# Indian test set build report

228 clauses from 11 documents.

| Category | clauses | model has training data? |
|---|---|---|
| Assignment | 5 | yes |
| Commercial Terms | 17 | yes |
| Dispute Resolution | 10 | yes |
| Governing Law / Jurisdiction | 7 | yes |
| Intellectual Property | 9 | yes |
| Liability | 5 | yes |
| Non-Solicitation | 1 | yes |
| Other | 32 | yes |
| Penalties | 15 | yes |
| Termination | 12 | yes |
| Warranty | 16 | yes |
| Confidentiality | 39 | no (never seen in training) |
| Force Majeure | 13 | no (never seen in training) |
| Indemnification | 14 | no (never seen in training) |
| Modification | 5 | no (never seen in training) |
| Payment | 22 | no (never seen in training) |
| Privacy / Data Protection | 6 | no (never seen in training) |

Categories with fewer than 3 clauses (too thin for per-category scores): Non-Solicitation

CSV encoding read as: utf-8

## Clauses per document

- mea_pmc_services_agreement.txt: 58  <-- more than 6; one document may dominate
- stqc_non_disclosure_agreement.txt: 41  <-- more than 6; one document may dominate
- noida_airport_concession_agreement.txt: 23  <-- more than 6; one document may dominate
- dot_unified_licence_agreement.txt: 21  <-- more than 6; one document may dominate
- meity_msa_cloud_services_template.txt: 19  <-- more than 6; one document may dominate
- mea_architect_consultancy_agreement.txt: 15  <-- more than 6; one document may dominate
- gem_general_terms_and_conditions.txt: 14  <-- more than 6; one document may dominate
- himachal_hemodialysis_ppp_concession_agreement.txt: 13  <-- more than 6; one document may dominate
- mha_rfp_project_management_consultants.txt: 12  <-- more than 6; one document may dominate
- aera_rfp_consultant_2025.txt: 10  <-- more than 6; one document may dominate
- tender_consultancy_scc_1349.txt: 2

Clauses in never-seen categories: 99 of 228
NOTE: Dispute Resolution is only 'seen' via CUAD's Covenant Not To Sue (no arbitration/forum clauses), so score it separately from the other seen categories.

## Warnings (rows kept, but check them)
- line 3: text ends in dot leaders or a page number -- probably a table-of-contents line

## Problems (these rows were skipped)
- line 8: DUPLICATE/overlapping row for the same category as line 3; skipped so it is not double counted
- line 25: DUPLICATE/overlapping row for the same category as line 2; skipped so it is not double counted

## Ambiguous matches
- line 146: text occurs 3 times in dot_unified_licence_agreement.txt; first occurrence used
- line 189: text occurs 2 times in himachal_hemodialysis_ppp_concession_agreement.txt; first occurrence used