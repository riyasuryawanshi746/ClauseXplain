# Indian test set build report

32 clauses from 9 documents.

| Category | clauses | model has training data? |
|---|---|---|
| Assignment | 1 | yes |
| Commercial Terms | 1 | yes |
| Non-Solicitation | 1 | yes |
| Penalties | 9 | yes |
| Confidentiality | 9 | no (never seen in training) |
| Force Majeure | 5 | no (never seen in training) |
| Indemnification | 6 | no (never seen in training) |

Categories with fewer than 3 clauses (too thin for per-category scores): Assignment, Commercial Terms, Non-Solicitation

CSV encoding read as: utf-8

## Clauses per document

- noida_airport_concession_agreement.txt: 7  <-- more than 6; one document may dominate
- dot_unified_licence_agreement.txt: 6
- aera_rfp_consultant_2025.txt: 4
- meity_msa_cloud_services_template.txt: 4
- stqc_non_disclosure_agreement.txt: 4
- mha_rfp_project_management_consultants.txt: 3
- gem_general_terms_and_conditions.txt: 2
- mea_pmc_services_agreement.txt: 1
- himachal_hemodialysis_ppp_concession_agreement.txt: 1

Clauses in never-seen categories: 20 of 32
NOTE: Dispute Resolution is only 'seen' via CUAD's Covenant Not To Sue (no arbitration/forum clauses), so score it separately from the other seen categories.

## Warnings (rows kept, but check them)
- line 3: text ends in dot leaders or a page number -- probably a table-of-contents line

## Problems (these rows were skipped)
- line 8: DUPLICATE/overlapping row for the same category as line 3; skipped so it is not double counted
- line 13: DUPLICATE/overlapping row for the same category as line 2; skipped so it is not double counted
- line 26: DUPLICATE/overlapping row for the same category as line 2; skipped so it is not double counted