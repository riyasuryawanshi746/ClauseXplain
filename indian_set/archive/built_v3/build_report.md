# Indian test set build report

334 clauses from 13 documents.

| Category | clauses | model has training data? |
|---|---|---|
| Assignment | 6 | yes |
| Commercial Terms | 22 | yes |
| Dispute Resolution | 18 | yes |
| Governing Law / Jurisdiction | 16 | yes |
| Intellectual Property | 9 | yes |
| Liability | 10 | yes |
| Non-Solicitation | 1 | yes |
| Other | 47 | yes |
| Penalties | 20 | yes |
| Termination | 26 | yes |
| Warranty | 17 | yes |
| Confidentiality | 47 | no (never seen in training) |
| Force Majeure | 19 | no (never seen in training) |
| Indemnification | 19 | no (never seen in training) |
| Modification | 6 | no (never seen in training) |
| Payment | 44 | no (never seen in training) |
| Privacy / Data Protection | 7 | no (never seen in training) |

Categories with fewer than 3 clauses (too thin for per-category scores): Non-Solicitation

CSV encoding read as: utf-8

## Clauses per document

- sbm_waste_to_compost_mca.txt: 67  <-- more than 6; one document may dominate
- mea_pmc_services_agreement.txt: 58  <-- more than 6; one document may dominate
- stqc_non_disclosure_agreement.txt: 41  <-- more than 6; one document may dominate
- cpwd_gcc_2023.txt: 39  <-- more than 6; one document may dominate
- noida_airport_concession_agreement.txt: 23  <-- more than 6; one document may dominate
- dot_unified_licence_agreement.txt: 21  <-- more than 6; one document may dominate
- meity_msa_cloud_services_template.txt: 19  <-- more than 6; one document may dominate
- mea_architect_consultancy_agreement.txt: 15  <-- more than 6; one document may dominate
- gem_general_terms_and_conditions.txt: 14  <-- more than 6; one document may dominate
- himachal_hemodialysis_ppp_concession_agreement.txt: 13  <-- more than 6; one document may dominate
- mha_rfp_project_management_consultants.txt: 12  <-- more than 6; one document may dominate
- aera_rfp_consultant_2025.txt: 10  <-- more than 6; one document may dominate
- tender_consultancy_scc_1349.txt: 2

Clauses in never-seen categories: 142 of 334
NOTE: Dispute Resolution is only 'seen' via CUAD's Covenant Not To Sue (no arbitration/forum clauses), so score it separately from the other seen categories.

## Warnings (rows kept, but check them)
- line 3: text ends in dot leaders or a page number -- probably a table-of-contents line
- line 246: text ends in dot leaders or a page number -- probably a table-of-contents line
- line 287: text ends in dot leaders or a page number -- probably a table-of-contents line

## Problems (these rows were skipped)
- line 8: DUPLICATE/overlapping row for the same category as line 3; skipped so it is not double counted
- line 25: DUPLICATE/overlapping row for the same category as line 2; skipped so it is not double counted
- line 295: DUPLICATE/overlapping row for the same category as line 292; skipped so it is not double counted

## Ambiguous matches
- line 146: text occurs 3 times in dot_unified_licence_agreement.txt; first occurrence used
- line 189: text occurs 2 times in himachal_hemodialysis_ppp_concession_agreement.txt; first occurrence used
- line 280: text occurs 2 times in sbm_waste_to_compost_mca.txt; first occurrence used
- line 292: text occurs 3 times in sbm_waste_to_compost_mca.txt; first occurrence used
- line 295: text occurs 3 times in sbm_waste_to_compost_mca.txt; first occurrence used
- line 310: text occurs 2 times in sbm_waste_to_compost_mca.txt; first occurrence used