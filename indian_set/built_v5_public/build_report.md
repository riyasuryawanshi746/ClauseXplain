# Indian test set build report

769 clauses from 26 documents.

| Category | clauses | model has training data? |
|---|---|---|
| Assignment | 25 | yes |
| Commercial Terms | 79 | yes |
| Dispute Resolution | 35 | yes |
| Governing Law / Jurisdiction | 20 | yes |
| Intellectual Property | 15 | yes |
| Liability | 18 | yes |
| Non-Compete | 12 | yes |
| Non-Solicitation | 11 | yes |
| Other | 172 | yes |
| Penalties | 21 | yes |
| Termination | 42 | yes |
| Warranty | 46 | yes |
| Confidentiality | 70 | no (never seen in training) |
| Force Majeure | 23 | no (never seen in training) |
| Indemnification | 43 | no (never seen in training) |
| Modification | 38 | no (never seen in training) |
| Payment | 79 | no (never seen in training) |
| Privacy / Data Protection | 20 | no (never seen in training) |

CSV encoding read as: utf-8

## Clauses per document

- sbm_waste_to_compost_mca.txt: 67  <-- more than 6; one document may dominate
- mea_pmc_services_agreement.txt: 59  <-- more than 6; one document may dominate
- share_subscription_purchase_executed.txt: 58  <-- more than 6; one document may dominate
- infosys_springboard_mou_2023.txt: 49  <-- more than 6; one document may dominate
- kaynes_share_purchase_2022.txt: 47  <-- more than 6; one document may dominate
- hinduja_business_transfer_2021.txt: 44  <-- more than 6; one document may dominate
- mmfsl_trust_deed_amendment_2023.txt: 44  <-- more than 6; one document may dominate
- cyient_master_services_2022.txt: 44  <-- more than 6; one document may dominate
- stqc_non_disclosure_agreement.txt: 41  <-- more than 6; one document may dominate
- majesco_master_services_2016.txt: 40  <-- more than 6; one document may dominate
- cpwd_gcc_2023.txt: 39  <-- more than 6; one document may dominate
- dec2022_share_subscription_purchase.txt: 27  <-- more than 6; one document may dominate
- azure_employment_2023.txt: 25  <-- more than 6; one document may dominate
- noida_airport_concession_agreement.txt: 23  <-- more than 6; one document may dominate
- dot_unified_licence_agreement.txt: 21  <-- more than 6; one document may dominate
- dec2022_shareholders_agreement.txt: 20  <-- more than 6; one document may dominate
- meity_msa_cloud_services_template.txt: 19  <-- more than 6; one document may dominate
- azure_employment_2022.txt: 15  <-- more than 6; one document may dominate
- gem_general_terms_and_conditions.txt: 14  <-- more than 6; one document may dominate
- mea_architect_consultancy_agreement.txt: 14  <-- more than 6; one document may dominate
- azure_executive_employment_2022.txt: 14  <-- more than 6; one document may dominate
- himachal_hemodialysis_ppp_concession_agreement.txt: 13  <-- more than 6; one document may dominate
- mha_rfp_project_management_consultants.txt: 12  <-- more than 6; one document may dominate
- aera_rfp_consultant_2025.txt: 10  <-- more than 6; one document may dominate
- concord_sha_amendment_2021.txt: 8  <-- more than 6; one document may dominate
- tender_consultancy_scc_1349.txt: 2

Clauses in never-seen categories: 273 of 769
NOTE: Dispute Resolution is only 'seen' via CUAD's Covenant Not To Sue (no arbitration/forum clauses), so score it separately from the other seen categories.

## Warnings (rows kept, but check them)
- line 3: text ends in dot leaders or a page number -- probably a table-of-contents line
- line 246: text ends in dot leaders or a page number -- probably a table-of-contents line
- line 287: text ends in dot leaders or a page number -- probably a table-of-contents line
- line 488: text ends in dot leaders or a page number -- probably a table-of-contents line
- line 507: only 5 words -- looks like a heading, not a clause (check it is not a table-of-contents line)
- line 531: only 1 words -- looks like a heading, not a clause (check it is not a table-of-contents line)
- line 553: only 10 words -- looks like a heading, not a clause (check it is not a table-of-contents line)
- line 623: only 1 words -- looks like a heading, not a clause (check it is not a table-of-contents line)
- line 718: only 7 words -- looks like a heading, not a clause (check it is not a table-of-contents line)
- line 763: only 3 words -- looks like a heading, not a clause (check it is not a table-of-contents line)

## Problems (these rows were skipped)
- line 8: DUPLICATE/overlapping row for the same category as line 3; skipped so it is not double counted
- line 25: DUPLICATE/overlapping row for the same category as line 2; skipped so it is not double counted
- line 295: DUPLICATE/overlapping row for the same category as line 292; skipped so it is not double counted

## Ambiguous matches
- line 144: text occurs 3 times in dot_unified_licence_agreement.txt; first occurrence used
- line 187: text occurs 2 times in himachal_hemodialysis_ppp_concession_agreement.txt; first occurrence used
- line 280: text occurs 2 times in sbm_waste_to_compost_mca.txt; first occurrence used
- line 292: text occurs 3 times in sbm_waste_to_compost_mca.txt; first occurrence used
- line 295: text occurs 3 times in sbm_waste_to_compost_mca.txt; first occurrence used
- line 310: text occurs 2 times in sbm_waste_to_compost_mca.txt; first occurrence used
- line 506: text occurs 2 times in hinduja_business_transfer_2021.txt; first occurrence used
- line 516: text occurs 2 times in azure_employment_2022.txt; first occurrence used