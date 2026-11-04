# Indian test set build report

886 clauses from 34 documents.

| Category | clauses | model has training data? |
|---|---|---|
| Assignment | 45 | yes |
| Commercial Terms | 89 | yes |
| Dispute Resolution | 38 | yes |
| Governing Law / Jurisdiction | 23 | yes |
| Intellectual Property | 19 | yes |
| Liability | 20 | yes |
| Non-Compete | 12 | yes |
| Non-Solicitation | 11 | yes |
| Other | 183 | yes |
| Penalties | 24 | yes |
| Termination | 46 | yes |
| Warranty | 53 | yes |
| Confidentiality | 76 | no (never seen in training) |
| Force Majeure | 27 | no (never seen in training) |
| Indemnification | 47 | no (never seen in training) |
| Modification | 42 | no (never seen in training) |
| Payment | 91 | no (never seen in training) |
| Privacy / Data Protection | 40 | no (never seen in training) |

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
- indusface_data_processing_agreement.txt: 20  <-- more than 6; one document may dominate
- meity_msa_cloud_services_template.txt: 19  <-- more than 6; one document may dominate
- mrn_canepower_master_services.txt: 19  <-- more than 6; one document may dominate
- hero_yamaha_jv_agreement.txt: 17  <-- more than 6; one document may dominate
- ssa_spa_jan2025_part1.txt: 17  <-- more than 6; one document may dominate
- selling_distribution_gujarat_2025.txt: 17  <-- more than 6; one document may dominate
- dhoot_business_transfer_2020.txt: 16  <-- more than 6; one document may dominate
- azure_employment_2022.txt: 15  <-- more than 6; one document may dominate
- gem_general_terms_and_conditions.txt: 14  <-- more than 6; one document may dominate
- mea_architect_consultancy_agreement.txt: 14  <-- more than 6; one document may dominate
- azure_executive_employment_2022.txt: 14  <-- more than 6; one document may dominate
- himachal_hemodialysis_ppp_concession_agreement.txt: 13  <-- more than 6; one document may dominate
- mha_rfp_project_management_consultants.txt: 12  <-- more than 6; one document may dominate
- aera_rfp_consultant_2025.txt: 10  <-- more than 6; one document may dominate
- concord_sha_amendment_2021.txt: 8  <-- more than 6; one document may dominate
- ssa_spa_jan2025_part2.txt: 8  <-- more than 6; one document may dominate
- dhoot_employment_2025.txt: 3
- tender_consultancy_scc_1349.txt: 2

Clauses in never-seen categories: 323 of 886
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
- line 852: text occurs 2 times in hero_yamaha_jv_agreement.txt; first occurrence used
- line 865: text occurs 2 times in hero_yamaha_jv_agreement.txt; first occurrence used
- line 869: text occurs 2 times in hero_yamaha_jv_agreement.txt; first occurrence used
- line 888: text occurs 2 times in hero_yamaha_jv_agreement.txt; first occurrence used