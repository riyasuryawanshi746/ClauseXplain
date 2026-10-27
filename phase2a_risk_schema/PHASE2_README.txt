PHASE 2 README -- ClauseXplain: Risk Annotation & Indian-Law Alignment
=========================================================================

1. OBJECTIVE
------------
Design (not yet populate) a rigorous risk + legal annotation schema that
will eventually turn Phase 1's mapped CUAD spans into a dataset suitable
for explainable Indian contract-risk assessment. Phase 2 explicitly does
NOT train any model, does NOT compute real risk scores, and does NOT
claim any legal validation.

2. INPUT FROM PHASE 1
----------------------
clauses_mapped.jsonl (13,823 rows, 15 fields) -- verified unchanged before
Phase 2 work began: 510 contracts, 41 CUAD categories, 18 ClauseXplain
categories, zero row loss, zero duplicate clause_ids (after the slugify
bugfix from Phase 1's own validation pass).

3. NEW ANNOTATION SCHEMA
--------------------------
17 new fields added (all null/default until real annotation happens),
formally defined in phase2_annotation_schema.json. Grouped as:
  - Risk level & score: risk_level, risk_score
  - Five risk dimensions: unfairness_score, ambiguity_score,
    non_compliance_score, precedent_match_score, enforceability_score
  - Legal basis: legal_risk_type, indian_law_relevant, legal_provision,
    legal_basis, legal_confidence, requires_legal_review
  - Provenance: annotation_source, annotation_confidence, annotator_id,
    annotation_timestamp

4. RISK DIMENSIONS
--------------------
Defined and exemplified (with synthetic examples clearly labeled as such)
in PHASE2_ANNOTATION_GUIDELINES.txt. Five dimensions, each 0.0-1.0:
unfairness, ambiguity, non-compliance (must trace to a specific rule in
indian_legal_rules.csv), precedent-match (currently un-scoreable -- no
precedent corpus exists), enforceability.

5. RISK FORMULA
------------------
Risk = w1*U + w2*A + w3*NC + w4*P + w5*E, thresholds Low [0,0.33] /
Medium (0.33,0.66] / High (0.66,1.0]. Full specification, including the
explicit statement that weights are UNSET and thresholds are proposed
operational values (not derived from Indian law), in
risk_scoring_specification.txt.

6. INDIAN-LAW ALIGNMENT
--------------------------
indian_legal_rules.csv: 15 candidate rules across 11 of the 18 target
categories, each with a confidence rating and a requires_review flag.
Provisions covered: Indian Contract Act 1872 (Sections 23, 27, 28, 39, 56,
73, 74, 75, 124-125), MSMED Act 2006 (Sections 15-16), Digital Personal
Data Protection Act 2023 (Sections 6, 8). Every rule is written
conservatively -- general-applicability provisions (S.10, S.23, S.37,
S.39, S.55, S.73, S.75) are marked as context/low-confidence rather than
forced into a precise trigger, since the statutory text itself does not
support that level of precision. Only S.27, S.28, S.74, and MSMED S.15-16
are marked Medium/High confidence, reflecting a clear, checkable trigger
condition in the statute's actual text.

7. SIX ZERO-COVERAGE CATEGORIES -- STRATEGY PER CATEGORY
-------------------------------------------------------------
| Category | Strategy | Rule(s) created |
|---|---|---|
| Payment | Symbolic rule (MSMED Act) | MSMED_S15_01, MSMED_S16_01 |
| Privacy / Data Protection | Symbolic rule (DPDPA) | DPDPA_S06_01, DPDPA_S08_01, DPDPA_S08_02 |
| Indemnification | Symbolic rule (ICA, low confidence) | ICA_S124_01 |
| Force Majeure | Symbolic rule (ICA, low confidence, contract-level absence flag) | ICA_S56_01 |
| Confidentiality | No rule created -- no specific, checkable statutory trigger exists without overreaching. Strategy: supplementary annotation only. | none |
| Modification | No rule created -- same reasoning as Confidentiality. Strategy: supplementary annotation only. | none |

Note: this deliberately does not force all six into having a rule.
Confidentiality and Modification are genuinely not well-served by a
specific Indian statute the way Payment or Privacy/Data Protection are --
manufacturing a rule for them would misrepresent what the law actually
says.

8. ANNOTATION METHODOLOGY
----------------------------
Researcher-annotation (not lawyer-validated), per PHASE2_ANNOTATION_GUIDELINES.txt.
Every record defaults to annotation_source="not_yet_annotated" and
requires_legal_review=True. No record in this phase's output has been
actually annotated by a human or a validated rule engine.

9. HUMAN VALIDATION PLAN
----------------------------
Inter-annotator agreement design specified in PHASE2_RESEARCH_QUESTIONS.txt
(minimum 2 independent annotators, Cohen's kappa for categorical fields,
ICC for continuous dimension scores, third-annotator adjudication for
disagreements). NOT YET EXECUTED -- this is a design for future work, not
a completed process. No kappa/ICC value exists anywhere in this project.

10. FILES GENERATED IN PHASE 2
-----------------------------------
  - phase2_annotation_schema.json    (machine-readable field definitions)
  - clauses_risk_annotation.jsonl    (13,823 rows, 32 fields, all new
                                       fields null/default)
  - risk_annotation_sample.csv       (200-row stratified sample for
                                       manual annotation, seed=42)
  - indian_legal_rules.csv           (15 candidate legal rules)
  - risk_scoring_specification.txt
  - PHASE2_ANNOTATION_GUIDELINES.txt
  - PHASE2_RESEARCH_QUESTIONS.txt
  - PHASE2_README.txt                (this file)
  - generate_phase2_dataset.py, generate_risk_annotation_sample.py,
    generate_legal_rules.py          (reproducible generation scripts)

11. WHAT IS COMPLETED
--------------------------
- A formally specified, machine-readable annotation schema.
- A risk-scoring formula and threshold specification, explicitly labeled
  as experimental/unvalidated.
- A conservative Indian legal-rule table covering 11 of 18 categories,
  with explicit confidence ratings and required-review flags.
- A documented, category-by-category strategy for all six zero-coverage
  categories (not a one-size-fits-all answer).
- A 200-clause stratified sample ready for manual annotation.
- An inter-annotator agreement protocol design.
- clauses_risk_annotation.jsonl with the full schema applied and verified
  to contain ZERO fabricated values (checked programmatically: 0 non-null
  risk scores, 0 non-null risk levels, 0 records with any annotation_source
  other than "not_yet_annotated").

12. WHAT REMAINS FOR PHASE 3
---------------------------------
- ACTUALLY annotating the 200-clause sample (or a subset) with at least
  one human researcher-annotator, ideally two for agreement measurement.
- Running the inter-annotator agreement analysis and reporting real
  kappa/ICC values.
- Deciding, based on annotation results, whether rule-based preannotation
  (explicitly labeled annotation_source="rule_based_preannotation") is
  worth building from indian_legal_rules.csv before or alongside human
  annotation, and validating any such preannotation against the human
  labels rather than trusting it blindly.
- Fixing actual numeric weights (w1..w5) for the risk formula, only after
  enough annotated data exists to run a sensitivity/ablation analysis.
- Building or sourcing supplementary Indian-contract examples for the two
  categories with no rule at all (Confidentiality, Modification) and the
  four with only low/no-confidence rules.
- Only after the above: Phase 3's actual model-development work
  (Longformer or similar long-context transformer, neuro-symbolic fusion
  design, contradiction detection, generative advisory) -- explicitly out
  of scope for Phase 2, per the Phase 2 brief's own instructions.
