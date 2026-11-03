# ClauseXplain: Where We Are and Where We're Going

*Status review + plan built around "Option 1" (one multi-task Longformer). Written 7 Oct 2026 after reading the three phase zips and `dataset.py`, `model.py`, `metrics.py`. Nothing here has been executed; the sandbox had no torch.*

---

## 1. What you have

### 1.1 Data (Phases 1 to 2B)

| Asset | What it is | Status |
|---|---|---|
| CUAD v1 spans | 510 US contracts, 13,823 expert-highlighted spans, 41 categories | Real, expert-labeled |
| `category_mapping.csv` | 41 CUAD categories mapped to your 18 Indian-context categories, each with reason and legal note | Solid, documented |
| `clauses_mapped.jsonl` | 9,786 spans mapped to categories (3,998 metadata, 39 excluded) | Real labels, derived from the mapping |
| Category coverage | 12 of 18 categories have data; **6 have zero** (Confidentiality, Force Majeure, Indemnification, Modification, Payment, Privacy). Dispute Resolution is thin (173 spans) | Known gap |
| `indian_legal_rules.csv` | 15 rules (ICA s.23/27/28/39/56/73/74/75/124-125, MSMED s.15-16, DPDPA s.6/8), typed as risk_trigger / contextual / gap_detection | Good start, never reviewed by a lawyer |
| Risk schema + guidelines | 4 scored dimensions (unfairness, ambiguity, non-compliance, enforceability) on a 5-point scale; precedent dimension has no data | Well documented |
| 200-clause sample | Stratified sample; 195 completed, 5 skipped | Sample is fine |
| 195 AI risk labels | Single AI annotator, no human review | **Silver at best. Not used (see section 2)** |

### 1.2 Code

| File | Does | Notes |
|---|---|---|
| `dataset.py` | Loads mapped spans, attaches labels, builds contract-grouped stratified splits, builds span examples, collator | Field names verified against the real data |
| `model.py` | Longformer with 3 heads: 18-way category, risk (ordinal 4x5), span start/end. Masked losses, fixed or uncertainty weighting | Never run |
| `metrics.py` | Classification, risk (with mean/median baselines), span (token-F1, EM, IoU), markdown tables | Good honesty conventions (n/a for unsupported classes) |

### 1.3 Things that are genuinely good

- Provenance discipline: nothing fabricated, nulls until populated, AI vs human labels tracked.
- Contract-grouped splits (no leakage across clauses from the same contract).
- Baselines built into the metrics.

---

## 2. What needs to change

### 2.1 Problems found

| # | Problem | Why it matters | Fix |
|---|---|---|---|
| 1 | **No contract text in the zips.** `char_start`/`char_end` point into full CUAD contracts that aren't included | Real span training (`cuad_context`) can't run; the code silently falls back to `synthetic`, which is only a pipeline proxy | Re-attach contract text from the original CUAD JSON; assert `text[start:end] == clause_text` for every row |
| 2 | **`train_phase3.py` and `smoke_test.py` are referenced but not in your upload** | Can't train or verify anything | Upload them, or they get rewritten |
| 3 | **Risk labels are AI-generated, single annotator, unreviewed** | Reviewers will not accept them as ground truth; they also skew heavily (non_compliance 0 for 176 of 195; no High-risk clauses at all) | Don't train on them. Replace with the approach in section 4 |
| 4 | **Classification is near-circular** | The 18 labels come from CUAD's category via your mapping, so the model mostly relearns the mapping | Treat classification as a secondary result; make span extraction the headline |
| 5 | **CUAD spans locate clauses; they don't explain risk** | `dataset.py` already warns about this. Calling span output "explanations" invites rejection | Call it "clause localization / evidence span"; add a faithfulness test (section 3, step 8) |
| 6 | **US contracts, Indian-law claim** | Only about 1% of CUAD contracts even mention India | Add a small Indian test set (30 to 50 clauses) |
| 7 | **Mapping noise** | Annotation batches found arbitration filed under Governing Law, pricing under Minimum Commitment, etc. | Document as a known limitation; don't claim the mapping is gold |
| 8 | **Annotation app has no highlight fields** | Option 1's note assumed them | Redesign the app (section 4) |

### 2.2 Decisions made

- **Go with Option 1**: one multi-task Longformer, not RAG-lite.
- **Drop AI-generated risk labels** from training. Keep the file on disk, unused.
- **Span extraction is the headline task**, category classification is secondary, risk is handled separately (section 4).

---

## 3. Step-by-step roadmap (end goal: a defensible paper)

### The end goal

> **ClauseXplain: a multi-task, rule-grounded model for clause understanding in Indian-context contracts.**

**Four contributions you can honestly claim:**

1. **Taxonomy and coverage analysis.** A documented 41-to-18 mapping from CUAD to an Indian-law-oriented taxonomy, with a quantified account of which categories CUAD cannot support.
2. **Joint model.** One Longformer jointly doing category classification and clause localization (plus an optional risk-flag head), compared against separate single-task models: quality and speed.
3. **Neuro-symbolic risk flagging.** A statutory rule layer (Indian Contract Act, MSMED, DPDPA) applied on top of model outputs, evaluated against neural-only.
4. **Span head vs LIME.** Speed (ms per clause) and agreement, plus a faithfulness test.

**Honest limits you state upfront:** US-sourced training data, small Indian test set, rules not legally adjudicated.

### Phases

| Phase | Goal | Key tasks | Rough time |
|---|---|---|---|
| **A. Foundations** | Make the pipeline run | 1) Get CUAD_v1.json, rebuild contract text, verify offsets. 2) Set up GPU (Colab or Kaggle T4 is enough with gradient checkpointing and about 6,000-char windows). 3) Run a tiny random-init smoke test. | Week 1 |
| **B. Core results (no new annotation)** | Real numbers on real labels | 4) Category-only baseline (single-task). 5) Span-only baseline on CUAD context. 6) Joint cls+span model. 7) Grouped 5-fold CV, report mean ± std. | Weeks 2-3 |
| **C. Ablations and explainability** | The "make or break" experiment | 8) Single-task x2 vs joint (quality and latency). 9) Span head vs LIME: ms per clause, overlap, and a deletion/faithfulness check. 10) Fixed vs uncertainty loss weighting. | Weeks 3-4 |
| **D. Fill the coverage gaps** | Handle the 6 empty categories | 11) Pull labeled data from public sets (LEDGAR covers indemnification, confidentiality, payments, force majeure and amendments; verify its label list before committing). 12) Rule-based weak labels as a fallback. Report these separately from CUAD results. | Weeks 4-5 |
| **E. Risk and Indian test set** | Section 4 | Build the checklist annotations, rule-derived flags, and Indian test set | Weeks 5-7 |
| **F. Writing** | | Tables from `metrics.py`, limitations section, release the taxonomy and checklist | Weeks 7-9 |

If time is tight, **A to C alone is a coherent, honest paper** (taxonomy + joint model + LIME comparison). D and E make it stronger and more "Indian".

---

## 4. Risk without being a legal expert

You don't need to judge legal risk to build a defensible risk component. The trick is to **separate what is textual from what is legal**:

### 4.1 Split the work

| Question | Needs a lawyer? | Who does it |
|---|---|---|
| "Does this clause state a duration? A geographic limit? A named sum? Is it mutual? Is there a cap?" | **No.** It's a reading task: the answer is in the text | You and a classmate, using a checklist |
| "Given those features, does Indian law flag it?" (e.g. unlimited-duration restraint, so ICA s.27 concern) | Yes, but it's **encoded once** in the rule table | The 15 rules, reviewed once (see 4.3) |

So risk becomes **rules applied to annotated facts**, not a score someone has to guess.

### 4.2 Concrete design

1. **Redesign the annotation app as a checklist** (yes/no/unclear per feature), plus the highlight start/end that Option 1 needs. Per category, 4 to 6 features, for example:
   - Non-compete: duration stated? geography stated? scope limited to a named business? post-termination?
   - Penalties: sum named? fixed amount or formula? one-sided?
   - Liability: cap present? carve-outs? mutual?
2. **Derive risk flags with the rule table.** Feature pattern leads to rule trigger, which yields a flag such as "may raise issues under ICA s.27". The wording stays cautious ("may raise issues"), as your rule file already does.
3. **Train an optional risk-flag head** on those derived flags (multi-label, not a 0-to-1 score). This is programmatic weak supervision, labeled as such.
4. **Evaluate on a small hand-annotated set** (60 to 100 clauses): checklist answers from you, plus a second person on at least 40 of them, so you can report agreement (Cohen's kappa). Checklist questions are objective, so agreement should be high.

### 4.3 The one place a human with legal background helps (small, optional)

Ask a law student, a faculty contact, or any lawyer in your network to look over **the 15-rule table only**, roughly an hour of work. If you can't get this, say so in the limitations section and keep the "may raise issues" language. It strengthens the paper but is not a blocker.

### 4.4 What not to do

- Don't present a number like "risk score 0.5" as legal truth.
- Don't use AI labels for training or call them ground truth.
- Don't claim legal compliance outcomes. Claim "rule-grounded flags".

### 4.5 Fallback if even the checklist is too much

Drop the learned risk head entirely. The model becomes **two neural heads (category + span) plus the symbolic rule layer** for flags. Weaker story, but fully honest and finishable.

---

## 5. What to do next (this week)

| # | Action | Output |
|---|---|---|
| 1 | Download CUAD_v1.json and write the contract-text re-attachment step (reuse the paragraph-joining logic in `inspect_and_convert_cuad.py`) | `contract_text` per contract; offset assertion passes for all rows |
| 2 | Upload `train_phase3.py` and `smoke_test.py` (or say they're missing) | Training and smoke-test scripts in hand |
| 3 | Open a Colab/Kaggle GPU notebook; run the smoke test with a tiny random-init Longformer | Pipeline confirmed end to end |
| 4 | Run the category-only and span-only baselines on one grouped split | First real numbers |
| 5 | Decide the risk route: checklist (4.2) or drop head (4.5) | One-line decision recorded |

**Suggested order of my help:** (1) loader and offset check, (2) fix or rewrite the training script, (3) checklist redesign of the annotation app.

---

## 6. Risks to keep in mind

- **GPU limits:** Longformer at long windows is heavy; use gradient checkpointing, mixed precision, and 4,096-token or shorter windows.
- **Metric trap:** high classification F1 may just reflect the mapping. Lead with span results.
- **Reviewer question:** "Why not just fine-tune a standard CUAD QA model?" Your answer is the joint training, the Indian taxonomy and rules layer, and the speed/faithfulness comparison against LIME.
- **Scope creep:** finish A to C before touching D and E.
