"""
generate_ai_pilot_annotations.py
ClauseXplain Phase 2B -- AI-assisted preannotation of the 10-clause pilot.

PROVENANCE: every record written here is annotation_source=
"ai_assisted_preannotation", reviewed_by_human=False, annotator_id=
"claude_opus_4_5__pilot_2026_09". These are UNREVIEWED MACHINE DRAFTS.
They are not research data and must not be used for agreement statistics.

Writes to risk_annotations_ai_pilot.jsonl ONLY.
risk_annotations_human.jsonl is never opened by this script.
"""
import json, csv
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
OUT = BASE / "risk_annotations_ai_pilot.jsonl"
HUMAN_FILE = BASE / "risk_annotations_human.jsonl"   # referenced only to assert we never write it

ANNOTATOR = "claude_opus_4_5__pilot_2026_09"
SOURCE = "ai_assisted_preannotation"

pilot_ids = json.load(open(BASE / "pilot_clause_ids.json", encoding="utf-8"))["pilot_clause_ids"]
sample = {r["clause_id"]: r for r in
          csv.DictReader(open(BASE / "risk_annotation_sample.csv", newline="", encoding="utf-8"))
          if r.get("clause_id")}

def lvl(x):
    if x is None: return None
    if x <= 0.33: return "Low"
    if x <= 0.66: return "Medium"
    return "High"

def prov(u, a, nc, e):
    """provisional_risk_score_4dim per risk_scoring_specification.txt s.4b"""
    return round((u + a + nc + e) / 4, 4)

# ---------------------------------------------------------------------------
# Per-clause annotations. Index matches pilot_clause_ids.json order.
# ---------------------------------------------------------------------------
A = {}

# [1] Assignment <- Change Of Control : "Licensor may terminate this Agreement:"
A[0] = dict(
    status="skipped",
    skip_reason=(
        "SPAN IS AN UNUSABLE FRAGMENT. The entire CUAD-highlighted span is 38 "
        "characters and ends in a colon: 'Licensor may terminate this Agreement:'. "
        "The operative content -- the enumerated conditions that follow the colon -- "
        "is not in the span. No risk dimension can be assessed from this text: it is "
        "impossible to say whether termination is unilateral-and-unconditional "
        "(potentially high unfairness) or tightly conditioned (low). Scoring any "
        "dimension here, including scoring 0.00, would manufacture certainty from "
        "absent text. Per PHASE2_ANNOTATION_GUIDELINES.txt ('Do not annotate a "
        "clause you don't have enough context to assess'), this is skipped with a "
        "reason rather than guessed. NOTE ALSO: the span is assigned category "
        "'Assignment' (from CUAD 'Change Of Control') but its visible text is about "
        "termination, so the category assignment is itself unverifiable here."
    ),
)

# [2] Commercial Terms <- Revenue/Profit Sharing
A[1] = dict(
    status="completed", u=0.25, a=0.50, nc=0.00, e=0.25,
    indian_law_relevant=False,
    legal_provision=None, legal_basis=None, legal_confidence=None,
    legal_risk_type=["ambiguous"],
    confidence="Low", review=True,
    notes=(
        "UNFAIRNESS 0.25: the fee-adjustment mechanism runs in one direction only -- "
        "missing the Quarterly Minimum Details reduces the Applicable Percentage, with "
        "no reciprocal uplift for exceeding it in the visible text. That is a real but "
        "minor asymmetry, and performance-linked fee adjustment is normal in promotion "
        "agreements. Cannot go higher because the actual reduced percentage is redacted "
        "as '[***]' -- proportionality, which is the thing that would distinguish a mild "
        "adjustment from a punitive one, is not observable. "
        "AMBIGUITY 0.50: three defined terms ('Details', 'Quarterly Minimum Details', "
        "'Applicable Percentage'), a cross-reference to Section 6.1.1, and the operative "
        "number itself are all outside the span. A reader of this span alone cannot "
        "determine what happens. "
        "NON_COMPLIANCE 0.00: no rule in indian_legal_rules.csv has a trigger_condition "
        "matching a performance-linked fee adjustment. "
        "ENFORCEABILITY 0.25: ICA_S74_01 was considered and NOT applied -- its trigger "
        "requires a sum named as payable ON BREACH, whereas this is a reduction in fee "
        "earned for services not rendered, which is a different legal construct. Residual "
        "0.25 records that if the redacted reduction were severe enough to be "
        "characterised as a stipulated consequence of breach, the S.74 question could "
        "arise; that is unresolvable on the visible text."
    ),
    uncertainty=(
        "The redaction is decisive. Whether this clause is benign or aggressive depends "
        "entirely on the value hidden behind '[***]'. Confidence is Low for that reason "
        "alone."
    ),
    guideline_issue=None,
)

# [3] Dispute Resolution <- Covenant Not To Sue (substantively an INDEMNITY)
A[2] = dict(
    status="completed", u=0.50, a=0.25, nc=0.00, e=0.25,
    indian_law_relevant=True,
    legal_provision=None, legal_basis=(
        "ICA_S124_01 (Indian Contract Act ss.124-125, indemnity) was considered because "
        "the clause is substantively an indemnity, but its trigger_condition ('indemnity "
        "obligation is uncapped or unusually broad in scope') is NOT met -- see notes. No "
        "rule is asserted."
    ),
    legal_confidence="Low",
    legal_risk_type=["potentially unfair", "excessive liability concern"],
    confidence="Medium", review=True,
    notes=(
        "CATEGORY MISMATCH -- PRIMARY FINDING. This span is assigned 'Dispute Resolution' "
        "via CUAD 'Covenant Not To Sue', but its text is a release / defend / indemnify / "
        "hold-harmless clause. It contains no forum-selection, arbitration, or "
        "bar-on-proceedings language. The assigned category does not describe the clause. "
        "UNFAIRNESS 0.50: obligations run one way only (Customer indemnifies Transporter "
        "and its members, directors, officers, agents and employees; no reciprocal "
        "obligation appears). Held at Moderate rather than High because two genuine "
        "limits are present in the text: the indemnity is confined to Claims arising from "
        "Customer's OWN breach of its representations/warranties/covenants, and it is "
        "proportionately carved back for Transporter's gross negligence or wilful "
        "misconduct. Those are meaningful protections, not symbolic ones. "
        "AMBIGUITY 0.25: unusually precise for its type -- 'Claims' is expressly defined "
        "inline and the carve-out is explicit. Residual 0.25 for 'proportionate extent', "
        "which requires apportionment machinery not visible here. "
        "NON_COMPLIANCE 0.00: no applicable rule; ICA_S124_01 trigger not met. "
        "ENFORCEABILITY 0.25: low residual only. The gross-negligence/wilful-misconduct "
        "carve-out is precisely the feature whose ABSENCE would raise concern."
    ),
    uncertainty=(
        "Because the clause's true subject (indemnity) is not its assigned category "
        "(Dispute Resolution), the rule inventory was searched by substance rather than "
        "by category label. A human should confirm that this is the right way to resolve "
        "a category/substance conflict -- the guidelines do not currently say."
    ),
    guideline_issue=(
        "Guidelines do not state whether to annotate against the ASSIGNED category or the "
        "clause's ACTUAL subject matter when they conflict, nor whether an annotator may "
        "cite a rule scoped to a different category."
    ),
)

# [4] Governing Law / Jurisdiction <- Governing Law (Delaware)
A[3] = dict(
    status="completed", u=0.00, a=0.00, nc=0.00, e=0.00,
    indian_law_relevant=False,
    legal_provision=None, legal_basis=None, legal_confidence=None,
    legal_risk_type=["jurisdictional concern"],
    confidence="High", review=True,
    notes=(
        "A clean, standard Delaware choice-of-law clause with an express exclusion of "
        "renvoi/conflicts rules. "
        "UNFAIRNESS 0.00: applies identically to both parties; no asymmetry. "
        "AMBIGUITY 0.00: unambiguous, and the conflicts-of-laws carve-out removes the one "
        "interpretive question such clauses usually raise. "
        "NON_COMPLIANCE 0.00 and ENFORCEABILITY 0.00: NOT because foreign governing law is "
        "risk-free for an Indian party, but because indian_legal_rules.csv contains no "
        "rule covering choice of law or forum. Scoring above zero here would require "
        "importing legal knowledge from outside the sanctioned inventory, which the pilot "
        "brief forbids. The concern is recorded instead as a legal-rule gap and as a "
        "POTENTIAL FUTURE LEGAL-CONTEXT NOTE (see pilot report). "
        "legal_risk_type 'jurisdictional concern' is tagged because the tag vocabulary "
        "supports it even though no numeric dimension could be scored -- this is itself a "
        "schema observation."
    ),
    uncertainty=(
        "This clause demonstrates the CUAD jurisdiction mismatch concretely: a Delaware "
        "governing-law clause is exactly what a ClauseXplain user would most want flagged, "
        "and the current Indian rule inventory is silent on it."
    ),
    guideline_issue=(
        "A legal_risk_type tag can be justifiably applied when every numeric dimension is "
        "0.00. Guidelines do not address whether that combination is permitted."
    ),
)

# [5] Intellectual Property <- Non-Transferable License
A[4] = dict(
    status="completed", u=0.25, a=0.00, nc=0.00, e=0.00,
    indian_law_relevant=False,
    legal_provision=None, legal_basis=None, legal_confidence=None,
    legal_risk_type=[],
    confidence="High", review=True,
    notes=(
        "The clearest clause in the pilot and a useful low-risk anchor. "
        "UNFAIRNESS 0.25: all three restrictions (non-exclusive, non-assignable, "
        "non-transferable) bind the licensee only. Per the guidelines, asymmetry that is "
        "standard and expected for the clause type is not by itself unfair, so this sits "
        "at Low rather than Moderate -- but the guidelines also direct using 0.25 rather "
        "than 0.00 where SOME asymmetry exists, which it does. "
        "AMBIGUITY 0.00: three standard licensing terms of art, no undefined operative "
        "language, no cross-reference. "
        "NON_COMPLIANCE 0.00 / ENFORCEABILITY 0.00: no rule applies; nothing in the text "
        "restrains trade, assigns pre-existing IP, or creates perpetual obligations. "
        "NOTE: requires_legal_review was initially drafted as False here -- this is the "
        "one clause in the pilot where an annotator would plausibly say no legal review is "
        "needed. Validation REJECTED that: PHASE2_ANNOTATION_GUIDELINES.txt requires True "
        "on every record, and the validator enforces it. Corrected to True in deference to "
        "the authoritative guidelines. Recorded as PILOT ISSUE #4 -- a field that is always "
        "True carries no information."
    ),
    uncertainty="Minimal. The span is short but complete and self-contained.",
    guideline_issue=None,
)

# [6] Liability <- Insurance
A[5] = dict(
    status="completed", u=0.25, a=0.50, nc=0.00, e=0.00,
    indian_law_relevant=False,
    legal_provision=None, legal_basis=(
        "ICA_S73_01 is the only Liability-scoped rule in the inventory and is rule_type="
        "contextual (statutory default for compensation where the contract is silent). It "
        "does not address insurance or additional-insured status, so it is not cited."
    ),
    legal_confidence=None,
    legal_risk_type=["ambiguous"],
    confidence="Medium", review=True,
    notes=(
        "Insurance-allocation clause: Owner Indemnified Parties' additional-insured status "
        "is preserved alongside Operator's indemnity, and Operator's policies are primary. "
        "UNFAIRNESS 0.25: structurally favours the Owner (Operator's insurer pays first), "
        "but primary-coverage wording is conventional where one party performs operations "
        "on another's asset. "
        "AMBIGUITY 0.50: the double negative 'will not limit the application of insurance "
        "protection as required by this Agreement which arises out of the Operator's "
        "indemnity obligations' is genuinely hard to parse, and 'Owner Indemnified "
        "Parties' plus the underlying indemnity obligations are defined outside the span. "
        "NON_COMPLIANCE 0.00 / ENFORCEABILITY 0.00: no rule in the inventory reaches "
        "insurance terms."
    ),
    uncertainty=(
        "Whether this clause is favourable or unfavourable depends on the indemnity "
        "obligations it cross-references, which are not in the span."
    ),
    guideline_issue=(
        "The Liability category's only rule is contextual and concerns damages, not "
        "insurance. Category-to-rule coverage is thinner than the category list implies."
    ),
)

# [7] Non-Compete <- Competitive Restriction Exception
A[6] = dict(
    status="completed", u=0.25, a=0.50, nc=0.00, e=0.25,
    indian_law_relevant=True,
    legal_provision=None, legal_basis=(
        "ICA_S27_01 (restraint of trade) was considered and NOT applied. Its "
        "trigger_condition requires that the restriction is 'not limited to a narrow, "
        "reasonable scope'. Here the restriction IS bounded on three axes: a numeric cap "
        "(no more than two other agreements), a temporal limit ('During the Term'), and "
        "express carve-outs. Trigger not met; no rule asserted."
    ),
    legal_confidence="Low",
    legal_risk_type=["restraint-of-trade concern", "ambiguous"],
    confidence="Low", review=True,
    notes=(
        "POLARITY CONFIRMED EMPIRICALLY. Phase 1 flagged CUAD 'Competitive Restriction "
        "Exception' as a risk-REDUCING carve-out wrongly grouped with risk-increasing "
        "Non-Compete. This clause shows the flag was right but incomplete: the span is "
        "BOTH at once. Its first half imposes a restriction (CNET may not enter more than "
        "two competing-retailer promotion agreements); its second half carves out standard "
        "advertisements, COMPUTERS.COM and Snap!. A single polarity label could not "
        "describe this span. "
        "UNFAIRNESS 0.25: binds CNET only, so it is one-directional -- but it is a "
        "negotiated in-term commercial cap with substantial express exclusions, not a "
        "blanket prohibition. "
        "AMBIGUITY 0.50: 'standard advertisements', 'fixed promotions' and 'permanent "
        "links' are undefined and carry real weight, since they mark the boundary between "
        "prohibited and permitted conduct. "
        "NON_COMPLIANCE 0.00: ICA_S27_01 trigger not met (see legal_basis). "
        "ENFORCEABILITY 0.25: residual only. Note this is an IN-TERM restriction on "
        "contracting with third parties, not a post-termination restraint on CNET's own "
        "trade; the rule's own exception_condition treats that distinction as "
        "context-dependent, and it is not resolved here."
    ),
    uncertainty=(
        "Whether an in-term exclusivity-style cap engages s.27 at all is context-dependent "
        "and the inventory does not settle it. Confidence Low."
    ),
    guideline_issue=(
        "A span can simultaneously restrict AND carve out. Neither the guidelines nor the "
        "schema provide a way to record mixed polarity within one record."
    ),
)

# [8] Non-Solicitation <- No-Solicit Of Employees
A[7] = dict(
    status="completed", u=0.25, a=0.50, nc=0.25, e=0.50,
    indian_law_relevant=True,
    legal_provision="Indian Contract Act, 1872, Section 27",
    legal_basis=(
        "ICA_S27_02 (rule_type=risk_trigger) applied. Its trigger_condition is that the "
        "restriction is 'broad/indefinite in scope or duration'. Duration here is definite "
        "(two years) and does NOT meet the trigger, but scope does: limb (ii) prohibits "
        "interference 'in any manner' with any employment relationship, which is open-ended "
        "in a way limb (i) is not. Applied on the scope limb only. Consistent with the "
        "rule's exception_condition, this is flagged as a matter for review rather than "
        "treated as a conclusion -- Indian courts assess reasonableness on the facts, and "
        "post-termination employee non-solicitation is treated differently from a restraint "
        "on a party's own trade."
    ),
    legal_confidence="Medium",
    legal_risk_type=["restraint-of-trade concern", "potentially unenforceable", "ambiguous"],
    confidence="Medium", review=True,
    notes=(
        "The only clause in the pilot where a rule's trigger_condition is affirmatively "
        "met, and the only Medium provisional level. "
        "UNFAIRNESS 0.25: expressly mutual ('Both Parties ... both Parties'), so the "
        "burden is symmetric -- this is why unfairness stays Low despite the restriction "
        "being substantive. "
        "AMBIGUITY 0.50: 'interfere in any manner' is vague and load-bearing; "
        "'take any other action, which is intended, directly or indirectly, to induce' is "
        "broad; and the text contains an apparent drafting error ('between the Parties and "
        "any their employees'). "
        "NON_COMPLIANCE 0.25: partial trigger only -- scope limb met, duration limb not. "
        "Deliberately not higher, because a two-year, mutual, employee-focused non-solicit "
        "is not the paradigm s.27 case. "
        "ENFORCEABILITY 0.50: the breadth of limb (ii) is a genuine and reviewable concern, "
        "stated as a concern rather than a conclusion."
    ),
    uncertainty=(
        "The gap between limb (i) (narrow, conventional) and limb (ii) (broad) is the whole "
        "question. A human may reasonably score non_compliance 0.00 by reading the clause "
        "as a single conventional non-solicit, or 0.50 by weighting limb (ii) more heavily. "
        "This is the clause most likely to produce genuine inter-annotator disagreement."
    ),
    guideline_issue=(
        "No instruction exists for partial trigger satisfaction -- where a rule's "
        "trigger_condition has two limbs and only one is met."
    ),
)

# [9] Other <- Audit Rights
A[8] = dict(
    status="completed", u=0.25, a=0.75, nc=0.00, e=0.00,
    indian_law_relevant=False,
    legal_provision=None, legal_basis=None, legal_confidence=None,
    legal_risk_type=["ambiguous"],
    confidence="Low", review=True,
    notes=(
        "MOST HEAVILY REDACTED SPAN IN THE PILOT. Five separate '[*]' redactions remove "
        "the notice period, the discrepancy threshold percentage, and -- critically -- the "
        "IDENTITY OF THE PARTY bearing the audit expense in each branch ('at [*] expense', "
        "'[*] shall pay for, or reimburse [*]'). "
        "UNFAIRNESS 0.25: the cost-shifting structure (auditing party pays unless a "
        "material discrepancy is found) is a recognised and reasonably balanced mechanism. "
        "But this score is close to unassessable, because which party sits on which side "
        "of the mechanism is redacted. "
        "AMBIGUITY 0.75: High. Nearly every operative variable is masked. This is the "
        "highest single dimension score in the pilot and it is driven by redaction, i.e. "
        "by a property of the SOURCE DOCUMENT, not of the underlying clause -- which the "
        "schema has no way to distinguish. "
        "NON_COMPLIANCE 0.00 / ENFORCEABILITY 0.00: no rule in the inventory covers audit "
        "rights. ICA_S23_01 is the only 'Other'-scoped rule and is a general contextual "
        "validity backstop that plainly does not apply."
    ),
    uncertainty=(
        "These 0.00 scores mean 'no concern identified within the rule inventory', NOT "
        "'assessed as risk-free'. The schema cannot currently express that difference."
    ),
    guideline_issue=(
        "Redaction-driven ambiguity is scored identically to drafting-driven ambiguity, "
        "which conflates a source-data artifact with a clause defect."
    ),
)

# [10] Penalties <- Liquidated Damages
A[9] = dict(
    status="completed", u=0.25, a=0.50, nc=0.00, e=0.25,
    indian_law_relevant=True,
    legal_provision=None, legal_basis=(
        "ICA_S74_01 (rule_type=risk_trigger) was considered and NOT applied. Its "
        "trigger_condition requires a monetary sum or formula named as payable ON BREACH. "
        "On the visible text the payment trigger is Dova's own termination under Section "
        "12.3.1, and the payment is consideration for promotion services already performed "
        "during the Term -- not a stipulated consequence of a breach by Valeant. Applying "
        "s.74 here would be exactly the 'liquidated damages = illegal penalty' error the "
        "methodology warns against. No rule asserted."
    ),
    legal_confidence="Low",
    legal_risk_type=["ambiguous"],
    confidence="Low", review=True,
    notes=(
        "CATEGORY QUESTIONABLE -- SECONDARY FINDING. CUAD labels this 'Liquidated Damages' "
        "and the mapping sends it to 'Penalties', but the text reads as a post-termination "
        "tail-compensation provision protecting the SERVICE PROVIDER (Valeant) for work "
        "performed, payable when the CLIENT (Dova) terminates. It is compensatory in "
        "direction, not punitive. "
        "UNFAIRNESS 0.25: the clause favours Valeant, and the party bearing the obligation "
        "(Dova) is the one who triggers it by its own election to terminate. Low. "
        "AMBIGUITY 0.50: the multiplier is redacted '[***]', and 'Tail Period', Section "
        "12.3.1 and Section 6.1 are all outside the span. "
        "NON_COMPLIANCE 0.00: ICA_S74_01 trigger not met (see legal_basis). "
        "ENFORCEABILITY 0.25: residual only, recording that if the redacted multiplier were "
        "large enough and the surrounding sections recharacterised the payment as a "
        "consequence of breach, the s.74 question could arise. Not resolvable here."
    ),
    uncertainty=(
        "Mapping a clause to 'Penalties' predisposes an annotator toward s.74. Reading the "
        "text on its own terms points the other way. A human should confirm this reading."
    ),
    guideline_issue=(
        "Category assignment can prime the annotator toward a specific statutory rule. "
        "Guidelines give no instruction to read the text independently of its category "
        "label first."
    ),
)

# ---------------------------------------------------------------------------
records = []
ts = datetime.now(timezone.utc).isoformat()

for i, cid in enumerate(pilot_ids):
    row = sample[cid]
    ann = A[i]
    base = {
        "clause_id": cid,
        "contract_id": row["contract_id"],
        "cuad_category": row["cuad_category"],
        "clausexplain_category": row["clausexplain_category"],
        "clause_text": row["clause_text"],
        "mapping_reason": row["mapping_reason"],
        "needs_review_phase1": row["needs_review"],
        "is_pilot_clause": True,
        "annotation_source": SOURCE,
        "reviewed_by_human": False,
        "annotator_id": ANNOTATOR,
        "annotation_timestamp": ts,
    }
    if ann["status"] == "skipped":
        base.update({"status": "skipped", "annotation_notes": ann["skip_reason"]})
    else:
        u, a, nc, e = ann["u"], ann["a"], ann["nc"], ann["e"]
        p = prov(u, a, nc, e)
        base.update({
            "status": "completed",
            "unfairness_score": u, "ambiguity_score": a,
            "non_compliance_score": nc, "enforceability_score": e,
            "precedent_match_score": None,
            "risk_score": None, "risk_level": None,
            "provisional_risk_score_4dim": p,
            "provisional_risk_level_4dim": lvl(p),
            "legal_risk_type": ann["legal_risk_type"],
            "indian_law_relevant": ann["indian_law_relevant"],
            "legal_provision": ann["legal_provision"],
            "legal_basis": ann["legal_basis"],
            "legal_confidence": ann["legal_confidence"],
            "requires_legal_review": ann["review"],
            "annotation_confidence": ann["confidence"],
            "annotation_notes": ann["notes"],
            "ai_uncertainty_note": ann["uncertainty"],
            "ai_guideline_issue": ann["guideline_issue"],
        })
    records.append(base)

with open(OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(records)} records to {OUT.name}")
comp = [r for r in records if r["status"] == "completed"]
print(f"  completed: {len(comp)} | skipped: {len(records)-len(comp)}")
for r in comp:
    print(f"   {r['provisional_risk_score_4dim']:.4f} {r['provisional_risk_level_4dim']:<7} {r['clausexplain_category']}")
print(f"\nHuman file untouched -> {HUMAN_FILE.stat().st_size} bytes")
