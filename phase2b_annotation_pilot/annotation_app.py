"""
annotation_app.py -- ClauseXplain Phase 2B Human Annotation Interface
=======================================================================

Run locally with:
    streamlit run annotation_app.py

WHAT THIS DOES
--------------
Displays one CUAD-highlighted span at a time from risk_annotation_sample.csv
and lets a single researcher-annotator (you) record risk-dimension scores,
Indian-law relevance, notes, and confidence for it. Completed annotations
are appended to risk_annotations_human.jsonl. The original sample CSV is
never modified.

RESUME BEHAVIOUR
-----------------
Every time you start the app, it re-reads risk_annotations_human.jsonl,
figures out which clause_ids already have a completed annotation, and
starts you at the first clause that doesn't. You can also jump to any
clause, filter by category, or move forward/back freely with Prev/Next.

WHAT THIS DOES NOT DO
----------------------
- It does not compute or guess risk scores for you.
- It does not fabricate a legal citation you didn't select.
- It does not set annotation_source to anything other than
  "human_researcher_annotation" -- there is no rule engine and no
  qualified legal reviewer in this project yet. Records written by THIS
  app are by definition human-entered, so they also carry
  reviewed_by_human=True (schema v2.2). Machine-generated drafts use
  annotation_source="ai_assisted_preannotation" with
  reviewed_by_human=False and live in a SEPARATE file
  (risk_annotations_ai_pilot.jsonl) -- this app never reads or writes them.
- It never sets precedent_match_score -- that field has no curated
  corpus to draw on (see risk_scoring_specification.txt) and stays null.

PROVISIONAL RISK SCORE -- IMPORTANT DESIGN DECISION
-----------------------------------------------------
phase2_annotation_schema.json states that risk_score/risk_level must
NEVER be set without all five dimension scores being non-null. Since
precedent_match_score cannot legitimately be scored in Phase 2, this app
does NOT set the canonical risk_score / risk_level fields -- they are
written as null, exactly as the schema requires.

Instead, this app computes and stores two clearly-separate,
non-canonical fields for your own triage use while annotating:
    provisional_risk_score_4dim  -- equal-weight average of the four
                                     dimensions that CAN be scored
                                     (unfairness, ambiguity,
                                     non_compliance, enforceability),
                                     i.e. each re-normalized to weight
                                     0.25. This is NOT the same formula
                                     as the eventual 5-dimension
                                     risk_score and must never be
                                     reported as if it were.
    provisional_risk_level_4dim -- Low/Medium/High banding of the above
                                     using the same 0.33/0.66 thresholds,
                                     labeled "provisional".
Rationale: averaging over the four available dimensions (rather than
treating the missing precedent score as 0) avoids silently biasing every
provisional score downward, and is the more statistically defensible
placeholder discussed in risk_scoring_specification.txt (equal-weight
placeholder, w=0.2 each -> renormalized to 0.25 each over 4 dims). It is
explicitly labeled non-canonical everywhere it appears so it can never be
mistaken for the real, eventual 5-dimension risk_score.

DISCRETE SCORING SCALE -- NOTE ON A TENSION WITH THE EXISTING GUIDELINES
--------------------------------------------------------------------------
PHASE2_ANNOTATION_GUIDELINES.txt describes the five dimensions as
continuous (0.0-1.0) with Low/Medium/High bands used only as anchoring
examples. The Phase 2B brief instead specifies a discrete 5-point scale
(0.0 / 0.25 / 0.50 / 0.75 / 1.0). This app implements the discrete scale
as instructed, since it is the more recent, explicit instruction -- but
this is a real tension between two project documents and should be
resolved (pick one, update the other) before scaling past the pilot.
See DECISIONS_AND_OPEN_QUESTIONS.txt.
"""

import json
import csv
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

# ---------------------------------------------------------------------------
# Paths (all relative to this file, so it runs regardless of cwd)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = BASE_DIR / "risk_annotation_sample.csv"
LEGAL_RULES_CSV = BASE_DIR / "indian_legal_rules.csv"
OUTPUT_JSONL = BASE_DIR / "risk_annotations_human.jsonl"

ANNOTATOR_DEFAULT = "researcher_1"
SCORE_OPTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCORE_LABELS = {
    0.0: "0.0 - none/negligible",
    0.25: "0.25 - low",
    0.5: "0.50 - moderate",
    0.75: "0.75 - high",
    1.0: "1.0 - very high",
}
LEGAL_RISK_TYPES = [
    "potentially unenforceable", "potentially non-compliant",
    "potentially unfair", "ambiguous", "commercially unfavorable",
    "jurisdictional concern", "privacy/data-protection concern",
    "restraint-of-trade concern", "excessive liability concern",
    "penalty/damages concern",
]
PILOT_SIZE = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_sample():
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("clause_id")]
    return rows


@st.cache_data
def load_legal_provisions():
    provisions = ["None", "Uncertain"]
    if LEGAL_RULES_CSV.exists():
        with open(LEGAL_RULES_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = r.get("legal_provision", "").strip()
                if p and p not in provisions:
                    provisions.append(p)
    return provisions


def load_completed():
    """Return {clause_id: record} for every completed (non-skipped)
    annotation currently in the output file, keeping the LAST record
    written for a given clause_id (so re-annotation overwrites logically,
    even though the file itself is append-only)."""
    completed = {}
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completed[rec["clause_id"]] = rec
    return completed


def append_record(record: dict):
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def provisional_risk(u, a, nc, e):
    score = round((u + a + nc + e) / 4.0, 4)
    if score <= 0.33:
        level = "Low"
    elif score <= 0.66:
        level = "Medium"
    else:
        level = "High"
    return score, level


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ClauseXplain Phase 2B Annotation", layout="wide")

sample = load_sample()
legal_provisions = load_legal_provisions()
completed_map = load_completed()

pilot_ids_path = BASE_DIR / "pilot_clause_ids.json"
if pilot_ids_path.exists():
    with open(pilot_ids_path, "r", encoding="utf-8") as f:
        pilot_ids = set(json.load(f)["pilot_clause_ids"])
else:
    # Fallback only -- see generate_pilot_set.py for why this isn't the
    # preferred path (first-10 rows cover only 6/12 categories, vs. 10/12
    # for the deterministic category-diverse set that script produces).
    pilot_ids = {r["clause_id"] for r in sample[:PILOT_SIZE]}

st.sidebar.title("ClauseXplain -- Phase 2B")
st.sidebar.caption(
    "Researcher-preliminary annotation. Not lawyer-validated. "
    "requires_legal_review is forced True on every record."
)

pilot_mode = st.sidebar.toggle(
    "Pilot mode (first 10 clauses only)",
    value=False,
    help="Per the Phase 2B plan, annotate these 10 first, review the "
         "interface, then switch this off to continue with the full 200.",
)

categories = sorted({r["clausexplain_category"] for r in sample if r["clausexplain_category"]})
category_filter = st.sidebar.multiselect("Filter by category", categories, default=[])

annotator_id = st.sidebar.text_input("Annotator ID", value=ANNOTATOR_DEFAULT)

# Build the working queue given current filters
queue = [r for r in sample if r["clause_id"] in pilot_ids] if pilot_mode else sample
if category_filter:
    queue = [r for r in queue if r["clausexplain_category"] in category_filter]

if not queue:
    st.warning("No clauses match the current filter.")
    st.stop()

n_total = len(queue)
n_done = sum(1 for r in queue if r["clause_id"] in completed_map)
st.sidebar.metric("Progress", f"{n_done} / {n_total} completed")
st.sidebar.progress(n_done / n_total if n_total else 0)

with st.sidebar.expander("Zero-coverage categories (not in this sample)"):
    st.write(
        "Payment, Privacy / Data Protection, Indemnification, Force "
        "Majeure, Confidentiality, and Modification have zero CUAD-"
        "derived spans and are correctly absent from this sample -- "
        "see PHASE2_README.txt, section 7. Do not annotate synthetic "
        "examples for them here."
    )

# ---------------------------------------------------------------------------
# Index / navigation state
# ---------------------------------------------------------------------------
queue_ids = [r["clause_id"] for r in queue]

if "current_id" not in st.session_state or st.session_state["current_id"] not in queue_ids:
    # Land on the first not-yet-completed clause in this queue, else the first.
    first_unannotated = next((cid for cid in queue_ids if cid not in completed_map), queue_ids[0])
    st.session_state["current_id"] = first_unannotated

current_idx = queue_ids.index(st.session_state["current_id"])

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 2])
with nav_col1:
    if st.button("< Prev", use_container_width=True) and current_idx > 0:
        st.session_state["current_id"] = queue_ids[current_idx - 1]
        st.rerun()
with nav_col2:
    if st.button("Next >", use_container_width=True) and current_idx < n_total - 1:
        st.session_state["current_id"] = queue_ids[current_idx + 1]
        st.rerun()
with nav_col3:
    jump_choice = st.selectbox(
        "Jump to clause",
        options=list(range(n_total)),
        index=current_idx,
        format_func=lambda i: f"{i+1}/{n_total} -- {queue[i]['clause_id'][:50]}"
                              + (" [done]" if queue[i]["clause_id"] in completed_map else ""),
        label_visibility="collapsed",
    )
    if jump_choice != current_idx:
        st.session_state["current_id"] = queue_ids[jump_choice]
        st.rerun()
with nav_col4:
    if st.button("Jump to first unannotated", use_container_width=True):
        first_unannotated = next((cid for cid in queue_ids if cid not in completed_map), queue_ids[0])
        st.session_state["current_id"] = first_unannotated
        st.rerun()

row = queue[current_idx]
clause_id = row["clause_id"]
already_done = clause_id in completed_map
prior = completed_map.get(clause_id, {})

st.divider()

# ---------------------------------------------------------------------------
# Clause display
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.subheader(f"Clause {current_idx + 1} of {n_total}")
    if already_done:
        st.info("Already annotated. Editing will overwrite your prior record for this clause.")
    if clause_id in pilot_ids:
        st.caption("PILOT CLAUSE (part of the 10-clause pilot set)")

    st.markdown(f"**Contract:** `{row['contract_id']}`")
    st.markdown(f"**ClauseXplain category:** {row['clausexplain_category']}")
    st.markdown(f"**CUAD category:** {row['cuad_category']}")
    st.markdown(f"**Phase 1 mapping note:** {row.get('mapping_reason', '')}")
    st.markdown(f"**Contract length band:** {row.get('contract_length_band', 'unknown')}")
    if str(row.get("needs_review", "")).lower() == "true":
        st.caption("Flagged in Phase 1 as a judgment-call mapping (needs_review=True).")

    st.markdown("**Clause text (CUAD-highlighted span -- may be a partial excerpt):**")
    st.text_area("clause_text", value=row["clause_text"], height=180,
                 disabled=True, label_visibility="collapsed")

with right:
    st.markdown("##### Relevant Indian legal rules for this category")
    matches = []
    if LEGAL_RULES_CSV.exists():
        with open(LEGAL_RULES_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("clause_category") == row["clausexplain_category"]:
                    matches.append(r)
    if matches:
        for m in matches:
            rtype = m.get("rule_type", "unclassified")
            rtype_note = {
                "risk_trigger": "may directly increase risk",
                "contextual": "background info, not itself a risk flag",
                "gap_detection": "concerns ABSENCE of a provision",
            }.get(rtype, "")
            st.caption(
                f"**{m['rule_id']}** [{rtype}{': ' + rtype_note if rtype_note else ''}] "
                f"({m['confidence']} confidence) -- "
                f"{m['legal_provision']}: {m['trigger_condition']}"
            )
    else:
        st.caption("No candidate rule in indian_legal_rules.csv for this category. "
                   "non_compliance_score should stay 0.0 unless you can cite one.")

st.divider()

# ---------------------------------------------------------------------------
# Annotation form
# ---------------------------------------------------------------------------
st.markdown("### Annotation")

REQUIRE_SELECTION = "-- select --"


def score_select(label, key, help_text):
    default = prior.get(key)
    options = [REQUIRE_SELECTION] + SCORE_OPTIONS
    idx = options.index(default) if default in SCORE_OPTIONS else 0
    choice = st.selectbox(
        label,
        options=options,
        index=idx,
        format_func=lambda v: v if v == REQUIRE_SELECTION else SCORE_LABELS[v],
        key=f"{key}_{clause_id}",
        help=help_text,
    )
    return None if choice == REQUIRE_SELECTION else choice


c1, c2 = st.columns(2)
with c1:
    unfairness = score_select(
        "unfairness_score", "unfairness_score",
        "Degree of asymmetric/one-sided obligations. See guidelines for anchors.",
    )
    ambiguity = score_select(
        "ambiguity_score", "ambiguity_score",
        "Degree of vague/undefined/multiply-interpretable language.",
    )
with c2:
    non_compliance = score_select(
        "non_compliance_score", "non_compliance_score",
        "Only score above 0.0 if you can cite a specific row in indian_legal_rules.csv.",
    )
    enforceability = score_select(
        "enforceability_score", "enforceability_score",
        "Concern the clause (or part of it) may be unenforceable under Indian law.",
    )

st.selectbox(
    "precedent_match_score",
    options=["null -- no precedent corpus exists yet (Phase 3+)"],
    index=0,
    disabled=True,
    help="Left null for every record in Phase 2. Do not attempt to score this.",
)

# Live provisional score preview
if None not in (unfairness, ambiguity, non_compliance, enforceability):
    prov_score, prov_level = provisional_risk(unfairness, ambiguity, non_compliance, enforceability)
    st.info(
        f"Provisional risk (4-dimension, non-canonical): **{prov_score:.2f} -> {prov_level}**. "
        f"This is NOT the final risk_score -- precedent_match_score is still missing."
    )
else:
    prov_score, prov_level = None, None
    st.caption("Provisional risk score will appear once all four dimensions are scored.")

st.markdown("##### Indian-law relevance")
lc1, lc2, lc3 = st.columns(3)
with lc1:
    indian_law_relevant = st.selectbox(
        "indian_law_relevant",
        options=["-- select --", "Yes", "No", "Uncertain"],
        index=(["-- select --", "Yes", "No", "Uncertain"].index(
            {True: "Yes", False: "No"}.get(prior.get("indian_law_relevant"), "-- select --")
            if "indian_law_relevant" in prior and prior.get("indian_law_relevant") is not None
            else "-- select --"
        )),
        key=f"ilr_{clause_id}",
    )
with lc2:
    legal_provision = st.selectbox(
        "legal_provision", options=legal_provisions,
        index=legal_provisions.index(prior["legal_provision"]) if prior.get("legal_provision") in legal_provisions else 0,
        key=f"lp_{clause_id}",
    )
with lc3:
    legal_confidence = st.selectbox(
        "legal_confidence", options=["-- select --", "High", "Medium", "Low"],
        key=f"lconf_{clause_id}",
    )

legal_risk_type = st.multiselect(
    "legal_risk_type (zero or more)", options=LEGAL_RISK_TYPES,
    default=prior.get("legal_risk_type") or [],
    key=f"lrt_{clause_id}",
)

legal_basis = st.text_input(
    "legal_basis (short, cautious -- 'may raise issues under...', never 'violates')",
    value=prior.get("legal_basis") or "", key=f"lbasis_{clause_id}",
)

annotation_notes = st.text_area(
    "annotation_notes -- explain your reasoning",
    value=prior.get("annotation_notes") or "", height=100, key=f"notes_{clause_id}",
)

cc1, cc2 = st.columns(2)
with cc1:
    annotation_confidence = st.selectbox(
        "annotation_confidence", options=["-- select --", "High", "Medium", "Low"],
        index=(["-- select --", "High", "Medium", "Low"].index(prior["annotation_confidence"])
               if prior.get("annotation_confidence") in ["High", "Medium", "Low"] else 0),
        key=f"aconf_{clause_id}",
    )
with cc2:
    st.checkbox("requires_legal_review", value=True, disabled=True,
                help="Always True in this project -- no qualified legal reviewer yet.")

st.divider()
b1, b2, b3 = st.columns([1, 1, 3])
with b1:
    save_clicked = st.button("Save annotation", type="primary", use_container_width=True)
with b2:
    skip_clicked = st.button("Skip this clause", use_container_width=True)

if save_clicked:
    missing = []
    if None in (unfairness, ambiguity, non_compliance, enforceability):
        missing.append("all four risk dimension scores")
    if indian_law_relevant == "-- select --":
        missing.append("indian_law_relevant")
    if annotation_confidence == "-- select --":
        missing.append("annotation_confidence")
    if not annotation_notes.strip():
        missing.append("annotation_notes")

    if missing:
        st.error("Cannot save -- missing: " + ", ".join(missing))
    else:
        prov_score, prov_level = provisional_risk(unfairness, ambiguity, non_compliance, enforceability)
        record = {
            "clause_id": row["clause_id"],
            "contract_id": row["contract_id"],
            "cuad_category": row["cuad_category"],
            "clausexplain_category": row["clausexplain_category"],
            "clause_text": row["clause_text"],
            "mapping_reason": row.get("mapping_reason", ""),
            "contract_length_band": row.get("contract_length_band", ""),
            "needs_review_phase1": row.get("needs_review", ""),
            "is_pilot_clause": clause_id in pilot_ids,
            # canonical schema fields -- left null, per schema's own rule
            # that they must never be set without all five dimensions:
            "risk_level": None,
            "risk_score": None,
            "unfairness_score": unfairness,
            "ambiguity_score": ambiguity,
            "non_compliance_score": non_compliance,
            "precedent_match_score": None,
            "enforceability_score": enforceability,
            # non-canonical provisional fields -- see module docstring:
            "provisional_risk_score_4dim": prov_score,
            "provisional_risk_level_4dim": prov_level,
            "legal_risk_type": legal_risk_type,
            "indian_law_relevant": {"Yes": True, "No": False, "Uncertain": None}[indian_law_relevant],
            "legal_provision": None if legal_provision == "None" else legal_provision,
            "legal_basis": legal_basis.strip() or None,
            "legal_confidence": None if legal_confidence == "-- select --" else legal_confidence,
            "requires_legal_review": True,
            "annotation_source": "human_researcher_annotation",
            "reviewed_by_human": True,   # v2.2: a human is entering this in the UI
            "annotation_confidence": annotation_confidence,
            "annotation_notes": annotation_notes.strip(),
            "annotator_id": annotator_id.strip() or ANNOTATOR_DEFAULT,
            "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
        }
        append_record(record)
        st.success(f"Saved annotation for {clause_id}.")
        if current_idx < n_total - 1:
            st.session_state["current_id"] = queue_ids[current_idx + 1]
        st.rerun()

if skip_clicked:
    record = {
        "clause_id": row["clause_id"],
        "contract_id": row["contract_id"],
        "cuad_category": row["cuad_category"],
        "clausexplain_category": row["clausexplain_category"],
        "clause_text": row["clause_text"],
        "annotator_id": annotator_id.strip() or ANNOTATOR_DEFAULT,
        "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "skipped",
        "annotation_notes": annotation_notes.strip() or None,
    }
    append_record(record)
    st.warning(f"Marked {clause_id} as skipped.")
    if current_idx < n_total - 1:
        st.session_state["current_id"] = queue_ids[current_idx + 1]
    st.rerun()
