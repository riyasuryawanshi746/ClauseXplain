"""
validate_phase2_annotations.py -- ClauseXplain Phase 2B
==========================================================

Checks risk_annotations_human.jsonl for structural and provenance
integrity. Run after any annotation session:

    python3 validate_phase2_annotations.py

This script does NOT judge whether a score is "correct" -- there is no
ground truth yet. It only checks internal consistency: required fields
present, values within allowed sets/ranges, no duplicate clause_ids
among completed records, and that provenance fields (contract_id,
cuad_category, clause_text) match the original sample row for that
clause_id (i.e. nothing was corrupted between sample and annotation).
"""

import json
import csv
import sys
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = BASE_DIR / "risk_annotation_sample.csv"
ANNOTATIONS_JSONL = BASE_DIR / "risk_annotations_human.jsonl"

ALLOWED_CATEGORIES = {
    "Termination", "Payment", "Liability", "Indemnification",
    "Intellectual Property", "Confidentiality", "Privacy / Data Protection",
    "Non-Compete", "Non-Solicitation", "Penalties", "Dispute Resolution",
    "Governing Law / Jurisdiction", "Assignment", "Modification",
    "Warranty", "Force Majeure", "Other", "Commercial Terms",
}
ALLOWED_SCORES = {0.0, 0.25, 0.5, 0.75, 1.0}
ALLOWED_CONFIDENCE = {"High", "Medium", "Low", None}
ALLOWED_SOURCE = {
    "not_yet_annotated", "rule_based_preannotation",
    "human_researcher_annotation", "human_legal_expert_annotation",
    "ai_assisted_preannotation",   # v2.2 -- unreviewed machine drafts
}
REQUIRED_FOR_COMPLETED = [
    "clause_id", "contract_id", "cuad_category", "clausexplain_category",
    "clause_text", "unfairness_score", "ambiguity_score",
    "non_compliance_score", "enforceability_score", "indian_law_relevant",
    "requires_legal_review", "annotation_source", "annotation_confidence",
    "annotator_id", "annotation_notes", "reviewed_by_human",
]


def load_sample_index():
    idx = {}
    if not SAMPLE_CSV.exists():
        return idx
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("clause_id"):
                idx[r["clause_id"]] = r
    return idx


def main():
    errors = []
    warnings = []

    sample_index = load_sample_index()
    if not sample_index:
        errors.append("risk_annotation_sample.csv not found or empty -- cannot verify provenance.")

    if not ANNOTATIONS_JSONL.exists():
        print("No risk_annotations_human.jsonl file found yet. Nothing to validate.")
        print(f"Sample has {len(sample_index)} clauses available for annotation.")
        return 0

    records = []
    with open(ANNOTATIONS_JSONL, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((line_no, json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_no}: invalid JSON ({e})")

    completed = [(ln, r) for ln, r in records if r.get("status") == "completed"]
    skipped = [(ln, r) for ln, r in records if r.get("status") == "skipped"]
    other_status = [(ln, r) for ln, r in records if r.get("status") not in ("completed", "skipped")]
    for ln, r in other_status:
        warnings.append(f"Line {ln}: unrecognized status {r.get('status')!r} for clause_id={r.get('clause_id')}")

    # Duplicate clause_ids among completed records (last-write-wins is fine
    # for the app, but flag it so it's a conscious, not accidental, fact)
    completed_ids = [r["clause_id"] for _, r in completed if "clause_id" in r]
    dupes = {cid: n for cid, n in Counter(completed_ids).items() if n > 1}
    if dupes:
        warnings.append(
            f"{len(dupes)} clause_id(s) annotated more than once "
            f"(most recent entry is authoritative): {list(dupes)[:10]}"
            + (" ..." if len(dupes) > 10 else "")
        )

    for ln, r in completed:
        cid = r.get("clause_id")
        for field in REQUIRED_FOR_COMPLETED:
            if field not in r or r[field] in (None, ""):
                # annotation_notes/annotator_id must be non-empty strings;
                # boolean/False values (e.g. indian_law_relevant=False) are valid.
                if field in ("indian_law_relevant", "reviewed_by_human") and r.get(field) is False:
                    continue
                errors.append(f"Line {ln} (clause_id={cid}): missing required field '{field}'")

        for score_field in ("unfairness_score", "ambiguity_score",
                             "non_compliance_score", "enforceability_score"):
            v = r.get(score_field)
            if v is not None and v not in ALLOWED_SCORES:
                errors.append(f"Line {ln} (clause_id={cid}): {score_field}={v} not in {sorted(ALLOWED_SCORES)}")

        if r.get("precedent_match_score") is not None:
            errors.append(f"Line {ln} (clause_id={cid}): precedent_match_score must be null in Phase 2, "
                           f"got {r.get('precedent_match_score')}")

        if r.get("risk_score") is not None or r.get("risk_level") is not None:
            errors.append(f"Line {ln} (clause_id={cid}): canonical risk_score/risk_level must stay null "
                           f"until all five dimensions (incl. precedent) are scored -- found a non-null value.")

        cat = r.get("clausexplain_category")
        if cat and cat not in ALLOWED_CATEGORIES:
            errors.append(f"Line {ln} (clause_id={cid}): invalid clausexplain_category '{cat}'")

        conf = r.get("annotation_confidence")
        if conf not in ALLOWED_CONFIDENCE:
            errors.append(f"Line {ln} (clause_id={cid}): invalid annotation_confidence '{conf}'")

        src = r.get("annotation_source")
        if src not in ALLOWED_SOURCE:
            errors.append(f"Line {ln} (clause_id={cid}): invalid annotation_source '{src}'")
        if src == "human_legal_expert_annotation":
            errors.append(f"Line {ln} (clause_id={cid}): annotation_source='human_legal_expert_annotation' "
                           f"is not permitted -- this project has no qualified legal reviewer.")

        # --- v2.2 provenance consistency ---
        reviewed = r.get("reviewed_by_human")
        if reviewed not in (True, False):
            errors.append(f"Line {ln} (clause_id={cid}): reviewed_by_human must be a boolean, got {reviewed!r}")
        if src == "ai_assisted_preannotation":
            if reviewed is not False:
                errors.append(f"Line {ln} (clause_id={cid}): ai_assisted_preannotation records must have "
                               f"reviewed_by_human=false until a human actually reviews them.")
            aid = str(r.get("annotator_id") or "")
            if aid.startswith("researcher") or aid == "rule_engine_v1":
                errors.append(f"Line {ln} (clause_id={cid}): annotator_id '{aid}' implies a human or rule "
                               f"engine, but annotation_source is ai_assisted_preannotation.")
        if src == "human_researcher_annotation" and reviewed is not True:
            errors.append(f"Line {ln} (clause_id={cid}): human_researcher_annotation implies a human "
                           f"produced/adopted the record, so reviewed_by_human must be true.")

        if r.get("requires_legal_review") is not True:
            errors.append(f"Line {ln} (clause_id={cid}): requires_legal_review must be True.")

        # Provenance check against the original sample row
        orig = sample_index.get(cid)
        if orig is None:
            warnings.append(f"Line {ln} (clause_id={cid}): clause_id not found in current "
                             f"risk_annotation_sample.csv (sample file may have changed).")
        else:
            for f_ in ("contract_id", "cuad_category", "clause_text"):
                if r.get(f_) != orig.get(f_):
                    errors.append(f"Line {ln} (clause_id={cid}): {f_} does not match the original "
                                  f"sample row -- provenance mismatch.")

    # --- Summary ---
    total_available = len(sample_index) if sample_index else None
    n_completed_unique = len(set(completed_ids))
    n_skipped_unique = len({r["clause_id"] for _, r in skipped if "clause_id" in r})

    print("=" * 70)
    print("PHASE 2B ANNOTATION VALIDATION")
    print("=" * 70)
    if total_available is not None:
        print(f"Total clauses available in sample: {total_available}")
        print(f"Completed (unique clause_id):       {n_completed_unique} "
              f"({n_completed_unique/total_available:.1%})")
        print(f"Skipped (unique clause_id):         {n_skipped_unique} "
              f"({n_skipped_unique/total_available:.1%})")
    print(f"Total annotation-log lines:          {len(records)}")

    if completed:
        cat_dist = Counter(r.get("clausexplain_category") for _, r in completed)
        print("\nCategory distribution (completed):")
        for cat, n in cat_dist.most_common():
            print(f"  {n:4d}  {cat}")

        prov_level_dist = Counter(r.get("provisional_risk_level_4dim") for _, r in completed)
        print("\nProvisional (4-dim) risk level distribution:")
        for lvl, n in prov_level_dist.most_common():
            print(f"  {n:4d}  {lvl}")

    print(f"\nErrors:   {len(errors)}")
    for e in errors:
        print(f"  [ERROR]   {e}")
    print(f"\nWarnings: {len(warnings)}")
    for w in warnings:
        print(f"  [WARNING] {w}")

    print("\nReminder: these are researcher-generated preliminary annotations "
          "and require independent validation. No inter-annotator agreement "
          "or legal-expert sign-off exists yet.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
