"""
generate_phase2_dataset.py
ClauseXplain Phase 2 -- adds the risk/legal annotation schema fields to
every Phase 1 record, WITHOUT fabricating any values. Every new field is
null and annotation_source='not_yet_annotated' for every record, because
no actual human or validated rule-based annotation has been performed yet.

This deliberately does NOT run indian_legal_rules.csv's trigger_conditions
against clause text automatically -- that would BE a rule-based
preannotation step, which is a real methodological choice (weak
supervision) that should be a separate, explicitly-labeled script run
with its own validation, not silently bundled into dataset construction.
See PHASE2_README.txt, "What remains for Phase 3," for this as a named
next step.

Run:
    python generate_phase2_dataset.py

Expects:
    data/processed/clauses_mapped.jsonl   (untouched, from Phase 1)

Produces:
    data/processed/clauses_risk_annotation.jsonl

clauses_mapped.jsonl is never modified.
"""
import json
from pathlib import Path

MAPPED_IN = Path("data/processed/clauses_mapped.jsonl")
OUT = Path("data/processed/clauses_risk_annotation.jsonl")

# Every new Phase 2 field, all null/default until real annotation happens.
PHASE2_BLANK_FIELDS = {
    "risk_level": None,
    "risk_score": None,
    "unfairness_score": None,
    "ambiguity_score": None,
    "non_compliance_score": None,
    "precedent_match_score": None,
    "enforceability_score": None,
    "legal_risk_type": None,
    "indian_law_relevant": None,
    "legal_provision": None,
    "legal_basis": None,
    "legal_confidence": None,
    "requires_legal_review": True,   # defaults True until an actual reviewer signs off
    "annotation_source": "not_yet_annotated",
    "annotation_confidence": None,
    "annotator_id": None,
    "annotation_timestamp": None,
}


def main():
    if not MAPPED_IN.exists():
        raise FileNotFoundError(f"{MAPPED_IN} not found. Run Phase 1 first.")

    n = 0
    with open(MAPPED_IN, "r", encoding="utf-8") as fin, \
         open(OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            row.update(PHASE2_BLANK_FIELDS)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} rows to {OUT}")
    print("Every row: annotation_source='not_yet_annotated', all risk/legal fields null.")
    print("This file adds structure; it adds NO labels. See PHASE2_README.txt.")


if __name__ == "__main__":
    main()
