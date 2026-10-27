"""
apply_category_mapping.py
ClauseXplain -- Phase 1b: apply the reviewed CUAD -> ClauseXplain category
mapping to every clause span. This step does NOT touch risk, does NOT
generate any legal conclusions, and does NOT modify the original CUAD
category field. It renames it to `cuad_category` for clarity (CUAD-native
label) and adds `clausexplain_category` (the mapped target).

Run:
    python apply_category_mapping.py

Expects:
    data/processed/clauses.jsonl        (untouched, from Phase 1)
    category_mapping.csv                (human-reviewed mapping table,
                                          columns: cuad_category,
                                          clausexplain_category,
                                          category_status, mapping_reason,
                                          legal_basis_or_note, needs_review)

Produces:
    data/processed/clauses_mapped.jsonl
    Prints mapping statistics to the console.

FAILS LOUDLY (raises, does not silently skip) if any CUAD category found
in the data has no corresponding row in category_mapping.csv -- an
unmapped category is a pipeline bug, not something to paper over.

The original clauses.jsonl is never written to.
"""

from __future__ import annotations
import json
import csv
import sys
from pathlib import Path
from collections import Counter

CLAUSES_IN = Path("data/processed/clauses.jsonl")
MAPPING_CSV = Path("category_mapping.csv")
CLAUSES_OUT = Path("data/processed/clauses_mapped.jsonl")


def load_mapping(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Mapping file not found at {path}. "
            "This script will not guess a mapping on your behalf."
        )
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "cuad_category", "clausexplain_category", "category_status",
            "mapping_reason", "legal_basis_or_note", "needs_review",
        }
        missing_cols = required_cols - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"category_mapping.csv is missing columns: {missing_cols}")
        for row in reader:
            mapping[row["cuad_category"]] = row
    return mapping


def apply_mapping():
    mapping = load_mapping(MAPPING_CSV)

    if not CLAUSES_IN.exists():
        raise FileNotFoundError(
            f"{CLAUSES_IN} not found. Run inspect_and_convert_cuad.py first."
        )

    CLAUSES_OUT.parent.mkdir(parents=True, exist_ok=True)

    status_counts = Counter()
    clausexplain_counts = Counter()
    original_category_counts = Counter()
    needs_review_count = 0
    unmapped_categories = set()
    total_rows = 0

    with open(CLAUSES_IN, "r", encoding="utf-8") as fin, \
         open(CLAUSES_OUT, "w", encoding="utf-8") as fout:

        for line in fin:
            raw_row = json.loads(line)
            total_rows += 1
            original_category = raw_row["category"]
            original_category_counts[original_category] += 1

            entry = mapping.get(original_category)
            if entry is None:
                unmapped_categories.add(original_category)
                sys.stderr.write(
                    f"FATAL: category '{original_category}' found in "
                    f"clauses.jsonl has no row in {MAPPING_CSV}.\n"
                )
                continue

            needs_review = str(entry["needs_review"]).strip().lower() in ("true", "1", "yes")
            if needs_review:
                needs_review_count += 1

            out_row = {
                "clause_id": raw_row["clause_id"],
                "contract_id": raw_row["contract_id"],
                "cuad_category": original_category,
                "clausexplain_category": entry["clausexplain_category"] or None,
                "category_status": entry["category_status"],
                "category_description": raw_row["category_description"],
                "clause_text": raw_row["clause_text"],
                "char_start": raw_row["char_start"],
                "char_end": raw_row["char_end"],
                "is_multi_span": raw_row["is_multi_span"],
                "needs_review": needs_review,
                "mapping_reason": entry["mapping_reason"],
                "legal_basis_or_note": entry["legal_basis_or_note"],
                "source": "CUAD_v1",
                "span_type": "CUAD-highlighted span",
            }

            status_counts[entry["category_status"]] += 1
            if out_row["clausexplain_category"]:
                clausexplain_counts[out_row["clausexplain_category"]] += 1

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    if unmapped_categories:
        raise RuntimeError(
            f"{len(unmapped_categories)} categories in the data have no "
            f"mapping entry: {sorted(unmapped_categories)}. "
            f"Fix {MAPPING_CSV} and re-run. No output file was completed "
            f"validly -- do not use the partial clauses_mapped.jsonl."
        )

    print("=" * 70)
    print("MAPPING APPLIED")
    print("=" * 70)
    print(f"Total clause rows processed        : {total_rows}")
    print(f"Original CUAD categories seen       : {len(original_category_counts)}")
    print()
    print("Rows by category_status:")
    for status, n in status_counts.most_common():
        pct = round(100 * n / total_rows, 1)
        print(f"  {n:5d}  ({pct:5.1f}%)  {status}")
    print()
    print(f"Rows flagged needs_review=True      : {needs_review_count} "
          f"({round(100*needs_review_count/total_rows,1)}%)")
    print()
    print("Rows by clausexplain_category (all MAPPED rows, including needs_review=True):")
    for cat, n in clausexplain_counts.most_common():
        print(f"  {n:5d}  {cat}")
    print()
    print(f"Written: {CLAUSES_OUT}")

    return {
        "total_rows": total_rows,
        "status_counts": status_counts,
        "clausexplain_counts": clausexplain_counts,
        "needs_review_count": needs_review_count,
        "original_category_counts": original_category_counts,
    }


if __name__ == "__main__":
    apply_mapping()
