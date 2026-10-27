"""
generate_review_sample.py
Deterministic ~100-row sample of clauses_mapped.jsonl for manual review.
Fixed seed (42) -- re-running this script always produces the same sample.
"""
import json
import csv
import random
from pathlib import Path

MAPPED_JSONL = Path("data/processed/clauses_mapped.jsonl")
OUT = Path("data/samples/mapping_review_sample.csv")

random.seed(42)

with open(MAPPED_JSONL, "r", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]

# Oversample needs_review=True rows deliberately (they're the ones that
# actually need a human's eyes) while still including a base of confident
# rows so the reviewer sees normal cases too, not only edge cases.
review_rows = [r for r in rows if r["needs_review"]]
confident_rows = [r for r in rows if not r["needs_review"] and r["clausexplain_category"]]

n_review = min(60, len(review_rows))
n_confident = min(40, len(confident_rows))

sample = random.sample(review_rows, n_review) + random.sample(confident_rows, n_confident)
random.shuffle(sample)

fieldnames = [
    "clause_id", "contract_id", "cuad_category", "clausexplain_category",
    "clause_text", "mapping_reason", "needs_review",
]

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in sample:
        writer.writerow({k: r[k] for k in fieldnames})

print(f"Wrote {len(sample)} rows to {OUT}")
print(f"  {n_review} needs_review=True rows, {n_confident} confident rows")
print("  (seed=42, reproducible)")
