"""
validate.py
Runs every validation check requested for this regeneration step.
Prints PASS/FAIL for each; does not silently continue past a failure.
"""
import json
import csv
from pathlib import Path
from collections import Counter

RAW = Path("data/raw/CUAD_v1.json")
CLAUSES = Path("data/processed/clauses.jsonl")
MAPPED = Path("data/processed/clauses_mapped.jsonl")
MAPPING_CSV = Path("category_mapping.csv")

failures = []

def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        failures.append(label)

print("=" * 70)
print("VALIDATION")
print("=" * 70)

with open(RAW) as f:
    raw = json.load(f)
raw_cats = set()
raw_span_count = 0
for c in raw["data"]:
    for p in c["paragraphs"]:
        for qa in p["qas"]:
            raw_cats.add(qa["id"].split("__")[-1])
            raw_span_count += len(qa["answers"])

print("\n-- Mapping validation --")
check("Exactly 41 CUAD categories in raw dataset", len(raw_cats) == 41)

with open(MAPPING_CSV) as f:
    mapping_rows = list(csv.DictReader(f))
mapping_cats = [r["cuad_category"] for r in mapping_rows]
check("category_mapping.csv has exactly 41 rows", len(mapping_rows) == 41)
check("No duplicate cuad_category rows in mapping CSV", len(mapping_cats) == len(set(mapping_cats)))
check("Every raw CUAD category has exactly one mapping row", raw_cats == set(mapping_cats))

all_18 = {
    "Termination", "Payment", "Liability", "Indemnification",
    "Intellectual Property", "Confidentiality", "Privacy / Data Protection",
    "Non-Compete", "Non-Solicitation", "Penalties", "Dispute Resolution",
    "Governing Law / Jurisdiction", "Assignment", "Modification", "Warranty",
    "Force Majeure", "Other", "Commercial Terms",
}
check("Final taxonomy has exactly 18 named categories", len(all_18) == 18)
targets_used = set(r["clausexplain_category"] for r in mapping_rows if r["clausexplain_category"])
check("No target category used that isn't one of the 18", targets_used.issubset(all_18))

print("\n-- Row validation --")
with open(CLAUSES) as f:
    clause_rows = [json.loads(l) for l in f]
with open(MAPPED) as f:
    mapped_rows = [json.loads(l) for l in f]

check("clauses.jsonl row count == raw CUAD span count",
      len(clause_rows) == raw_span_count == 13823)
check("clauses_mapped.jsonl row count == clauses.jsonl row count",
      len(mapped_rows) == len(clause_rows))

orig_texts = set((r["clause_id"], r["clause_text"]) for r in clause_rows)
mapped_texts = set((r["clause_id"], r["clause_text"]) for r in mapped_rows)
check("No clause text lost or altered (clause_id, clause_text pairs identical)",
      orig_texts == mapped_texts)

orig_contract_ids = set(r["contract_id"] for r in clause_rows)
mapped_contract_ids = set(r["contract_id"] for r in mapped_rows)
check("No contract IDs lost", orig_contract_ids == mapped_contract_ids)

clause_ids = [r["clause_id"] for r in mapped_rows]
check("No duplicate clause_id values introduced", len(clause_ids) == len(set(clause_ids)))

print("\n-- Provenance validation --")
check("Every mapped record carries source='CUAD_v1'",
      all(r.get("source") == "CUAD_v1" for r in mapped_rows))
check("Every mapped record retains its original cuad_category",
      all(r.get("cuad_category") for r in mapped_rows))
check("Every mapped record labeled span_type='CUAD-highlighted span' (not 'clause')",
      all(r.get("span_type") == "CUAD-highlighted span" for r in mapped_rows))

print("\n-- Commercial Terms validation --")
ct_sources = Counter(r["cuad_category"] for r in mapped_rows if r["clausexplain_category"] == "Commercial Terms")
print("  CUAD categories mapped to Commercial Terms:")
for cat, n in ct_sources.most_common():
    print(f"    {n:5d}  {cat}")
expected_ct_sources = {"Most Favored Nation", "Revenue/Profit Sharing", "Price Restrictions",
                        "Minimum Commitment", "Volume Restriction"}
check("Commercial Terms sourced only from the 5 intended CUAD categories",
      set(ct_sources.keys()) == expected_ct_sources)

print("\n-- Payment validation --")
pay_sources = Counter(r["cuad_category"] for r in mapped_rows if r["clausexplain_category"] == "Payment")
check("Payment is NOT used as a generic commercial catch-all (zero CUAD sources)",
      len(pay_sources) == 0)

print("\n-- Other validation --")
other_sources = Counter(r["cuad_category"] for r in mapped_rows if r["clausexplain_category"] == "Other")
print("  CUAD categories mapped to Other:")
for cat, n in other_sources.most_common():
    print(f"    {n:5d}  {cat}")

print("\n" + "=" * 70)
if failures:
    print(f"RESULT: {len(failures)} CHECK(S) FAILED:")
    for f_ in failures:
        print(f"  - {f_}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 70)
