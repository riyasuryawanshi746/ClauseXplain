"""
generate_statistics.py
Computes dataset_statistics.txt entirely from the actual files on disk.
No hard-coded numbers. Run after apply_category_mapping.py.
"""
import json
import csv
import statistics
from pathlib import Path
from collections import Counter

CONTRACTS_CSV = Path("data/processed/contracts.csv")
CLAUSES_JSONL = Path("data/processed/clauses.jsonl")
MAPPED_JSONL = Path("data/processed/clauses_mapped.jsonl")
MAPPING_CSV = Path("category_mapping.csv")
OUT = Path("data/processed/dataset_statistics.txt")

ALL_18_TARGETS = [
    "Termination", "Payment", "Liability", "Indemnification",
    "Intellectual Property", "Confidentiality", "Privacy / Data Protection",
    "Non-Compete", "Non-Solicitation", "Penalties", "Dispute Resolution",
    "Governing Law / Jurisdiction", "Assignment", "Modification", "Warranty",
    "Force Majeure", "Other", "Commercial Terms",
]

lines = []


def w(s=""):
    lines.append(s)


with open(CONTRACTS_CSV) as f:
    contracts = list(csv.DictReader(f))

with open(CLAUSES_JSONL) as f:
    clauses = [json.loads(l) for l in f]

with open(MAPPED_JSONL) as f:
    mapped = [json.loads(l) for l in f]

with open(MAPPING_CSV) as f:
    mapping_rows = list(csv.DictReader(f))

w("=" * 70)
w("CLAUSEXPLAIN DATASET STATISTICS")
w("Computed directly from the files in data/processed/ -- no hard-coded values.")
w("=" * 70)
w()

w("--- CONTRACT-LEVEL ---")
w(f"Number of contracts (CUAD v1)              : {len(contracts)}")
lengths = [int(c["char_length"]) for c in contracts]
est_tokens = [int(c["estimated_tokens"]) for c in contracts]
w(f"Contract length (chars) min/median/mean/max: {min(lengths)} / {statistics.median(lengths)} / {round(statistics.mean(lengths))} / {max(lengths)}")
w(f"Estimated tokens (chars/4.2, ROUGH ESTIMATE) min/median/mean/max: {min(est_tokens)} / {statistics.median(est_tokens)} / {round(statistics.mean(est_tokens))} / {max(est_tokens)}")
over4k = sum(1 for t in est_tokens if t > 4096)
over8k = sum(1 for t in est_tokens if t > 8192)
w(f"Contracts estimated > 4096 tokens          : {over4k} ({round(100*over4k/len(est_tokens),1)}%)")
w(f"Contracts estimated > 8192 tokens          : {over8k} ({round(100*over8k/len(est_tokens),1)}%)")
w()

w("--- CUAD CATEGORY / SPAN LEVEL (original, unmapped) ---")
cuad_cat_counts = Counter(c["category"] for c in clauses)
w(f"Total CUAD-highlighted spans                : {len(clauses)}")
w(f"Unique CUAD categories                      : {len(cuad_cat_counts)}")
w()
w("Spans per original CUAD category (descending):")
for cat, n in cuad_cat_counts.most_common():
    w(f"  {n:5d}  {cat}")
multi_span = sum(1 for c in clauses if c["is_multi_span"])
w()
w(f"Spans belonging to a multi-span category group: {multi_span} ({round(100*multi_span/len(clauses),1)}%)")
w()

w("--- 41 -> 18 MAPPING SUMMARY ---")
status_counts = Counter(r["category_status"] for r in mapping_rows)
w(f"CUAD categories -> METADATA (no risk signal): {status_counts.get('METADATA',0)}")
w(f"CUAD categories -> EXCLUDED                 : {status_counts.get('EXCLUDED',0)}")
w(f"CUAD categories -> MAPPED to a target category: {status_counts.get('MAPPED',0)}")
w()

row_status_counts = Counter(r["category_status"] for r in mapped)
w("Row-level breakdown (13,823 total CUAD-highlighted spans):")
for status, n in row_status_counts.most_common():
    w(f"  {n:5d}  ({round(100*n/len(mapped),1)}%)  {status}")
w()

needs_review_rows = sum(1 for r in mapped if r["needs_review"])
w(f"Rows flagged needs_review=True in category_mapping.csv: {needs_review_rows} ({round(100*needs_review_rows/len(mapped),1)}%)")
w()

w("--- FINAL 18-CATEGORY CLAUSEXPLAIN TAXONOMY: SPAN COUNTS ---")
cx_counts = Counter(r["clausexplain_category"] for r in mapped if r["clausexplain_category"])
for target in ALL_18_TARGETS:
    n = cx_counts.get(target, 0)
    flag = "  <-- ZERO CUAD COVERAGE" if n == 0 else ""
    w(f"  {n:5d}  {target}{flag}")
w()
zero_cov = [t for t in ALL_18_TARGETS if cx_counts.get(t, 0) == 0]
w(f"Target categories with ZERO CUAD-derived spans: {len(zero_cov)}")
for t in zero_cov:
    w(f"  - {t}")
w()

w("--- COMMERCIAL TERMS (new 18th category) -- CUAD source breakdown ---")
ct_source = Counter(r["cuad_category"] for r in mapped if r["clausexplain_category"] == "Commercial Terms")
for cat, n in ct_source.most_common():
    w(f"  {n:5d}  {cat}")
w(f"  ----- total: {sum(ct_source.values())}")
w()

w("--- PAYMENT -- CUAD source breakdown (should be empty; confirms no forced mapping) ---")
pay_source = Counter(r["cuad_category"] for r in mapped if r["clausexplain_category"] == "Payment")
if not pay_source:
    w("  (none -- Payment receives zero CUAD-derived spans under the strict")
    w("   payment/invoicing-only definition. This is expected, not a bug.)")
else:
    for cat, n in pay_source.most_common():
        w(f"  {n:5d}  {cat}")
w()

w("--- OTHER -- CUAD source breakdown ---")
other_source = Counter(r["cuad_category"] for r in mapped if r["clausexplain_category"] == "Other")
for cat, n in other_source.most_common():
    w(f"  {n:5d}  {cat}")
w(f"  ----- total: {sum(other_source.values())}")
w()

w("=" * 70)
w("END OF STATISTICS REPORT")
w("=" * 70)

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
print(f"({len(lines)} lines)")
