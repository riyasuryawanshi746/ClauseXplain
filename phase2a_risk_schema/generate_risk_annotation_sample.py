"""
generate_risk_annotation_sample.py
Builds a deterministic ~200-clause sample for manual Phase 2 annotation,
stratified across ClauseXplain categories, CUAD categories, and contract
length bands, using a fixed seed for reproducibility.

Payment has zero CUAD-derived spans (Phase 1 finding) -- this sample does
NOT include a Payment stratum, and does not pretend to.
"""
import json
import csv
import random
from pathlib import Path
from collections import defaultdict

MAPPED_JSONL = Path("data/processed/clauses_risk_annotation.jsonl")
CONTRACTS_CSV = Path("data/processed/contracts.csv")
OUT = Path("data/samples/risk_annotation_sample.csv")

random.seed(42)

with open(MAPPED_JSONL, "r", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]

with open(CONTRACTS_CSV, "r", encoding="utf-8") as f:
    contracts = {r["contract_id"]: r for r in csv.DictReader(f)}

# Only sample from MAPPED rows with a real target category (skip
# METADATA/EXCLUDED -- there's nothing to risk-annotate there).
candidates = [r for r in rows if r["category_status"] == "MAPPED" and r["clausexplain_category"]]

by_category = defaultdict(list)
for r in candidates:
    by_category[r["clausexplain_category"]].append(r)

TARGET_TOTAL = 200
categories = sorted(by_category.keys())
per_category_target = max(1, TARGET_TOTAL // len(categories))

sample = []
for cat in categories:
    pool = by_category[cat]
    n = min(per_category_target, len(pool))
    sample.extend(random.sample(pool, n))

# Top up to ~200 from the overall pool if stratified sampling undershot
# (small categories like Penalties, 121 spans, cap the per-category draw).
if len(sample) < TARGET_TOTAL:
    already_ids = set(r["clause_id"] for r in sample)
    remaining_pool = [r for r in candidates if r["clause_id"] not in already_ids]
    top_up = min(TARGET_TOTAL - len(sample), len(remaining_pool))
    sample.extend(random.sample(remaining_pool, top_up))

random.shuffle(sample)

fieldnames = [
    "clause_id", "contract_id", "cuad_category", "clausexplain_category",
    "clause_text", "needs_review", "mapping_reason",
    "contract_char_length", "contract_length_band",
    "risk_level", "unfairness_score", "ambiguity_score",
    "non_compliance_score", "precedent_match_score", "enforceability_score",
    "legal_risk_type", "legal_provision", "legal_confidence",
    "annotation_confidence", "annotator_id", "annotation_source",
]


def length_band(chars: int) -> str:
    if chars < 10000:
        return "short (<10k chars)"
    if chars < 40000:
        return "medium (10k-40k chars)"
    return "long (>40k chars)"


OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in sample:
        contract = contracts.get(r["contract_id"], {})
        char_len = int(contract.get("char_length", 0))
        writer.writerow({
            "clause_id": r["clause_id"],
            "contract_id": r["contract_id"],
            "cuad_category": r["cuad_category"],
            "clausexplain_category": r["clausexplain_category"],
            "clause_text": r["clause_text"],
            "needs_review": r["needs_review"],
            "mapping_reason": r["mapping_reason"],
            "contract_char_length": char_len,
            "contract_length_band": length_band(char_len),
            "risk_level": "",             # blank for the human annotator to fill in
            "unfairness_score": "",
            "ambiguity_score": "",
            "non_compliance_score": "",
            "precedent_match_score": "",
            "enforceability_score": "",
            "legal_risk_type": "",
            "legal_provision": "",
            "legal_confidence": "",
            "annotation_confidence": "",
            "annotator_id": "",
            "annotation_source": "",       # annotator must fill this in explicitly
        })

print(f"Wrote {len(sample)} rows to {OUT}")
from collections import Counter
cat_counts = Counter(r["clausexplain_category"] for r in sample)
print("Category distribution in sample:")
for cat, n in cat_counts.most_common():
    print(f"  {n:4d}  {cat}")
print()
print("NOTE: Payment has no stratum -- zero CUAD-derived spans exist for it (Phase 1 finding).")
