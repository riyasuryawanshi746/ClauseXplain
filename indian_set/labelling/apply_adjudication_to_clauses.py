#!/usr/bin/env python
"""
apply_adjudication_to_clauses.py -- carry the adjudicated labels into the clause set.

clauses_v2.csv is NOT a plain conversion of label_sheet_labelled.csv: near-duplicate clauses of the MEA
architect agreement (same template as the MEA PMC agreement) were removed by hand when v2 was built.  So instead
of re-deriving the set, this edits the clauses_v2 rows of every unit whose label changed:
    category -> category   relabel the row
    clause -> not a clause drop the row
    not a clause -> clause add the row, unless it is a near-duplicate of the MEA PMC template (listed below)
Units whose row was removed as a near-duplicate stay removed.  Writes clauses_v4.csv = edited v2 + batch 2.

    python indian_set/labelling/apply_adjudication_to_clauses.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# MEA architect units that repeat MEA PMC template text, kept out like the 25 removed when v2 was built
NEAR_DUP = {"mea_architect_consultancy_agreement:3": "recitals; same as mea_pmc_services_agreement:4",
            "mea_architect_consultancy_agreement:43": "nominated officers; same as mea_pmc_services_agreement:55-56"}


def load(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


lab = lambda r: "NOT" if (r.get("not_a_clause (y)") or "").strip() else r["your_category"].strip()
old = {r["unit_id"]: r for r in load(ROOT / "label_sheet_labelled.csv")}
new = {r["unit_id"]: r for r in load(ROOT / "label_sheet_adjudicated.csv")}
v2 = load(ROOT / "archive" / "clauses_v2.csv")
log = []
for uid, o in old.items():
    a, b = lab(o), lab(new[uid])
    if a == b:
        continue
    doc, txt = o["document_file"], o["clause_text"]
    hit = [r for r in v2 if r["document_file"] == doc and r["clause_text"] == txt]
    if uid in NEAR_DUP:
        log.append(f"{uid}: {a} -> {b}; kept out ({NEAR_DUP[uid]})")
    elif hit and b == "NOT":
        v2.remove(hit[0]); log.append(f"{uid}: {a} -> not a clause; row dropped")
    elif hit:
        hit[0]["category"] = b; hit[0]["source_note"] += "|adjudicated"; log.append(f"{uid}: {a} -> {b}; relabelled")
    elif b != "NOT":
        v2.append({"document_file": doc, "category": b, "clause_text": txt,
                   "source_note": f"label_sheet:{o['pool']}|adjudicated"}); log.append(f"{uid}: not a clause -> {b}; row added")
    else:
        log.append(f"{uid}: {a} -> {b}; not in clauses_v2, nothing to do")
batch2 = load(ROOT / "labelling" / "clauses_batch2.csv")
with open(ROOT / "clauses_v4.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["document_file", "category", "clause_text", "source_note"])
    w.writeheader(); w.writerows(v2 + batch2)
print("\n".join(log))
print(f"clauses_v4.csv: {len(v2)} original-pool rows + {len(batch2)} batch-2 rows")
