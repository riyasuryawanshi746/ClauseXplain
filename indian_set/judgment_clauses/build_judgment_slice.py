#!/usr/bin/env python
"""
build_judgment_slice.py -- turn judgment_labelled.csv into the same files build_indian_testset.py
writes, so eval_indian.py can score the judgment slice on its own.

We do not keep the full judgment texts, so each clause is its own "document" and its span is the
whole text.  Score classification only on this slice; span numbers would be meaningless.

    python indian_set/judgment_clauses/build_judgment_slice.py \
        --csv indian_set/judgment_clauses/judgment_labelled.csv --out_dir indian_set/built_judgments
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()

with open(a.csv, newline="", encoding="utf-8-sig") as f:
    src = [r for r in csv.DictReader(f) if not (r.get("not_a_clause (y)") or "").strip()]
rows, texts = [], {}
for r in src:
    cid, txt, cat = f"judgment_{r['unit_id']}", r["clause_text"], r["your_category"].strip()
    texts[cid] = txt
    rows.append({"clause_id": f"{cid}__{cat}__0", "contract_id": cid, "cuad_category": "INDIAN_JUDGMENT",
                 "clausexplain_category": cat, "clause_text": txt, "char_start": 0, "char_end": len(txt),
                 "category_status": "MAPPED", "source": "indian_judgment",
                 "source_note": f"{r['source_case']}; {r['source_url']} (accessed {r['accessed']})"})
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "indian_mapped.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
(out / "indian_contracts.json").write_text(json.dumps(texts), encoding="utf-8")
cnt = Counter(r["clausexplain_category"] for r in rows)
rep = ["# Judgment-sourced slice", "", f"{len(rows)} clauses from {len({r['source_case'] for r in src})} judgments. "
       "Hand-picked, so they over-represent restrictive covenants and arbitration. Classification only.", "",
       "| Category | clauses |", "|---|---|"] + [f"| {c} | {n} |" for c, n in cnt.most_common()]
(out / "build_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
print("\n".join(rep))
