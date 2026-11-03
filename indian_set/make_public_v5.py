#!/usr/bin/env python
"""
make_public_v5.py -- public version of the v5 clause set (clauses_v5.csv stays local).

clauses_v5.csv (local) -> clauses_v5_public.csv: the 19 clauses from the two leave-and-license deeds are dropped and
batch 3 clause text is masked with the same patterns as documents_batch3_redacted/ (redact_batch3.py), so the clauses
can be located in the redacted documents. Then build it:

    python indian_set/redact_batch3.py
    python indian_set/make_public_v5.py
    python phase3_model/build_indian_testset.py --docs_dir indian_set/documents indian_set/documents_batch2 \
        indian_set/documents_batch3_redacted --clauses_csv indian_set/clauses_v5_public.csv --out_dir indian_set/built_v5_public

The reported v5 runs used the full 788 clauses; this public set has 769.
"""
import csv
from pathlib import Path

from redact_batch3 import SKIP, redact

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "clauses_v5.csv", encoding="utf-8")))
out, dropped, masked = [], 0, 0
for r in rows:
    if not r["source_note"].startswith("batch3"):
        out.append(r); continue
    if Path(r["document_file"]).stem in SKIP:
        dropped += 1; continue
    r = dict(r)
    r["clause_text"], n = redact(r["clause_text"]); masked += n > 0
    r["source_note"] = ":".join(r["source_note"].split(":")[:2])    # batch3:<pool>; unit ids point into local sheets
    out.append(r)
with open(HERE / "clauses_v5_public.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(out)
print(f"{len(out)} clauses -> clauses_v5_public.csv ({dropped} leave-and-license clauses dropped, {masked} masked)")
