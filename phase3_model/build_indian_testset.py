#!/usr/bin/env python
"""
build_indian_testset.py -- turn your hand-collected Indian clauses into a test set
the Phase 3 code can evaluate.

YOU PROVIDE
    indian_set/documents/*.txt   full plain text of each Indian agreement / template
    indian_set/clauses.csv       one row per clause you picked
        document_file   file name inside documents/ (e.g. nda_sample_1.txt)
        category        one of the 18 ClauseXplain categories (exact spelling, see --list_categories)
        clause_text     the clause, COPIED from that document (whitespace differences are fine)
        source_note     where the document came from (public template, tender, ...)

THIS SCRIPT
    finds each clause inside its document (so we get exact character offsets, which
    the span evaluation needs), and writes
        built/indian_mapped.jsonl      same format as clauses_mapped.jsonl
        built/indian_contracts.json    {contract_id: full text}
        built/build_report.md          counts, problems, coverage warnings

    python build_indian_testset.py --make_template indian_set      # starter folder + CSV
    python build_indian_testset.py --docs_dir indian_set/documents --clauses_csv indian_set/clauses.csv --out_dir indian_set/built

GUIDANCE FOR COLLECTING (so the set is honest)
    * Use documents you are allowed to use (public templates, government tender
      documents, sample agreements). Do not include confidential or personal data.
    * Pick clauses by CATEGORY MEANING, before looking at what any model predicts.
    * Aim for 3+ clauses per category you can cover, 30-50 in total, from several
      different documents.  Some categories (Confidentiality, Indemnification,
      Payment, Force Majeure, Modification, Privacy) are valuable to include: the
      model has never seen them, and the report will say so.
    * If a clause could fit two categories, choose the closest and add a source_note.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from dataset import CATEGORIES

NO_TRAINING_DATA = ["Confidentiality", "Force Majeure", "Indemnification", "Modification",
                    "Payment", "Privacy / Data Protection"]


def slug(name: str) -> str:
    return "indian_" + re.sub(r"[^a-zA-Z0-9]+", "_", Path(name).stem).strip("_").lower()


def locate(doc: str, clause: str):
    toks = clause.split()
    if not toks:
        return None, 0
    rx = re.compile(r"\s+".join(re.escape(t) for t in toks))
    ms = list(rx.finditer(doc))
    return (ms[0].start(), ms[0].end()) if ms else None, len(ms)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs_dir")
    ap.add_argument("--clauses_csv")
    ap.add_argument("--out_dir", default="indian_set/built")
    ap.add_argument("--make_template", metavar="DIR", help="create DIR/documents and DIR/clauses.csv")
    ap.add_argument("--list_categories", action="store_true")
    a = ap.parse_args(argv)

    if a.list_categories:
        print("\n".join(CATEGORIES)); return
    if a.make_template:
        d = Path(a.make_template); (d / "documents").mkdir(parents=True, exist_ok=True)
        with open(d / "clauses.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["document_file", "category", "clause_text", "source_note"])
            w.writerow(["example_nda.txt", "Termination", "Either party may terminate this Agreement by giving thirty (30) days written notice.", "public template (example row, delete me)"])
        print(f"created {d}/documents and {d}/clauses.csv; put plain-text agreements in documents/"); return

    docs_dir, out = Path(a.docs_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in sorted(docs_dir.glob("*.txt"))}
    rows, problems, multi = [], [], []
    per_doc = Counter()
    raw = Path(a.clauses_csv).read_bytes()
    try:
        csv_text, enc_used = raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        csv_text, enc_used = raw.decode("cp1252", errors="replace"), "cp1252 (Excel default; re-save as CSV UTF-8 if text looks wrong)"
    import io
    seen_spans, per_cat_doc = [], Counter()
    notes = []
    if True:
        f = io.StringIO(csv_text, newline="")
        for n, r in enumerate(csv.DictReader(f), start=2):
            fn, cat, txt = (r.get("document_file") or "").strip(), (r.get("category") or "").strip(), r.get("clause_text") or ""
            if fn.startswith("example_nda"):
                continue
            if fn not in docs:
                problems.append(f"line {n}: document '{fn}' not found in {docs_dir}"); continue
            if cat not in CATEGORIES:
                problems.append(f"line {n}: category '{cat}' is not one of the 18 (use --list_categories)"); continue
            loc, count = locate(docs[fn], txt)
            if loc is None:
                problems.append(f"line {n}: clause text not found in {fn}; copy it from the document"); continue
            if count > 1:
                multi.append(f"line {n}: text occurs {count} times in {fn}; first occurrence used")
            s, e = loc
            if len(txt.split()) < 12:
                notes.append(f"line {n}: only {len(txt.split())} words -- looks like a heading, not a clause (check it is not a table-of-contents line)")
            if re.search(r"\.{4,}|\s\d{1,3}\s*$", docs[fn][s:e]):
                notes.append(f"line {n}: text ends in dot leaders or a page number -- probably a table-of-contents line")
            clash = [(o_fn, o_n, o_cat) for (o_fn, o_s, o_e, o_n, o_cat) in seen_spans
                     if o_fn == fn and s < o_e and o_s < e]
            if clash:
                same = [c for c in clash if c[2] == cat]
                if same:
                    problems.append(f"line {n}: DUPLICATE/overlapping row for the same category as line {same[0][1]}; skipped so it is not double counted")
                    continue
                notes.append(f"line {n}: span overlaps line {clash[0][1]} ({clash[0][2]}) but has a different category -- confirm both labels are intended")
            seen_spans.append((fn, s, e, n, cat))
            cid = slug(fn)
            i = per_doc[fn]; per_doc[fn] += 1
            rows.append({"clause_id": f"{cid}__{cat}__{i}", "contract_id": cid, "cuad_category": "INDIAN_SET",
                         "clausexplain_category": cat, "clause_text": docs[fn][s:e], "char_start": s,
                         "char_end": e, "category_status": "MAPPED", "source": "indian_set",
                         "source_note": r.get("source_note", "")})
    (out / "indian_mapped.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    used = {r["contract_id"] for r in rows}
    (out / "indian_contracts.json").write_text(json.dumps({slug(n): t for n, t in docs.items() if slug(n) in used}),
                                               encoding="utf-8")
    cnt = Counter(r["clausexplain_category"] for r in rows)
    rep = ["# Indian test set build report", "", f"{len(rows)} clauses from {len(used)} documents.", "",
           "| Category | clauses | model has training data? |", "|---|---|---|"]
    for c in CATEGORIES:
        if cnt[c]:
            rep.append(f"| {c} | {cnt[c]} | {'no (never seen in training)' if c in NO_TRAINING_DATA else 'yes'} |")
    thin = [c for c in CATEGORIES if 0 < cnt[c] < 3]
    if thin:
        rep += ["", "Categories with fewer than 3 clauses (too thin for per-category scores): " + ", ".join(thin)]
    rep += ["", f"CSV encoding read as: {enc_used}"]
    rep += ["", "## Clauses per document", ""] + [f"- {d}: {c}" + ("  <-- more than 6; one document may dominate" if c > 6 else "")
                                                for d, c in per_doc.most_common()]
    nz = [c for c in CATEGORIES if cnt[c]]
    unseen_n = sum(cnt[c] for c in NO_TRAINING_DATA)
    rep += ["", f"Clauses in never-seen categories: {unseen_n} of {len(rows)}",
            "NOTE: Dispute Resolution is only 'seen' via CUAD's Covenant Not To Sue (no arbitration/forum clauses), "
            "so score it separately from the other seen categories."]
    if notes:
        rep += ["", "## Warnings (rows kept, but check them)"] + [f"- {p}" for p in notes]
    if problems:
        rep += ["", "## Problems (these rows were skipped)"] + [f"- {p}" for p in problems]
    if multi:
        rep += ["", "## Ambiguous matches"] + [f"- {p}" for p in multi]
    (out / "build_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    return rows


if __name__ == "__main__":
    main()
