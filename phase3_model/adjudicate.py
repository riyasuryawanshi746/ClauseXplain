#!/usr/bin/env python
"""
adjudicate.py -- settle the second-annotator disagreements, blind to who said what.

WHY BLIND
    You are annotator 1, so when you adjudicate you are tempted to side with yourself.
    The sheet shows the two labels as A and B in a random order per row; the key that
    says which one was yours goes to a separate file.  Do not open the key until you
    are done.  The paper can then report how often the final label matched each side.

STEPS
    python phase3_model/adjudicate.py sheet \
        --disagreements indian_set/labelling/second_annotator_disagreements.csv \
        --primary indian_set/label_sheet_labelled.csv --docs_dir indian_set/documents \
        --out indian_set/labelling/adjudication_sheet.csv
    ... fill final_category (exact spelling) OR not_a_clause (y), plus a short note ...
    python phase3_model/adjudicate.py apply \
        --sheet indian_set/labelling/adjudication_sheet.csv \
        --primary indian_set/label_sheet_labelled.csv \
        --out indian_set/label_sheet_adjudicated.csv

    Then run make_label_sheet.py convert on label_sheet_adjudicated.csv as before.

    final_category can be A's label, B's label, or a third category if both were wrong.
    Context columns are the neighbouring clause units of the same document, only to help
    you read the row; label the clause_text column alone.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

from make_label_sheet import clause_units

NOT = "NOT_A_CLAUSE"
GUIDE = Path(__file__).resolve().parents[1] / "indian_set" / "labelling" / "LABELLING_GUIDE.md"


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def meanings():
    """Category -> one-line definition, read from the labelling guide's table."""
    out = {NOT: "heading, definition, party/date block, TOC line, signature block or unreadable fragment"}
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$", line)
        if m and m.group(1) not in ("Category", "---"):
            out[m.group(1)] = m.group(2)
    return out


def sheet(a):
    primary = {r["unit_id"]: r for r in load(a.primary)}
    docs = {}
    means = meanings()
    rng = random.Random(a.seed)
    rows, key = [], []
    for d in load(a.disagreements):
        uid = d["unit_id"]
        p = primary[uid]
        name = p["document_file"]
        if name not in docs:
            docs[name] = clause_units((Path(a.docs_dir) / name).read_text(encoding="utf-8"))
        units, i = docs[name], int(uid.rsplit(":", 1)[1])
        if units[i] != p["clause_text"]:
            raise SystemExit(f"{uid}: document text no longer matches the label sheet; re-segmentation changed it")
        labs = [("annotator1", d["annotator1"]), ("annotator2", d["annotator2"])]
        rng.shuffle(labs)
        (wa, la), (wb, lb) = labs
        rows.append([uid, name, units[i - 1] if i > 0 else "", p["clause_text"],
                     units[i + 1] if i + 1 < len(units) else "",
                     la, lb, means.get(la, "?"), means.get(lb, "?"), "", "", ""])
        key.append([uid, wa, wb])
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "document_file", "context_before", "clause_text", "context_after",
                    "label_A", "label_B", "meaning_A", "meaning_B",
                    "final_category", "not_a_clause (y)", "adjudication_note"])
        w.writerows(rows)
    key_path = Path(a.out).with_name("adjudication_key_DO_NOT_OPEN.csv")
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["unit_id", "label_A_is", "label_B_is"]); w.writerows(key)
    print(f"{len(rows)} rows -> {a.out}\nkey -> {key_path} (leave it closed until you have finished)")


def apply(a):
    from dataset import CATEGORIES
    allowed = set(CATEGORIES)
    adj = load(a.sheet)
    key = {r["unit_id"]: r for r in load(Path(a.sheet).with_name("adjudication_key_DO_NOT_OPEN.csv"))}
    final, problems = {}, []
    for r in adj:
        cat = (r.get("final_category") or "").strip()
        nac = (r.get("not_a_clause (y)") or "").strip().lower() in ("y", "yes", "1")
        if nac and cat:
            problems.append(f"{r['unit_id']}: both final_category and not_a_clause filled")
        elif nac:
            final[r["unit_id"]] = NOT
        elif not cat:
            problems.append(f"{r['unit_id']}: not adjudicated yet")
        elif cat not in allowed:
            problems.append(f"{r['unit_id']}: '{cat}' is not one of the 18 categories")
        else:
            final[r["unit_id"]] = cat
    if problems:
        raise SystemExit("Fix these first, nothing was written:\n  " + "\n  ".join(problems))

    side = Counter()
    for r in adj:
        f, k = final[r["unit_id"]], key[r["unit_id"]]
        hit = [k["label_A_is"]] if f == r["label_A"] else [k["label_B_is"]] if f == r["label_B"] else []
        side[hit[0] if hit else "neither"] += 1

    rows = load(a.primary)
    changed = 0
    for r in rows:
        f = final.get(r["unit_id"])
        if f is None:
            continue
        old = NOT if (r.get("not_a_clause (y)") or "").strip().lower() in ("y", "yes", "1") else r["your_category"]
        changed += f != old
        r["your_category"], r["not_a_clause (y)"] = ("", "y") if f == NOT else (f, "")
        r["notes"] = (r.get("notes", "") + " [adjudicated]").strip()
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n = len(adj)
    rep = ["# Adjudication", "",
           f"{n} disagreements adjudicated blind. Final label matched annotator 1 in {side['annotator1']}, "
           f"annotator 2 in {side['annotator2']}, neither in {side['neither']}.",
           f"{changed} of {len(rows)} primary labels changed -> {a.out}"]
    Path(a.out).with_name("adjudication_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--disagreements", required=True)
    s.add_argument("--primary", required=True)
    s.add_argument("--docs_dir", required=True)
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--out", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--sheet", required=True)
    p.add_argument("--primary", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sheet(a) if a.cmd == "sheet" else apply(a)


if __name__ == "__main__":
    main()
