#!/usr/bin/env python
"""
recheck.py -- second round on annotator disagreements after the guideline changed (batch 3, guide v2).

WHY
    Batch 3 round 1 was labelled before guide v2 was final (kappa 0.57). Instead of adjudicating every
    disagreement, each annotator re-labels only the units the two disagreed on, under v2. Each sees their OWN
    round-1 label (to reconsider it) but never the other annotator's. Remaining disagreements are then
    adjudicated blind with adjudicate.py. The paper reports both rounds.

STEPS
    python recheck.py make --sheet1 a1_filled.csv --sheet2 a2_filled.csv --docs_dir ../indian_set/documents_batch3 \
        --out_dir ../indian_set/labelling/batch3
        # -> recheck_annotator1.csv, recheck_annotator2.csv (give each file only to its annotator)
    ... each annotator fills your_category or not_a_clause (y) on every row, using LABELLING_GUIDE.md v2 ...
    python recheck.py merge --sheet1 a1_filled.csv --sheet2 a2_filled.csv \
        --recheck1 recheck_annotator1_filled.csv --recheck2 recheck_annotator2_filled.csv \
        --out_dir ../indian_set/labelling/batch3
        # -> round2_annotator{1,2}.csv (round-1 sheets with re-checked rows replaced) + recheck_report.md
    Then: make_label_sheet.py convert --sheet round2_annotator1.csv --second round2_annotator2.csv ...
          adjudicate.py sheet on the new disagreements.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from make_label_sheet import clause_units, kappa

NOT = "NOT_A_CLAUSE"


def load(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return {r["unit_id"]: r for r in csv.DictReader(f)}


def label(r):
    if (r.get("not_a_clause (y)") or "").strip().lower() in ("y", "yes", "1"):
        return NOT
    return (r.get("your_category") or "").strip()


def make(a):
    s1, s2 = load(a.sheet1), load(a.sheet2)
    dis = [u for u in s1 if u in s2 and label(s1[u]) != label(s2[u])]
    units = {}
    for p in Path(a.docs_dir).glob("*.txt"):
        units[p.stem] = clause_units(p.read_text(encoding="utf-8"))
    random.Random(a.seed).shuffle(dis)
    cols = ["unit_id", "document_file", "context_before", "clause_text", "context_after", "your_round1_label",
            "your_category", "not_a_clause (y)", "notes"]
    for who, s in (("annotator1", s1), ("annotator2", s2)):
        with open(Path(a.out_dir) / f"recheck_{who}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(cols)
            for u in dis:
                stem, i = u.rsplit(":", 1); i = int(i); us = units.get(stem, [])
                before = us[i - 1][:300] if 0 < i <= len(us) else ""
                after = us[i + 1][:300] if i + 1 < len(us) else ""
                w.writerow([u, s[u]["document_file"], before, s[u]["clause_text"], after, label(s[u]), "", "", ""])
    print(f"{len(dis)} disagreement units -> recheck_annotator1.csv / recheck_annotator2.csv in {a.out_dir}")


def merge(a):
    s1, s2 = load(a.sheet1), load(a.sheet2)
    r1, r2 = load(a.recheck1), load(a.recheck2)
    assert set(r1) == set(r2), "the two re-check sheets cover different units"
    missing = [u for u in r1 if not label(r1[u]) or not label(r2[u])]
    if missing:
        raise SystemExit(f"{len(missing)} re-check rows are unlabelled, e.g. {missing[:3]}")
    out = Path(a.out_dir)
    for who, s, r in (("annotator1", s1, r1), ("annotator2", s2, r2)):
        rows = []
        for u, row in s.items():
            row = dict(row)
            if u in r:
                row["your_category"] = r[u].get("your_category", "")
                row["not_a_clause (y)"] = r[u].get("not_a_clause (y)", "")
            rows.append(row)
        with open(out / f"round2_{who}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    common = list(s1)
    k1 = kappa([label(s1[u]) for u in common], [label(s2[u]) for u in common])
    a1 = sum(label(s1[u]) == label(s2[u]) for u in common) / len(common)
    f1 = {u: label(r1[u]) if u in r1 else label(s1[u]) for u in common}
    f2 = {u: label(r2[u]) if u in r2 else label(s2[u]) for u in common}
    k2 = kappa([f1[u] for u in common], [f2[u] for u in common])
    a2 = sum(f1[u] == f2[u] for u in common) / len(common)
    re_agree = sum(label(r1[u]) == label(r2[u]) for u in r1)
    ch1 = sum(label(r1[u]) != label(s1[u]) for u in r1); ch2 = sum(label(r2[u]) != label(s2[u]) for u in r2)
    to_other1 = sum(label(r1[u]) == label(s2[u]) for u in r1)      # moved to the other's round-1 label
    to_other2 = sum(label(r2[u]) == label(s1[u]) for u in r2)
    pairs = Counter(tuple(sorted((label(r1[u]), label(r2[u])))) for u in r1 if label(r1[u]) != label(r2[u]))
    rep = ["# Batch 3 re-check (guide v2)", "",
           f"Round 1 (guide v1): {len(common)} units, raw agreement {a1:.2f}, kappa {k1:.2f}.",
           f"Re-checked {len(r1)} round-1 disagreements under v2, each annotator blind to the other's labels.",
           f"Round 2: raw agreement {a2:.2f}, kappa {k2:.2f}. {re_agree} of {len(r1)} re-checked units now agree; "
           f"{len(r1) - re_agree} remain for blind adjudication.",
           f"Annotator 1 changed {ch1} labels ({to_other1} to annotator 2's round-1 label); annotator 2 changed {ch2} "
           f"({to_other2} to annotator 1's round-1 label).", "",
           "Remaining disagreement pairs:", ""] + [f"- {n} {p}" for p, n in pairs.most_common(15)]
    (out / "recheck_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--sheet1", required=True); m.add_argument("--sheet2", required=True)
    m.add_argument("--docs_dir", required=True); m.add_argument("--out_dir", required=True)
    m.add_argument("--seed", type=int, default=13)
    g = sub.add_parser("merge")
    g.add_argument("--sheet1", required=True); g.add_argument("--sheet2", required=True)
    g.add_argument("--recheck1", required=True); g.add_argument("--recheck2", required=True)
    g.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    make(a) if a.cmd == "make" else merge(a)


if __name__ == "__main__":
    main()
