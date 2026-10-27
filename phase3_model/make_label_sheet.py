#!/usr/bin/env python
"""
make_label_sheet.py -- grow the Indian test set to 100+ clauses WITHOUT keyword bias.

WHY THIS EXISTS
    The first 32 clauses were picked from keyword shortlists, so they are mostly the
    categories whose keywords are easy to search (Penalties, Confidentiality,
    Indemnification, Force Majeure).  20 of 32 fall in categories the model never
    trained on, and only 12 are scorable, 9 of them Penalties.  Keyword-picked clauses
    are also the easy ones for keyword baselines AND for the rule engine, so scores on
    them flatter both.

TWO POOLS
    exhaustive : EVERY clause-sized paragraph of a few short documents, in document order.
                 You label them all (category, or "not a clause").  This gives the natural
                 mix of categories, including Other, and no selection bias.
    topup      : clause-sized paragraphs from the other documents that mention a keyword
                 of a category you still need.  Keyword-assisted, so say so in the paper;
                 the label is still YOUR reading of the clause, not the keyword.

STEPS
    python make_label_sheet.py build --docs_dir indian_set/documents \
        --exclude indian_set/clauses.csv \
        --exhaustive mea_pmc_services_agreement.txt stqc_non_disclosure_agreement.txt \
                     mea_architect_consultancy_agreement.txt \
        --out indian_set/label_sheet.csv
    ... fill your_category (exact spelling, or "Other") and not_a_clause (y) in Excel/Sheets ...
    python make_label_sheet.py convert --sheet indian_set/label_sheet_filled.csv \
        --out indian_set/clauses_from_sheet.csv
    # optional second annotator on the same sheet -> agreement (Cohen's kappa)
    python make_label_sheet.py convert --sheet label_sheet_filled.csv --second label_sheet_annotator2.csv ...

Then concatenate clauses_from_sheet.csv with your earlier clauses.csv rows and run
build_indian_testset.py as before (it drops overlapping duplicates).

LABELLING RULES (write these in the paper's dataset section)
    * Label by what the clause DOES, using the 18 category names (see --list_categories).
    * "Other" = a real operative clause that fits none of the 18 (e.g. scope of work,
      reporting duties).  "not a clause" = definitions, headings leftovers, tables,
      signature blocks, recitals, anything with no obligation or right.
    * Ignore the selected_by_keyword column; it only says why a paragraph was shown.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

LEAD_NUM = re.compile(r"^(?:\(?\d{1,3}(?:\.\d+)*[.)]?|\([a-z]{1,4}\)|\([ivx]+\)|[A-Z]\.)\s+")
SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")

HEAD = {  # keyword -> category it hints at (used ONLY to build the top-up pool)
    "Termination": r"terminat", "Liability": r"liabilit|limitation of", "Assignment": r"\bassign|novation",
    "Warranty": r"warrant|defect", "Intellectual Property": r"intellectual property|copyright|ownership of",
    "Governing Law / Jurisdiction": r"governing law|applicable law|jurisdiction|laws of india",
    "Dispute Resolution": r"arbitrat|conciliation|dispute", "Commercial Terms": r"minimum|royalt|revenue share|escalation",
    "Non-Compete": r"non-?compete|exclusiv|conflict of interest", "Non-Solicitation": r"solicit|non-?poach",
    "Penalties": r"liquidated damages|penalt", "Modification": r"amendment|variation|waiver",
    "Payment": r"\bpayment|invoice", "Privacy / Data Protection": r"personal data|data protection|privacy",
    "Confidentiality": r"confidential|non-?disclosure", "Indemnification": r"indemnif|hold harmless",
    "Force Majeure": r"force majeure",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def clause_units(doc: str, lo: int = 100, hi: int = 900, target: int = 700):
    """Clause-sized units, verbatim substrings of the document (leading numbering removed)."""
    units = []
    for p in doc.split("\n\n"):
        p = LEAD_NUM.sub("", p.strip(), count=1).strip()
        if len(p) < lo or "..." in p or sum(c.isdigit() for c in p) > 0.18 * len(p):
            continue
        if len(p) <= hi:
            units.append(p); continue
        cur = ""
        for s in SENT.split(p):
            if cur and len(cur) + len(s) > target:
                if len(cur) >= lo:
                    units.append(cur.strip())
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if len(cur) >= lo:
            units.append(cur.strip())
    return units


def read_excluded(paths):
    ex = []
    for p in paths or []:
        with open(p, newline="", encoding="utf-8-sig") as f:
            ex += [norm(r.get("clause_text", "")) for r in csv.DictReader(f)]
    return [e for e in ex if e]


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def build(a):
    from dataset import CATEGORIES  # noqa: F401  (validates the environment)
    docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(a.docs_dir).glob("*.txt"))}
    excluded = read_excluded(a.exclude)
    overlaps = lambda t: any(e and (e in norm(t) or norm(t) in e) for e in excluded)
    rows = []
    rng = random.Random(a.seed)
    for name in a.exhaustive:
        for i, u in enumerate(clause_units(docs[name])):
            if not overlaps(u):
                rows.append([f"{Path(name).stem}:{i}", "exhaustive", name, u, "", "", "", ""])
    per_doc = defaultdict(list)
    for name, text in docs.items():
        if name in a.exhaustive:
            continue
        for i, u in enumerate(clause_units(text)):
            if overlaps(u):
                continue
            hits = [c for c, rx in HEAD.items() if re.search(rx, u[:220], re.I)]
            if hits:
                per_doc[name].append((i, u, hits[0]))
    # balanced top-up: round-robin over categories, spread across documents
    by_cat = defaultdict(list)
    for name, lst in per_doc.items():
        for i, u, c in lst:
            by_cat[c].append((name, i, u))
    cats = [c for c in a.topup_categories] if a.topup_categories else list(HEAD)
    for c in cats:
        pool = by_cat.get(c, [])
        rng.shuffle(pool)
        seen_docs = Counter()
        pool.sort(key=lambda x: seen_docs.update([x[0]]) or seen_docs[x[0]])   # prefer unused docs
        for name, i, u in pool[: a.topup_per_category]:
            rows.append([f"{Path(name).stem}:{i}", "topup", name, u, c, "", "", ""])
    seen_ids, out = set(), []
    for r in rows:
        if r[0] not in seen_ids:
            seen_ids.add(r[0]); out.append(r)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "pool", "document_file", "clause_text", "selected_by_keyword (ignore)",
                    "your_category", "not_a_clause (y)", "notes"])
        w.writerows(out)
    cnt = Counter(r[1] for r in out)
    print(f"{len(out)} units -> {a.out}  ({dict(cnt)}). Expect to mark many as 'not a clause'.")


def convert(a):
    from dataset import CATEGORIES
    allowed = set(CATEGORIES)

    def load(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            return {r["unit_id"]: r for r in csv.DictReader(f)}

    s1 = load(a.sheet)

    def label(r):
        if (r.get("not_a_clause (y)") or "").strip().lower() in ("y", "yes", "1"):
            return "NOT_A_CLAUSE"
        return (r.get("your_category") or "").strip()

    out, problems, unlabeled = [], [], 0
    for uid, r in s1.items():
        lab = label(r)
        if not lab:
            unlabeled += 1; continue
        if lab == "NOT_A_CLAUSE":
            continue
        if lab not in allowed:
            problems.append(f"{uid}: '{lab}' is not one of the 18 categories"); continue
        out.append([r["document_file"], lab, r["clause_text"], f"label_sheet:{r['pool']}"])
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["document_file", "category", "clause_text", "source_note"]); w.writerows(out)
    cnt = Counter(o[1] for o in out)
    rep = [f"# Label sheet conversion", "", f"{len(out)} clauses kept, {unlabeled} units left unlabeled, "
           f"{sum(1 for r in s1.values() if label(r) == 'NOT_A_CLAUSE')} marked not-a-clause.", "",
           "| Category | clauses |", "|---|---|"] + [f"| {c} | {n} |" for c, n in cnt.most_common()]
    if problems:
        rep += ["", "## Problems (rows skipped)"] + [f"- {p}" for p in problems]
    if a.second:
        s2 = load(a.second)
        common = [u for u in s1 if u in s2 and label(s1[u]) and label(s2[u])]
        l1, l2 = [label(s1[u]) for u in common], [label(s2[u]) for u in common]
        k = kappa(l1, l2)
        agree = sum(x == y for x, y in zip(l1, l2)) / max(1, len(common))
        rep += ["", "## Annotator agreement", "", f"{len(common)} units labelled by both: raw agreement "
                f"{agree:.2f}, Cohen's kappa {k:.2f} (labels include NOT_A_CLAUSE).",
                "Kappa above 0.6 is usually read as substantial agreement; below that, tighten the labelling rules."]
        dis = [[u, s1[u]["document_file"], l1[i], l2[i], s1[u]["clause_text"][:300]]
               for i, u in enumerate(common) if l1[i] != l2[i]]
        with open(Path(a.out).with_name("disagreements.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["unit_id", "document_file", "annotator1", "annotator2", "clause_text"])
            w.writerows(dis)
        rep.append(f"{len(dis)} disagreements written to disagreements.csv for you to adjudicate.")
    Path(a.out).with_name("sheet_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--docs_dir", required=True)
    b.add_argument("--exclude", nargs="*", default=[])
    b.add_argument("--exhaustive", nargs="+", default=[])
    b.add_argument("--topup_per_category", type=int, default=6)
    b.add_argument("--topup_categories", nargs="*", default=None)
    b.add_argument("--seed", type=int, default=13)
    b.add_argument("--out", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--sheet", required=True)
    c.add_argument("--second", default=None)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    build(a) if a.cmd == "build" else convert(a)


if __name__ == "__main__":
    main()
