#!/usr/bin/env python
"""Pull contract wording QUOTED INSIDE court judgments (Indian Kanoon) into a candidate sheet.
Only text the judgment reproduces as the contract's own words is taken; every row is cut from the
cleaned judgment by start/end anchors and re-checked as an exact substring. Categories are left
blank: the human annotator labels them. Usage: python extract_judgment_clauses.py SRC_DIR OUT_CSV"""
import csv, glob, re, sys
src, out = sys.argv[1], sys.argv[2]
ACCESSED = "2026-10-29"
def clean(path, header_rx):
    keep = []
    for ln in open(path, encoding="utf-8", errors="ignore").read().splitlines():
        s = ln.strip()
        if re.match(r"^Indian Kanoon\s*-", s) or re.match(r"^Page\s+\d+$", s): continue
        keep.append(re.sub(r"https?://www\.judis\.nic\.in", "", s))
    t = re.sub(r"\s+", " ", " ".join(keep))
    return re.sub(header_rx, "", t) if header_rx else t
# (file glob, case, header regex to strip, doc id, [(label, start, end)])
SPEC = [
 ("03_Super*", "Superintendence Co. of India v Krishan Murgai (SC, 1980)", r"Superintendence Company Of India \(P\) \.\.\. vs Krishan Murgai on 9 May, 1980 ", [
   ("Cl.10 post-employment non-compete", r"10\. That you will not be permitted", r"(?i)after you leave the company\.")]),
 ("04_Wipro*", "Wipro v Beckman Coulter (Del HC, 2006)", r"Wipro Limited vs Beckman Coulter International S\.A\. on 11 July, 2006 ", [
   ("Cl.5 non-solicitation of employees", r"5\. Non-Solicitation of Employees:", r"employees of the other party\."),
   ("Cl.22 arbitration (ICC Geneva)", r"22\. Arbitration 22\.1\.", r"breach of this Agreement\.")]),
 ("05_Complex*", "Complex v Johnwin Manavalan (Mad HC, 2009)", r"Complex vs Johnwin Manavalan S/O Shri on 6 July, 2009 ", [
   ("Cl.12.1-12.4 non-compete / non-solicit", r"12\.1: The Shareholders covenant", r"or entity which is a competitor of the Company\."),
   ("Cl.12.5 severability of restraint", r"12\.5: The shareholder acknowledges", r"valid and enforceable\."),
   ("Cl.14 dispute resolution (negotiation, conciliation)", r"14\.1: Settlement of Disputes", r"Arbitration and Conciliation Act, 1996\. 14\.2")]),
 ("06_Indep*", "Independent News Service v Anuraag Muskaan (Del HC, 2013)", r"Independent News Service Pvt\. Ltd\. vs Mr\. Anuraag Muskaan on 2 April, 2013 ", [
   ("Termination clause", r"Termination The company has right", r"six months professional fee\."),
   ("Exclusivity ii", r"ii\) The Presenter shall not present any other show", r"occupation\."),
   ("Exclusivity / restraint iii", r"iii\) The Presenter, during the validity", r"expiry of this contract\."),
   ("Negotiation bar vii", r"vii\) The Presenter agrees that at any point", r"30 days thereafter\.")]),
 ("07_Limited*", "Anantara v Aumkar (judgment of 1 Apr 2015)", r"", [
   ("Cl.4 non-compete / non-solicit of customers", r"4\.0\. NON-COMPETE AGREEMENT", r"covered by this clause\.")]),
]
rows = []
for pat, case, hdr, items in SPEC:
    p = glob.glob(f"{src}/{pat}")[0]
    url = re.search(r"https?://indiankanoon\.org/doc/\d+/?", open(p, encoding="utf-8", errors="ignore").read()).group(0)
    t = clean(p, hdr)
    for label, s, e in items:
        m = re.search(s, t); assert m, (case, label, "start")
        m2 = re.search(e, t[m.start():]); assert m2, (case, label, "end")
        txt = t[m.start(): m.start() + m2.end()]
        if label.startswith("Cl.14"): txt = re.sub(r" 14\.2$", "", txt)
        assert txt in t
        rows.append([f"jud{len(rows)+1:02d}", case, label, txt, url, ACCESSED, "", ""])
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["unit_id","source_case","what_it_is","clause_text","source_url","accessed","your_category","not_a_clause (y)"]); w.writerows(rows)
print(len(rows), "clauses ->", out)
