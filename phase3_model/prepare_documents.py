#!/usr/bin/env python
"""
prepare_documents.py -- PDF -> clean plain text for the Indian test set, plus a
browsing index that points you at candidate clauses.

    # one PDF -> one cleaned .txt
    python prepare_documents.py --pdf contracts/GTC_on_GeM.pdf --out indian_set/documents/gem_gtc.txt

    # build the browsing index for every .txt in a folder
    python prepare_documents.py --browse indian_set/documents --browse_out indian_set/clause_browser.csv

WHAT THE CLEANING DOES (and does not do)
    * extracts text with pdftotext (poppler); falls back to pypdf if poppler is missing
    * removes lines that repeat on most pages (running headers / footers) and bare page numbers
    * rejoins hard-wrapped lines into paragraphs, and rejoins words split by a line-end
      hyphen ONLY when the joined word appears elsewhere in the same document
    * turns bullet glyphs into "- " and dot leaders ("........") into " ... "
    * does NOT fix OCR errors, translate, or change wording.  If a PDF has no text layer
      or a bad OCR layer, the output will be empty or garbage: check it before use.

THE BROWSING INDEX is built by keyword matching on headings and body text.  It is a
finding aid, NOT a label: category_hint only says which keyword matched.  Decide each
clause's category yourself from its meaning.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

PAGE_NUM = re.compile(r"^\s*(?:page\s*)?[-–]?\s*(?:\d{1,4}|[ivxlc]{1,6})\s*(?:[-–]|(?:of|/)\s*\d{1,4})?\s*$", re.I)
NEW_PARA = re.compile(r"^(?:\(?\d{1,3}(?:\.\d+)*[.)]?\s|\(?[a-z]{1,4}\)\s|\([ivx]+\)\s|[A-Z]\.\s|"
                      r"(?:article|clause|section|schedule|annexure|appendix|chapter)\b|[-•–]\s)", re.I)
BULLETS = "\uf0b7\uf0a7\uf0d8\uf076\u2022\u25cf\u25aa\u2023"


def extract_pages(pdf: str) -> list[str]:
    try:
        out = subprocess.run(["pdftotext", "-enc", "UTF-8", pdf, "-"], capture_output=True, text=True, check=True).stdout
        return out.split("\f")
    except (FileNotFoundError, subprocess.CalledProcessError):
        from pypdf import PdfReader
        return [(p.extract_text() or "") for p in PdfReader(pdf).pages]


def clean_pages(pages: list[str]) -> tuple[str, dict]:
    pages = [unicodedata.normalize("NFKC", p) for p in pages]
    n = len([p for p in pages if p.strip()]) or 1
    key = lambda l: re.sub(r"\d+", "#", re.sub(r"\s+", " ", l.strip().lower()))
    counts = Counter()
    for p in pages:
        counts.update({key(l) for l in p.splitlines() if l.strip()})
    thresh = max(4, 0.35 * n)
    removed = Counter()
    lines: list[str] = []
    for p in pages:
        ls = p.splitlines()
        nonempty = [i for i, l in enumerate(ls) if l.strip()]
        edge = set(nonempty[:2] + nonempty[-2:])
        for i, l in enumerate(ls):
            s = l.strip()
            if not s:
                lines.append(""); continue
            k = key(s)
            if counts[k] >= thresh and len(s) >= 5 and sum(c.isalpha() for c in s) >= 4:
                removed[s[:60]] += 1; continue
            if i in edge and PAGE_NUM.match(s):
                removed["<page number>"] += 1; continue
            lines.append(s)
        lines.append("")
    # bullets and dot leaders
    bullet_rx = re.compile("[%s]" % re.escape(BULLETS))
    for i, l in enumerate(lines):
        l = bullet_rx.sub("-", l)
        l = re.sub(r"\.{4,}|…{2,}", " ... ", l)
        lines[i] = re.sub(r"[ \t\u00a0]+", " ", l).strip()

    vocab = Counter(w.lower() for w in re.findall(r"[A-Za-z]{3,}", " ".join(lines)))
    paras, cur = [], ""

    def flush():
        nonlocal cur
        if cur.strip():
            paras.append(cur.strip())
        cur = ""

    for s in lines:
        if not s:
            flush(); continue
        if cur:
            prev_end = cur[-1]
            new = bool(NEW_PARA.match(s)) or (prev_end in ".;:!?" and (s[0].isupper() or s[0].isdigit())) \
                or (s.isupper() and len(s) < 80) or (cur.isupper() and len(cur) < 80)
            if new:
                flush()
        if cur and cur.endswith("-") and len(cur) > 1 and cur[-2].isalpha() and s[:1].islower():
            m = re.search(r"([A-Za-z]+)-$", cur)
            first = re.match(r"[A-Za-z]+", s)
            if m and first and vocab[(m.group(1) + first.group(0)).lower()] >= 2:
                cur = cur[:-1] + s          # 'termi-' + 'nation' -> 'termination'
            else:
                cur = cur + s               # keep the real hyphen: 'non-' + 'compete'
        else:
            cur = (cur + " " + s) if cur else s
    flush()
    text = "\n\n".join(paras)
    stats = {"pages": len(pages), "chars_out": len(text), "paragraphs": len(paras),
             "removed_top": removed.most_common(4)}
    return text, stats


# ---------------------------------------------------------------------------------
HEAD = {
    "Termination": r"terminat|expiry|expiration",
    "Liability": r"liabilit|limitation of",
    "Indemnification": r"indemn",
    "Force Majeure": r"force majeure|act of god",
    "Confidentiality": r"confidential|non-?disclosure",
    "Dispute Resolution": r"arbitrat|dispute|conciliation|mediation",
    "Governing Law / Jurisdiction": r"governing law|applicable law|jurisdiction|laws of india",
    "Penalties": r"liquidated damages|penalt",
    "Assignment": r"assign|novation|subcontract",
    "Warranty": r"warrant|defect liability|guarantee",
    "Payment": r"payment|invoice|fees?\b|price",
    "Modification": r"amendment|modification|variation|waiver",
    "Privacy / Data Protection": r"data protection|personal data|privacy|data security|data breach",
    "Intellectual Property": r"intellectual property|copyright|ownership of",
    "Non-Solicitation": r"solicit|non-?poach|employment of",
    "Non-Compete": r"non-?compete|exclusiv|conflict of interest",
    "Commercial Terms": r"minimum|royalt|revenue share|most favou?red",
}
BODY = {
    "Penalties": r"liquidated damages|\bpenalty\b|\bpenalties\b",
    "Non-Solicitation": r"\bsolicit",
    "Non-Compete": r"\bcompet(?:e|ing|ition)\b",
    "Indemnification": r"shall indemnify|hold harmless|indemnified",
    "Force Majeure": r"force majeure",
    "Confidentiality": r"confidential information|shall keep .{0,40}confidential",
    "Privacy / Data Protection": r"personal data|data protection|data breach",
}


def browse(docs_dir: str, out_csv: str, per_cat: int = 6, per_body: int = 4) -> int:
    rows = []
    for f in sorted(Path(docs_dir).glob("*.txt")):
        paras = f.read_text(encoding="utf-8").split("\n\n")
        for cat, rx in HEAD.items():
            hits = [i for i, p in enumerate(paras) if len(p) <= 110 and re.search(r"\b(?:" + rx + ")", p, re.I)]
            step = max(1, len(hits) // per_cat)
            for i in hits[::step][:per_cat]:
                prev = " ".join(paras[i:i + 3])[:480]
                rows.append([f.name, cat, i, "heading", paras[i][:100], prev])
        for cat, rx in BODY.items():
            hits = [i for i, p in enumerate(paras) if len(p) > 110 and re.search(r"\b(?:" + rx + ")", p, re.I)]
            step = max(1, len(hits) // per_body)
            for i in hits[::step][:per_body]:
                rows.append([f.name, cat, i, "body", "", paras[i][:480]])
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["document_file", "category_hint (keyword only, not a label)", "paragraph_no", "kind",
                    "heading", "preview"])
        w.writerows(rows)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf")
    ap.add_argument("--out")
    ap.add_argument("--browse")
    ap.add_argument("--browse_out", default="clause_browser.csv")
    a = ap.parse_args(argv)
    if a.pdf:
        text, st = clean_pages(extract_pages(a.pdf))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"{a.pdf} -> {a.out}: {st['pages']} pages, {st['chars_out']} chars, {st['paragraphs']} paragraphs; "
              f"removed repeating lines e.g. {st['removed_top']}")
    if a.browse:
        print(f"{browse(a.browse, a.browse_out)} browser rows -> {a.browse_out}")


if __name__ == "__main__":
    main()
