#!/usr/bin/env python
"""
redact_batch3.py -- public copy of the batch 3 documents for the repo.

documents_batch3/ (local only) -> documents_batch3_redacted/ (in git). The two leave-and-license deeds are left out
(private individuals' names and home addresses). In the rest, PAN, Aadhaar, mobile numbers and e-mail addresses are
replaced with [REDACTED]. Labels and clause sets keep using the unredacted local copies, so clause text may differ
from these files where a number was masked.

    python indian_set/redact_batch3.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "documents_batch3", HERE / "documents_batch3_redacted"
SKIP = {"leave_license_mh_2024a", "leave_license_mh_2024b"}
PATTERNS = [
    re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),                       # PAN
    re.compile(r"\b[0-9]{4}[ -]?[0-9]{4}[ -]?[0-9]{4}\b"),          # Aadhaar
    re.compile(r"(\+91[ -]?)?\b[6-9][0-9]{9}\b"),                   # mobile number
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # e-mail
    re.compile(r"(?<=\bPAN )(?:No\.?:? |number:? )?[A-Z0-9]{8,11}\b"),  # malformed PAN after the word PAN
    re.compile(r"(?<=PAN)(?:\s*No\.?)?(?:\s*\(i[ft] [Aa]pplicable\))?\s*[|:=—-]+\s*[A-Za-z0-9]{8,16}\b"),  # OCR'd PAN fields
    re.compile(r"\b[0-9]{9,}\b"),                                    # bank account and other long numbers
    re.compile(r"\b[0-9]{8}\b(?=.{0,80}\bIN30)|(?<=IN30[0-9]{4} )(?:[^\n]{0,60} )?[0-9]{8}\b"),  # demat client ID
    re.compile(r"\bIN30[0-9]{4}\b"),                                 # depository participant ID (after client ID)
]



def redact(text):
    n = 0
    for pat in PATTERNS:
        text, k = pat.subn("[REDACTED]", text); n += k
    return text, n


if __name__ == "__main__":
    DST.mkdir(exist_ok=True)
    for p in sorted(SRC.glob("*.txt")):
        if p.stem in SKIP:
            continue
        text, n = redact(p.read_text(encoding="utf-8"))
        (DST / p.name).write_text(text, encoding="utf-8")
        print(f"{p.name}: {n} masked")
