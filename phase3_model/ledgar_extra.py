#!/usr/bin/env python
"""
ledgar_extra.py -- extra training clauses from LEDGAR (LexGLUE version) mapped to ClauseXplain categories (step 5).

WHY
    CUAD has no data for Confidentiality, Force Majeure, Indemnification, Modification, Payment and Privacy, and only
    "Covenant Not To Sue" for Dispute Resolution.  LEDGAR (Tuggener et al. 2020; LexGLUE split: 60k train provisions,
    100 labels, SEC filings) has labels for several of these.  It has NO Force Majeure or Privacy label, and nothing
    for Liability, Penalties, Non-Compete, Non-Solicitation or Commercial Terms.

MAPPING
    Fixed on 2026-11-03 from the category definitions in indian_set/labelling/LABELLING_GUIDE.md, before any LEDGAR
    result was seen.  Only LEDGAR labels that clearly match a guide definition are used; everything else is dropped.
    Uses the LEDGAR train split only (never its validation/test splits, never Indian data).

USAGE
    python ledgar_extra.py --out ledgar_mapped.jsonl            # downloads the train parquet (~21 MB) if needed
    finetune_indian.py then samples --extra_cap clauses per category from it (--extra_jsonl).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

URL = ("https://huggingface.co/datasets/coastalcph/lex_glue/resolve/refs%2Fconvert%2Fparquet/"
       "ledgar/train/0000.parquet")

# LEDGAR label -> ClauseXplain category (guide wording in comments)
MAP = {
    # Modification: "amending the agreement, waiver, entire-agreement style change rules"
    "Amendments": "Modification", "Modifications": "Modification", "Waivers": "Modification",
    "No Waivers": "Modification", "Entire Agreements": "Modification", "Integration": "Modification",
    # Payment: "fees, invoices, retention money, taxes, when and how money is paid"
    "Payments": "Payment", "Fees": "Payment", "Taxes": "Payment",
    # Confidentiality: "keeping information secret, non-disclosure"
    "Confidentiality": "Confidentiality",
    # Indemnification: "compensating/holding harmless the other for losses or claims"
    "Indemnifications": "Indemnification", "Indemnity": "Indemnification",
    # Dispute Resolution: "arbitration, mediation, ..."
    "Arbitration": "Dispute Resolution",
    # Governing Law / Jurisdiction: "which law applies, which courts have jurisdiction"
    "Governing Laws": "Governing Law / Jurisdiction", "Applicable Laws": "Governing Law / Jurisdiction",
    "Jurisdictions": "Governing Law / Jurisdiction", "Consent To Jurisdiction": "Governing Law / Jurisdiction",
    "Submission To Jurisdiction": "Governing Law / Jurisdiction", "Venues": "Governing Law / Jurisdiction",
    # Assignment: "transferring the agreement or rights; successors/assigns"
    "Assignments": "Assignment", "Assigns": "Assignment", "Successors": "Assignment",
    # Intellectual Property, Termination, Warranty
    "Intellectual Property": "Intellectual Property",
    "Terminations": "Termination",
    "Warranties": "Warranty",
    # Other: "notices, severability, survival, language"
    "Notices": "Other", "Severability": "Other", "Survival": "Other",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet", default="ledgar_train.parquet", help="local copy; downloaded if missing")
    a = ap.parse_args(argv)
    import pandas as pd
    from dataset import CAT2ID
    if not Path(a.parquet).exists():
        urllib.request.urlretrieve(URL, a.parquet)
    df = pd.read_parquet(a.parquet)
    names = None
    try:                                           # label names from the parquet schema metadata
        import pyarrow.parquet as pq
        meta = json.loads(pq.read_schema(a.parquet).metadata[b"huggingface"])
        names = meta["info"]["features"]["label"]["names"]
    except Exception as e:
        raise SystemExit(f"could not read LEDGAR label names from parquet metadata: {e}")
    assert len(names) == 100, len(names)
    missing = [k for k in MAP if k not in names]
    assert not missing, missing
    for c in set(MAP.values()):
        assert c in CAT2ID, c
    n = Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for i, (text, lab) in enumerate(zip(df["text"], df["label"])):
            src = names[int(lab)]
            if src in MAP and text and text.strip():
                f.write(json.dumps({"id": f"ledgar_train_{i}", "clause_text": text.strip(),
                                    "clausexplain_category": MAP[src], "ledgar_label": src}) + "\n")
                n[MAP[src]] += 1
    print(f"{sum(n.values())} of {len(df)} LEDGAR train provisions mapped -> {a.out}")
    for c, k in n.most_common():
        print(f"  {c:<30} {k}")


if __name__ == "__main__":
    main()
