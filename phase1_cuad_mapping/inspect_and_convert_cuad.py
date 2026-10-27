"""
inspect_and_convert_cuad.py
ClauseXplain — Phase 1: inspect the raw CUAD v1 JSON and convert it into two
clean, reproducible tables (contracts + clauses). This script makes NO
assumptions about risk, Indian law, or anything CUAD doesn't actually contain.
It only extracts what's really there.

Run:
    python inspect_and_convert_cuad.py

Expects:
    data/raw/CUAD_v1.json

Produces:
    data/processed/contracts.csv
    data/processed/clauses.jsonl
    data/samples/sample_clauses.csv   (50 random rows, for manual eyeballing)
    Prints a full statistics report to the console.

WHY TWO TABLES, NOT ONE:
CUAD's file structure is "per contract, per category: is this category
present, and if so what span(s) of text represent it." That's naturally a
contract-level table (510 rows) and a clause/span-level table (~13,800 rows).
Flattening them into one table loses the "this category was checked for and
found ABSENT" information, which matters later for building any presence/
absence classifier.
"""

from __future__ import annotations
import json
import csv
import re
import random
import statistics
from pathlib import Path
from collections import Counter

RAW_PATH = Path("data/raw/CUAD_v1.json")
OUT_CONTRACTS = Path("data/processed/contracts.csv")
OUT_CLAUSES = Path("data/processed/clauses.jsonl")
OUT_SAMPLE = Path("data/samples/sample_clauses.csv")

# Rough chars-per-token ratio for legal English. This is an ESTIMATE for
# planning purposes only — NOT a substitute for running the actual tokenizer
# you'll eventually use (e.g. Longformer's tokenizer). Documented here so
# nobody mistakes this for a precise number.
CHARS_PER_TOKEN_ESTIMATE = 4.2


def slugify(title: str) -> str:
    """
    Turn a CUAD title into a safe, stable contract_id.

    BUGFIX (found during validation of the 41->18 category mapping step):
    truncating to a fixed character length can silently collide two
    different contracts whose titles are identical up to that length.
    This actually happened in CUAD v1: two Playboy Enterprises filings
    differ only in a trailing "1"/"2" beyond the old 120-char cutoff,
    which produced duplicate clause_ids downstream. Fix: keep a longer
    human-readable prefix AND append a short hash of the FULL title, so
    two different titles can never produce the same contract_id even if
    their readable prefixes are identical.
    """
    import hashlib
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    prefix = s[:100]
    suffix_hash = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{suffix_hash}"


def guess_contract_type(title: str) -> str:
    """
    Heuristic ONLY. CUAD does not provide a contract_type field.
    We take the text after the last '-' in the title, which is usually
    (not always) the agreement type as filed with the SEC.
    This is a derived (category B) field — treat it as unvalidated.
    """
    parts = title.split("-")
    guess = parts[-1].strip().upper() if parts else "UNKNOWN"
    return guess if guess else "UNKNOWN"


def load_raw(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected CUAD file at {path}. Place CUAD_v1.json in data/raw/ first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_schema(raw: dict) -> None:
    print("=" * 70)
    print("SCHEMA INSPECTION")
    print("=" * 70)
    print(f"Top-level keys      : {list(raw.keys())}")
    print(f"CUAD version field  : {raw.get('version')}")
    print(f"Number of contracts : {len(raw.get('data', []))}")
    if raw.get("data"):
        first = raw["data"][0]
        print(f"Per-contract keys   : {list(first.keys())}")
        if first.get("paragraphs"):
            para = first["paragraphs"][0]
            print(f"Per-paragraph keys  : {list(para.keys())}")
            if para.get("qas"):
                qa = para["qas"][0]
                print(f"Per-qa keys         : {list(qa.keys())}")
    print()


def convert(raw: dict):
    """
    Walk every contract -> paragraph -> qa -> answer, and build:
      - contract_rows: one dict per contract
      - clause_rows:   one dict per answered span (is_impossible == False)
      - category_names: the 41 unique category labels found
      - presence_counter: how many contracts have each category present
    """
    contract_rows = []
    clause_rows = []
    category_names = set()
    presence_counter = Counter()
    absence_counter = Counter()
    multi_span_count = 0
    total_spans = 0

    for contract in raw["data"]:
        title = contract["title"]
        contract_id = slugify(title)

        # CUAD always has exactly one paragraph per contract (verified on
        # the actual file — but we don't hard-assume this holds for every
        # future version, so we loop properly instead of indexing [0]).
        full_text_parts = []
        categories_present_here = 0

        contract_qas_summary = []

        for para in contract["paragraphs"]:
            full_text_parts.append(para["context"])
            for qa in para["qas"]:
                # category name = the suffix after "__" in the qa id
                category = qa["id"].split("__")[-1] if "__" in qa["id"] else qa["question"]
                category_names.add(category)

                description = ""
                if "Details:" in qa["question"]:
                    description = qa["question"].split("Details:", 1)[1].strip()

                is_impossible = qa.get("is_impossible", len(qa["answers"]) == 0)

                if is_impossible or not qa["answers"]:
                    absence_counter[category] += 1
                    continue

                presence_counter[category] += 1
                categories_present_here += 1
                n_spans = len(qa["answers"])
                if n_spans > 1:
                    multi_span_count += 1

                for i, ans in enumerate(qa["answers"]):
                    total_spans += 1
                    clause_rows.append({
                        "clause_id": f"{contract_id}__{category}__{i}",
                        "contract_id": contract_id,
                        "category": category,
                        "category_description": description,
                        "clause_text": ans["text"],
                        "char_start": ans["answer_start"],
                        "char_end": ans["answer_start"] + len(ans["text"]),
                        "is_multi_span": n_spans > 1,
                    })

        full_text = "\n".join(full_text_parts)
        contract_rows.append({
            "contract_id": contract_id,
            "title": title,
            "contract_type_guess": guess_contract_type(title),
            "char_length": len(full_text),
            "estimated_tokens": round(len(full_text) / CHARS_PER_TOKEN_ESTIMATE),
            "num_categories_present": categories_present_here,
        })

    return {
        "contract_rows": contract_rows,
        "clause_rows": clause_rows,
        "category_names": sorted(category_names),
        "presence_counter": presence_counter,
        "absence_counter": absence_counter,
        "multi_span_count": multi_span_count,
        "total_spans": total_spans,
    }


def print_statistics(result: dict) -> None:
    contracts = result["contract_rows"]
    clauses = result["clause_rows"]

    lengths = [c["char_length"] for c in contracts]
    est_tokens = [c["estimated_tokens"] for c in contracts]
    span_lengths = [len(c["clause_text"]) for c in clauses]

    print("=" * 70)
    print("DATASET STATISTICS  (computed from the actual file — not hard-coded)")
    print("=" * 70)
    print(f"contracts                         : {len(contracts)}")
    print(f"unique clause categories          : {len(result['category_names'])}")
    print(f"total answered (category,contract) pairs : {sum(result['presence_counter'].values())}")
    print(f"total absent  (category,contract) pairs  : {sum(result['absence_counter'].values())}")
    print(f"total individual clause spans     : {result['total_spans']}")
    print(f"categories with >1 span in a contract    : {result['multi_span_count']}")
    print()
    print("Contract length (characters):")
    print(f"  min / median / mean / max       : {min(lengths)} / {statistics.median(lengths)} / {round(statistics.mean(lengths))} / {max(lengths)}")
    print("Estimated tokens per contract (chars / 4.2 — rough estimate, not exact):")
    print(f"  min / median / mean / max       : {min(est_tokens)} / {statistics.median(est_tokens)} / {round(statistics.mean(est_tokens))} / {max(est_tokens)}")
    over_4096 = sum(1 for t in est_tokens if t > 4096)
    over_8192 = sum(1 for t in est_tokens if t > 8192)
    print(f"  contracts est. > 4096 tokens     : {over_4096} ({round(100*over_4096/len(est_tokens))}%)")
    print(f"  contracts est. > 8192 tokens     : {over_8192} ({round(100*over_8192/len(est_tokens))}%)")
    print()
    print("Clause span length (characters):")
    print(f"  min / median / mean / max       : {min(span_lengths)} / {statistics.median(span_lengths)} / {round(statistics.mean(span_lengths))} / {max(span_lengths)}")
    print()

    titles = [c["title"] for c in contracts]
    dup_rate = 1 - (len(set(titles)) / len(titles))
    print(f"duplicate contract titles (rate)  : {dup_rate:.4f}")
    print()

    print("Top 10 categories by number of spans:")
    span_counts = Counter(c["category"] for c in clauses)
    for cat, n in span_counts.most_common(10):
        print(f"  {n:5d}  {cat}")
    print()


def write_outputs(result: dict) -> None:
    OUT_CONTRACTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_CLAUSES.parent.mkdir(parents=True, exist_ok=True)
    OUT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)

    contracts = result["contract_rows"]
    clauses = result["clause_rows"]

    with open(OUT_CONTRACTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(contracts[0].keys()))
        writer.writeheader()
        writer.writerows(contracts)

    with open(OUT_CLAUSES, "w", encoding="utf-8") as f:
        for row in clauses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    random.seed(42)  # reproducible sample
    sample = random.sample(clauses, min(50, len(clauses)))
    with open(OUT_SAMPLE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        writer.writeheader()
        writer.writerows(sample)

    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)
    print(f"  {OUT_CONTRACTS}   ({len(contracts)} rows)")
    print(f"  {OUT_CLAUSES}   ({len(clauses)} rows)")
    print(f"  {OUT_SAMPLE}   (50-row random sample for manual review, seed=42)")


def main():
    raw = load_raw(RAW_PATH)
    inspect_schema(raw)
    result = convert(raw)
    print_statistics(result)
    write_outputs(result)


if __name__ == "__main__":
    main()
