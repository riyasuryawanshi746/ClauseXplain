#!/usr/bin/env python
"""
summarise_predictions.py -- slice indian_predictions.csv by document / exclusions.

    python summarise_predictions.py indian_predictions.csv
    python summarise_predictions.py indian_predictions.csv --exclude noida --exclude_other

Scores only categories the model was trained on. --exclude drops documents whose id
contains the given text (e.g. OCR-noisy ones). --exclude_other drops true 'Other'.
Intervals are Wilson 95%; with a few dozen clauses they are wide.
"""
import argparse, csv, math
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--exclude_other", action="store_true")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv_path, encoding="utf-8")))
    rows = [r for r in rows if r["seen_category"] == "1"
            and not any(x.lower() in r["document"].lower() for x in a.exclude)
            and not (a.exclude_other and r["true_category"] == "Other")]
    def line(name, rs):
        k = sum(int(r["correct"]) for r in rs); n = len(rs); lo, hi = wilson(k, n)
        print(f"| {name} | {n} | {k/n:.3f} | {lo:.3f} to {hi:.3f} |" if n else f"| {name} | 0 | n/a | |")
    print("| Slice | n | accuracy | 95% interval |\n|---|---|---|---|")
    line("ALL (seen categories)", rows)
    by = defaultdict(list)
    for r in rows: by[r["document"]].append(r)
    for d, rs in sorted(by.items(), key=lambda x: -len(x[1])): line(d, rs)
    print("\n| True category | n | accuracy | most common wrong prediction |\n|---|---|---|---|")
    bc = defaultdict(list)
    for r in rows: bc[r["true_category"]].append(r)
    for c, rs in sorted(bc.items(), key=lambda x: -len(x[1])):
        wrong = defaultdict(int)
        for r in rs:
            if not int(r["correct"]): wrong[r["pred_category"]] += 1
        top = max(wrong, key=wrong.get) if wrong else "-"
        print(f"| {c} | {len(rs)} | {sum(int(r['correct']) for r in rs)/len(rs):.2f} | {top} |")

if __name__ == "__main__":
    main()
