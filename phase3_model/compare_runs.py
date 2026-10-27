#!/usr/bin/env python
"""
compare_runs.py -- are the differences between models real, or noise?

    python compare_runs.py runs/stageB

Reads <out_dir>/partial/*.json written by train_phase3.py and compares models
FOLD BY FOLD (paired: same test set for both models in each fold):

    joint_cls_span  vs cls_only    on classification metrics
    joint_cls_span  vs span_only   on span metrics
    cls_only        vs tfidf_logreg
    span_only       vs lexical_centroid_baseline / whole_window_baseline
    joint_cls_span  vs the same two non-neural baselines

For every pair it prints the mean difference, a 95% t-interval over folds, and
how many folds the first model won.  With 1 fold nothing can be concluded; with
3 to 5 folds intervals are wide, and this script says so instead of hiding it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np


def paired_summary(diffs: List[float]) -> dict:
    d = np.asarray([x for x in diffs if x is not None and not math.isnan(x)], dtype=float)
    n = len(d)
    out = {"n_folds": n, "mean_diff": float(d.mean()) if n else float("nan"),
           "wins": int((d > 0).sum()), "ci95": None}
    if n >= 2:
        from scipy import stats
        sd = float(d.std(ddof=1))
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        out["sd"] = sd
        out["ci95"] = (out["mean_diff"] - half, out["mean_diff"] + half)
    return out


def verdict(s: dict) -> str:
    if s["n_folds"] < 3:
        return "too few folds to conclude anything"
    lo, hi = s["ci95"]
    if lo > 0:
        return "first model better (95% CI above 0; still only a few folds)"
    if hi < 0:
        return "first model worse (95% CI below 0; still only a few folds)"
    return "no evidence of a difference (CI includes 0)"


def load_runs(partial: Path) -> Dict[str, Dict[int, dict]]:
    runs: Dict[str, Dict[int, dict]] = {}
    for f in sorted(partial.glob("*_fold*.json")):
        name, fold = f.stem.rsplit("_fold", 1)
        if name in ("baselines", "span_baselines"):
            continue
        runs.setdefault(name, {})[int(fold)] = json.loads(f.read_text())
    return runs


def load_baselines(partial: Path, prefix: str) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = {}
    for f in sorted(partial.glob(f"{prefix}_fold*.json")):
        fold = int(f.stem.rsplit("_fold", 1)[1])
        for name, m in json.loads(f.read_text()).items():
            out.setdefault(name, {})[fold] = m
    return out


def get(d: dict, *path) -> Optional[float]:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


CLS_METRICS = {"macro-F1": ("cls", "macro_f1"), "micro-F1": ("cls", "micro_f1")}
SPAN_METRICS = {"span token-F1 (best-match)": ("span", "token_f1"),
                "span exact-match": ("span", "exact_match"),
                "span token-F1 (own label)": ("span_own_label", "token_f1")}


def compare(a: Dict[int, dict], b: Dict[int, dict], path) -> dict:
    folds = sorted(set(a) & set(b))
    diffs = []
    for k in folds:
        va = get(a[k], *path)
        vb = get(b[k], *path)
        if va is not None and vb is not None:
            diffs.append(va - vb)
    return paired_summary(diffs)


def baseline_view(base: Dict[int, dict], path) -> Dict[int, dict]:
    """baseline files hold bare metric dicts; wrap them so the same path works."""
    key = path[0]
    return {k: {key: m} for k, m in base.items()}


def build_report(out_dir: Path) -> str:
    partial = out_dir / "partial"
    runs = load_runs(partial)
    cls_base = load_baselines(partial, "baselines")
    span_base = load_baselines(partial, "span_baselines")
    rows = []

    def add(label_a, A, label_b, B, metric_name, path):
        if A is None or B is None:
            return
        s = compare(A, B, path)
        if s["n_folds"] == 0:
            return
        ci = "n/a" if s["ci95"] is None else f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
        rows.append(f"| {label_a} vs {label_b} | {metric_name} | {s['mean_diff']:+.3f} | {ci} | "
                    f"{s['wins']}/{s['n_folds']} | {verdict(s)} |")

    for m, path in CLS_METRICS.items():
        add("joint_cls_span", runs.get("joint_cls_span"), "cls_only", runs.get("cls_only"), m, path)
        for n, b in cls_base.items():
            if n == "majority":
                continue
            add("cls_only", runs.get("cls_only"), n, baseline_view(b, path), m, path)
            add("joint_cls_span", runs.get("joint_cls_span"), n, baseline_view(b, path), m, path)
    for m, path in SPAN_METRICS.items():
        add("joint_cls_span", runs.get("joint_cls_span"), "span_only", runs.get("span_only"), m, path)
        if path[0] == "span":                      # baselines only scored with best-match gold
            for n, b in span_base.items():
                bv = {k: {"span": v} for k, v in b.items()}
                add("span_only", runs.get("span_only"), n, bv, m, path)
                add("joint_cls_span", runs.get("joint_cls_span"), n, bv, m, path)

    lines = ["# Paired comparisons (difference = first minus second, per fold)", "",
             "| Pair | Metric | Mean diff | 95% CI over folds | First wins | Reading |",
             "|---|---|---|---|---|---|"] + rows
    n_folds = max((len(v) for v in runs.values()), default=0)
    lines += ["", f"Folds available: {n_folds}. Intervals use a t-distribution over folds, so they "
              "are wide when there are few folds and they ignore seed-to-seed variance within a fold. "
              "Treat 'better' as a hint to investigate, not as a proven result."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    a = ap.parse_args()
    out = Path(a.out_dir)
    rep = build_report(out)
    (out / "comparisons.md").write_text(rep, encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
