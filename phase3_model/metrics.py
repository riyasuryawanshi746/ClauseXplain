"""
metrics.py -- evaluation metrics + paper-ready tables (numpy/sklearn only).

Honesty conventions baked in:
  * Classes with ZERO test support are reported as "n/a", never as F1 = 0.
  * Macro-F1 averages over classes that have support in y_true or y_pred.
  * Every risk table includes trivial baselines (train-mean, train-median)
    because the silver labels are heavily skewed (e.g. non_compliance is 0.0
    for ~90% of clauses): a model must beat the median predictor to mean
    anything.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from dataset import CATEGORIES, RISK_DIMS, RISK_LEVELS


# ------------------------------------------------------------------ classification
def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(CATEGORIES)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    present = [i for i in labels if s[i] > 0 or (y_pred == i).any()]
    micro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0)[2]
    nan = float("nan")
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "micro_f1": float(micro),
        "macro_f1": float(np.mean([f[i] for i in present])) if present else float("nan"),
        "macro_f1_n_classes": len(present),
        # F1 is undefined (NaN -> "n/a") for classes with no test support
        "per_class": {CATEGORIES[i]: {
            "precision": float(p[i]) if s[i] > 0 else nan,
            "recall": float(r[i]) if s[i] > 0 else nan,
            "f1": float(f[i]) if s[i] > 0 else nan,
            "support": int(s[i])} for i in labels},
    }


# ------------------------------------------------------------------ risk
def risk_metrics(true_levels: np.ndarray, pred_scores: np.ndarray,
                 train_levels: np.ndarray | None = None) -> dict:
    """
    true_levels : (N, 4) ints 0..4          pred_scores : (N, 4) floats in [0,1]
    train_levels: optional (M, 4) ints, used for mean/median baselines.
    Errors are computed on the 0..1 score scale (level / 4).
    """
    t = np.asarray(true_levels, dtype=float) / 4.0
    p = np.asarray(pred_scores, dtype=float)
    out: dict = {"n": int(len(t)), "dims": {}}
    for j, d in enumerate(RISK_DIMS):
        e = p[:, j] - t[:, j]
        row = {"mae": float(np.abs(e).mean()), "mse": float((e ** 2).mean()),
               "mae_rounded_to_scale": float(np.abs(
                   np.round(p[:, j] * 4) / 4 - t[:, j]).mean())}
        if train_levels is not None and len(train_levels):
            tr = np.asarray(train_levels, dtype=float)[:, j] / 4.0
            for name, c in (("mean", tr.mean()), ("median", np.median(tr))):
                b = c - t[:, j]
                row[f"baseline_{name}_mae"] = float(np.abs(b).mean())
                row[f"baseline_{name}_mse"] = float((b ** 2).mean())
        out["dims"][d] = row
    out["avg_mae"] = float(np.mean([v["mae"] for v in out["dims"].values()]))
    out["avg_mse"] = float(np.mean([v["mse"] for v in out["dims"].values()]))
    return out


# ------------------------------------------------------------------ span
def _token_f1(pred: Tuple[int, int], gold: Tuple[int, int]) -> float:
    ps, pe = pred
    gs, ge = gold
    if pe < ps:                       # empty / invalid prediction
        return 0.0
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    if inter == 0:
        return 0.0
    prec = inter / (pe - ps + 1)
    rec = inter / (ge - gs + 1)
    return 2 * prec * rec / (prec + rec)


def _iou(pred: Tuple[int, int], gold: Tuple[int, int]) -> float:
    ps, pe = pred
    gs, ge = gold
    if pe < ps:
        return 0.0
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    union = (pe - ps + 1) + (ge - gs + 1) - inter
    return inter / union if union else 0.0


def span_metrics(preds: Sequence[Tuple[int, int]],
                 golds: Sequence[Tuple[int, int]]) -> dict:
    """Token-level F1, Exact Match and IoU. Inclusive token indices."""
    if not len(golds):
        return {"n": 0, "token_f1": float("nan"), "exact_match": float("nan"),
                "iou": float("nan")}
    f1 = [_token_f1(p, g) for p, g in zip(preds, golds)]
    em = [float(tuple(p) == tuple(g)) for p, g in zip(preds, golds)]
    iou = [_iou(p, g) for p, g in zip(preds, golds)]
    return {"n": len(golds), "token_f1": float(np.mean(f1)),
            "exact_match": float(np.mean(em)), "iou": float(np.mean(iou))}


def decode_span(start_logits: np.ndarray, end_logits: np.ndarray,
                context_mask: np.ndarray, max_answer_tokens: int = 512
                ) -> Tuple[int, int]:
    """
    Best (start, end) with start <= end, length <= max_answer_tokens, both
    inside the context (mask==True).  O(L * max_len) via a vectorised sweep.
    """
    s = np.where(context_mask, start_logits, -1e9)
    e = np.where(context_mask, end_logits, -1e9)
    L = len(s)
    best, best_sc = (0, 0), -1e18
    for i in np.argsort(-s)[:20]:          # top-20 starts is plenty in practice
        hi = min(L, i + max_answer_tokens)
        j = i + int(np.argmax(e[i:hi]))
        sc = s[i] + e[j]
        if sc > best_sc:
            best, best_sc = (int(i), int(j)), sc
    return best


# ------------------------------------------------------------------ tables
def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def classification_table(results: Dict[str, dict]) -> str:
    """results: {model_name: classification_metrics(...)}  -> markdown."""
    names = list(results)
    lines = ["| Category | support | " + " | ".join(f"{n} F1" for n in names) + " |",
             "|---|---|" + "---|" * len(names)]
    ref = results[names[0]]["per_class"]
    for c in CATEGORIES:
        sup = ref[c]["support"]
        cells = ["n/a (no support)" if sup == 0 else _fmt(results[n]["per_class"][c]["f1"])
                 for n in names]
        lines.append(f"| {c} | {sup} | " + " | ".join(cells) + " |")
    lines.append("| **Micro-F1** | | " + " | ".join(
        f"**{_fmt(results[n]['micro_f1'])}**" for n in names) + " |")
    lines.append("| **Macro-F1** (supported classes) | | " + " | ".join(
        f"**{_fmt(results[n]['macro_f1'])}**" for n in names) + " |")
    return "\n".join(lines)


def risk_table(results: Dict[str, dict]) -> str:
    names = list(results)
    first = results[names[0]]["dims"]
    cols = []
    for n in names:
        cols += [f"{n} MAE", f"{n} MSE"]
    has_base = "baseline_median_mae" in next(iter(first.values()))
    if has_base:
        cols += ["median-baseline MAE", "mean-baseline MAE"]
    lines = ["| Dimension | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for d in RISK_DIMS:
        cells = []
        for n in names:
            cells += [_fmt(results[n]["dims"][d]["mae"]), _fmt(results[n]["dims"][d]["mse"])]
        if has_base:
            cells += [_fmt(first[d]["baseline_median_mae"]), _fmt(first[d]["baseline_mean_mae"])]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines.append("| **Average** | " + " | ".join(
        c for n in names for c in (f"**{_fmt(results[n]['avg_mae'])}**",
                                   f"**{_fmt(results[n]['avg_mse'])}**")) +
        (" | | |" if has_base else " |"))
    return "\n".join(lines)


def span_table(results: Dict[str, dict]) -> str:
    lines = ["| Model | n | Token-F1 | Exact Match | IoU |", "|---|---|---|---|---|"]
    for n, m in results.items():
        lines.append(f"| {n} | {int(round(m['n']))} | {_fmt(m['token_f1'])} | "
                     f"{_fmt(m['exact_match'])} | {_fmt(m['iou'])} |")
    return "\n".join(lines)


def aggregate(dicts: list):
    """Recursively (mean, std) over a list of identically-shaped metric dicts (NaN-aware)."""
    import warnings
    first = dicts[0]
    if isinstance(first, dict):
        m, s = {}, {}
        for k in first:
            m[k], s[k] = aggregate([d[k] for d in dicts])
        return m, s
    if isinstance(first, (int, float, np.floating, np.integer)):
        a = np.asarray(dicts, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = float(np.nanmean(a)) if not np.all(np.isnan(a)) else float("nan")
            n_ok = int((~np.isnan(a)).sum())
            std = float(np.nanstd(a, ddof=1)) if n_ok > 1 else 0.0
        return mean, std
    return first, None


def mean_std_over_folds(per_fold: List[float]) -> str:
    a = np.asarray([x for x in per_fold if x is not None and not np.isnan(x)])
    return "n/a" if len(a) == 0 else f"{a.mean():.3f} ± {a.std(ddof=1) if len(a) > 1 else 0:.3f}"
