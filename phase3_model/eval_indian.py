#!/usr/bin/env python
"""
eval_indian.py -- how do the CUAD-trained models behave on Indian-context clauses?

    python eval_indian.py --run_dir /kaggle/working/stageB --fold 0 \
        --ckpt ckpt_joint_cls_span_fold0.pt \
        --indian_dir indian_set/built \
        --mapped_jsonl clauses_mapped.jsonl --cuad_json CUAD_v1.json \
        --rules indian_legal_rules.csv

(build the Indian set first with build_indian_testset.py)

WHAT IT REPORTS
    1. Category classification on Indian clauses whose category the model was trained
       on (12 of 18), with a Wilson 95% interval, next to the TF-IDF baseline trained
       on the same CUAD training split.
    2. Span localisation on those clauses (windowed, same settings as training),
       next to the whole-window and lexical-centroid baselines.
    3. What the model predicts for clauses in the 6 categories it has NEVER seen
       (Confidentiality, Force Majeure, Indemnification, Modification, Payment,
       Privacy).  It cannot be right on these; the table shows what it does instead.
    4. Rule-layer flags on every Indian clause (indian_rule_flags.csv, for review).

HONEST LIMITS
    A test set of a few dozen clauses gives wide intervals, and all numbers come from
    ONE checkpoint.  Differences of a few points are noise.  The point of this test is
    the size of the gap to the CUAD numbers (domain shift), not a leaderboard.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import metrics as M
from build_indian_testset import NO_TRAINING_DATA
from dataset import (CATEGORIES, ClauseDataset, attach_contract_text, load_mapped, make_collator)
from rules_engine import apply_rules, load_rules
from scan_vs_span import load_model
from train_phase3 import choose_amp_dtype, classification_baselines, evaluate, span_baselines


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", required=True, help="checkpoint used for category AND span (e.g. the joint model)")
    ap.add_argument("--ckpt_span", default=None, help="optional different checkpoint for the span head")
    ap.add_argument("--indian_dir", required=True)
    ap.add_argument("--mapped_jsonl", default=None, help="CUAD clauses_mapped.jsonl (for baselines)")
    ap.add_argument("--cuad_json", default=None)
    ap.add_argument("--rules", default=None)
    ap.add_argument("--clause_batch", type=int, default=32)
    ap.add_argument("--span_batch", type=int, default=4)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)

    run, ind = Path(a.run_dir), Path(a.indian_dir)
    cfg = json.loads((run / "partial" / "config.json").read_text())
    cfg.setdefault("max_len_clause", 512)
    if a.tiny:
        cfg["tiny"] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = choose_amp_dtype(device, "none" if a.tiny else "auto")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(run / "tiny_tokenizer") if a.tiny else cfg["encoder"])
    pad_mult = 32 if a.tiny else 512

    # ---- Indian data ---------------------------------------------------------------
    ex_map = load_mapped(ind / "indian_mapped.jsonl")
    texts = json.loads((ind / "indian_contracts.json").read_text(encoding="utf-8"))
    attach_contract_text(ex_map, texts, strict=True)
    everything = list(ex_map.values())
    seen = [e for e in everything if e.category not in NO_TRAINING_DATA]
    unseen = [e for e in everything if e.category in NO_TRAINING_DATA]
    print(f"{len(everything)} Indian clauses: {len(seen)} in categories the model was trained on, "
          f"{len(unseen)} in categories it never saw")

    # ---- CUAD training split for baselines -------------------------------------------
    train_ex = None
    if a.mapped_jsonl and a.cuad_json:
        from dataset import load_contract_texts
        cmap = load_mapped(a.mapped_jsonl)
        attach_contract_text(cmap, load_contract_texts(a.cuad_json), strict=True)
        pool = [e for e in cmap.values() if e.has_cls]
        if cfg.get("max_contracts"):
            raise SystemExit("baselines need the full CUAD pool; rerun without --max_contracts runs")
        c2i = {e.clause_id: i for i, e in enumerate(pool)}
        sp = run / "partial" / "splits.json"
        if sp.exists():
            train_ex = [pool[c2i[c]] for c in json.loads(sp.read_text())[a.fold][0]]
        else:
            print("WARNING: splits.json missing; baselines skipped")

    # ---- model --------------------------------------------------------------------
    m_cls = load_model(a.ckpt, cfg, tok, device)
    m_span = m_cls if not a.ckpt_span else load_model(a.ckpt_span, cfg, tok, device)
    collate = make_collator(tok.pad_token_id, pad_to_multiple_of=pad_mult)

    def loaders_for(examples, views):
        ds = ClauseDataset(examples, tok, max_len_clause=cfg["max_len_clause"], max_len_span=cfg["max_len_span"],
                           span_mode="cuad_context", window_chars=cfg["window_chars"], span_pool=examples,
                           views=views, use_cls=True, use_risk=False, seed=cfg["seed"] + a.fold)
        out = {}
        for v in views:
            ids = [i for i, (vv, _) in enumerate(ds.items) if vv == v]
            if ids:
                out[v] = DataLoader(Subset(ds, ids), batch_size=a.clause_batch if v == "clause" else a.span_batch,
                                    shuffle=False, collate_fn=collate)
        return ds, out

    res: dict = {"n_total": len(everything), "n_seen": len(seen), "n_unseen": len(unseen)}
    train_chars = ([e.char_end - e.char_start for e in train_ex] if train_ex
                   else [e.char_end - e.char_start for e in seen])
    med = float(np.median(train_chars)) if train_chars else None

    if seen:
        _, lo = loaders_for(seen, ("clause", "span"))
        r1 = evaluate(m_cls, {k: v for k, v in lo.items() if k == "clause"}, ("cls",), device, amp_dtype, med)
        r2 = evaluate(m_span, {k: v for k, v in lo.items() if k == "span"}, ("span",), device, amp_dtype, med)
        cls = r1.get("cls")
        res["cls"] = cls
        k = int(round(cls["accuracy"] * cls["n"])) if cls else 0
        res["cls_acc_ci95"] = wilson(k, cls["n"]) if cls else None
        res["span"] = r2.get("span")
        res["span_own_label"] = r2.get("span_own_label")
        res["span_random_baseline"] = r2.get("span_random_baseline")
        if train_ex:
            res["baselines_cls"] = classification_baselines(train_ex, seen)
            tb = res["baselines_cls"].get("tfidf_logreg")
            if tb:
                kt = int(round(tb["accuracy"] * tb["n"]))
                res["tfidf_acc_ci95"] = wilson(kt, tb["n"])
            _, lo2 = loaders_for(seen, ("span",))
            res["baselines_span"] = span_baselines(train_ex, lo2["span"]) if "span" in lo2 else {}

    # ---- what the model does on categories it never saw -------------------------------
    confusion = defaultdict(Counter)
    if unseen:
        _, lu = loaders_for(unseen, ("clause",))
        m_cls.eval()
        with torch.no_grad():
            for batch in lu.get("clause", []):
                b = {k: v.to(device) for k, v in batch.items() if k != "meta"}
                lab = batch["cls_labels"].tolist()
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=device.type == "cuda" and amp_dtype is not None):
                    pred = m_cls(**b)["cls_logits"].argmax(-1).tolist()
                for t, p in zip(lab, pred):
                    confusion[CATEGORIES[t]][CATEGORIES[p]] += 1
    res["unseen_category_predictions"] = {k: dict(v) for k, v in confusion.items()}

    # ---- rule flags ---------------------------------------------------------------------
    if a.rules:
        rules = load_rules(a.rules)
        rows, tally = [], Counter()
        for e in everything:
            for f in apply_rules(e.text, e.category, rules):
                tally[(f.rule_id, f.status)] += 1
                if f.status != "none":
                    ev = "; ".join(t for _, _, t in f.evidence[:4])
                    rows.append([e.clause_id, e.category, f.rule_id, f.status, " | ".join(f.reasons), ev,
                                 f.note, e.text[:300]])
        with open(ind / "indian_rule_flags.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["clause_id", "category", "rule_id", "status", "reasons", "evidence", "note", "clause_text"])
            w.writerows(rows)
        res["rule_status_counts"] = {f"{r}:{s}": n for (r, s), n in sorted(tally.items())}

    # ---- reference numbers from the CUAD run --------------------------------------------
    rj = run / "results.json"
    if rj.exists():
        agg = json.loads(rj.read_text()).get("aggregate", {})
        res["cuad_reference"] = {k: {"cls_macro_f1": (v.get("cls") or {}).get("mean", {}).get("macro_f1"),
                                     "cls_micro_f1": (v.get("cls") or {}).get("mean", {}).get("micro_f1"),
                                     "span_token_f1": (v.get("span") or {}).get("mean", {}).get("token_f1")}
                                 for k, v in agg.items()}

    (ind / "indian_eval.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    # ---- markdown ------------------------------------------------------------------------
    f3 = lambda x: "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
    L = ["# Indian-context evaluation", "",
         f"{res['n_total']} clauses: {res['n_seen']} in categories seen in training, {res['n_unseen']} in unseen categories. "
         "Small sample: read gaps, not decimals.", ""]
    if res.get("cls"):
        c = res["cls"]
        L += ["## Category classification (seen categories only)", "",
              "| Model | n | accuracy | 95% interval | macro-F1 |", "|---|---|---|---|---|",
              f"| ClauseXplain checkpoint | {c['n']} | {f3(c['accuracy'])} | "
              f"{f3(res['cls_acc_ci95'][0])} to {f3(res['cls_acc_ci95'][1])} | {f3(c['macro_f1'])} |"]
        for n, b in res.get("baselines_cls", {}).items():
            ci = res.get("tfidf_acc_ci95") if n == "tfidf_logreg" else None
            L.append(f"| {n} (trained on CUAD fold {a.fold}) | {b['n']} | {f3(b['accuracy'])} | "
                     + (f"{f3(ci[0])} to {f3(ci[1])}" if ci else "") + f" | {f3(b['macro_f1'])} |")
        L.append("")
    if res.get("span"):
        L += ["## Span localisation (windowed)", "", "| Method | n | token-F1 | exact match | IoU |", "|---|---|---|---|---|"]
        for name, m in [("span head (best-match gold)", res["span"]), ("span head (own label)", res.get("span_own_label")),
                        ("random span", res.get("span_random_baseline"))] + list(res.get("baselines_span", {}).items()):
            if m:
                L.append(f"| {name} | {int(m['n'])} | {f3(m['token_f1'])} | {f3(m['exact_match'])} | {f3(m['iou'])} |")
        L.append("")
    if confusion:
        L += ["## Categories the model never saw: what it predicted instead", "", "| True category | predictions |", "|---|---|"]
        for t, ctr in sorted(confusion.items()):
            L.append(f"| {t} | " + ", ".join(f"{p} x{n}" for p, n in ctr.most_common()) + " |")
        L.append("")
    if res.get("rule_status_counts"):
        L += ["## Rule-layer findings (see indian_rule_flags.csv)", ""] + \
             [f"- {k}: {v}" for k, v in res["rule_status_counts"].items()] + [""]
    if res.get("cuad_reference"):
        L += ["## Reference: CUAD cross-validated test numbers (mean over folds)", "",
              "| Model | macro-F1 | micro-F1 | span token-F1 |", "|---|---|---|---|"]
        for k, v in res["cuad_reference"].items():
            L.append(f"| {k} | {f3(v['cls_macro_f1'])} | {f3(v['cls_micro_f1'])} | {f3(v['span_token_f1'])} |")
    (ind / "indian_eval.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return res


if __name__ == "__main__":
    main()
