#!/usr/bin/env python
"""
scan_vs_span.py -- is the span head a better / cheaper localiser than scanning the
window with the category classifier?

    python scan_vs_span.py --run_dir /kaggle/working/cuad_project/stageB --fold 0 \
        --ckpt_span ckpt_joint_cls_span_fold0.pt --ckpt_cls ckpt_joint_cls_span_fold0.pt \
        --mapped_jsonl clauses_mapped.jsonl --cuad_json CUAD_v1.json

THE TWO METHODS (same windows, same gold spans, same scoring as train_phase3.py)
    span head       ONE forward pass over [category question + window] -> start/end.
    classifier scan the window is cut into sentence-aligned candidate spans; EVERY
                    candidate goes through the category classifier; the candidate
                    with the highest probability for the target category wins.
                    This is the obvious non-span-head way to localise with a
                    classifier, and costs one forward pass per candidate.

ALSO REPORTED
    candidate ceiling   best token-F1 any candidate could achieve (upper bound for scan)
    paired bootstrap    95% interval of (span head - scan) token-F1 over windows

WHY NOT LIME?  LIME explains which WORDS drove a classification.  It does not
localise a clause inside a long window, so it answers a different question and
is not a like-for-like baseline here.  If you still want a LIME timing, measure it
separately and describe it as an explanation method, not a localiser.

HONEST LIMITS
    * One fold, a random sample of --n_windows test windows (default 500).
    * Windowed task: the answer is guaranteed to be inside the window.
    * The classifier was trained on whole clauses, so fragments score imperfectly;
      that is part of why scanning is a weak localiser, and part of the comparison.
    * Latency is wall-clock on whatever GPU you run on, batch settings included.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import metrics as M
from dataset import (CAT2ID, ClauseDataset, attach_contract_text, char_to_token_span,
                     load_contract_texts, load_mapped, make_splits)
from model import ClauseXplainMTL, MTLConfig
from train_phase3 import (candidate_spans, choose_amp_dtype, tiny_encoder_kwargs)


def load_model(ckpt: str, cfg: dict, tok, device) -> ClauseXplainMTL:
    tiny = None
    if cfg.get("tiny"):
        tiny = tiny_encoder_kwargs(len(tok), max(cfg["max_len_span"], cfg["max_len_clause"]))
    model = ClauseXplainMTL(MTLConfig(encoder_name=cfg["encoder"], active_tasks=("cls", "span"),
                                      tiny_encoder=tiny))
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in list(missing) + list(unexpected) if not k.startswith("log_vars")]
    if bad:
        raise RuntimeError(f"checkpoint {ckpt} does not match the model: {bad[:5]}")
    return model.to(device).eval()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def span_head_predict(model, tok, item_batches, device, amp_dtype):
    preds, secs = [], 0.0
    use = device.type == "cuda" and amp_dtype is not None
    for batch in item_batches:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k != "meta"}
        sync(device); t = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use):
            out = model(**b)
        sync(device); secs += time.perf_counter() - t
        sl = out["start_logits"].float().cpu().numpy()
        el = out["end_logits"].float().cpu().numpy()
        for i, meta in enumerate(batch["meta"]):
            sids = meta["seq_ids"]
            mask = np.array([x == 1 for x in sids])
            preds.append(M.decode_span(sl[i, :len(sids)], el[i, :len(sids)], mask))
    return preds, secs


@torch.no_grad()
def scan_predict(model, tok, metas, cats, med_chars, device, amp_dtype, max_cands, max_len_clause,
                 pad_multiple, batch_size):
    from dataset import make_collator
    collate = make_collator(tok.pad_token_id, pad_to_multiple_of=pad_multiple)
    use = device.type == "cuda" and amp_dtype is not None
    preds, ceilings, n_cands, secs = [], [], [], 0.0
    for meta, cat in zip(metas, cats):
        ctx = meta["context"]
        cands = candidate_spans(ctx, med_chars, max_cands)
        n_cands.append(len(cands))
        items = []
        for a, b in cands:
            enc = tok(ctx[a:b], truncation=True, max_length=max_len_clause)
            ids = enc["input_ids"]
            items.append({"input_ids": ids, "global_attention_mask": [1] + [0] * (len(ids) - 1),
                          "cls_label": -100, "risk_levels": [-100] * 4, "start_pos": -100,
                          "end_pos": -100, "meta": {}})
        scores = []
        sync(device); t = time.perf_counter()
        for s in range(0, len(items), batch_size):
            batch = collate(items[s:s + batch_size])
            b = {k: v.to(device) for k, v in batch.items() if k != "meta"}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use):
                out = model(**b)
            scores += out["cls_logits"].float().softmax(-1)[:, CAT2ID[cat]].cpu().tolist()
        sync(device); secs += time.perf_counter() - t
        order = int(np.argmax(scores))
        spans = [char_to_token_span(meta["offsets"], meta["seq_ids"], a, b) or (1, 0) for a, b in cands]
        preds.append(spans[order])
        ceilings.append(spans)
    return preds, ceilings, n_cands, secs


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="the train_phase3.py --out_dir (has partial/config.json)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt_span", required=True)
    ap.add_argument("--ckpt_cls", required=True)
    ap.add_argument("--mapped_jsonl", default="clauses_mapped.jsonl")
    ap.add_argument("--cuad_json", default="CUAD_v1.json")
    ap.add_argument("--n_windows", type=int, default=500)
    ap.add_argument("--max_candidates", type=int, default=60)
    ap.add_argument("--scan_batch", type=int, default=32)
    ap.add_argument("--span_batch", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)

    run = Path(a.run_dir)
    cfg = json.loads((run / "partial" / "config.json").read_text())
    cfg.setdefault("max_len_clause", 512)
    if a.tiny:
        cfg["tiny"] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = choose_amp_dtype(device, "none" if a.tiny else "auto")

    # ---- data: exactly the examples / split / windows used in training -------------
    ex_map = load_mapped(a.mapped_jsonl)
    attach_contract_text(ex_map, load_contract_texts(a.cuad_json), strict=True)
    examples = [e for e in ex_map.values() if e.has_cls or e.has_risk]
    if cfg.get("max_contracts"):
        keep = sorted({e.contract_id for e in examples})
        random.Random(cfg["seed"]).shuffle(keep)
        keep = set(keep[: cfg["max_contracts"]])
        examples = [e for e in examples if e.contract_id in keep]
    sp = run / "partial" / "splits.json"
    if sp.exists():
        c2i = {e.clause_id: i for i, e in enumerate(examples)}
        split = [[c2i[c] for c in part] for part in json.loads(sp.read_text())[a.fold]]
    else:
        print("WARNING: partial/splits.json missing; regenerating the split. Check that the fold sizes "
              "match your training log.")
        split = [list(map(int, x)) for x in make_splits(examples, cfg["n_parts"], cfg["val_frac"],
                                                         cfg["test_frac"], cfg["seed"])[a.fold]]
    train_idx, _, test_idx = split
    print(f"fold {a.fold}: train {len(train_idx)}, test {len(test_idx)}")

    if a.tiny:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(run / "tiny_tokenizer"))
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["encoder"])

    ds = ClauseDataset([examples[i] for i in test_idx], tok, max_len_clause=cfg["max_len_clause"],
                       max_len_span=cfg["max_len_span"], span_mode=cfg["span_mode"],
                       window_chars=cfg["window_chars"], span_pool=examples, views=("span",),
                       use_cls=False, use_risk=False, seed=cfg["seed"] + a.fold)
    ids = list(range(len(ds)))
    random.Random(0).shuffle(ids)
    items = []
    for i in ids:
        it = ds[i]
        if it["meta"]["gold_tokens"] and it["meta"].get("category") in CAT2ID:
            items.append(it)
        if len(items) >= a.n_windows:
            break
    print(f"{len(items)} windows")
    train_chars = [examples[i].char_end - examples[i].char_start for i in train_idx
                   if examples[i].char_start is not None]
    med = float(np.median(train_chars))

    from dataset import make_collator
    collate = make_collator(tok.pad_token_id, pad_to_multiple_of=32 if a.tiny else 512)
    batches = [collate(items[s:s + a.span_batch]) for s in range(0, len(items), a.span_batch)]

    m_span = load_model(a.ckpt_span, cfg, tok, device)
    m_cls = m_span if a.ckpt_cls == a.ckpt_span else load_model(a.ckpt_cls, cfg, tok, device)

    p_span, t_span = span_head_predict(m_span, tok, batches, device, amp_dtype)
    metas = [it["meta"] for it in items]
    cats = [it["meta"]["category"] for it in items]
    p_scan, cand_spans, n_cands, t_scan = scan_predict(
        m_cls, tok, metas, cats, med, device, amp_dtype, a.max_candidates, cfg["max_len_clause"],
        32 if a.tiny else 512, a.scan_batch)

    golds = [it["meta"]["gold_tokens"] for it in items]
    best = lambda p, gs: max(gs, key=lambda g: M._token_f1(p, g))
    gold_best_span = [best(p, g) for p, g in zip(p_span, golds)]
    gold_best_scan = [best(p, g) for p, g in zip(p_scan, golds)]
    f1_span = np.array([M._token_f1(p, g) for p, g in zip(p_span, gold_best_span)])
    f1_scan = np.array([M._token_f1(p, g) for p, g in zip(p_scan, gold_best_scan)])
    f1_ceiling = np.array([max(max(M._token_f1(c, g) for g in gs) for c in cs)
                           for cs, gs in zip(cand_spans, golds)])
    rng = np.random.default_rng(0)
    diff = f1_span - f1_scan
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(a.n_boot)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    n = len(items)
    res = {
        "n_windows": n, "fold": a.fold,
        "span_head": {**M.span_metrics(p_span, gold_best_span), "forward_passes_per_window": 1.0,
                      "ms_per_window": 1000 * t_span / n},
        "classifier_scan": {**M.span_metrics(p_scan, gold_best_scan),
                            "forward_passes_per_window": float(np.mean(n_cands)),
                            "ms_per_window": 1000 * t_scan / n},
        "candidate_ceiling_token_f1": float(f1_ceiling.mean()),
        "span_minus_scan_token_f1": {"mean": float(diff.mean()), "ci95": ci},
        "settings": {k: v for k, v in vars(a).items()},
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    out = run / f"scan_vs_span_fold{a.fold}"
    (Path(str(out) + ".json")).write_text(json.dumps(res, indent=1, default=float))
    sh, sc = res["span_head"], res["classifier_scan"]
    md = [f"# Span head vs classifier scan (fold {a.fold}, {n} windows)", "",
          "| Method | forward passes / window | ms / window | token-F1 | exact match | IoU |",
          "|---|---|---|---|---|---|",
          f"| span head | 1 | {sh['ms_per_window']:.0f} | {sh['token_f1']:.3f} | {sh['exact_match']:.3f} | {sh['iou']:.3f} |",
          f"| classifier scan | {sc['forward_passes_per_window']:.0f} | {sc['ms_per_window']:.0f} | "
          f"{sc['token_f1']:.3f} | {sc['exact_match']:.3f} | {sc['iou']:.3f} |", "",
          f"Candidate ceiling (best any candidate could score): token-F1 {res['candidate_ceiling_token_f1']:.3f}.",
          f"Span head minus scan, token-F1: {diff.mean():+.3f} (95% bootstrap interval {ci[0]:+.3f} to {ci[1]:+.3f}).",
          "", "One fold, a random sample of test windows, windowed task. Speed is wall-clock on this hardware."]
    Path(str(out) + ".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return res


if __name__ == "__main__":
    main()
