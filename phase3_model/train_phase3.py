#!/usr/bin/env python
"""
train_phase3.py -- ClauseXplain Phase 3: single-task vs joint multi-task training.

WHAT THIS SCRIPT DOES
    Trains the Longformer in model.py on real CUAD supervision and compares
        cls_only        category classifier alone
        span_only       span localiser alone
        joint_cls_span  ONE encoder, both heads (the "Option 1" model)
    plus, ONLY if you supply risk labels:  risk_only, joint_all.
    Evaluation is contract-grouped, category-stratified, optionally K-fold.
    Every run also reports trivial baselines (majority, TF-IDF+LogReg for
    classification; random same-length span for localisation).

WHAT IT DELIBERATELY DOES NOT DO (read before you quote any number)
    * No AI-generated risk labels are used by default.  Risk experiments run
      only with --risk_jsonl AND --risk_label_source, and that source is
      printed in results.md.  "ai_silver" results are never legal ground truth.
    * The span task is WINDOWED LOCALISATION: the model gets a ~6000-character
      window that is guaranteed to contain the answer and must find it.  That is
      easier than CUAD's full-contract search.  Do not compare it with
      published full-contract CUAD numbers.
    * Classification labels come from your 41->18 mapping of CUAD categories,
      so a good score partly measures how learnable the mapping is.  The
      TF-IDF baseline shows how much of that is "easy".
    * "Joint" here means one shared encoder updated by alternating clause-view
      and span-view batches.  Each batch only trains the heads it has labels for.
    * Six of the 18 categories have no CUAD data and are reported as n/a.
    * There is no pretrained-model weight download inside the authoring
      sandbox, so numbers from --tiny are pipeline checks, not results.

TYPICAL USAGE
    # pipeline check on CPU (random-init tiny model, ~minutes)
    python train_phase3.py --tiny --max_contracts 40 --epochs 1

    # real run on a GPU (Colab/Kaggle T4 is enough)
    python train_phase3.py --cuad_json CUAD_v1.json --mapped_jsonl clauses_mapped.jsonl \
        --experiments cls_only span_only joint_cls_span --cv_folds 5 --epochs 3
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import metrics as M
from dataset import (CAT2ID, CATEGORIES, RISK_DIMS, ClauseDataset, Example,
                     attach_contract_text, attach_risk_annotations,
                     attach_weak_labels, char_to_token_span, load_contract_texts,
                     load_mapped, make_collator, make_splits, split_report)
from model import IGNORE, ClauseXplainMTL, MTLConfig

def n_parts(cv_folds: int, test_frac: float) -> int:
    """Number of partitions the data is cut into. --cv_folds 1 uses the FIRST of
    round(1/test_frac) partitions, which is exactly fold 0 of --cv_folds 5, so a
    single-split run can be extended to 5 folds with --resume."""
    return cv_folds if cv_folds and cv_folds > 1 else max(2, round(1 / test_frac))


RESUME_KEYS = ["encoder", "epochs", "lr", "head_lr", "seed", "test_frac", "val_frac",
               "max_contracts", "span_mode", "window_chars", "max_len_span", "max_len_clause",
               "bs_clause", "bs_span", "grad_accum", "class_weights", "loss_weighting",
               "span_train_frac", "tiny"]

EXPERIMENTS: Dict[str, dict] = {
    "cls_only":       dict(tasks=("cls",),              views=("clause",)),
    "span_only":      dict(tasks=("span",),             views=("span",)),
    "joint_cls_span": dict(tasks=("cls", "span"),       views=("clause", "span")),
    "risk_only":      dict(tasks=("risk",),             views=("clause",), needs_risk=True),
    "joint_all":      dict(tasks=("cls", "risk", "span"), views=("clause", "span"), needs_risk=True),
}


# ----------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_tiny_tokenizer(texts: Sequence[str], vocab_size: int = 4000):
    """Offline byte-level BPE tokenizer with RoBERTa-style specials (ids 0..3 =
    <s> <pad> </s> <unk>), so the tiny random-init Longformer and the pair-encoding
    / offset logic can be exercised without downloading anything."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                  special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(texts, trainer)
    tk.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0), True, False)
    tk.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>",
                                   sep_token="</s>", cls_token="<s>", pad_token="<pad>",
                                   unk_token="<unk>")


def tiny_encoder_kwargs(vocab_size: int, max_len: int, window: int = 32) -> dict:
    return dict(vocab_size=vocab_size, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=128,
                attention_window=[window, window], max_position_embeddings=max_len + window + 4,
                pad_token_id=1, bos_token_id=0, eos_token_id=2, sep_token_id=2)


def choose_amp_dtype(device, requested: str = "auto", capability=None):
    """bf16 only on Ampere or newer (compute capability >= 8). A T4 is 7.5: PyTorch
    reports bf16 as 'supported' there but EMULATES it, which is roughly an order of
    magnitude slower than fp16. fp16 (with GradScaler) is the right choice on a T4."""
    if device.type != "cuda":
        return None
    cap = capability if capability is not None else torch.cuda.get_device_capability()
    if requested == "auto":
        return torch.bfloat16 if cap[0] >= 8 else torch.float16
    dt = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[requested]
    if dt is torch.bfloat16 and cap[0] < 8:
        print(f"WARNING: bf16 requested on compute capability {cap}; it is emulated and very slow here.")
    return dt


def interleave(loaders: Dict[str, DataLoader], rng: random.Random):
    """Yield (view, batch) with views mixed in proportion to their remaining batches."""
    its = {v: iter(l) for v, l in loaders.items()}
    left = {v: len(l) for v, l in loaders.items()}
    while any(left.values()):
        views = [v for v, n in left.items() if n > 0]
        v = rng.choices(views, weights=[left[x] for x in views])[0]
        left[v] -= 1
        yield v, next(its[v])


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def model_inputs(batch: dict) -> dict:
    return {k: v for k, v in batch.items() if k != "meta"}


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: ClauseXplainMTL, loaders: Dict[str, DataLoader], tasks: Sequence[str],
             device, amp_dtype, train_span_chars: Optional[float] = None,
             rng_seed: int = 0, max_items: Optional[int] = None) -> dict:
    model.eval()
    res: dict = {"latency_ms_per_item": {}}
    use_cuda = device.type == "cuda"

    def timed(fn):
        if use_cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        if use_cuda:
            torch.cuda.synchronize()
        return out, time.perf_counter() - t

    # ---- clause view: category + risk
    if "clause" in loaders:
        yt, yp, rt, rp = [], [], [], []
        n_items, t_sum = 0, 0.0
        for batch in loaders["clause"]:
            b = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None and use_cuda):
                out, dt = timed(lambda: model(**model_inputs(b)))
            t_sum += dt
            n_items += b["input_ids"].shape[0]
            if "cls_logits" in out:
                lab = b["cls_labels"]
                ok = lab != IGNORE
                yt += lab[ok].tolist()
                yp += out["cls_logits"].argmax(-1)[ok].tolist()
            if "risk_scores" in out:
                lv = b["risk_levels"]
                ok = (lv != IGNORE).all(dim=1)
                if ok.any():
                    rt += lv[ok].tolist()
                    rp += out["risk_scores"][ok].float().cpu().tolist()
        res["latency_ms_per_item"]["clause"] = 1000 * t_sum / max(1, n_items)
        if "cls" in tasks and yt:
            res["cls"] = M.classification_metrics(yt, yp)
        if "risk" in tasks and rt:
            res["risk"] = {"scores": np.asarray(rp).tolist(), "levels": rt}  # finalised by caller

    # ---- span view
    if "span" in tasks and "span" in loaders:
        rng = random.Random(rng_seed)
        pred_any, gold_any, pred_own, gold_own, rb_pred, rb_gold = [], [], [], [], [], []
        n_items, t_sum, n_trunc = 0, 0.0, 0
        for batch in loaders["span"]:
            b = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None and use_cuda):
                out, dt = timed(lambda: model(**model_inputs(b)))
            t_sum += dt
            n_items += b["input_ids"].shape[0]
            sl = out["start_logits"].float().cpu().numpy()
            el = out["end_logits"].float().cpu().numpy()
            for i, meta in enumerate(batch["meta"]):
                seq_ids = meta["seq_ids"]
                n = len(seq_ids)
                ctx_mask = np.array([sid == 1 for sid in seq_ids])
                if not ctx_mask.any():
                    continue
                pred = M.decode_span(sl[i, :n], el[i, :n], ctx_mask)
                golds = meta["gold_tokens"]
                if not golds:
                    n_trunc += 1
                    continue
                best = max(golds, key=lambda g: M._token_f1(pred, g))
                pred_any.append(pred); gold_any.append(best)
                own = (int(batch["start_positions"][i]), int(batch["end_positions"][i]))
                if own[0] != IGNORE:
                    pred_own.append(pred); gold_own.append(own)
                # baseline: random span of the train-median character length
                if train_span_chars:
                    cl = int(train_span_chars)
                    ctx_len = max(o1 for sid, (o0, o1) in zip(seq_ids, meta["offsets"]) if sid == 1)
                    a = rng.randint(0, max(0, ctx_len - cl))
                    tk = char_to_token_span(meta["offsets"], seq_ids, a, min(ctx_len, a + cl))
                    if tk is not None:
                        rb_pred.append(tk)
                        rb_gold.append(best)
            if max_items and n_items >= max_items:
                break
        res["latency_ms_per_item"]["span"] = 1000 * t_sum / max(1, n_items)
        res["span"] = M.span_metrics(pred_any, gold_any)           # best-matching gold span
        res["span_own_label"] = M.span_metrics(pred_own, gold_own)  # the example's own span only
        res["span"]["n_truncated_no_gold"] = n_trunc
        if rb_pred:
            res["span_random_baseline"] = M.span_metrics(rb_pred, rb_gold)
    return res


def composite(res: dict, tasks: Sequence[str]) -> float:
    """Mean of the headline score of every active task (higher is better)."""
    vals = []
    if "cls" in tasks and "cls" in res:
        vals.append(res["cls"]["macro_f1"])
    if "span" in tasks and "span" in res:
        vals.append(res["span"]["token_f1"])
    if "risk" in tasks and "risk" in res and "avg_mae" in res["risk"]:
        vals.append(1.0 - res["risk"]["avg_mae"])
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def finalize_risk(res: dict, train_levels: Optional[np.ndarray]) -> None:
    if "risk" in res and "scores" in res["risk"]:
        r = res["risk"]
        res["risk"] = M.risk_metrics(np.asarray(r["levels"]), np.asarray(r["scores"]), train_levels)


# ----------------------------------------------------------------------------
# baselines (no neural net)
# ----------------------------------------------------------------------------
def classification_baselines(train: Sequence[Example], test: Sequence[Example]) -> Dict[str, dict]:
    tr = [e for e in train if e.has_cls]
    te = [e for e in test if e.has_cls]
    if not tr or not te:
        return {}
    y_tr = [CAT2ID[e.category] for e in tr]
    y_te = [CAT2ID[e.category] for e in te]
    out = {}
    maj = Counter(y_tr).most_common(1)[0][0]
    out["majority"] = M.classification_metrics(y_te, [maj] * len(y_te))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform([e.text for e in tr])
    clf = LogisticRegression(max_iter=1000, C=10.0).fit(Xtr, y_tr)
    out["tfidf_logreg"] = M.classification_metrics(y_te, clf.predict(vec.transform([e.text for e in te])))
    return out


def candidate_spans(ctx: str, med: float, max_candidates: int = 400) -> List[Tuple[int, int]]:
    """
    Sentence-aligned candidate character spans of a window: single segments first,
    then merges of 2, 3, ... consecutive segments (total length <= 2x the typical
    clause length, >= 0.4x for merges).  Shared by the lexical baseline and by
    scan_vs_span.py so both use exactly the same candidate set.
    """
    import re
    cuts = sorted(set([0] + [m.end() for m in re.finditer(r"(?<=[.;:])\s+|\n+", ctx)] + [len(ctx)]))
    segs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if ctx[a:b].strip()]
    out: List[Tuple[int, int]] = list(segs)
    for k in range(1, len(segs)):                       # k = extra segments merged
        added = False
        for i in range(len(segs) - k):
            a, b = segs[i][0], segs[i + k][1]
            if b - a > 2 * med:
                continue
            if b - a >= 0.4 * med:
                out.append((a, b)); added = True
        if not added and k > 1:
            break
        if len(out) >= max_candidates:
            break
    return out[:max_candidates] or [(0, len(ctx))]


def span_baselines(train: Sequence[Example], loader) -> Dict[str, dict]:
    """
    Non-neural span baselines, scored exactly like the model (best-match gold):
      whole_window_baseline  : predict the entire window
      lexical_centroid_baseline : among sentence-aligned candidate spans of roughly
          the typical length, pick the one most similar (TF-IDF cosine) to the
          centroid of the training clauses of that category.
    A neural span head should beat BOTH before its score means much.
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr = [e for e in train if e.has_cls and e.char_start is not None]
    if not tr:
        return {}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    vec.fit([e.text for e in tr])
    cent = {}
    for c in {e.category for e in tr}:
        v = np.asarray(vec.transform([e.text for e in tr if e.category == c]).mean(axis=0)).ravel()
        n = np.linalg.norm(v)
        cent[c] = v / n if n else v
    med = float(np.median([e.char_end - e.char_start for e in tr]))
    p_lex, p_all, golds = [], [], []
    for batch in loader:
        for meta in batch["meta"]:
            g = meta["gold_tokens"]
            if not g or meta.get("category") not in cent:
                continue
            ctx, offs, sids = meta["context"], meta["offsets"], meta["seq_ids"]
            ctx_tok = [t for t, sid in enumerate(sids) if sid == 1]
            whole = (ctx_tok[0], ctx_tok[-1])
            cands = candidate_spans(ctx, med, 400)
            X = vec.transform([ctx[a:b] for a, b in cands])
            best = cands[int(np.argmax(X.dot(cent[meta["category"]])))]
            tk = char_to_token_span(offs, sids, best[0], best[1]) or (1, 0)
            p_lex.append(tk); p_all.append(whole); golds.append(g)
    def score(preds):
        bests = [max(gs, key=lambda x: M._token_f1(p, x)) for p, gs in zip(preds, golds)]
        return M.span_metrics(preds, bests)
    return {"whole_window_baseline": score(p_all), "lexical_centroid_baseline": score(p_lex)}


# ----------------------------------------------------------------------------
# one training run
# ----------------------------------------------------------------------------
def make_loaders(examples: Sequence[Example], idx: Sequence[int], tok, exp: dict, args,
                 pool: Sequence[Example], shuffle: bool, bs: Dict[str, int], seed: int,
                 span_frac: float = 1.0) -> Tuple[ClauseDataset, Dict[str, DataLoader]]:
    sub = [examples[i] for i in idx]
    ds = ClauseDataset(sub, tok, max_len_clause=args.max_len_clause, max_len_span=args.max_len_span,
                       span_mode=args.span_mode, window_chars=args.window_chars, span_pool=pool,
                       views=exp["views"], use_cls="cls" in exp["tasks"],
                       use_risk="risk" in exp["tasks"], seed=seed)
    collate = make_collator(tok.pad_token_id, pad_to_multiple_of=args.pad_multiple)
    loaders = {}
    for view in exp["views"]:
        ids = [i for i, (v, _) in enumerate(ds.items) if v == view]
        if not ids:
            continue
        if view == "span" and span_frac < 1.0:      # compute saver (train split only)
            ids = random.Random(seed).sample(ids, max(1, int(len(ids) * span_frac)))
        g = torch.Generator().manual_seed(seed)
        loaders[view] = DataLoader(Subset(ds, ids), batch_size=bs[view], shuffle=shuffle,
                                   collate_fn=collate, num_workers=args.num_workers, generator=g)
    return ds, loaders


def run_one(exp_name: str, exp: dict, fold: int, examples: List[Example], split, tok, args,
            device, amp_dtype, tiny_kwargs: Optional[dict]) -> dict:
    tr_idx, va_idx, te_idx = split
    seed = args.seed + fold
    set_seed(seed)
    bs = {"clause": args.bs_clause, "span": args.bs_span}
    ev_bs = {"clause": args.bs_clause * 2, "span": max(1, args.bs_span * 2)}
    tasks = exp["tasks"]

    _, tr_loaders = make_loaders(examples, tr_idx, tok, exp, args, examples, True, bs, seed,
                                    span_frac=args.span_train_frac)
    _, va_loaders = make_loaders(examples, va_idx, tok, exp, args, examples, False, ev_bs, seed)
    _, te_loaders = make_loaders(examples, te_idx, tok, exp, args, examples, False, ev_bs, seed)
    if not tr_loaders:
        raise RuntimeError(f"{exp_name}: no training items (missing labels or contract text?)")

    cw = None
    if args.class_weights == "sqrt_inv" and "cls" in tasks:
        cnt = Counter(CAT2ID[examples[i].category] for i in tr_idx if examples[i].has_cls)
        w = np.array([1 / math.sqrt(cnt[c]) if cnt[c] else 0.0 for c in range(len(CATEGORIES))])
        cw = (w / w[w > 0].mean()).tolist()
    cfg = MTLConfig(encoder_name=args.encoder, active_tasks=tasks, risk_head=args.risk_head,
                    loss_weighting=args.loss_weighting, w_cls=args.w_cls, w_risk=args.w_risk,
                    w_span=args.w_span, cls_class_weights=cw, tiny_encoder=tiny_kwargs,
                    gradient_checkpointing=args.grad_ckpt)
    model = ClauseXplainMTL(cfg).to(device)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.head_parameters())
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": head_params, "lr": args.head_lr}],
                            weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(sum(len(l) for l in tr_loaders.values()) / args.grad_accum)
    total = max(1, steps_per_epoch * args.epochs)
    warm = int(args.warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16 and use_cuda))

    train_span_chars = None
    if "span" in tasks:
        L = [e.char_end - e.char_start for e in (examples[i] for i in tr_idx)
             if e.char_start is not None and e.char_end is not None]
        train_span_chars = float(np.median(L)) if L else None
    train_levels = None
    if "risk" in tasks:
        tl = [examples[i].risk_levels for i in tr_idx if examples[i].has_risk]
        train_levels = np.asarray(tl) if tl else None

    best, best_state, bad, history = -1e9, None, 0, []
    rng = random.Random(seed)
    t0 = time.time()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    for ep in range(args.epochs):
        model.train()
        run_loss, n_loss, step_in_acc = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for view, batch in interleave(tr_loaders, rng):
            b = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None and use_cuda):
                out = model(**model_inputs(b))
            if "loss" not in out:          # batch without usable labels (e.g. truncated span)
                continue
            loss = out["loss"] / args.grad_accum
            scaler.scale(loss).backward()
            run_loss += float(out["loss"].detach()); n_loss += 1; step_in_acc += 1
            if step_in_acc % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
        # flush a partial accumulation at epoch end
        if step_in_acc % args.grad_accum != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(opt); scaler.update(); sched.step()
            opt.zero_grad(set_to_none=True)

        va = evaluate(model, va_loaders, tasks, device, amp_dtype, train_span_chars, seed,
                      args.val_max_items)
        finalize_risk(va, train_levels)
        score = composite(va, tasks)
        history.append({"epoch": ep + 1, "train_loss": run_loss / max(1, n_loss), "val_composite": score})
        print(f"  [{exp_name} fold{fold}] epoch {ep + 1}/{args.epochs} "
              f"loss={run_loss / max(1, n_loss):.4f} val_composite={score:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if ep == 0:
            print(f"  -> ~{(time.time() - t0) * args.epochs / 60:.0f} min for this run "
                  f"(upper bound; early stopping may cut it)", flush=True)
        if not math.isnan(score) and score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > args.patience:
                print("  early stop", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if args.save_ckpt and best_state is not None:
        torch.save(best_state, Path(args.out_dir) / f"ckpt_{exp_name}_fold{fold}.pt")
    te = evaluate(model, te_loaders, tasks, device, amp_dtype, train_span_chars, seed)
    finalize_risk(te, train_levels)
    te["best_val_composite"] = best
    te["history"] = history
    te["n_params"] = int(sum(p.numel() for p in model.parameters()))
    te["train_seconds"] = time.time() - t0
    te["peak_gpu_mem_gb"] = (torch.cuda.max_memory_allocated() / 1e9) if use_cuda else None
    del model, opt
    if use_cuda:
        torch.cuda.empty_cache()
    return te


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def aggregate_folds(per_fold: List[dict], key: str):
    ds = [d[key] for d in per_fold if key in d]
    if not ds:
        return None
    mean, std = M.aggregate(ds)
    if key == "cls":     # support should be summed over folds, not averaged
        for c in CATEGORIES:
            mean["per_class"][c]["support"] = int(sum(d["per_class"][c]["support"] for d in ds))
    return {"mean": mean, "std": std}


def write_report(out_dir: Path, results: dict, args, notes: List[str]) -> None:
    L = ["# ClauseXplain Phase 3 results", ""]
    L += [f"*Run: {results['meta']['timestamp']}  |  device: {results['meta']['device']}  |  "
          f"encoder: {results['meta']['encoder']}  |  folds: {results['meta']['n_folds']}*", ""]
    if args.tiny:
        L += ["> **TINY RANDOM-INIT PIPELINE TEST. These numbers are meaningless.**", ""]
    L += ["## Read this first"] + [f"- {n}" for n in notes] + [""]

    cls_tab = {}
    for name, b in results["baselines_cls"].items():
        cls_tab[f"{name}"] = b["mean"]
    for exp, r in results["aggregate"].items():
        if r.get("cls"):
            cls_tab[exp] = r["cls"]["mean"]
    if cls_tab:
        L += ["## Category classification (mean over folds; support = total over folds)", "",
              M.classification_table(cls_tab), ""]
        if results["meta"]["n_folds"] > 1:
            L += ["Std of macro-F1 across folds: " + ", ".join(
                f"{n}={r['cls']['std']['macro_f1']:.3f}" for n, r in results['aggregate'].items()
                if r.get('cls')), ""]
        else:
            L += ["Only one fold: no variance estimate, so differences of a few points are not evidence.", ""]
    span_tab = {}
    for exp, r in results["aggregate"].items():
        if r.get("span"):
            span_tab[f"{exp} (best-match gold)"] = r["span"]["mean"]
        if r.get("span_own_label"):
            span_tab[f"{exp} (own label only)"] = r["span_own_label"]["mean"]
        if r.get("span_random_baseline") and "random-span baseline" not in span_tab:
            span_tab["random-span baseline"] = r["span_random_baseline"]["mean"]
    for bn, b in results.get("baselines_span", {}).items():
        span_tab[bn] = b["mean"]
    if span_tab:
        L += ["## Span localisation (windowed; answer guaranteed inside window)", "",
              M.span_table(span_tab), ""]
    risk_tab = {e: r["risk"]["mean"] for e, r in results["aggregate"].items() if r.get("risk")}
    if risk_tab:
        L += [f"## Risk (label source: **{args.risk_label_source}**)", "", M.risk_table(risk_tab), ""]
    lat = {e: r["latency_ms_per_item"] for e, r in results["aggregate"].items() if r.get("latency_ms_per_item")}
    if lat:
        L += ["## Inference latency, ms per item (batch eval, this hardware)", "",
              "| experiment | clause view | span view |", "|---|---|---|"]
        for e, d in lat.items():
            L.append(f"| {e} | {d.get('clause', {}).get('mean', float('nan')):.1f} | "
                     f"{d.get('span', {}).get('mean', float('nan')):.1f} |")
        L.append("")
    (out_dir / "results.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--mapped_jsonl", default="clauses_mapped.jsonl")
    g.add_argument("--cuad_json", default="CUAD_v1.json")
    g.add_argument("--out_dir", default="runs/phase3")
    g.add_argument("--risk_jsonl", default=None, help="OPTIONAL risk labels (default: none)")
    g.add_argument("--risk_label_source", default=None,
                   choices=["human", "checklist_rules", "ai_silver"],
                   help="required with --risk_jsonl; printed in the report")
    g.add_argument("--weak_labels_jsonl", default=None, help="optional rule/LEDGAR labels for zero-coverage categories")
    g.add_argument("--max_contracts", type=int, default=None, help="subsample contracts (debugging)")
    g.add_argument("--span_mode", default="cuad_context", choices=["cuad_context", "synthetic"])
    g.add_argument("--window_chars", type=int, default=6000)
    g = p.add_argument_group("experiments")
    g.add_argument("--experiments", nargs="+", default=["cls_only", "span_only", "joint_cls_span"],
                   choices=list(EXPERIMENTS))
    g.add_argument("--cv_folds", type=int, default=1, help="1 = a single split")
    g.add_argument("--val_frac", type=float, default=0.1)
    g.add_argument("--folds", type=int, nargs="+", default=None,
                   help="only run these fold indices (e.g. --folds 1 2); default: all")
    g.add_argument("--test_frac", type=float, default=0.2)
    g.add_argument("--skip_baselines", action="store_true")
    g = p.add_argument_group("model / optimisation")
    g.add_argument("--encoder", default="allenai/longformer-base-4096")
    g.add_argument("--tiny", action="store_true", help="random-init tiny model + offline tokenizer (pipeline test only)")
    g.add_argument("--risk_head", default="ordinal", choices=["ordinal", "regression"])
    g.add_argument("--loss_weighting", default="fixed", choices=["fixed", "uncertainty"])
    g.add_argument("--w_cls", type=float, default=1.0)
    g.add_argument("--w_risk", type=float, default=1.0)
    g.add_argument("--w_span", type=float, default=1.0)
    g.add_argument("--class_weights", default="none", choices=["none", "sqrt_inv"])
    g.add_argument("--epochs", type=int, default=3)
    g.add_argument("--patience", type=int, default=1)
    g.add_argument("--lr", type=float, default=2e-5)
    g.add_argument("--head_lr", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--warmup_frac", type=float, default=0.1)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--bs_clause", type=int, default=16)
    g.add_argument("--bs_span", type=int, default=2)
    g.add_argument("--grad_accum", type=int, default=1)
    g.add_argument("--max_len_clause", type=int, default=512)
    g.add_argument("--max_len_span", type=int, default=2048)
    g.add_argument("--pad_multiple", type=int, default=512)
    g.add_argument("--amp", default="auto", choices=["auto", "none", "bf16", "fp16"])
    g.add_argument("--grad_ckpt", action="store_true")
    g.add_argument("--span_train_frac", type=float, default=1.0,
                   help="train on this random fraction of span items (compute saver; stated in report)")
    g.add_argument("--resume", action="store_true",
                   help="skip (experiment, fold) runs already saved in <out_dir>/partial")
    g.add_argument("--val_max_items", type=int, default=None)
    g.add_argument("--num_workers", type=int, default=0)
    g.add_argument("--save_ckpt", action="store_true")
    g.add_argument("--seed", type=int, default=13)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = choose_amp_dtype(device, args.amp)
    print(f"device={device} amp={amp_dtype}", flush=True)
    if device.type == "cpu" and not args.tiny:
        print("WARNING: no GPU found. Longformer-base on CPU will take days. In Colab: "
              "Runtime > Change runtime type > T4 GPU, then restart.", flush=True)

    # ---- guards: refuse to produce misleading numbers silently ------------
    for name in args.experiments:
        if EXPERIMENTS[name].get("needs_risk") and not args.risk_jsonl:
            raise SystemExit(f"experiment '{name}' needs --risk_jsonl (+ --risk_label_source)")
    if args.risk_jsonl and not args.risk_label_source:
        raise SystemExit("--risk_jsonl requires --risk_label_source {human,checklist_rules,ai_silver}")
    if args.span_mode == "synthetic":
        print("WARNING: span_mode=synthetic is a pipeline proxy, NOT a CUAD localisation result.")

    # ---- data -------------------------------------------------------------
    ex_map = load_mapped(args.mapped_jsonl)
    if args.span_mode == "cuad_context":
        stats = attach_contract_text(ex_map, load_contract_texts(args.cuad_json), strict=True)
        print(f"contract text attached + offsets verified: {stats}", flush=True)
    if args.weak_labels_jsonl:
        print("weak labels added:", attach_weak_labels(ex_map, args.weak_labels_jsonl))
    if args.risk_jsonl:
        print("risk attach:", attach_risk_annotations(ex_map, args.risk_jsonl,
                                                      label_source=args.risk_label_source))
    examples = [e for e in ex_map.values() if e.has_cls or e.has_risk]
    if args.max_contracts:
        keep = sorted({e.contract_id for e in examples})
        random.Random(args.seed).shuffle(keep)
        keep = set(keep[: args.max_contracts])
        examples = [e for e in examples if e.contract_id in keep]
    print(f"{len(examples)} examples from {len({e.contract_id for e in examples})} contracts; "
          f"{Counter(e.category for e in examples).most_common()}", flush=True)

    splits = make_splits(examples, args.cv_folds, args.val_frac, args.test_frac, args.seed)
    # Persist the exact splits: StratifiedGroupKFold output can differ between
    # scikit-learn versions, and Kaggle/Colab sessions may not have the same one.
    (out_dir / "partial").mkdir(exist_ok=True)
    sp = out_dir / "partial" / "splits.json"
    cid2i = {e.clause_id: i for i, e in enumerate(examples)}
    if args.resume and sp.exists():
        try:
            splits = [tuple([cid2i[c] for c in part] for part in fold)
                      for fold in json.loads(sp.read_text())]
        except KeyError as e:
            raise SystemExit(f"--resume: saved splits.json refers to a clause not in the current "
                             f"data ({e}); use the same --max_contracts / data, or a new --out_dir")
        print("splits loaded from partial/splits.json (identical to the original run)", flush=True)
    else:
        sp.write_text(json.dumps([[[examples[i].clause_id for i in part] for part in fold]
                                  for fold in splits]))
    for k, (tr, va, te) in enumerate(splits):
        print(f"fold {k}: " + split_report(examples, {"train": tr, "val": va, "test": te}).replace("\n", " | "))

    tiny_kwargs = None
    if args.tiny:
        tok = build_tiny_tokenizer([e.text for e in examples], vocab_size=4000)
        tok.save_pretrained(str(out_dir / "tiny_tokenizer"))     # scan_vs_span.py needs the same tokenizer
        tiny_kwargs = tiny_encoder_kwargs(len(tok), max(args.max_len_span, args.max_len_clause))
        args.pad_multiple = 32
        args.encoder = "tiny-random-longformer"
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.encoder)

    # ---- run ---------------------------------------------------------------
    partial = out_dir / "partial"; partial.mkdir(exist_ok=True)
    cfg = {k: getattr(args, k) for k in RESUME_KEYS}
    cfg["n_parts"] = n_parts(args.cv_folds, args.test_frac)
    cfg_path = partial / "config.json"
    if args.resume and cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if "n_parts" not in old and "cv_folds" in old:        # config written by an older version
            old["n_parts"] = n_parts(old["cv_folds"], old.get("test_frac", args.test_frac))
        old.pop("cv_folds", None)
        diff = {k: (old.get(k), cfg[k]) for k in cfg if old.get(k) != cfg[k]}
        if diff:
            raise SystemExit(f"--resume: settings differ from the saved run {diff}. "
                             f"Use a new --out_dir (or delete {partial}).")
    cfg_path.write_text(json.dumps(cfg, indent=1))

    per_exp: Dict[str, List[dict]] = {e: [] for e in args.experiments}
    baselines_cls: Dict[str, List[dict]] = {}
    baselines_span: Dict[str, List[dict]] = {}
    for k, split in enumerate(splits):
        if args.folds is not None and k not in args.folds:
            continue
        if not args.skip_baselines:
            bf = partial / f"baselines_fold{k}.json"
            if args.resume and bf.exists():
                bl = json.loads(bf.read_text())
            else:
                tr = [examples[i] for i in split[0]]; te = [examples[i] for i in split[2]]
                bl = classification_baselines(tr, te)
                bf.write_text(json.dumps(bl, default=float))
            for n, m in bl.items():
                baselines_cls.setdefault(n, []).append(m)
            sf = partial / f"span_baselines_fold{k}.json"
            if args.resume and sf.exists():
                sb = json.loads(sf.read_text())
            else:
                sexp = {"views": ("span",), "tasks": ("span",)}
                _, sl = make_loaders(examples, split[2], tok, sexp, args, examples, False,
                                     {"clause": 1, "span": max(1, args.bs_span * 2)}, args.seed + k)
                sb = span_baselines([examples[i] for i in split[0]], sl["span"]) if "span" in sl else {}
                sf.write_text(json.dumps(sb, default=float))
            for n, m in sb.items():
                baselines_span.setdefault(n, []).append(m)
        for name in args.experiments:
            f = partial / f"{name}_fold{k}.json"
            if args.resume and f.exists():
                print(f"=== {name} | fold {k + 1}/{len(splits)} === already done, loading", flush=True)
                per_exp[name].append(json.loads(f.read_text()))
                continue
            print(f"=== {name} | fold {k + 1}/{len(splits)} ===", flush=True)
            r = run_one(name, EXPERIMENTS[name], k, examples, split, tok, args,
                        device, amp_dtype, tiny_kwargs)
            f.write_text(json.dumps(r, default=float))      # survives a Colab disconnect
            per_exp[name].append(r)

    aggregate = {}
    for name, runs in per_exp.items():
        aggregate[name] = {key: aggregate_folds(runs, key) for key in
                           ("cls", "span", "span_own_label", "span_random_baseline", "risk")}
        aggregate[name]["latency_ms_per_item"] = {
            v: {"mean": float(np.mean([r["latency_ms_per_item"][v] for r in runs if v in r["latency_ms_per_item"]]))}
            for v in ("clause", "span") if any(v in r["latency_ms_per_item"] for r in runs)}
    agg_span_base = {n: {"mean": M.aggregate(ms)[0], "std": M.aggregate(ms)[1]}
                     for n, ms in baselines_span.items()}
    agg_base = {n: {"mean": M.aggregate(ms)[0], "std": M.aggregate(ms)[1]} for n, ms in baselines_cls.items()}
    for n, ms in baselines_cls.items():
        for c in CATEGORIES:
            agg_base[n]["mean"]["per_class"][c]["support"] = int(sum(m["per_class"][c]["support"] for m in ms))

    notes = [
        "Span task is WINDOWED localisation (answer guaranteed in a "
        f"{args.window_chars}-char window); not comparable to full-contract CUAD numbers.",
        "Category labels derive from the 41->18 CUAD mapping; compare against the TF-IDF baseline before claiming anything.",
        "6 categories (Confidentiality, Force Majeure, Indemnification, Modification, Payment, "
        "Privacy / Data Protection) have no data unless --weak_labels_jsonl is supplied; they are n/a, not 0.",
        "'Joint' = shared encoder trained on alternating clause/span batches; each batch only trains heads it has labels for.",
        "Span 'best-match gold' scores against the closest of all labelled spans of that category inside the "
        "window (a window can hold several valid answers); 'own label only' is the stricter version.",
        f"Risk labels: {args.risk_label_source or 'none used'}."
        + (" These are UNREVIEWED model-generated labels, not legal ground truth." if args.risk_label_source == "ai_silver" else ""),
        f"Folds run: {max(len(v) for v in per_exp.values())} of {len(splits)}; seed {args.seed}. "
        "With one fold there is no variance estimate.",
        *([f"Span training used a random {args.span_train_frac:.0%} of span items (compute saver)."]
          if args.span_train_frac < 1.0 else []),
    ]
    results = {
        "meta": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(device),
                 "encoder": args.encoder, "n_folds": max(len(v) for v in per_exp.values()), "args": vars(args),
                 "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                 "amp_dtype": str(amp_dtype),
                 "versions": {"torch": torch.__version__, "python": platform.python_version()}},
        "notes": notes, "runs": per_exp, "baselines_cls": agg_base, "baselines_span": agg_span_base, "aggregate": aggregate,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    write_report(out_dir, results, args, notes)
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'results.md'}")
    return results


if __name__ == "__main__":
    main()
