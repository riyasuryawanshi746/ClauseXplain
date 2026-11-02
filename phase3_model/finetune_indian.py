#!/usr/bin/env python
"""
finetune_indian.py -- can models learn the Indian labels? Document-held-out training on the Indian set.

WHY
    Zero-shot CUAD models score ~0.34 macro-F1 on Indian clauses.  That number answers "does CUAD transfer?"
    It does not answer "are the Indian labels learnable/consistent?"  This script trains on Indian clauses and
    tests on Indian DOCUMENTS the model has not seen, so every clause is predicted exactly once by a model that
    never saw its document.

PROTOCOL (fixed on 2026-11-02, before any Indian fine-tuning result was seen; do not tune on these results)
    * Leave-one-group-out over documents.  Group = document, except the two MEA agreements, which share one
      template and form one group ("mea").  13 documents -> 12 folds.
    * Pooled metrics over all 334 held-out predictions: accuracy (Wilson 95%), macro-F1 over categories present
      (bootstrap 95% over clauses), seen (192) / unseen (142) / per-pool / per-category / per-group breakdowns.
    * The judgment slice (11) is scored by a model trained on all 334 clauses (it never enters training).
    * No validation-based selection: fixed epochs, last-epoch model.  Neural settings: 8 epochs, lr 2e-5,
      head lr 1e-4, batch 16, warmup 10%, weight decay 0.01, max 512 tokens, seed 13, fp16 on GPU.
    * "--tricks" (step 7) = class weights (inverse frequency on the training fold, mean 1) + label smoothing 0.1
      + layer-wise LR decay 0.9.  Without it: plain cross-entropy, uniform encoder LR.  TF-IDF --tricks = balanced
      class weights.

USAGE
    python finetune_indian.py run --method tfidf --name tfidf_indian --indian_dir ../indian_set/built_v4 \
        --judgments_dir ../indian_set/built_judgments --out_dir runs_ft
    python finetune_indian.py run --method tfidf --name tfidf_cuad_indian --cuad_mapped ../phase1_cuad_mapping/clauses_mapped.jsonl ...
    python finetune_indian.py run --method neural --name legalbert_indian_tricks --encoder nlpaueb/legal-bert-base-uncased --tricks ...
    python finetune_indian.py run --method neural --name longformer_cuad_indian --encoder allenai/longformer-base-4096 \
        --init_ckpt ckpt_joint_cls_span_fold0.pt ...
    python finetune_indian.py run --method neural --name legalbert_cuad_indian_ledgar --encoder nlpaueb/legal-bert-base-uncased         --init_ckpt ckpt_cls_only_fold0.pt --extra_jsonl ledgar_mapped.jsonl --extra_cap 100 ...
    python finetune_indian.py summarise --out_dir runs_ft      # comparison table of every run in out_dir
    python finetune_indian.py final --encoder nlpaueb/legal-bert-base-uncased --init_ckpt ckpt_cls_only_fold0.pt         --indian_dir ../indian_set/built_v4 --judgments_dir ../indian_set/built_judgments --save_dir demo_model
        # demo model: same settings, trained on ALL Indian clauses (no held-out test; only the judgment slice is
        # scored). Saves encoder + tokenizer (save_pretrained, fp16) + heads.pt + meta.json for pipeline.py --cls_model.

EXTRA DATA (step 5, fixed 2026-11-03 before any LEDGAR result was seen)
    --extra_jsonl: mapped LEDGAR provisions from ledgar_extra.py.  --extra_cap N samples at most N per category
    (sampling seed 13, independent of --seed), added to every training fold for TF-IDF and neural alike.
    --extra_categories restricts the extra data to the listed categories.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dataset import CATEGORIES, CAT2ID

NO_DATA = {"Confidentiality", "Force Majeure", "Indemnification", "Modification", "Payment",
           "Privacy / Data Protection"}
BATCH2 = {"indian_cpwd_gcc_2023", "indian_sbm_waste_to_compost_mca"}


# ---------------------------------------------------------------------------- data
def load_set(d):
    rows = [json.loads(l) for l in open(Path(d) / "indian_mapped.jsonl", encoding="utf-8") if l.strip()]
    return [{"clause_id": r["clause_id"], "doc": r["contract_id"], "text": r["clause_text"],
             "y": CAT2ID[r["clausexplain_category"]]} for r in rows]


def group_of(doc):
    return "mea" if doc.startswith("indian_mea_") else doc


def load_extra(path, cap, cats, seed=13):
    by = defaultdict(list)
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if not cats or r["clausexplain_category"] in cats:
            by[r["clausexplain_category"]].append({"text": r["clause_text"], "y": CAT2ID[r["clausexplain_category"]]})
    rng = random.Random(seed); out = []
    for c in sorted(by):
        xs = by[c][:]; rng.shuffle(xs); out += xs[:cap] if cap else xs
    return out


def load_cuad(path):
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("category_status") == "MAPPED" and r.get("clausexplain_category"):
            out.append({"text": r["clause_text"], "y": CAT2ID[r["clausexplain_category"]]})
    return out


# ---------------------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    labels = sorted(set(y))                       # categories present in the gold labels
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0)) if y else float("nan")


def boot_ci(y, p, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); vals.append(fn(list(y[i]), list(p[i])))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def summarise_preds(rows):
    """rows: dicts with y, pred, doc. Returns the metric dict."""
    def block(rs):
        if not rs:
            return None
        y = [r["y"] for r in rs]; p = [r["pred"] for r in rs]; k = sum(a == b for a, b in zip(y, p))
        return {"n": len(rs), "accuracy": k / len(rs), "accuracy_ci95": wilson(k, len(rs)),
                "macro_f1": macro_f1(y, p)}
    from sklearn.metrics import precision_recall_fscore_support
    y = [r["y"] for r in rows]; p = [r["pred"] for r in rows]
    res = {"all": block(rows),
           "seen": block([r for r in rows if CATEGORIES[r["y"]] not in NO_DATA]),
           "unseen": block([r for r in rows if CATEGORIES[r["y"]] in NO_DATA]),
           "original_pool": block([r for r in rows if r["doc"] not in BATCH2]),
           "batch2": block([r for r in rows if r["doc"] in BATCH2])}
    res["all"]["macro_f1_ci95"] = boot_ci(y, p, macro_f1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(len(CATEGORIES))), zero_division=0)
    res["per_category"] = {CATEGORIES[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                                           "support": int(sup[i])} for i in range(len(CATEGORIES)) if sup[i]}
    by = defaultdict(list)
    for r in rows:
        by[group_of(r["doc"])].append(r)
    res["per_group"] = {g: block(rs) for g, rs in sorted(by.items())}
    conf = defaultdict(Counter)
    for r in rows:
        if r["y"] != r["pred"]:
            conf[CATEGORIES[r["y"]]][CATEGORIES[r["pred"]]] += 1
    res["top_confusions"] = sorted(((t, q, n) for t, c in conf.items() for q, n in c.items()), key=lambda x: -x[2])[:10]
    return res


# ---------------------------------------------------------------------------- TF-IDF
def tfidf_fit_predict(train, test, balanced):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True)
    X = vec.fit_transform([e["text"] for e in train])
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced" if balanced else None)
    clf.fit(X, [e["y"] for e in train])
    P = clf.predict_proba(vec.transform([e["text"] for e in test]))
    idx = P.argmax(1)
    return [int(clf.classes_[i]) for i in idx], [float(P[j, i]) for j, i in enumerate(idx)]


# ---------------------------------------------------------------------------- neural
class Neural:
    def __init__(self, a):
        import torch
        from transformers import AutoTokenizer
        self.a, self.torch = a, torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(a.tokenizer or a.encoder)

    def build(self):
        from model import ClauseXplainMTL, MTLConfig
        m = ClauseXplainMTL(MTLConfig(encoder_name=self.a.encoder, active_tasks=("cls",)))
        if self.a.init_ckpt:
            state = self.torch.load(self.a.init_ckpt, map_location="cpu")
            missing, unexpected = m.load_state_dict(state, strict=False)
            bad = [k for k in list(missing) + list(unexpected) if not k.startswith("log_vars")]
            if bad:
                raise RuntimeError(f"checkpoint does not match the model: {bad[:5]}")
        return m.to(self.dev)

    def param_groups(self, m):
        a = self.a
        head = [p for n, p in m.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        if not a.tricks:
            enc = [p for n, p in m.named_parameters() if n.startswith("encoder.")]
            return [{"params": enc, "lr": a.lr}, {"params": head, "lr": a.head_lr}]
        L = m.encoder.config.num_hidden_layers
        groups = defaultdict(list)
        for n, p in m.named_parameters():
            if not n.startswith("encoder."):
                continue
            if ".layer." in n:
                i = int(n.split(".layer.")[1].split(".")[0]); depth = L - 1 - i
            elif "embeddings" in n:
                depth = L
            else:
                depth = 0
            groups[depth].append(p)
        out = [{"params": ps, "lr": a.lr * (a.llrd ** d)} for d, ps in groups.items()]
        return out + [{"params": head, "lr": a.head_lr}]

    def batches(self, items, shuffle, rng):
        idx = list(range(len(items)))
        if shuffle:
            rng.shuffle(idx)
        for s in range(0, len(idx), self.a.batch):
            chunk = [items[i] for i in idx[s:s + self.a.batch]]
            enc = self.tok([e["text"] for e in chunk], truncation=True, max_length=self.a.max_len,
                           padding=True, return_tensors="pt")
            g = self.torch.zeros_like(enc["input_ids"]); g[:, 0] = 1
            yield ({"input_ids": enc["input_ids"].to(self.dev), "attention_mask": enc["attention_mask"].to(self.dev),
                    "global_attention_mask": g.to(self.dev)},
                   self.torch.tensor([e["y"] for e in chunk], device=self.dev))

    def save(self, m, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        m.encoder.half().save_pretrained(d); self.tok.save_pretrained(d)
        self.torch.save({k: v.float().cpu() for k, v in m.state_dict().items() if not k.startswith("encoder.")},
                        d / "heads.pt")

    def fit_predict(self, train, test, seed, save_dir=None):
        torch, a = self.torch, self.a
        from transformers import get_linear_schedule_with_warmup
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        m = self.build()
        weight = None
        if a.tricks:
            cnt = Counter(e["y"] for e in train); K = len(CATEGORIES)
            w = np.array([1.0 / cnt[c] if cnt[c] else 0.0 for c in range(K)])
            w = w / (sum(w[e["y"]] for e in train) / len(train))       # mean weight over training samples = 1
            weight = torch.tensor(w, dtype=torch.float, device=self.dev)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1 if a.tricks else 0.0)
        opt = torch.optim.AdamW(self.param_groups(m), weight_decay=0.01)
        steps = a.epochs * math.ceil(len(train) / a.batch)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        use_amp = self.dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = random.Random(seed)
        for ep in range(a.epochs):
            m.train(); tot = 0.0; nb = 0
            for x, y in self.batches(train, True, rng):
                with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=use_amp):
                    logits = m(**x)["cls_logits"]
                loss = loss_fn(logits.float(), y)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                tot += float(loss); nb += 1
        m.eval(); preds, confs = [], []
        with torch.no_grad():
            for x, _ in self.batches(test, False, rng):
                with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=use_amp):
                    pr = torch.softmax(m(**x)["cls_logits"].float(), -1)
                c, i = pr.max(-1); preds += i.tolist(); confs += c.tolist()
        last_loss = tot / max(1, nb)
        if save_dir:
            self.save(m, save_dir)
        del m, opt
        if use_amp:
            torch.cuda.empty_cache()
        return preds, confs, last_loss


# ---------------------------------------------------------------------------- run
def run(a):
    data = load_set(a.indian_dir)
    judg = load_set(a.judgments_dir) if a.judgments_dir else []
    extra = load_cuad(a.cuad_mapped) if a.cuad_mapped else []
    if a.extra_jsonl:
        ext = load_extra(a.extra_jsonl, a.extra_cap, set(a.extra_categories or []))
        print(f"extra training data: {len(ext)} clauses", dict(Counter(CATEGORIES[e['y']] for e in ext)), flush=True)
        extra += ext
    groups = sorted({group_of(e["doc"]) for e in data})
    out = Path(a.out_dir) / a.name; out.mkdir(parents=True, exist_ok=True)
    nn = Neural(a) if a.method == "neural" else None
    rows, log = [], []
    t0 = time.time()
    for k, g in enumerate(groups):
        test = [e for e in data if group_of(e["doc"]) == g]
        train = [e for e in data if group_of(e["doc"]) != g] + extra
        if a.method == "tfidf":
            preds, confs = tfidf_fit_predict(train, test, a.tricks); info = ""
        else:
            preds, confs, last = nn.fit_predict(train, test, a.seed + k); info = f" last-epoch loss {last:.3f}"
        acc = sum(p == e["y"] for p, e in zip(preds, test)) / len(test)
        log.append(f"fold {k:2d} {g:<55} n={len(test):3d} acc={acc:.3f}{info} ({time.time() - t0:.0f}s)")
        print(log[-1], flush=True)
        for e, p, c in zip(test, preds, confs):
            rows.append({**e, "group": g, "pred": p, "conf": c})
    res = {"name": a.name, "method": a.method, "tricks": a.tricks, "encoder": a.encoder, "init_ckpt": a.init_ckpt,
           "with_cuad_train": bool(a.cuad_mapped),
           "extra_jsonl": a.extra_jsonl, "extra_cap": a.extra_cap, "extra_categories": a.extra_categories,
           "n_extra": len(extra), "n_folds": len(groups), "settings": {k: getattr(a, k) for k in
           ["epochs", "lr", "head_lr", "batch", "max_len", "seed", "llrd"]}, "fold_log": log,
           "heldout": summarise_preds(rows)}
    if judg:
        train = data + extra
        if a.method == "tfidf":
            jp, jc = tfidf_fit_predict(train, judg, a.tricks)
        else:
            jp, jc, _ = nn.fit_predict(train, judg, a.seed + 100)
        jrows = [{**e, "group": "judgment", "pred": p, "conf": c} for e, p, c in zip(judg, jp, jc)]
        k = sum(r["pred"] == r["y"] for r in jrows)
        res["judgments"] = {"n": len(jrows), "accuracy": k / len(jrows), "accuracy_ci95": wilson(k, len(jrows)),
                            "macro_f1": macro_f1([r["y"] for r in jrows], [r["pred"] for r in jrows])}
        rows_j = jrows
    else:
        rows_j = []
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["clause_id", "document", "group", "true_category", "pred_category",
                                       "correct", "confidence", "seen_category", "clause_text"])
        for r in rows + rows_j:
            w.writerow([r["clause_id"], r["doc"], r["group"], CATEGORIES[r["y"]], CATEGORIES[r["pred"]],
                        int(r["y"] == r["pred"]), f"{r['conf']:.3f}", int(CATEGORIES[r["y"]] not in NO_DATA),
                        r["text"][:300].replace("\n", " ")])
    (out / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    h = res["heldout"]; f = lambda b: "n/a" if b is None else f"{b['accuracy']:.3f} / {b['macro_f1']:.3f} (n={b['n']})"
    print(f"\n{a.name}: held-out all acc/macro-F1 {f(h['all'])}; seen {f(h['seen'])}; unseen {f(h['unseen'])}; "
          f"original {f(h['original_pool'])}; batch2 {f(h['batch2'])}"
          + (f"; judgments acc {res['judgments']['accuracy']:.3f}" if judg else ""))


def final(a):
    """Train once on all Indian clauses (+ extra data) and save the model for the demo."""
    data = load_set(a.indian_dir)
    judg = load_set(a.judgments_dir) if a.judgments_dir else []
    extra = load_extra(a.extra_jsonl, a.extra_cap, set(a.extra_categories or [])) if a.extra_jsonl else []
    nn = Neural(a)
    test = judg or data[:1]
    preds, _, last = nn.fit_predict(data + extra, test, a.seed, save_dir=a.save_dir)
    meta = {"categories": CATEGORIES, "encoder": a.encoder, "init_ckpt": a.init_ckpt, "n_indian": len(data),
            "n_extra": len(extra), "extra_jsonl": a.extra_jsonl, "extra_cap": a.extra_cap, "last_epoch_loss": last,
            "settings": {k: getattr(a, k) for k in ["epochs", "lr", "head_lr", "batch", "max_len", "seed", "llrd", "tricks"]},
            "note": "trained on all Indian clauses; held-out estimates are in experiments/indian_finetune*/"}
    if judg:
        k = sum(p == e["y"] for p, e in zip(preds, judg)); meta["judgment_accuracy"] = k / len(judg)
    (Path(a.save_dir) / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved demo model to {a.save_dir}; last-epoch loss {last:.3f}"
          + (f"; judgment slice acc {meta['judgment_accuracy']:.3f}" if judg else ""))


def summarise(a):
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(a.out_dir).glob("*/metrics.json"))]
    f3 = lambda x: "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
    L = ["# Indian set: document-held-out training (leave-one-document-group-out, 12 folds)", "",
         "Every clause is predicted by a model that never saw its document. Zero-shot CUAD models (no Indian training)",
         "scored 0.347 acc / 0.338 macro-F1 (Longformer) on the 192 seen-category clauses only.", "",
         "| Run | all acc (95% CI) | all macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |",
         "|---|---|---|---|---|---|---|"]
    for r in runs:
        h = r["heldout"]; al = h["all"]; ci = al["accuracy_ci95"]; mci = al["macro_f1_ci95"]
        L.append(f"| {r['name']} | {f3(al['accuracy'])} ({f3(ci[0])}-{f3(ci[1])}) | {f3(al['macro_f1'])} "
                 f"({f3(mci[0])}-{f3(mci[1])}) | {f3(h['seen']['accuracy'])} / {f3(h['seen']['macro_f1'])} | "
                 f"{f3(h['unseen']['accuracy'])} / {f3(h['unseen']['macro_f1'])} | "
                 f"{f3(h['original_pool']['accuracy'])} / {f3(h['batch2']['accuracy'])} | "
                 f"{f3(r.get('judgments', {}).get('accuracy'))} |")
    (Path(a.out_dir) / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p, fi = sub.add_parser("run"), sub.add_parser("final")
    run_p.add_argument("--method", choices=["tfidf", "neural"], required=True)
    run_p.add_argument("--name", required=True)
    run_p.add_argument("--out_dir", required=True)
    for r in (run_p, fi):
        r.add_argument("--indian_dir", required=True)
        r.add_argument("--judgments_dir", default=None)
        r.add_argument("--cuad_mapped", default=None, help="also train on all mapped CUAD spans (TF-IDF only)")
        r.add_argument("--extra_jsonl", default=None, help="extra mapped clauses (ledgar_extra.py output)")
        r.add_argument("--extra_cap", type=int, default=0, help="max extra clauses per category (0 = all)")
        r.add_argument("--extra_categories", nargs="*", default=None)
        r.add_argument("--encoder", default=None)
        r.add_argument("--tokenizer", default=None)
        r.add_argument("--init_ckpt", default=None, help="start from a CUAD-trained ClauseXplain checkpoint")
        r.add_argument("--tricks", action="store_true")
        r.add_argument("--epochs", type=int, default=8)
        r.add_argument("--lr", type=float, default=2e-5)
        r.add_argument("--head_lr", type=float, default=1e-4)
        r.add_argument("--llrd", type=float, default=0.9)
        r.add_argument("--batch", type=int, default=16)
        r.add_argument("--max_len", type=int, default=512)
        r.add_argument("--seed", type=int, default=13)
    fi.add_argument("--save_dir", required=True)
    s = sub.add_parser("summarise")
    s.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "final":
        if not a.encoder:
            raise SystemExit("--encoder is required")
        final(a)
    elif a.cmd == "run":
        if a.method == "neural" and not a.encoder:
            raise SystemExit("--encoder is required for --method neural")
        if a.method == "neural" and a.cuad_mapped:
            raise SystemExit("--cuad_mapped is for TF-IDF; for neural use --init_ckpt (a CUAD-trained checkpoint)")
        run(a)
    else:
        summarise(a)


if __name__ == "__main__":
    main()
