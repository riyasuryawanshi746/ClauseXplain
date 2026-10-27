#!/usr/bin/env python
"""
pipeline.py -- the ClauseXplain demo: contract in, categorised clauses and Indian-law
rule flags out, as a self-contained HTML report (+ JSON).

    python pipeline.py --input contract.pdf --ckpt ckpt_joint_cls_span_fold0.pt \
        --rules indian_legal_rules.csv --out report.html
    python pipeline.py --input contract.txt ... --find "Termination" "Liability"

WHAT IT DOES
    1. PDF -> cleaned text (same cleaner as the test set), or plain .txt as is.
    2. Splits the text into clause-sized units (100-900 characters, headings removed).
    3. Classifies every unit with the category head.  Only the 12 categories that have
       CUAD training data can ever be predicted; the other 6 are masked out.
    4. Applies the rule layer (rules_engine.py) to units predicted as a category that
       has rules, but only when the classifier is confident (--min_prob).
    5. Contract-level check: no force-majeure wording anywhere (ICA s.56 gap flag).
    6. Optional --find CATEGORY: the span head proposes a clause in each window of the
       contract and the classifier VERIFIES it.  The span head has no "no answer"
       option, so without the verification step it would always return something.

WHAT THE REPORT CAN AND CANNOT TELL YOU  (also printed in the report itself)
    * The model was trained on US contracts (CUAD) and checked on a few dozen Indian
      clauses; expect lower accuracy on Indian text.
    * It cannot recognise Confidentiality, Force Majeure, Indemnification, Modification,
      Payment or Privacy clauses; those get mislabelled.  The keyword-based rule layer
      only sees units the classifier assigned to a ruled category.
    * Rule flags are candidates for review ("may raise issues"), never findings that a
      clause is void, and nothing here is legal advice.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import metrics as M
from dataset import CAT2ID, CATEGORIES, category_question, make_collator
from make_label_sheet import LEAD_NUM, SENT
from rules_engine import apply_rules, force_majeure_gap, load_rules
from scan_vs_span import load_model
from train_phase3 import choose_amp_dtype

NO_DATA = ["Confidentiality", "Force Majeure", "Indemnification", "Modification", "Payment",
           "Privacy / Data Protection"]
SEVERITY = {"flag": 0, "possible": 1, "gap": 2, "needs_fact": 3, "context": 4, "none": 9}


def _emit(units, body, cs, ce, base, lo):
    """Append body[cs:ce] with offsets trimmed to the stripped text, so text[start:end] == unit text."""
    seg = body[cs:ce]
    a = cs + (len(seg) - len(seg.lstrip()))
    b = ce - (len(seg) - len(seg.rstrip()))
    if b - a >= lo:
        units.append({"start": base + a, "end": base + b, "text": body[a:b]})


def segment(text: str, lo: int = 40, hi: int = 900, target: int = 700) -> List[dict]:
    """Clause-sized units with absolute character offsets into `text`."""
    units, pos = [], 0
    for p in text.split("\n\n"):
        pstart = text.find(p, pos)
        if pstart < 0:
            continue
        pos = pstart + len(p)
        m = LEAD_NUM.match(p)
        off = m.end() if m else 0
        body = p[off:]
        off += len(body) - len(body.lstrip())
        body = body.strip()
        if len(body) < lo or "..." in body or sum(c.isdigit() for c in body) > 0.25 * len(body):
            continue
        if len(body) <= hi:
            units.append({"start": pstart + off, "end": pstart + off + len(body), "text": body}); continue
        bounds = [0] + [mm.end() for mm in SENT.finditer(body)] + [len(body)]
        sents = list(zip(bounds[:-1], bounds[1:]))
        cs = ce = None
        for a, b in sents:
            if cs is not None and b - cs > target:
                _emit(units, body, cs, ce, pstart + off, lo)
                cs = a
            elif cs is None:
                cs = a
            ce = b
        if cs is not None:
            _emit(units, body, cs, ce, pstart + off, lo)
    return units


class ClauseXplain:
    def __init__(self, ckpt: str, rules_csv: Optional[str] = None, encoder: str = "allenai/longformer-base-4096",
                 max_len_clause: int = 512, max_len_span: int = 1024, window_chars: int = 3000,
                 tiny_tokenizer_dir: Optional[str] = None, tiny: bool = False):
        self.cfg = {"encoder": encoder, "tiny": tiny, "max_len_clause": max_len_clause,
                    "max_len_span": max_len_span, "window_chars": window_chars}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = choose_amp_dtype(self.device, "none" if tiny else "auto")
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tiny_tokenizer_dir if tiny else encoder)
        self.model = load_model(ckpt, self.cfg, self.tok, self.device)
        self.rules = load_rules(rules_csv) if rules_csv else None
        self.collate = make_collator(self.tok.pad_token_id, pad_to_multiple_of=32 if tiny else 512)
        self.mask = torch.tensor([c in NO_DATA for c in CATEGORIES])

    # ---------------------------------------------------------------- helpers
    def _item(self, ids):
        return {"input_ids": ids, "global_attention_mask": [1] + [0] * (len(ids) - 1), "cls_label": -100,
                "risk_levels": [-100] * 4, "start_pos": -100, "end_pos": -100, "meta": {}}

    def _forward(self, batch):
        b = {k: v.to(self.device) for k, v in batch.items() if k != "meta"}
        with torch.autocast(device_type=self.device.type, dtype=self.amp,
                            enabled=self.device.type == "cuda" and self.amp is not None):
            return self.model(**b)

    @torch.no_grad()
    def classify(self, texts: List[str], bs: int = 32) -> np.ndarray:
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), len(CATEGORIES)), dtype=np.float32)
        for s in range(0, len(texts), bs):
            idx = order[s:s + bs]
            items = [self._item(self.tok(texts[i], truncation=True,
                                         max_length=self.cfg["max_len_clause"])["input_ids"]) for i in idx]
            logits = self._forward(self.collate(items))["cls_logits"].float().cpu()
            logits[:, self.mask] = -1e9
            out[idx] = logits.softmax(-1).numpy()
        return out

    @torch.no_grad()
    def find(self, text: str, category: str, top_k: int = 3, overlap: int = 500) -> List[dict]:
        """Span head proposes, classifier verifies."""
        w = self.cfg["window_chars"]
        step = max(1, w - overlap)
        wins = [(s, min(len(text), s + w)) for s in range(0, max(1, len(text) - overlap), step)]
        q = category_question(category)
        cands = []
        for s0 in range(0, len(wins), 4):
            chunk = wins[s0:s0 + 4]
            items, metas = [], []
            for a, b in chunk:
                enc = self.tok(q, text[a:b], truncation="only_second", max_length=self.cfg["max_len_span"],
                               return_offsets_mapping=True, return_token_type_ids=False)
                sids = enc.sequence_ids()
                g = [1 if sid != 1 else 0 for sid in sids]; g[0] = 1
                items.append({"input_ids": enc["input_ids"], "global_attention_mask": g, "cls_label": -100,
                              "risk_levels": [-100] * 4, "start_pos": -100, "end_pos": -100, "meta": {}})
                metas.append((a, sids, enc["offset_mapping"]))
            out = self._forward(self.collate(items))
            sl, el = out["start_logits"].float().cpu().numpy(), out["end_logits"].float().cpu().numpy()
            for i, (a, sids, offs) in enumerate(metas):
                n = len(sids)
                mask = np.array([x == 1 for x in sids])
                if not mask.any():
                    continue
                s, e = M.decode_span(sl[i, :n], el[i, :n], mask)
                cs, ce = a + offs[s][0], a + offs[e][1]
                if ce - cs >= 20:
                    cands.append({"start": cs, "end": ce, "text": text[cs:ce],
                                  "span_score": float(sl[i, s] + el[i, e])})
        if not cands:
            return []
        probs = self.classify([c["text"] for c in cands])[:, CAT2ID[category]]
        for c, p in zip(cands, probs):
            c["verify_prob"] = float(p)
        cands.sort(key=lambda c: -c["verify_prob"])
        kept = []
        for c in cands:                                   # drop near-duplicates from overlapping windows
            if all(min(c["end"], k["end"]) - max(c["start"], k["start"]) < 0.5 * (c["end"] - c["start"]) for k in kept):
                kept.append(c)
            if len(kept) >= top_k:
                break
        return kept

    # ---------------------------------------------------------------- main entry
    def analyze(self, text: str, find: Optional[List[str]] = None, min_prob: float = 0.6) -> dict:
        t0 = time.time()
        units = segment(text)
        probs = self.classify([u["text"] for u in units]) if units else np.zeros((0, len(CATEGORIES)))
        for u, p in zip(units, probs):
            top = np.argsort(-p)[:3]
            u["category"] = CATEGORIES[int(top[0])]
            u["prob"] = float(p[top[0]])
            u["top3"] = [(CATEGORIES[int(i)], float(p[i])) for i in top]
            u["confident"] = u["prob"] >= min_prob and u["category"] != "Other"
            u["flags"] = []
            if self.rules and u["confident"]:
                for f in apply_rules(u["text"], u["category"], self.rules):
                    if f.status == "none":
                        continue
                    u["flags"].append({"rule_id": f.rule_id, "provision": f.provision, "status": f.status,
                                       "reasons": f.reasons, "note": f.note, "rule_confidence": f.confidence,
                                       "evidence": [(s, e, t) for s, e, t in f.evidence]})
        contract_flags = []
        if self.rules:
            g = force_majeure_gap(text, self.rules)
            if g is not None and g.status != "none":
                contract_flags.append({"rule_id": g.rule_id, "provision": g.provision, "status": g.status,
                                       "reasons": g.reasons, "note": g.note})
        found = {c: self.find(text, c) for c in (find or [])}
        cnt: Dict[str, int] = {}
        for u in units:
            if u["confident"]:
                cnt[u["category"]] = cnt.get(u["category"], 0) + 1
        return {"n_chars": len(text), "n_units": len(units), "min_prob": min_prob, "units": units,
                "contract_flags": contract_flags, "found": found, "category_counts": cnt,
                "seconds": round(time.time() - t0, 1)}


# -------------------------------------------------------------------------------------
def highlight(text: str, spans) -> str:
    spans = sorted((max(0, s), min(len(text), e)) for s, e, *_ in spans if e > s)
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, pos = [], 0
    for s, e in merged:
        out += [html.escape(text[pos:s]), "<mark>", html.escape(text[s:e]), "</mark>"]
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def render_html(res: dict, title: str = "ClauseXplain report") -> str:
    E = html.escape
    flagged = [(u, f) for u in res["units"] for f in u["flags"] if f["status"] in ("flag", "possible", "gap", "needs_fact")]
    flagged.sort(key=lambda x: (SEVERITY[x[1]["status"]], x[0]["start"]))
    css = ("body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1c1c1c}"
           "h1{font-size:1.5rem}h2{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}"
           ".warn{background:#fff6dd;border:1px solid #ecd58a;padding:.8rem 1rem;border-radius:8px;font-size:.92rem}"
           ".card{border:1px solid #ddd;border-radius:8px;padding:.8rem 1rem;margin:.8rem 0}"
           ".tag{display:inline-block;padding:.1rem .5rem;border-radius:99px;font-size:.78rem;margin-right:.4rem;background:#eee}"
           ".flag{background:#fbd4d4}.possible{background:#fde9c8}.gap{background:#dbe7fb}.needs_fact{background:#e6e0f5}"
           "mark{background:#ffe27a}table{border-collapse:collapse;width:100%;font-size:.88rem}"
           "td,th{border-bottom:1px solid #e5e5e5;padding:.35rem .5rem;text-align:left;vertical-align:top}"
           "small{color:#666}details{margin:.4rem 0}")
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>{E(title)}</title><style>{css}</style></head><body>",
         f"<h1>{E(title)}</h1>",
         "<div class='warn'><b>Research prototype, not legal advice.</b> The category model was trained on US contracts "
         "(CUAD) and tested on a small Indian sample, so expect errors on Indian text. It cannot recognise "
         "Confidentiality, Force Majeure, Indemnification, Modification, Payment or Privacy clauses (it mislabels them). "
         "Rule flags are candidates for human review under the Indian statutes named, not findings that a clause is void.</div>",
         f"<p>{res['n_units']} clause units from {res['n_chars']:,} characters, analysed in {res['seconds']} s. "
         f"Categories shown only when the classifier confidence is at least {res['min_prob']:.2f}.</p>"]
    h.append("<h2>Categories detected</h2><table><tr><th>Category</th><th>Clauses</th></tr>")
    for c, n in sorted(res["category_counts"].items(), key=lambda x: -x[1]):
        h.append(f"<tr><td>{E(c)}</td><td>{n}</td></tr>")
    h.append("</table>")
    h.append("<h2>Contract-level flags</h2>")
    if res["contract_flags"]:
        for f in res["contract_flags"]:
            h.append(f"<div class='card'><span class='tag {f['status']}'>{f['status']}</span><b>{E(f['provision'])}</b><br>"
                     f"{E('; '.join(f['reasons']))}<br><small>{E(f['note'])}</small></div>")
    else:
        h.append("<p>None.</p>")
    h.append(f"<h2>Clause-level flags ({len(flagged)})</h2>")
    if not flagged:
        h.append("<p>No rule flags on confidently classified clauses.</p>")
    for u, f in flagged:
        ev = [(s, e) for s, e, _ in f["evidence"]]
        h.append(f"<div class='card'><span class='tag {f['status']}'>{f['status']}</span><b>{E(f['provision'])}</b> "
                 f"<small>(rule {E(f['rule_id'])}, rule confidence {E(f['rule_confidence'])})</small><br>"
                 f"<small>classified as <b>{E(u['category'])}</b> ({u['prob']:.2f})</small>"
                 f"<p>{highlight(u['text'], ev)}</p><small>{E('; '.join(f['reasons']))}. {E(f['note'])}</small></div>")
    if res["found"]:
        h.append("<h2>Clauses located by the span head (verified by the classifier)</h2>")
        for cat, lst in res["found"].items():
            h.append(f"<h3>{E(cat)}</h3>")
            if not lst:
                h.append("<p>Nothing proposed.</p>")
            for c in lst:
                ok = "verified" if c["verify_prob"] >= res["min_prob"] else "NOT verified (low classifier confidence)"
                h.append(f"<div class='card'><small>classifier probability for {E(cat)}: {c['verify_prob']:.2f} - {ok}</small>"
                         f"<p>{E(c['text'])}</p></div>")
    h.append("<h2>All clause units</h2><details><summary>show table</summary><table><tr><th>#</th><th>Predicted</th>"
             "<th>Conf.</th><th>Text</th></tr>")
    for i, u in enumerate(res["units"]):
        h.append(f"<tr><td>{i}</td><td>{E(u['category'])}</td><td>{u['prob']:.2f}</td><td>{E(u['text'][:260])}"
                 f"{'...' if len(u['text']) > 260 else ''}</td></tr>")
    h.append("</table></details></body></html>")
    return "".join(h)


def load_input(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from prepare_documents import clean_pages, extract_pages
        return clean_pages(extract_pages(path))[0]
    return Path(path).read_text(encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rules", default=None)
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--find", nargs="*", default=None)
    ap.add_argument("--min_prob", type=float, default=0.6)
    ap.add_argument("--encoder", default="allenai/longformer-base-4096")
    ap.add_argument("--window_chars", type=int, default=3000)
    ap.add_argument("--max_len_span", type=int, default=1024)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--tiny_tokenizer_dir", default=None)
    a = ap.parse_args(argv)
    cx = ClauseXplain(a.ckpt, a.rules, a.encoder, window_chars=a.window_chars, max_len_span=a.max_len_span,
                      tiny=a.tiny, tiny_tokenizer_dir=a.tiny_tokenizer_dir)
    res = cx.analyze(load_input(a.input), a.find, a.min_prob)
    Path(a.out).write_text(render_html(res, f"ClauseXplain: {Path(a.input).name}"), encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"{res['n_units']} units, {sum(len(u['flags']) for u in res['units'])} flags, "
          f"{len(res['contract_flags'])} contract-level flags -> {a.out}")
    return res


if __name__ == "__main__":
    main()
