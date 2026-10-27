#!/usr/bin/env python
"""
smoke_test.py -- correctness checks for the Phase 3 code. CPU only, no downloads.

    python smoke_test.py                       # fixtures only (~1-3 min)
    python smoke_test.py --real \
        --mapped_jsonl clauses_mapped.jsonl --cuad_json CUAD_v1.json   # + real-data checks

WHAT PASSING MEANS
    The plumbing is correct: labels line up with tokens, masked losses behave,
    the ordinal head maths is right, splits do not leak, the metrics handle
    missing classes, and a tiny model can actually LEARN both heads on an easy
    synthetic task (so a bug in the supervision path would show up here).

WHAT PASSING DOES NOT MEAN
    * Nothing here says the model will do well on real contracts.
    * Nothing here loads allenai/longformer-base-4096 (no download in tests).
      Run a real short training next: see the usage block in train_phase3.py.
    * The fixture contracts are synthetic, with one fixed template per category.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

import metrics as M
from dataset import (CAT2ID, CATEGORIES, ClauseDataset, Example, _slugify, assert_no_group_leak,
                     attach_contract_text, load_contract_texts, load_mapped, make_collator,
                     make_splits)
from model import IGNORE, ClauseXplainMTL, MTLConfig
from train_phase3 import build_tiny_tokenizer, main as train_main, tiny_encoder_kwargs

RESULTS = []


def check(name):
    def deco(fn):
        def run(*a, **k):
            try:
                fn(*a, **k)
                RESULTS.append((name, True, ""))
                print(f"PASS  {name}")
            except Exception as e:                                   # noqa: BLE001
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"FAIL  {name}\n      {type(e).__name__}: {e}")
                if "-v" in sys.argv:
                    traceback.print_exc()
        return run
    return deco


# ----------------------------------------------------------------------------
# synthetic fixture: CUAD-shaped JSON + clauses_mapped.jsonl
# ----------------------------------------------------------------------------
TEMPLATES = {
    "Liability": "In no event shall {A} be liable to {B} for any indirect or consequential damages arising out of this Agreement.",
    "Termination": "Either party may terminate this Agreement upon {N} days written notice to the other party.",
    "Non-Compete": "During the term and for {N} years thereafter {A} shall not engage in any competing business.",
    "Governing Law / Jurisdiction": "This Agreement shall be governed by the laws of the State of {S}.",
}
FILLER = [
    "The parties acknowledge that the schedules attached hereto form part of this document.",
    "Notices shall be delivered to the addresses set out on the signature page.",
    "Headings are for convenience only and do not affect interpretation.",
    "The Company shall provide the services described in Exhibit A in a professional manner.",
    "Payment of invoices is due within the period stated in the applicable order form.",
    "Each party shall bear its own costs in connection with the negotiation of this document.",
    "This document may be executed in counterparts, each of which is an original.",
    "The Supplier will maintain reasonable records relating to the deliverables.",
    "Any amendment must be in writing and signed by an authorised representative.",
    "Definitions used in this document have the meanings given in Section 1.",
    "The term of the engagement begins on the effective date shown above.",
    "Confidential materials shall be kept in a secure location at all times.",
]
NAMES = ["Alpha Corp", "Beta Ltd", "Gamma Inc", "Delta LLC", "Omega Co"]
STATES = ["Delaware", "New York", "California", "Texas"]


def make_fixture(root: Path, n_contracts: int = 20, seed: int = 0):
    rng = random.Random(seed)
    cats = list(TEMPLATES)
    data, rows = [], []
    for ci in range(n_contracts):
        title = f"FIXTURECO_{ci:02d}-EX-10-TEST AGREEMENT"
        cid = _slugify(title)
        chosen = rng.sample(cats, 3)
        sentences = [rng.choice(FILLER) for _ in range(30)]
        slots = sorted(rng.sample(range(len(sentences) + 1), 3))
        plan = []
        for slot, cat in zip(slots, chosen):
            plan.append((slot, cat))
        if ci == 0:                                   # two same-category spans, adjacent
            plan = [(5, "Liability"), (5, "Liability"), (20, "Termination")]
        inserts = []
        for slot, cat in plan:
            txt = TEMPLATES[cat].format(A=rng.choice(NAMES), B=rng.choice(NAMES),
                                        N=rng.choice([30, 60, 90, 2, 3]), S=rng.choice(STATES))
            inserts.append((slot, cat, txt))
        pieces, spans = [], []
        pos, k = 0, 0
        order = sorted(range(len(inserts)), key=lambda i: inserts[i][0])
        for s_i, sent in enumerate(sentences + [None]):
            for oi in order:
                if inserts[oi][0] == s_i:
                    pieces.append(("clause", inserts[oi][1], inserts[oi][2]))
            if sent is not None:
                pieces.append(("filler", None, sent))
        text, para_texts = "", []
        for kind, cat, t in pieces:
            start = len(text)
            text += t + ("\n" if rng.random() < 0.15 else " ")
            if kind == "clause":
                spans.append((cat, start, start + len(t), t))
        text = text.rstrip()
        data.append({"title": title, "paragraphs": [{"context": text, "qas": []}]})
        for si, (cat, a, b, t) in enumerate(spans):
            assert text[a:b] == t
            rows.append({"clause_id": f"{cid}__{cat}__{si}", "contract_id": cid, "cuad_category": cat,
                         "clausexplain_category": cat, "clause_text": t, "char_start": a, "char_end": b,
                         "category_status": "MAPPED", "source": "fixture"})
    cuad = root / "CUAD_fixture.json"
    cuad.write_text(json.dumps({"version": "fixture", "data": data}), encoding="utf-8")
    mapped = root / "clauses_mapped_fixture.jsonl"
    mapped.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return mapped, cuad


def fixture_examples(mapped, cuad):
    ex = load_mapped(mapped)
    attach_contract_text(ex, load_contract_texts(cuad), strict=True)
    return list(ex.values())


def tiny_model(tok, active=("cls", "risk", "span"), **kw):
    return ClauseXplainMTL(MTLConfig(active_tasks=active, tiny_encoder=tiny_encoder_kwargs(len(tok), 320),
                                     **kw))


def get_batch(ds, view, n=6):
    ids = [i for i, (v, _) in enumerate(ds.items) if v == view][:n]
    col = make_collator(1, pad_to_multiple_of=32)
    return col([ds[i] for i in ids])


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def run_fixture_tests(tmp: Path):
    mapped, cuad = make_fixture(tmp)
    examples = fixture_examples(mapped, cuad)
    tok = build_tiny_tokenizer([e.contract_text for e in examples] + [e.text for e in examples], 1500)
    ds = ClauseDataset(examples, tok, max_len_clause=96, max_len_span=256, span_mode="cuad_context",
                       window_chars=500, views=("clause", "span"), seed=0)

    @check("contract text re-attachment verifies every offset")
    def _():
        assert len(examples) == 60 and all(e.contract_text for e in examples)
        bad = Example("x", "nope", "t", category="Liability", char_start=0, char_end=3)
        try:
            attach_contract_text({"x": bad}, {}, strict=True)
        except AssertionError:
            return
        raise AssertionError("missing contract should raise in strict mode")
    _()

    @check("span window: always contains the answer exactly, and is NOT always centred")
    def _():
        rel = []
        for i, (v, ex) in enumerate(ds.items):
            if v != "span":
                continue
            m = ds[i]["meta"]
            a0, a1 = m["ans_char"]
            assert m["context"][a0:a1] == ex.text, "answer not at its recorded offsets"
            rel.append(a0 / len(m["context"]))
        assert np.std(rel) > 0.08, f"answer position spread too small ({np.std(rel):.3f}): window leak?"
    _()

    @check("span labels round-trip: tokens start..end decode back to the clause text")
    def _():
        n = 0
        for i, (v, ex) in enumerate(ds.items):
            if v != "span":
                continue
            it = ds[i]
            s, e = it["start_pos"], it["end_pos"]
            assert s != IGNORE and e >= s, "answer was truncated away in fixture"
            offs, ctx = it["meta"]["offsets"], it["meta"]["context"]
            rec = ctx[offs[s][0]:offs[e][1]]
            assert ex.text.strip() in rec and len(rec) - len(ex.text) < 12, (ex.text, rec)
            assert it["meta"]["seq_ids"][s] == 1 and it["meta"]["seq_ids"][e] == 1, "label points at question/special token"
            n += 1
        assert n == 60
    _()

    @check("multi-gold: two same-category spans in one window both counted as valid answers")
    def _():
        # big window (= whole fixture contract) so both spans are guaranteed inside it
        big = ClauseDataset(examples, tok, max_len_span=1024, span_mode="cuad_context",
                            window_chars=6000, views=("span",), seed=0)
        two = [i for i, (v, ex) in enumerate(big.items)
               if ex.category == "Liability" and ex.contract_id == examples[0].contract_id]
        assert len(two) == 2, len(two)
        for i in two:
            assert len(big[i]["meta"]["gold_tokens"]) == 2, big[i]["meta"]["gold_tokens"]
        # and with the small window, gold_tokens never contains a span outside the window
        for i, (v, ex) in enumerate(ds.items):
            if v == "span":
                assert 1 <= len(ds[i]["meta"]["gold_tokens"]) <= 2
    _()

    @check("span tokens: question/special tokens are global-attention, context is not")
    def _():
        it = ds[[i for i, (v, _) in enumerate(ds.items) if v == "span"][0]]
        g, sid = it["global_attention_mask"], it["meta"]["seq_ids"]
        assert g[0] == 1 and all((g[k] == 1) == (sid[k] != 1) for k in range(len(g)))
    _()

    @check("splits are contract-disjoint; a deliberate leak is caught")
    def _():
        for tr, va, te in make_splits(examples, 4, 0.15, 0.25, seed=0):
            assert_no_group_leak(examples, tr, va, te)
            assert len(set(tr) & set(te)) == 0 and len(te) > 0 and len(va) > 0
        try:
            assert_no_group_leak(examples, [0, 1], [2])
            ids = {examples[0].contract_id, examples[2].contract_id}
            if len(ids) == 1:
                raise RuntimeError("constructed pair was not a leak")
        except AssertionError:
            return
    _()

    @check("--cv_folds 1 is exactly fold 0 of --cv_folds 5 (so a 1-fold run can be extended with --resume)")
    def _():
        one = make_splits(examples, 1, 0.1, 0.2, seed=13)[0]
        five = make_splits(examples, 5, 0.1, 0.2, seed=13)[0]
        assert [list(map(int, x)) for x in one] == [list(map(int, x)) for x in five]
    _()

    @check("compare_runs: paired summary maths and verdict wording")
    def _():
        from compare_runs import paired_summary, verdict
        s = paired_summary([0.02, 0.03, 0.01, 0.04, 0.02])
        assert s["wins"] == 5 and s["ci95"][0] > 0 and "better" in verdict(s)
        s = paired_summary([0.02, -0.03, 0.01, -0.04, 0.02])
        assert "no evidence" in verdict(s)
        assert "too few" in verdict(paired_summary([0.5]))
        assert paired_summary([0.5])["ci95"] is None
    _()

    m = tiny_model(tok)

    @check("forward: output shapes for all three heads")
    def _():
        b = get_batch(ds, "span")
        o = m(**{k: v for k, v in b.items() if k != "meta"})
        B, L = b["input_ids"].shape
        assert o["cls_logits"].shape == (B, 18)
        assert o["risk_scores"].shape == (B, 4) and (0 <= o["risk_scores"]).all() and (o["risk_scores"] <= 1).all()
        assert o["start_logits"].shape == (B, L) and o["end_logits"].shape == (B, L)
    _()

    @check("gradients reach every head and the encoder; loss is finite")
    def _():
        b = get_batch(ds, "span")
        b["cls_labels"] = torch.randint(0, 12, (b["input_ids"].shape[0],))
        b["risk_levels"] = torch.randint(0, 5, (b["input_ids"].shape[0], 4))
        mm = tiny_model(tok)
        o = mm(**{k: v for k, v in b.items() if k != "meta"})
        assert set(o["loss_parts"]) == {"cls", "risk", "span"} and torch.isfinite(o["loss"])
        o["loss"].backward()
        for name, mod in (("cls", mm.cls_head), ("risk", mm.risk_head), ("span", mm.span_head),
                          ("encoder", mm.encoder)):
            gs = [p.grad for p in mod.parameters() if p.requires_grad]
            assert any(g is not None and g.abs().sum() > 0 for g in gs), f"no gradient reached {name}"
    _()

    @check("masked losses: unlabeled batch -> no loss; partial labels -> only those tasks")
    def _():
        b = get_batch(ds, "clause")
        inp = lambda bb: {k: v for k, v in bb.items() if k != "meta"}
        b0 = dict(b); b0["cls_labels"] = torch.full_like(b["cls_labels"], IGNORE)
        assert "loss" not in m(**inp(b0)), "a batch with no labels must not produce a loss"
        o = m(**inp(b))
        assert set(o["loss_parts"]) == {"cls"}, o["loss_parts"].keys()
        mu = tiny_model(tok, loss_weighting="uncertainty")
        mu(**inp(b))["loss"].backward()
        assert mu.log_vars["cls"].grad is not None and mu.log_vars["span"].grad is None \
            and mu.log_vars["risk"].grad is None, "unlabelled task leaked into uncertainty weighting"
    _()

    @check("ordinal risk head: score read-out == level/4, perfect logits give ~0 loss")
    def _():
        levels = torch.tensor([[0, 1, 2, 4]])
        z = torch.full((1, 4, 4), -30.0)
        for j, k in enumerate(levels[0].tolist()):
            z[0, j, :k] = 30.0
        sc = m.risk_scores_from_logits(z.view(1, -1))
        assert torch.allclose(sc, levels.float() / 4, atol=1e-4), sc
        assert m.loss_risk(z.view(1, -1), levels) < 1e-6
        wrong = m.loss_risk(-z.view(1, -1), levels)
        assert wrong > 5
    _()

    @check("decode_span: finds planted span, respects context mask, start<=end")
    def _():
        s = np.full(30, -5.0); e = np.full(30, -5.0)
        s[7], e[11] = 5, 5
        mask = np.zeros(30, bool); mask[5:25] = True
        assert M.decode_span(s, e, mask) == (7, 11)
        s2 = s.copy(); s2[1] = 99                      # outside context -> must be ignored
        assert M.decode_span(s2, e, mask) == (7, 11)
        s3 = np.full(30, -5.0); e3 = np.full(30, -5.0); s3[20], e3[8] = 5, 5
        a, b = M.decode_span(s3, e3, mask)
        assert a <= b
    _()

    @check("metrics: unsupported classes are n/a (not 0); NaN-aware aggregation; empty span set")
    def _():
        r = M.classification_metrics([0, 0, 1, 2], [0, 1, 1, 2])
        assert np.isnan(r["per_class"]["Payment"]["f1"]) and r["per_class"]["Payment"]["support"] == 0
        assert r["macro_f1_n_classes"] == 3
        r2 = M.classification_metrics([0, 1], [0, 1])
        mean, std = M.aggregate([r, r2])
        assert np.isnan(mean["per_class"]["Payment"]["f1"]) and not np.isnan(mean["per_class"]["Assignment"]["f1"])
        assert M._token_f1((2, 5), (4, 8)) > 0 and M._token_f1((0, 1), (5, 6)) == 0
        assert np.isnan(M.span_metrics([], [])["token_f1"])
        rm = M.risk_metrics(np.array([[0, 1, 2, 3], [1, 1, 1, 1]]), np.full((2, 4), 0.25),
                            train_levels=np.array([[0, 0, 0, 0]] * 3))
        assert "baseline_median_mae" in rm["dims"]["unfairness"]
    _()

    @check("amp choice: fp16 on T4-class GPUs (cc 7.5), bf16 only on Ampere+, none on CPU")
    def _():
        from train_phase3 import choose_amp_dtype
        cuda, cpu = torch.device("cuda"), torch.device("cpu")
        assert choose_amp_dtype(cuda, "auto", (7, 5)) is torch.float16
        assert choose_amp_dtype(cuda, "auto", (8, 0)) is torch.bfloat16
        assert choose_amp_dtype(cpu, "auto") is None
    _()

    @check("tiny model can LEARN both heads, and --resume reproduces results without retraining")
    def _():
        (tmp / "big").mkdir(exist_ok=True)
        mapped2, cuad2 = make_fixture(tmp / "big", n_contracts=40, seed=1)
        out = tmp / "run"
        argv = ["--tiny", "--mapped_jsonl", str(mapped2), "--cuad_json", str(cuad2),
                "--out_dir", str(out), "--experiments", "cls_only", "span_only", "joint_cls_span",
                "--epochs", "30", "--patience", "100", "--lr", "3e-3", "--head_lr", "3e-3",
                "--bs_clause", "8", "--bs_span", "4", "--max_len_clause", "96", "--amp", "none",
                "--max_len_span", "256", "--window_chars", "500", "--test_frac", "0.25",
                "--val_frac", "0.15", "--seed", "3", "--save_ckpt"]
        res = train_main(argv)
        assert (out / "results.json").exists() and (out / "results.md").exists()
        assert (out / "partial" / "joint_cls_span_fold0.json").exists()
        agg = res["aggregate"]
        maj = res["baselines_cls"]["majority"]["mean"]["micro_f1"]
        for exp in ("cls_only", "joint_cls_span"):
            f1 = agg[exp]["cls"]["mean"]["micro_f1"]
            assert f1 > maj + 0.3 and f1 > 0.7, f"{exp} failed to learn easy categories: micro-F1 {f1:.3f} (majority {maj:.3f})"
        for exp in ("span_only", "joint_cls_span"):
            sp = agg[exp]["span"]["mean"]["token_f1"]
            rb = agg[exp]["span_random_baseline"]["mean"]["token_f1"]
            assert sp > rb + 0.12, f"{exp} span head did not beat random baseline: {sp:.3f} vs {rb:.3f}"
        lex = res["baselines_span"]["lexical_centroid_baseline"]["mean"]["token_f1"]
        whole = res["baselines_span"]["whole_window_baseline"]["mean"]["token_f1"]
        rb0 = agg["span_only"]["span_random_baseline"]["mean"]["token_f1"]
        assert lex > rb0 + 0.2 and 0 < whole < 1, f"span baselines look broken: lexical {lex:.3f} whole {whole:.3f} random {rb0:.3f}"
        from compare_runs import build_report
        rep = build_report(out)
        assert "joint_cls_span vs cls_only" in rep and "lexical_centroid_baseline" in rep, rep
        print(f"      span baselines: lexical {lex:.2f}, whole-window {whole:.2f}, random {rb0:.2f}")
        print(f"      learned: cls micro-F1 {agg['cls_only']['cls']['mean']['micro_f1']:.2f}, "
              f"span token-F1 {agg['span_only']['span']['mean']['token_f1']:.2f} "
              f"(random {agg['span_only']['span_random_baseline']['mean']['token_f1']:.2f})")
        from scan_vs_span import main as scan_main
        ck = str(out / "ckpt_joint_cls_span_fold0.pt")
        sres = scan_main(["--run_dir", str(out), "--fold", "0", "--ckpt_span", ck, "--ckpt_cls", ck,
                          "--mapped_jsonl", str(mapped2), "--cuad_json", str(cuad2), "--n_windows", "40",
                          "--tiny", "--n_boot", "200"])
        assert (out / "scan_vs_span_fold0.md").exists() and sres["n_windows"] > 10
        assert sres["classifier_scan"]["forward_passes_per_window"] > 3 and sres["span_head"]["forward_passes_per_window"] == 1
        assert 0 <= sres["span_head"]["token_f1"] <= 1 and 0 <= sres["classifier_scan"]["token_f1"] <= 1
        assert sres["candidate_ceiling_token_f1"] >= sres["classifier_scan"]["token_f1"] - 1e-9, "ceiling must bound scan"
        # ---- demo pipeline end to end on a fixture contract
        import json as _j
        from pipeline import ClauseXplain, render_html, segment
        cj = _j.loads(Path(cuad2).read_text())
        first = cj["data"][0]
        ctext = first["paragraphs"][0]["context"]
        units = segment(ctext)
        assert units and all(ctext[u["start"]:u["end"]] == u["text"] for u in units), "segment offsets broken"
        cx = ClauseXplain(ck, rules_csv=str(Path(__file__).with_name("indian_legal_rules.csv")), tiny=True, tiny_tokenizer_dir=str(out / "tiny_tokenizer"),
                          max_len_clause=96, max_len_span=256, window_chars=500)
        pr = cx.classify([u["text"] for u in units[:5]])
        assert pr.shape[1] == 18 and abs(pr.sum(1) - 1).max() < 1e-4, "classify must return softmax rows"
        from pipeline import NO_DATA
        from dataset import CATEGORIES
        assert all(pr[:, [i for i, c in enumerate(CATEGORIES) if c in NO_DATA]].max(1) < 1e-6), "no-data categories must be masked"
        res_p = cx.analyze(ctext, find=["Termination"], min_prob=0.0)
        html = render_html(res_p, "t")
        assert "<html" in html.lower() and "not legal advice" in html.lower()
        print(f"      pipeline: {len(units)} clause units, report {len(html)} chars")
        print(f"      scan vs span: head {sres['span_head']['token_f1']:.2f}, scan {sres['classifier_scan']['token_f1']:.2f}, "
              f"ceiling {sres['candidate_ceiling_token_f1']:.2f}")
        # ---- Indian test-set tooling: build from documents + CSV, then evaluate a checkpoint
        import csv as _csv
        from build_indian_testset import main as build_main
        from eval_indian import main as eval_main
        indd = tmp / "indian"; (indd / "documents").mkdir(parents=True)
        raw = json.loads(Path(cuad2).read_text())["data"][:8]
        slug2file = {}
        for i, c in enumerate(raw):
            (indd / "documents" / f"doc{i}.txt").write_text(c["paragraphs"][0]["context"], encoding="utf-8")
            slug2file[_slugify(c["title"])] = f"doc{i}.txt"
        rows_in = [json.loads(l) for l in Path(mapped2).read_text().splitlines()]
        keep = [r for r in rows_in if r["contract_id"] in slug2file]
        with open(indd / "clauses.csv", "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh); w.writerow(["document_file", "category", "clause_text", "source_note"])
            for k, r in enumerate(keep):
                cat = "Indemnification" if k == 0 else r["clausexplain_category"]       # one never-seen category
                w.writerow([slug2file[r["contract_id"]], cat, r["clause_text"].replace(" ", "  "), "fixture"])
            w.writerow(["doc0.txt", "Liability", "this text is not in the document at all", "bad row"])
        built = build_main(["--docs_dir", str(indd / "documents"), "--clauses_csv", str(indd / "clauses.csv"),
                            "--out_dir", str(indd / "built")])
        assert len(built) == len(keep), (len(built), len(keep))
        assert "not found" in (indd / "built" / "build_report.md").read_text(), "bad row must be reported"
        ev = eval_main(["--run_dir", str(out), "--fold", "0", "--ckpt", ck, "--indian_dir", str(indd / "built"),
                        "--mapped_jsonl", str(mapped2), "--cuad_json", str(cuad2), "--tiny"]
                       + (["--rules", "indian_legal_rules.csv"] if Path("indian_legal_rules.csv").exists() else []))
        assert ev["n_unseen"] == 1 and ev["cls"]["n"] >= 5 and "Indemnification" in ev["unseen_category_predictions"]
        assert (indd / "built" / "indian_eval.md").exists() and "tfidf_logreg" in ev["baselines_cls"]
        print(f"      indian set: {ev['n_total']} clauses built and evaluated end to end")
        res2 = train_main(argv + ["--resume"])
        assert abs(res2["aggregate"]["span_only"]["span"]["mean"]["token_f1"]
                   - agg["span_only"]["span"]["mean"]["token_f1"]) < 1e-9, "resume changed results"
        try:
            train_main(argv + ["--resume", "--epochs", "5"])
        except SystemExit:
            return
        raise AssertionError("--resume with different settings must refuse")
    _()


def run_real_tests(mapped_path: str, cuad_path: str):
    ex_map = load_mapped(mapped_path)
    texts = load_contract_texts(cuad_path)

    @check("REAL: all clauses_mapped offsets reproduce clause text from CUAD json")
    def _():
        st = attach_contract_text(ex_map, texts, strict=True)
        assert st.get("verified") == len(ex_map), st
    _()

    examples = [e for e in ex_map.values() if e.has_cls]

    @check("REAL: 12 populated categories, 6 zero-coverage categories")
    def _():
        have = {e.category for e in examples}
        assert len(have) == 12, sorted(have)
        assert set(CATEGORIES) - have == {"Confidentiality", "Force Majeure", "Indemnification",
                                          "Modification", "Payment", "Privacy / Data Protection"}
    _()

    @check("REAL: 5-fold grouped splits have no contract leakage")
    def _():
        for tr, va, te in make_splits(examples, 5, 0.1, 0.2, seed=13):
            assert_no_group_leak(examples, tr, va, te)
    _()

    @check("REAL: span label round-trip on 300 random clauses (6000-char windows)")
    def _():
        rng = random.Random(0)
        sample = rng.sample(examples, 300)
        tok = build_tiny_tokenizer([e.text for e in sample], 3000)
        ds = ClauseDataset(sample, tok, max_len_span=4096, span_mode="cuad_context",
                           window_chars=6000, views=("span",), seed=0)
        bad = trunc = 0
        for i in range(len(ds)):
            it = ds[i]
            if it["start_pos"] == IGNORE:
                trunc += 1
                continue
            offs, ctx = it["meta"]["offsets"], it["meta"]["context"]
            rec = ctx[offs[it["start_pos"]][0]:offs[it["end_pos"]][1]]
            ex = ds.items[i][1]
            if ex.text.strip() not in rec:
                bad += 1
        assert bad == 0, f"{bad}/{len(ds)} spans do not round-trip"
        print(f"      {len(ds)} checked, {trunc} truncated by max length")
    _()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--mapped_jsonl", default="clauses_mapped.jsonl")
    ap.add_argument("--cuad_json", default="CUAD_v1.json")
    a, _ = ap.parse_known_args()
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        run_fixture_tests(Path(td))
    if a.real:
        run_real_tests(a.mapped_jsonl, a.cuad_json)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    if bad:
        print("FAILED:", *[f"\n  - {n}: {m}" for n, _, m in bad])
        sys.exit(1)
    print("Plumbing OK. This does not show the model performs well; see the header of smoke_test.py.")


if __name__ == "__main__":
    main()
