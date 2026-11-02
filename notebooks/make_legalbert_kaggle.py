# Regenerates notebooks/legalbert_baseline_v4_kaggle.ipynb (embeds current phase3 code + built_v4 + built_judgments).
# Run from repo root after the code or the Indian set changes.
import base64, io, json, zipfile
from pathlib import Path

CODE = ["dataset.py", "model.py", "metrics.py", "train_phase3.py", "eval_indian.py", "scan_vs_span.py",
        "rules_engine.py", "build_indian_testset.py", "summarise_predictions.py"]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in CODE:
        z.write(f"phase3_model/{f}", f)
    for d in ["built_v4", "built_judgments"]:
        for f in ["indian_mapped.jsonl", "indian_contracts.json", "build_report.md"]:
            z.write(f"indian_set/{d}/{f}", f"{d}/{f}")
blob = base64.b64encode(buf.getvalue()).decode()

md = """# ClauseXplain: LegalBERT classification baseline (CUAD folds 0-2 + Indian set)
Trains `nlpaueb/legal-bert-base-uncased` as a classifier only (`cls_only`) on the SAME CUAD folds as the Longformer runs
(the saved `splits.json` is reused), with the same epochs / learning rates / seed, then scores the Indian main set
(334 clauses) and the judgment slice. No span head: BERT only takes 512 tokens.

Settings: attach `clausexplain-indian-eval` and `scan-vs-span`; **GPU T4**; **Internet ON** (downloads LegalBERT from
Hugging Face). Run all, then download `legalbert_results_v4.zip` from the Output tab.
Roughly 15-30 min per fold. Set FOLDS = [0] for a quick first check."""

setup = f'''import os, shutil, base64, io, zipfile, subprocess, sys, json
BASE = "/kaggle/input/datasets/riyasuryawanshi"
if not os.path.exists(BASE): BASE = "/kaggle/input"   # older Kaggle mount layout
EVAL_SRC, MODEL_SRC = f"{{BASE}}/clausexplain-indian-eval", f"{{BASE}}/scan-vs-span"
WORK, RUN = "/kaggle/working/clausexplain", "/kaggle/working/legalbert"
ENCODER = "nlpaueb/legal-bert-base-uncased"
FOLDS = [0, 1, 2]
for d in (WORK, RUN):
    if os.path.exists(d): shutil.rmtree(d)
shutil.copytree(EVAL_SRC, WORK)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{blob}"))).extractall(WORK)   # current code + Indian set
os.makedirs(f"{{RUN}}/partial")
shutil.copy(f"{{WORK}}/partial/splits.json", f"{{RUN}}/partial/splits.json")   # identical CUAD folds
LF = json.load(open(f"{{WORK}}/partial/config.json"))
print("Longformer run settings reused:", {{k: LF.get(k) for k in ["epochs", "lr", "head_lr", "seed", "test_frac", "val_frac",
                                                                 "bs_clause", "grad_accum", "class_weights", "max_len_clause"]}})
assert os.path.exists(f"{{MODEL_SRC}}/CUAD_v1.json"), "attach the scan-vs-span dataset"
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(ENCODER)   # fails here if Internet is off
print("LegalBERT reachable")'''

train = '''os.chdir(WORK)
opt = lambda k, flag=None: [f"--{flag or k}", str(LF[k])] if LF.get(k) is not None else []
cmd = [sys.executable, "train_phase3.py", "--cuad_json", f"{MODEL_SRC}/CUAD_v1.json",
       "--mapped_jsonl", f"{MODEL_SRC}/clauses_mapped.jsonl", "--out_dir", RUN, "--encoder", ENCODER,
       "--experiments", "cls_only", "--cv_folds", "5", "--folds", *map(str, FOLDS),
       "--max_len_span", "512", "--pad_multiple", "64", "--resume", "--save_ckpt", "--skip_baselines"]
for k in ["epochs", "lr", "head_lr", "seed", "test_frac", "val_frac", "bs_clause", "grad_accum", "class_weights", "max_len_clause"]:
    cmd += opt(k)
print(" ".join(cmd))
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if "it/s" not in line and "s/it" not in line: print(line, end="")
assert p.wait() == 0, "training failed (see log above)"'''

cuad = '''# CUAD test numbers per fold (compare: Longformer cls_only 0.839 macro-F1, TF-IDF 0.802, 3-fold means)
import statistics as st
rows = [json.load(open(f"{RUN}/partial/cls_only_fold{k}.json"))["cls"] for k in FOLDS]
for k, c in zip(FOLDS, rows):
    print(f"fold {k}: macro-F1 {c['macro_f1']:.3f}  micro-F1 {c['micro_f1']:.3f}  n {c['n']}")
print(f"mean: macro-F1 {st.mean(c['macro_f1'] for c in rows):.3f}  micro-F1 {st.mean(c['micro_f1'] for c in rows):.3f}")'''

indian = '''SETS = {"main": "built_v4", "judgments": "built_judgments"}
for k in FOLDS:
    for tag, src in SETS.items():
        out = f"{WORK}/{tag}_fold{k}"
        shutil.copytree(f"{WORK}/{src}", out, dirs_exist_ok=True)
        cmd = [sys.executable, "eval_indian.py", "--run_dir", RUN, "--fold", str(k), "--cls_only",
               "--ckpt", f"{RUN}/ckpt_cls_only_fold{k}.pt", "--indian_dir", out,
               "--mapped_jsonl", f"{MODEL_SRC}/clauses_mapped.jsonl", "--cuad_json", f"{MODEL_SRC}/CUAD_v1.json",
               "--clause_batch", "16"]
        print("=== fold", k, tag, "==="); r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-200:], r.stderr[-1500:] if r.returncode else "")'''

summ = r'''# LegalBERT accuracy on seen categories by source (pre-adjudication LegalBERT means: all 0.349, original 0.375, batch 2 0.296, judgments 0.545)
import csv, math
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)
BATCH2 = ("indian_cpwd_gcc_2023", "indian_sbm_waste_to_compost_mca")
print("| fold | slice | n | accuracy | 95% interval | macro-F1 (whole set) |\n|---|---|---|---|---|---|")
for k in FOLDS:
    for tag in ["main", "judgments"]:
        rows = [r for r in csv.DictReader(open(f"{WORK}/{tag}_fold{k}/indian_predictions.csv", encoding="utf-8")) if r["seen_category"] == "1"]
        mf1 = json.load(open(f"{WORK}/{tag}_fold{k}/indian_eval.json"))["cls"]["macro_f1"]
        sl = {"all 334 (main)": rows, "original 228": [r for r in rows if r["document"] not in BATCH2],
              "batch 2 only": [r for r in rows if r["document"] in BATCH2]} if tag == "main" else {"judgments": rows}
        for name, rs in sl.items():
            n = len(rs); c = sum(int(r["correct"]) for r in rs); lo, hi = wilson(c, n)
            extra = f"{mf1:.3f}" if name in ("all 334 (main)", "judgments") else ""
            print(f"| {k} | {name} | {n} | " + (f"{c/n:.3f} | {lo:.3f} to {hi:.3f} | {extra} |" if n else "n/a | n/a | |"))'''

zipc = '''with zipfile.ZipFile("/kaggle/working/legalbert_results_v4.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["results.json", "results.md", "partial/config.json"] + [f"partial/cls_only_fold{k}.json" for k in FOLDS]:
        if os.path.exists(f"{RUN}/{f}"): z.write(f"{RUN}/{f}", f"cuad/{f}")
    for k in FOLDS:
        for tag in ["main", "judgments"]:
            for f in ["indian_eval.json", "indian_eval.md", "indian_predictions.csv"]:
                p = f"{WORK}/{tag}_fold{k}/{f}"
                if os.path.exists(p): z.write(p, f"{tag}/fold{k}/{f}")
print("download /kaggle/working/legalbert_results_v4.zip from the Output tab (checkpoints are not included)")'''


def cell(t, s):
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
    if t == "code":
        c.update(outputs=[], execution_count=None)
    return c


nb = {"cells": [cell("markdown", md)] + [cell("code", s) for s in (setup, train, cuad, indian, summ, zipc)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/legalbert_baseline_v4_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
for c in nb["cells"][1:]:
    compile("".join(c["source"]), "cell", "exec")
print("embedded", len(blob) // 1024, "KB; all cells compile")
