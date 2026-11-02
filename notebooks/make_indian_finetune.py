# Regenerates notebooks/indian_finetune_kaggle.ipynb (embeds current phase3 code + built_v4 + built_judgments).
# Run from repo root after the code or the Indian set changes.
import base64, io, json, zipfile
from pathlib import Path

CODE = ["dataset.py", "model.py", "metrics.py", "train_phase3.py", "eval_indian.py", "scan_vs_span.py",
        "rules_engine.py", "build_indian_testset.py", "finetune_indian.py"]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in CODE:
        z.write(f"phase3_model/{f}", f)
    for d in ["built_v4", "built_judgments"]:
        for f in ["indian_mapped.jsonl", "build_report.md"]:   # contract texts not needed (keeps the notebook < 1 MB for Save Version)
            z.write(f"indian_set/{d}/{f}", f"{d}/{f}")
blob = base64.b64encode(buf.getvalue()).decode()

md = """# ClauseXplain: training on Indian clauses, testing on held-out Indian documents
Leave-one-document-group-out (12 folds; the two MEA agreements share a template and are held out together).
Every Indian clause is predicted once by a model that never saw its document. Settings are fixed in
`finetune_indian.py` (8 epochs, no tuning on these results). Zero-shot results stay in the paper next to these.

Settings: attach `clausexplain-indian-eval` and `scan-vs-span`; **GPU T4**; **Internet ON** (downloads LegalBERT /
Longformer base weights). Run all; download `indian_finetune_results.zip`. About 1.5 h in total.
The run is resumable: re-running skips configurations whose `metrics.json` already exists."""

setup = f'''import os, shutil, base64, io, zipfile, subprocess, sys, json
BASE = "/kaggle/input/datasets/riyasuryawanshi"
if not os.path.exists(BASE): BASE = "/kaggle/input"
EVAL_SRC, MODEL_SRC = f"{{BASE}}/clausexplain-indian-eval", f"{{BASE}}/scan-vs-span"
WORK, OUT = "/kaggle/working/clausexplain", "/kaggle/working/finetune_runs"
LB_RUN = "/kaggle/working/legalbert_cuad"
LEGALBERT, LONGFORMER = "nlpaueb/legal-bert-base-uncased", "allenai/longformer-base-4096"
LF_CKPT = f"{{MODEL_SRC}}/ckpt_joint_cls_span_fold0.pt"          # CUAD-trained Longformer (fold 0)
if os.path.exists(WORK): shutil.rmtree(WORK)
shutil.copytree(EVAL_SRC, WORK)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{blob}"))).extractall(WORK)
os.makedirs(OUT, exist_ok=True)
assert os.path.exists(LF_CKPT), "attach the scan-vs-span dataset"
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(LEGALBERT); AutoTokenizer.from_pretrained(LONGFORMER)   # fails here if Internet is off
# (name, method, extra args). Comment out lines to run fewer.
RUNS = [
    ("tfidf_indian",                 "tfidf",  []),
    ("tfidf_indian_balanced",        "tfidf",  ["--tricks"]),
    ("tfidf_cuad_indian",            "tfidf",  ["--cuad_mapped", f"{{MODEL_SRC}}/clauses_mapped.jsonl"]),
    ("tfidf_cuad_indian_balanced",   "tfidf",  ["--cuad_mapped", f"{{MODEL_SRC}}/clauses_mapped.jsonl", "--tricks"]),
    ("legalbert_indian",             "neural", ["--encoder", LEGALBERT]),
    ("legalbert_indian_tricks",      "neural", ["--encoder", LEGALBERT, "--tricks"]),
    ("legalbert_cuad_indian",        "neural", ["--encoder", LEGALBERT, "--init_ckpt", f"{{LB_RUN}}/ckpt_cls_only_fold0.pt"]),
    ("legalbert_cuad_indian_tricks", "neural", ["--encoder", LEGALBERT, "--init_ckpt", f"{{LB_RUN}}/ckpt_cls_only_fold0.pt", "--tricks"]),
    ("longformer_cuad_indian",       "neural", ["--encoder", LONGFORMER, "--init_ckpt", LF_CKPT]),
    ("longformer_cuad_indian_tricks","neural", ["--encoder", LONGFORMER, "--init_ckpt", LF_CKPT, "--tricks"]),
]
print("ready;", len(RUNS), "configurations")'''

cuad_lb = '''# LegalBERT trained on CUAD fold 0 (same split and settings as the zero-shot LegalBERT run), ~4 min.
# Only needed for the legalbert_cuad_* configurations.
os.chdir(WORK)
need = any("legalbert_cuad" in n for n, _, _ in RUNS)
if need and not os.path.exists(f"{LB_RUN}/ckpt_cls_only_fold0.pt"):
    os.makedirs(f"{LB_RUN}/partial", exist_ok=True)
    shutil.copy(f"{WORK}/partial/splits.json", f"{LB_RUN}/partial/splits.json")
    LF = json.load(open(f"{WORK}/partial/config.json"))
    cmd = [sys.executable, "train_phase3.py", "--cuad_json", f"{MODEL_SRC}/CUAD_v1.json",
           "--mapped_jsonl", f"{MODEL_SRC}/clauses_mapped.jsonl", "--out_dir", LB_RUN, "--encoder", LEGALBERT,
           "--experiments", "cls_only", "--cv_folds", "5", "--folds", "0", "--max_len_span", "512",
           "--pad_multiple", "64", "--resume", "--save_ckpt", "--skip_baselines"]
    for k in ["epochs", "lr", "head_lr", "seed", "test_frac", "val_frac", "bs_clause", "grad_accum", "class_weights", "max_len_clause"]:
        if LF.get(k) is not None: cmd += [f"--{k}", str(LF[k])]
    r = subprocess.run(cmd, capture_output=True, text=True); print(r.stdout[-800:], r.stderr[-1500:] if r.returncode else "")
print("LegalBERT CUAD checkpoint:", os.path.exists(f"{LB_RUN}/ckpt_cls_only_fold0.pt"))'''

run_all = '''os.chdir(WORK)
for name, method, extra in RUNS:
    if os.path.exists(f"{OUT}/{name}/metrics.json"):
        print("skip (done):", name); continue
    cmd = [sys.executable, "finetune_indian.py", "run", "--method", method, "--name", name,
           "--indian_dir", f"{WORK}/built_v4", "--judgments_dir", f"{WORK}/built_judgments", "--out_dir", OUT] + extra
    print("=== ", name, flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if line.startswith(("fold", name)) or "Error" in line or "error" in line: print(line, end="", flush=True)
    print("exit", p.wait())'''

summ = '''r = subprocess.run([sys.executable, "finetune_indian.py", "summarise", "--out_dir", OUT], capture_output=True, text=True, cwd=WORK)
print(r.stdout, r.stderr[-1000:])'''

zipc = '''with zipfile.ZipFile("/kaggle/working/indian_finetune_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/summary.md", "summary.md")
    for name, _, _ in RUNS:
        for f in ["metrics.json", "predictions.csv"]:
            p = f"{OUT}/{name}/{f}"
            if os.path.exists(p): z.write(p, f"{name}/{f}")
print("download /kaggle/working/indian_finetune_results.zip from the Output tab")'''


def cell(t, s):
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
    if t == "code":
        c.update(outputs=[], execution_count=None)
    return c


nb = {"cells": [cell("markdown", md)] + [cell("code", s) for s in (setup, cuad_lb, run_all, summ, zipc)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/indian_finetune_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
for c in nb["cells"][1:]:
    compile("".join(c["source"]), "cell", "exec")
print("embedded", len(blob) // 1024, "KB; all cells compile")
