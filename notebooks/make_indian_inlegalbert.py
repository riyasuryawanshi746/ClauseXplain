# Regenerates notebooks/indian_inlegalbert_kaggle.ipynb (step 6: InLegalBERT encoder; embeds phase3 code + built_v4 + built_judgments).
# Run from repo root after the code or the Indian set changes.
import base64, io, json, zipfile
from pathlib import Path

CODE = ["dataset.py", "model.py", "metrics.py", "train_phase3.py", "eval_indian.py", "scan_vs_span.py",
        "rules_engine.py", "build_indian_testset.py", "finetune_indian.py", "ledgar_extra.py"]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in CODE:
        z.write(f"phase3_model/{f}", f)
    for d in ["built_v4", "built_judgments"]:
        for f in ["indian_mapped.jsonl", "build_report.md"]:   # contract texts not needed (keeps the notebook < 1 MB for Save Version)
            z.write(f"indian_set/{d}/{f}", f"{d}/{f}")
blob = base64.b64encode(buf.getvalue()).decode()

md = """# ClauseXplain step 6: InLegalBERT encoder, document-held-out Indian test
InLegalBERT (`law-ai/InLegalBERT`, Paul et al. 2023: LEGAL-BERT further pretrained on Indian court judgments; same
tokenizer and size as LEGAL-BERT). Same protocol and settings as the LEGAL-BERT runs: train on CUAD fold 0, then on
Indian clauses with whole documents held out (12 folds, 8 epochs, no tuning on these results). Three seeds for
CUAD -> Indian, plus one run with LEDGAR extra data (same mapping/cap as step 5).

Settings: attach `clausexplain-indian-eval` and `scan-vs-span`; **GPU T4**; **Internet ON**. Run all; download
`indian_inlegalbert_results.zip`. About 1 h. Resumable."""

setup = f'''import os, shutil, base64, io, zipfile, subprocess, sys, json
BASE = "/kaggle/input/datasets/riyasuryawanshi"
if not os.path.exists(BASE): BASE = "/kaggle/input"
EVAL_SRC, MODEL_SRC = f"{{BASE}}/clausexplain-indian-eval", f"{{BASE}}/scan-vs-span"
WORK, OUT = "/kaggle/working/clausexplain", "/kaggle/working/inlegalbert_runs"
LB_RUN = "/kaggle/working/inlegalbert_cuad"
LEGALBERT = "law-ai/InLegalBERT"   # variable name kept so the CUAD cell is unchanged
if os.path.exists(WORK): shutil.rmtree(WORK)
shutil.copytree(EVAL_SRC, WORK)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{blob}"))).extractall(WORK)
os.makedirs(OUT, exist_ok=True)
assert os.path.exists(f"{{MODEL_SRC}}/clauses_mapped.jsonl"), "attach the scan-vs-span dataset"
from transformers import AutoTokenizer
import torch, transformers
from transformers import AutoModel
print("torch", torch.__version__, "transformers", transformers.__version__)
AutoTokenizer.from_pretrained(LEGALBERT)   # fails here if Internet is off
AutoModel.from_pretrained(LEGALBERT)       # InLegalBERT ships pytorch_model.bin only; fails here if this torch/transformers pair refuses it
LEDGAR = "/kaggle/working/ledgar_mapped.jsonl"
r = subprocess.run([sys.executable, "ledgar_extra.py", "--out", LEDGAR, "--parquet", "/kaggle/working/ledgar_train.parquet"],
                   capture_output=True, text=True, cwd=WORK); print(r.stdout, r.stderr[-1500:])
assert os.path.exists(LEDGAR)
# (name, method, extra args). Comment out lines to run fewer.
LB = ["--encoder", LEGALBERT, "--init_ckpt", f"{{LB_RUN}}/ckpt_cls_only_fold0.pt"]
LG = ["--extra_jsonl", LEDGAR, "--extra_cap", "100"]
MISSING = ["--extra_categories", "Confidentiality", "Indemnification", "Modification", "Payment", "Dispute Resolution"]
RUNS = [
    ("inlegalbert_cuad_indian_s13",        "neural", LB + ["--seed", "13"]),
    ("inlegalbert_cuad_indian_s14",        "neural", LB + ["--seed", "14"]),
    ("inlegalbert_cuad_indian_s15",        "neural", LB + ["--seed", "15"]),
    ("inlegalbert_cuad_indian_ledgar_s13", "neural", LB + LG + ["--seed", "13"]),
]
print("ready;", len(RUNS), "configurations")'''

cuad_lb = '''# InLegalBERT trained on CUAD fold 0 (same split and settings as the zero-shot LegalBERT run), ~4 min.
os.chdir(WORK)
need = True
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
print("InLegalBERT CUAD checkpoint:", os.path.exists(f"{LB_RUN}/ckpt_cls_only_fold0.pt"))'''

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

zipc = '''with zipfile.ZipFile("/kaggle/working/indian_inlegalbert_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/summary.md", "summary.md")
    for f in ["results.json", "results.md"]:                  # CUAD fold-0 score of the encoder used as init
        if os.path.exists(f"{LB_RUN}/{f}"): z.write(f"{LB_RUN}/{f}", f"cuad_fold0_{f}")
    for name, _, _ in RUNS:
        for f in ["metrics.json", "predictions.csv"]:
            p = f"{OUT}/{name}/{f}"
            if os.path.exists(p): z.write(p, f"{name}/{f}")
print("download /kaggle/working/indian_inlegalbert_results.zip from the Output tab")'''


def cell(t, s):
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
    if t == "code":
        c.update(outputs=[], execution_count=None)
    return c


nb = {"cells": [cell("markdown", md)] + [cell("code", s) for s in (setup, cuad_lb, run_all, summ, zipc)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/indian_inlegalbert_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
for c in nb["cells"][1:]:
    compile("".join(c["source"]), "cell", "exec")
print("embedded", len(blob) // 1024, "KB; all cells compile")
