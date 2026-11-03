# Regenerates notebooks/indian_v5_kaggle.ipynb (v5 = v4 + batch 3, 788 clauses; embeds phase3 code + built_v5 + built_judgments).
# Run from repo root. The notebook embeds batch 3 clause text (private names), so it is gitignored: keep the Kaggle notebook private.
import base64, io, json, zipfile
from pathlib import Path

CODE = ["dataset.py", "model.py", "metrics.py", "train_phase3.py", "eval_indian.py", "scan_vs_span.py",
        "rules_engine.py", "build_indian_testset.py", "finetune_indian.py", "ledgar_extra.py"]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in CODE:
        z.write(f"phase3_model/{f}", f)
    for d in ["built_v5", "built_judgments"]:
        for f in ["indian_mapped.jsonl", "build_report.md"]:   # contract texts not needed (keeps the notebook < 1 MB for Save Version)
            z.write(f"indian_set/{d}/{f}", f"{d}/{f}")
blob = base64.b64encode(buf.getvalue()).decode()

md = """# ClauseXplain v5: 788 Indian clauses (v4 + batch 3), document-held-out
Same protocol and settings as `indian_ledgar_kaggle.ipynb` (8 epochs, no tuning on these results), now 23 folds
(batch 3 template groups azure / leave_license / dec2022 held out together). The summary also scores the 334 v4
clauses alone, so the effect of more training data can be compared with the v4 runs on the same test clauses.
**Keep this notebook private** (batch 3 clauses name private individuals).

Settings: attach `clausexplain-indian-eval` and `scan-vs-span`; **GPU T4**; **Internet ON** (LegalBERT weights and the
LEDGAR parquet, ~21 MB). Run all; download `indian_v5_results.zip`. About 1.5 h per run (2 runs; two optional seed-14 runs are commented out). Resumable."""

setup = f'''import os, shutil, base64, io, zipfile, subprocess, sys, json
BASE = "/kaggle/input/datasets/riyasuryawanshi"
if not os.path.exists(BASE): BASE = "/kaggle/input"
EVAL_SRC, MODEL_SRC = f"{{BASE}}/clausexplain-indian-eval", f"{{BASE}}/scan-vs-span"
WORK, OUT = "/kaggle/working/clausexplain", "/kaggle/working/v5_runs"
LB_RUN = "/kaggle/working/legalbert_cuad"
LEGALBERT, LONGFORMER = "nlpaueb/legal-bert-base-uncased", "allenai/longformer-base-4096"
if os.path.exists(WORK): shutil.rmtree(WORK)
shutil.copytree(EVAL_SRC, WORK)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{blob}"))).extractall(WORK)
os.makedirs(OUT, exist_ok=True)
assert os.path.exists(f"{{MODEL_SRC}}/clauses_mapped.jsonl"), "attach the scan-vs-span dataset"
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(LEGALBERT)   # fails here if Internet is off
LEDGAR = "/kaggle/working/ledgar_mapped.jsonl"
r = subprocess.run([sys.executable, "ledgar_extra.py", "--out", LEDGAR, "--parquet", "/kaggle/working/ledgar_train.parquet"],
                   capture_output=True, text=True, cwd=WORK); print(r.stdout, r.stderr[-1500:])
assert os.path.exists(LEDGAR)
# (name, method, extra args). Comment out lines to run fewer.
LB = ["--encoder", LEGALBERT, "--init_ckpt", f"{{LB_RUN}}/ckpt_cls_only_fold0.pt"]
LG = ["--extra_jsonl", LEDGAR, "--extra_cap", "100"]
MISSING = ["--extra_categories", "Confidentiality", "Indemnification", "Modification", "Payment", "Dispute Resolution"]
RUNS = [
    ("v5_legalbert_cuad_indian_s13",        "neural", LB + ["--seed", "13"]),
    ("v5_legalbert_cuad_indian_ledgar_s13", "neural", LB + LG + ["--seed", "13"]),
#   ("v5_legalbert_cuad_indian_s14",        "neural", LB + ["--seed", "14"]),
#   ("v5_legalbert_cuad_indian_ledgar_s14", "neural", LB + LG + ["--seed", "14"]),
]
print("ready;", len(RUNS), "configurations")'''

cuad_lb = '''# LegalBERT trained on CUAD fold 0 (same split and settings as the zero-shot LegalBERT run), ~4 min.
# Only needed for the legalbert_cuad_* configurations.
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
print("LegalBERT CUAD checkpoint:", os.path.exists(f"{LB_RUN}/ckpt_cls_only_fold0.pt"))'''

run_all = '''os.chdir(WORK)
for name, method, extra in RUNS:
    if os.path.exists(f"{OUT}/{name}/metrics.json"):
        print("skip (done):", name); continue
    cmd = [sys.executable, "finetune_indian.py", "run", "--method", method, "--name", name,
           "--indian_dir", f"{WORK}/built_v5", "--judgments_dir", f"{WORK}/built_judgments", "--out_dir", OUT] + extra
    print("=== ", name, flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if line.startswith(("fold", name)) or "Error" in line or "error" in line: print(line, end="", flush=True)
    print("exit", p.wait())'''

summ = '''r = subprocess.run([sys.executable, "finetune_indian.py", "summarise", "--out_dir", OUT], capture_output=True, text=True, cwd=WORK)
print(r.stdout, r.stderr[-1000:])
# the 334 v4 clauses alone (same test clauses as the v4 runs; batch 3 documents only add training data there)
from sklearn.metrics import f1_score
BATCH3 = ("indian_azure_", "indian_leave_license_", "indian_dec2022_", "indian_concord_", "indian_cyient_",
          "indian_hinduja_", "indian_infosys_springboard_", "indian_kaynes_", "indian_majesco_", "indian_mmfsl_",
          "indian_share_subscription_purchase_executed")
import csv
lines = ["", "## v4 clauses only (n=334) and batch 3 only", "", "| run | v4 acc | v4 macro-F1 | batch3 acc | batch3 macro-F1 |", "|---|---|---|---|---|"]
for name, _, _ in RUNS:
    p = f"{OUT}/{name}/predictions.csv"
    if not os.path.exists(p): continue
    rs = list(csv.DictReader(open(p, encoding="utf-8"))); cells = []
    for sub in ([r for r in rs if not r["document"].startswith(BATCH3)], [r for r in rs if r["document"].startswith(BATCH3)]):
        y, q = [r["true_category"] for r in sub], [r["pred_category"] for r in sub]
        cells += [f"{sum(a == b for a, b in zip(y, q)) / len(y):.3f} (n={len(y)})", f"{f1_score(y, q, average='macro', labels=sorted(set(y))):.3f}"]
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
open(f"{OUT}/summary.md", "a", encoding="utf-8").write("\\n".join(lines) + "\\n"); print("\\n".join(lines))'''

zipc = '''with zipfile.ZipFile("/kaggle/working/indian_v5_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(f"{OUT}/summary.md", "summary.md")
    for f in ["results.json", "results.md"]:                  # CUAD fold-0 score of the encoder used as init
        if os.path.exists(f"{LB_RUN}/{f}"): z.write(f"{LB_RUN}/{f}", f"cuad_fold0_{f}")
    for name, _, _ in RUNS:
        for f in ["metrics.json", "predictions.csv"]:
            p = f"{OUT}/{name}/{f}"
            if os.path.exists(p): z.write(p, f"{name}/{f}")
print("download /kaggle/working/indian_v5_results.zip from the Output tab")'''


def cell(t, s):
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
    if t == "code":
        c.update(outputs=[], execution_count=None)
    return c


nb = {"cells": [cell("markdown", md)] + [cell("code", s) for s in (setup, cuad_lb, run_all, summ, zipc)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/indian_v5_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
for c in nb["cells"][1:]:
    compile("".join(c["source"]), "cell", "exec")
print("embedded", len(blob) // 1024, "KB; all cells compile")
