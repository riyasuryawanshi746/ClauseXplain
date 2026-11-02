# Regenerates notebooks/demo_model_kaggle.ipynb (trains the demo classifier on all Indian clauses; embeds phase3 code + data).
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

md = """# ClauseXplain demo model: LEGAL-BERT trained on CUAD, then on all 334 Indian clauses
Trains the classifier used by the demo (`pipeline.py --cls_model`). Same settings as the document-held-out runs, but
on ALL Indian clauses, so it has no held-out score of its own: quote the held-out estimate from
`experiments/indian_finetune/` (and `indian_ledgar/`). Two variants: without and with LEDGAR extra data; pick one for
the demo after the step-5 results are in.

Settings: attach `clausexplain-indian-eval` and `scan-vs-span`; **GPU T4**; **Internet ON**. About 15 min.
Download `demo_model_legalbert.zip` and/or `demo_model_legalbert_ledgar.zip` (about 200 MB each)."""

setup = f'''import os, shutil, base64, io, zipfile, subprocess, sys, json
BASE = "/kaggle/input/datasets/riyasuryawanshi"
if not os.path.exists(BASE): BASE = "/kaggle/input"
EVAL_SRC, MODEL_SRC = f"{{BASE}}/clausexplain-indian-eval", f"{{BASE}}/scan-vs-span"
WORK, OUT = "/kaggle/working/clausexplain", "/kaggle/working/demo_models"
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
    ("demo_model_legalbert",        LB),
    ("demo_model_legalbert_ledgar", LB + LG),
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
for name, extra in RUNS:
    if os.path.exists(f"{OUT}/{name}/meta.json"):
        print("skip (done):", name); continue
    cmd = [sys.executable, "finetune_indian.py", "final", "--indian_dir", f"{WORK}/built_v4",
           "--judgments_dir", f"{WORK}/built_judgments", "--save_dir", f"{OUT}/{name}"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(name, r.stdout[-400:], r.stderr[-1500:] if r.returncode else "")'''

summ = '''for name, _ in RUNS:
    p = f"{OUT}/{name}/meta.json"
    print(name, open(p).read()[-600:] if os.path.exists(p) else "MISSING")'''

zipc = '''for name, _ in RUNS:
    d = f"{OUT}/{name}"
    if not os.path.exists(f"{d}/meta.json"): continue
    with zipfile.ZipFile(f"/kaggle/working/{name}.zip", "w", zipfile.ZIP_STORED) as z:
        for f in os.listdir(d): z.write(f"{d}/{f}", f"{name}/{f}")
    print("download", f"/kaggle/working/{name}.zip", round(os.path.getsize(f"/kaggle/working/{name}.zip") / 1e6), "MB")'''


def cell(t, s):
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
    if t == "code":
        c.update(outputs=[], execution_count=None)
    return c


nb = {"cells": [cell("markdown", md)] + [cell("code", s) for s in (setup, cuad_lb, run_all, summ, zipc)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/demo_model_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
for c in nb["cells"][1:]:
    compile("".join(c["source"]), "cell", "exec")
print("embedded", len(blob) // 1024, "KB; all cells compile")
