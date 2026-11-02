# Regenerates notebooks/indian_eval_v3_kaggle.ipynb (embeds built_v3 + built_judgments). Run from repo root after the Indian set changes.
import base64, io, json, zipfile
from pathlib import Path

def b64(p): return base64.b64encode(Path(p).read_bytes()).decode()
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ["built_v3", "built_judgments"]:
        for f in ["indian_mapped.jsonl", "indian_contracts.json", "build_report.md"]:
            z.write(f"indian_set/{d}/{f}", f"{d}/{f}")
data_b64 = base64.b64encode(buf.getvalue()).decode()
scripts = {n: b64(f"phase3_model/{n}") for n in ["eval_indian.py", "summarise_predictions.py"]}

md = """# ClauseXplain: Indian-set evaluation v3 (334 clauses + judgment slice)
Attach the SAME two datasets as v2: `clausexplain-indian-eval` and `scan-vs-span`. GPU T4. Internet not needed.
The new test data (built_v3 = 228 old + 106 batch-2 clauses; built_judgments = 11 clauses) is embedded in this notebook,
so nothing has to be re-uploaded. Run all cells, then download `indian_eval_results_v3.zip` from the Output tab.
Set FOLDS to `[0]` if it runs too slowly."""

setup = f'''import os, shutil, base64, io, zipfile, subprocess, sys
BASE = "/kaggle/input/datasets/riyasuryawanshi"
if not os.path.exists(BASE): BASE = "/kaggle/input"   # older Kaggle mount layout
EVAL_SRC, MODEL_SRC = f"{{BASE}}/clausexplain-indian-eval", f"{{BASE}}/scan-vs-span"
WORK = "/kaggle/working/clausexplain"
FOLDS = [0, 1, 2]
if os.path.exists(WORK): shutil.rmtree(WORK)
shutil.copytree(EVAL_SRC, WORK)
for name, b in {json.dumps(scripts)}.items():
    open(f"{{WORK}}/{{name}}", "wb").write(base64.b64decode(b))
zipfile.ZipFile(io.BytesIO(base64.b64decode("{data_b64}"))).extractall(WORK)
for d in ["built_v3", "built_judgments"]:
    n = sum(1 for _ in open(f"{{WORK}}/{{d}}/indian_mapped.jsonl", encoding="utf-8"))
    print(d, n, "clauses")
assert os.path.exists(f"{{MODEL_SRC}}/ckpt_joint_cls_span_fold0.pt"), "attach the scan-vs-span dataset"'''

run = '''os.chdir(WORK)
SETS = {"main": ("built_v3", True), "judgments": ("built_judgments", False)}   # (folder, run rule layer?)
for k in FOLDS:
    for tag, (src, rules) in SETS.items():
        out = f"{WORK}/{tag}_fold{k}"
        shutil.copytree(f"{WORK}/{src}", out, dirs_exist_ok=True)
        cmd = [sys.executable, "eval_indian.py", "--run_dir", WORK, "--fold", str(k),
               "--ckpt", f"{MODEL_SRC}/ckpt_joint_cls_span_fold{k}.pt", "--indian_dir", out,
               "--mapped_jsonl", f"{MODEL_SRC}/clauses_mapped.jsonl", "--cuad_json", f"{MODEL_SRC}/CUAD_v1.json",
               "--clause_batch", "2", "--span_batch", "1"] + (["--rules", f"{WORK}/indian_legal_rules.csv"] if rules else [])
        print("=== fold", k, tag, "==="); r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-300:], r.stderr[-1500:] if r.returncode else "")'''

summ = r'''# Accuracy on seen categories, split by where the clauses came from.
# "original 228" should land near the v2 numbers (0.377 mean acc); if not, something changed.
import csv, math
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), min(1, c+h)
BATCH2 = ("indian_cpwd_gcc_2023", "indian_sbm_waste_to_compost_mca")
def slices(rows):
    seen = [r for r in rows if r["seen_category"] == "1"]
    return {"all 334 (main)": seen,
            "original 228": [r for r in seen if r["document"] not in BATCH2],
            "batch 2 only": [r for r in seen if r["document"] in BATCH2]}
print("| fold | slice | n | accuracy | 95% interval |\n|---|---|---|---|---|")
for k in FOLDS:
    main = list(csv.DictReader(open(f"{WORK}/main_fold{k}/indian_predictions.csv", encoding="utf-8")))
    jud = list(csv.DictReader(open(f"{WORK}/judgments_fold{k}/indian_predictions.csv", encoding="utf-8")))
    s = slices(main); s["judgments (separate)"] = [r for r in jud if r["seen_category"] == "1"]
    for name, rs in s.items():
        n = len(rs); c = sum(int(r["correct"]) for r in rs); lo, hi = wilson(c, n)
        print(f"| {k} | {name} | {n} | " + (f"{c/n:.3f} | {lo:.3f} to {hi:.3f} |" if n else "n/a | n/a |"))
for k in FOLDS:
    print(f"\n##### fold {k}: main set, per document / category")
    print(subprocess.run([sys.executable, "summarise_predictions.py", f"{WORK}/main_fold{k}/indian_predictions.csv"],
                         capture_output=True, text=True).stdout)
print("Judgment slice: ignore its span numbers (each clause is its own 'document').")'''

zipc = '''with zipfile.ZipFile("/kaggle/working/indian_eval_results_v3.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for k in FOLDS:
        for tag in ["main", "judgments"]:
            for f in ["indian_eval.json", "indian_eval.md", "indian_predictions.csv", "indian_rule_flags.csv"]:
                p = f"{WORK}/{tag}_fold{k}/{f}"
                if os.path.exists(p): z.write(p, f"{tag}/fold{k}/{f}")
print("download /kaggle/working/indian_eval_results_v3.zip from the Output tab")'''

def cell(t, s):
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
    if t == "code": c.update(outputs=[], execution_count=None)
    return c
nb = {"cells": [cell("markdown", md), cell("code", setup), cell("code", run), cell("code", summ), cell("code", zipc)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("notebooks/indian_eval_v3_kaggle.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("embedded data", len(data_b64) // 1024, "KB")
