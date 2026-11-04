"""Compare v6 runs with v4 (3 seeds) and v5 on the same 334 v4 test clauses: scores, paired bootstrap, per-category F1.

    python experiments/indian_v6/compare_v6.py <repo root> <folder with unzipped v6 results>
"""
import csv, json, random, sys
from pathlib import Path
from sklearn.metrics import f1_score

REPO, V6 = Path(sys.argv[1]), Path(sys.argv[2])
ids = [json.loads(l)["clause_id"] for l in open(REPO / "indian_set/built_v4/indian_mapped.jsonl", encoding="utf-8") if l.strip()]
ids_set = set(ids)


def load(p):
    return {r["clause_id"]: r for r in csv.DictReader(open(p, encoding="utf-8")) if r["clause_id"] in ids_set}


K4, K5 = REPO / "experiments/indian_ledgar/kaggle", REPO / "experiments/indian_v5/kaggle"
runs = {f"v4{c}_s{s}": load(K4 / f"legalbert_cuad_indian{c}_s{s}/predictions.csv") for c in ("", "_ledgar") for s in (13, 14, 15)}
runs["v5_s13"] = load(K5 / "v5_legalbert_cuad_indian_s13/predictions.csv")
runs["v5_ledgar_s13"] = load(K5 / "v5_legalbert_cuad_indian_ledgar_s13/predictions.csv")
for n, sub in (("v6_s13", "v6_legalbert_cuad_indian_s13"), ("v6_ledgar_s13", "v6_legalbert_cuad_indian_ledgar_s13")):
    if (V6 / sub / "predictions.csv").exists():
        runs[n] = load(V6 / sub / "predictions.csv")
for n, r in runs.items():
    assert set(r) == ids_set, (n, len(r))
y = [runs["v5_s13"][i]["true_category"] for i in ids]
seen = [runs["v5_s13"][i]["seen_category"] == "1" for i in ids]


def score(run, idx):
    yt = [y[k] for k in idx]; yp = [runs[run][ids[k]]["pred_category"] for k in idx]
    return sum(a == b for a, b in zip(yt, yp)) / len(idx), f1_score(yt, yp, average="macro", labels=sorted(set(yt)), zero_division=0)


full = list(range(len(ids)))
S = [k for k in full if seen[k]]; U = [k for k in full if not seen[k]]
print(f"{'run':<16} acc    mF1   | seen mF1 unseen mF1  (334 v4 clauses)")
for n in runs:
    a, f = score(n, full)
    print(f"{n:<16} {a:.3f}  {f:.3f} | {score(n, S)[1]:.3f}    {score(n, U)[1]:.3f}")


def boot(a, b, B=2000, seed=0):
    rng = random.Random(seed); da, df = [], []
    for _ in range(B):
        idx = [rng.randrange(len(ids)) for _ in ids]
        (a1, f1), (a2, f2) = score(a, idx), score(b, idx)
        da.append(a2 - a1); df.append(f2 - f1)
    da.sort(); df.sort()
    (a1, f1), (a2, f2) = score(a, full), score(b, full)
    return f"acc {a2 - a1:+.3f} [{da[49]:+.3f}, {da[1949]:+.3f}]  mF1 {f2 - f1:+.3f} [{df[49]:+.3f}, {df[1949]:+.3f}]"


print()
pairs = [("v5_s13", "v6_s13")] + [(f"v4_s{s}", "v6_s13") for s in (13, 14, 15)]
if "v6_ledgar_s13" in runs:
    pairs += [("v5_ledgar_s13", "v6_ledgar_s13")] + [(f"v4_ledgar_s{s}", "v6_ledgar_s13") for s in (13, 14, 15)] + [("v6_s13", "v6_ledgar_s13")]
for a, b in pairs:
    if a in runs and b in runs:
        print(f"{a} -> {b}: {boot(a, b)}")

cols = [c for c in ("v4_s13", "v5_s13", "v6_s13", "v5_ledgar_s13", "v6_ledgar_s13") if c in runs]
print("\nper-category F1 on the 334:", " / ".join(cols))
for c in sorted(set(y)):
    row = [f1_score(y, [runs[n][i]["pred_category"] for i in ids], labels=[c], average="macro", zero_division=0) for n in cols]
    print(f"{c:<30} " + " ".join(f"{v:.2f}" for v in row) + f"  n={y.count(c)}")
