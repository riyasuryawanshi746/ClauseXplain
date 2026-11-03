import csv, json, random, sys
from collections import defaultdict
from sklearn.metrics import f1_score

REPO, V5 = sys.argv[1], sys.argv[2]
ids = [json.loads(l)["clause_id"] for l in open(f"{REPO}/indian_set/built_v4/indian_mapped.jsonl", encoding="utf-8") if l.strip()]
ids_set = set(ids)


def load(p):
    return {r["clause_id"]: r for r in csv.DictReader(open(p, encoding="utf-8")) if r["clause_id"] in ids_set}


K = f"{REPO}/experiments/indian_ledgar/kaggle"
runs = {f"v4_{c}_s{s}": load(f"{K}/legalbert_cuad_indian{c}_s{s}/predictions.csv") for c in ("", "_ledgar") for s in (13, 14, 15)}
runs["v5_s13"] = load(f"{V5}/v5_legalbert_cuad_indian_s13/predictions.csv")
runs["v5_ledgar_s13"] = load(f"{V5}/v5_legalbert_cuad_indian_ledgar_s13/predictions.csv")
for n, r in runs.items():
    assert set(r) == ids_set, (n, len(r))
y = [runs["v5_s13"][i]["true_category"] for i in ids]
labels = sorted(set(y))


def score(run, idx):
    yt = [y[k] for k in idx]; yp = [runs[run][ids[k]]["pred_category"] for k in idx]
    return sum(a == b for a, b in zip(yt, yp)) / len(idx), f1_score(yt, yp, average="macro", labels=sorted(set(yt)), zero_division=0)


full = list(range(len(ids)))
print("run                acc    mF1  (334 v4 clauses)")
for n in runs:
    a, f = score(n, full); print(f"{n:<18} {a:.3f}  {f:.3f}")


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
for a, b in [("v4__s13", "v5_s13"), ("v4__s14", "v5_s13"), ("v4__s15", "v5_s13"),
             ("v4__ledgar_s13", "v5_ledgar_s13"), ("v4__ledgar_s14", "v5_ledgar_s13"), ("v4__ledgar_s15", "v5_ledgar_s13"),
             ("v5_s13", "v5_ledgar_s13")]:
    print(f"{a} -> {b}: {boot(a, b)}")

print("\nper-category F1 on the 334 (v4 s13 / v5 s13 / v5+LEDGAR s13), n")
for c in labels:
    row = []
    for n in ("v4__s13", "v5_s13", "v5_ledgar_s13"):
        yp = [runs[n][i]["pred_category"] for i in ids]
        row.append(f1_score(y, yp, labels=[c], average="macro", zero_division=0))
    print(f"{c:<30} {row[0]:.2f} {row[1]:.2f} {row[2]:.2f}  n={y.count(c)}")
