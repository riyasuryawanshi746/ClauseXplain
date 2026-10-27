#!/usr/bin/env python
"""
rebuild_partial.py -- recover per-run result files from a surviving results.json.

    python rebuild_partial.py path/to/results.json  path/to/new_out_dir

train_phase3.py keeps one file per finished (experiment, fold) in <out_dir>/partial/.
If you only have an old results.json (e.g. from the first single-fold run), this
writes those per-run files back so that
    python train_phase3.py ... --cv_folds 5 --folds 0 1 2 --resume --out_dir <new_out_dir>
loads them instead of retraining.  Splits are re-created on resume with the same
seed; check the printed fold-0 sizes match your original run (6914/889/1983).
Only runs that are inside results.json are recovered; nothing is invented.
"""
import json
import sys
from pathlib import Path


def main(results_path: str, out_dir: str) -> None:
    r = json.loads(Path(results_path).read_text(encoding="utf-8"))
    part = Path(out_dir) / "partial"
    part.mkdir(parents=True, exist_ok=True)
    n = 0
    for exp, runs in r["runs"].items():
        for k, run in enumerate(runs):
            (part / f"{exp}_fold{k}.json").write_text(json.dumps(run, default=float))
            n += 1
    n_folds = r["meta"]["n_folds"]
    if n_folds == 1:     # aggregate over one fold == that fold's own values
        for fname, key in (("baselines_fold0.json", "baselines_cls"),
                           ("span_baselines_fold0.json", "baselines_span")):
            b = r.get(key) or {}
            if b:
                (part / fname).write_text(json.dumps({k: v["mean"] for k, v in b.items()}, default=float))
    else:
        print("NOTE: results.json aggregates several folds; baselines per fold cannot be recovered "
              "and will be recomputed on --resume (cheap).")
    print(f"recovered {n} run files into {part}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
