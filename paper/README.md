# Paper (IEEE conference format)

`main.tex` uses `\documentclass[conference]{IEEEtran}`. Compile on Overleaf (IEEEtran is built in): upload `main.tex`, set compiler to pdfLaTeX.

Red `[TODO: ...]` items are things that are NOT done yet (see top-level plan). Every number in the tables comes from:
- CUAD table: `experiments/stageB_3fold/results.json`
- Indian table: `experiments/indian_eval_v3/main/fold{0,1,2}/indian_eval.json` (pool rows and judgment slice: `experiments/indian_eval_v3/README.md`; v2 228-clause run: `experiments/indian_eval/`)
- Scan vs span: `experiments/scan_vs_span/README.md`
- LEGAL-BERT rows: `experiments/legalbert_baseline/` (README + per-fold JSON)

Not compiled against the real IEEEtran here (only a stand-in class), so check layout on Overleaf. Reference list is minimal; verify each citation's details before submission.
