# Paper (IEEE conference format)

`main.tex` uses `\documentclass[conference]{IEEEtran}`. Compile on Overleaf (IEEEtran is built in): upload `main.tex`, set compiler to pdfLaTeX.

Red `[TODO: ...]` items are things that are NOT done yet (see top-level plan). Every number in the tables comes from:
- CUAD table: `experiments/stageB_3fold/results.json`
- Indian table: `experiments/indian_eval_v4/` (final, adjudicated; README has every number incl. pool rows and judgment slice)
- Scan vs span: `experiments/scan_vs_span/README.md`
- LEGAL-BERT rows: `experiments/legalbert_baseline_v4/` (CUAD) and `experiments/indian_eval_v4/README.md` (Indian)

Not compiled against the real IEEEtran here (only a stand-in class), so check layout on Overleaf. References checked 2026-11-02 against the ACL Anthology / arXiv / Hugging Face pages (authors, venue, pages). InLegalBERT pages (ICAIL 2023, pp. 187-196) come from a DBLP listing seen via search, not opened directly; ICAT has no paper, so it is cited as a dataset; LAUKIN is an arXiv preprint (June 2026).
