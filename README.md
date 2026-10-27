# ClauseXplain

Clause understanding for Indian-context contracts: one multi-task Longformer (18-way clause category + evidence-span head) with a rule layer that flags clauses against Indian statutes (Indian Contract Act, MSMED Act, DPDPA).

**Status: research prototype, work in progress. Not legal advice.** Rule flags use cautious "may raise issues" wording and have not been reviewed by a lawyer.

## Repo map (in the order the work happened)

| Folder | What it is |
|---|---|
| `phase1_cuad_mapping/` | CUAD v1 (510 US contracts, 13,823 expert spans, 41 categories) mapped to an 18-category Indian taxonomy. `clauses_mapped.jsonl` = 9,786 mapped spans. Start with `FINAL_DATASET_README.txt`. |
| `phase2a_risk_schema/` | Risk schema, annotation guidelines, first version of the statutory rule table. |
| `phase2b_annotation_pilot/` | Annotation app and AI-assisted pilot labels. **The AI risk labels are not used for training** (single AI annotator, no human review). Kept for provenance. |
| `phase3_model/` | Dataset loader, multi-task model, training, metrics, CV comparison, scan-vs-span, Indian-set builder/evaluator, label-sheet tool, demo `pipeline.py`, smoke tests. |
| `indian_law/` | Current rule table (`indian_legal_rules.csv`), human feature-audit sheets and rule-coverage reports. The engine itself is `phase3_model/rules_engine.py`. |
| `indian_set/` | Small Indian test set: cleaned text of 11 public Indian contracts/tenders, clause drafts, built set and evaluation output. |
| `experiments/` | Result files. See `experiments/README.md`: only `stageB_3fold` and `scan_vs_span` are canonical, the rest is archived. |
| `notebooks/` | Kaggle notebooks (training, scan-vs-span, Indian set). |
| `docs/` | Status and plan. |

## What the results support so far

- Longformer category classifier beats TF-IDF+LogReg by about +2.2 micro-F1 / +3.7 macro-F1 (3 folds, interval above 0).
- Joint cls+span model is on par with the single-task models at roughly half the parameters. No speed-up.
- Span head beats scanning the window with the classifier by about +0.125 token-F1 using roughly 50x fewer forward passes (folds run so far).
- Indian test set evidence is anecdotal at this point (few scorable clauses, keyword-selected). Growing it to 100+ clauses via `make_label_sheet.py` is in progress.

## Known limits

- Training data is US (CUAD). Category labels come from a mapping, so classification is partly circular; span localisation is the headline task.
- 6 of 18 categories have no training data (Confidentiality, Force Majeure, Indemnification, Modification, Payment, Privacy / Data Protection). The pipeline masks them.
- Span task is windowed localisation, not comparable to full-contract CUAD numbers. Span results are noisy and probably undertrained; only folds 0-2 were run.
- No risk head is trained. Risk is handled by rules over extracted text features.

## Quick start

```bash
pip install -r requirements.txt
cd phase3_model
python smoke_test.py            # CPU, ~2 min, 16 checks, tiny random model
python test_rules_engine.py     # rule engine tests
```

Real training needs a GPU (Kaggle/Colab T4 works) plus `CUAD_v1.json` (not in this repo, download from the CUAD release; license CC BY 4.0):

```bash
python train_phase3.py --mapped_jsonl ../phase1_cuad_mapping/clauses_mapped.jsonl --cuad_json CUAD_v1.json \
  --experiments cls_only span_only joint_cls_span --cv_folds 5 --folds 0 1 2 --resume --save_ckpt --out_dir stageB
```

Demo on a contract (needs a trained checkpoint, which is not in the repo):

```bash
python pipeline.py --input contract.pdf --ckpt ckpt_joint_cls_span_fold0.pt \
  --rules indian_legal_rules.csv --out report.html --find "Termination" "Liability"
```

Grow the Indian set without keyword bias:

```bash
python make_label_sheet.py build --docs_dir ../indian_set/documents --out label_sheet.csv
python make_label_sheet.py convert --sheet label_sheet.csv [--second other_annotator.csv] --out clauses_new.csv
```

## Not in the repo (by design)

Model checkpoints (`*.pt`, ~0.6 GB each), `CUAD_v1.json`, the original contract PDFs, and the large regenerable JSONL files from Phase 1/2A (`clauses.jsonl`, `clauses_risk_annotation.jsonl`; rebuild with the scripts in those folders).

## Data notes

CUAD: Hendrycks et al., CC BY 4.0. Indian documents are publicly available government tenders/agreements; check each source's terms before making this repo public.
