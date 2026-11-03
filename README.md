# ClauseXplain

Clause classification for Indian contracts, and how badly models trained on US contracts (CUAD) transfer to them.
A contract goes in; each clause gets one of 18 categories, and a rule layer flags clauses that may raise issues
under Indian statutes (Indian Contract Act, MSMED Act, DPDPA).

**Research prototype, not legal advice.** The rule flags use cautious "may raise issues" wording and have not been
reviewed by a lawyer.

## Main results

Indian test set: 334 double-annotated clauses from 13 public Indian documents (kappa 0.65, disagreements adjudicated
blind). Training on Indian clauses is document-held-out: every document group is tested by a model that never saw it.

Zero-shot (trained on CUAD only, 3 CUAD folds), scored on the 192 Indian clauses whose category exists in CUAD:

| Model | accuracy | macro-F1 |
|---|---|---|
| TF-IDF + LogReg | 0.333 | 0.344 |
| Longformer (cls + span) | 0.347 | 0.338 |
| LegalBERT (CUAD macro-F1 0.870) | 0.356 | 0.342 |

Trained on Indian clauses too (document-held-out, all 334 clauses, mean ± sd over 3 seeds):

| Model | accuracy | macro-F1 |
|---|---|---|
| LegalBERT, CUAD -> Indian | 0.565 ± 0.007 | 0.524 ± 0.006 (seen 0.593) |
| + LEDGAR provisions | 0.600 ± 0.017 | 0.512 ± 0.017 |
| InLegalBERT, CUAD -> Indian | 0.537 ± 0.009 | 0.450 ± 0.007 |
| LegalBERT, CUAD -> Indian + batch 3 (1 seed) | 0.611 | 0.609 |
| + batch 3 + LEDGAR (1 seed) | 0.644 | 0.636 |

- On CUAD the models score well; on Indian clauses all three fall to the same level, TF-IDF included. Six of the 18
  categories have no CUAD training data at all.
- LEDGAR improves 13 of 17 categories; macro-F1 is flat only because the single Non-Solicitation clause flips.
  Indian-law pretraining (InLegalBERT) did not help.
- A few hundred Indian clauses do far more than a better encoder or extra US data.
- Batch 3 rows: 454 private-sector clauses added to training only, scored on the same 334 test clauses
  (`experiments/indian_v5/`). More Indian data gives the largest gain, mostly in the categories CUAD lacks.

Full numbers, intervals and caveats: `experiments/README.md` and the paper (`paper/main.tex`).

## Repository layout

| Folder | Contents |
|---|---|
| `phase1_cuad_mapping/` | CUAD v1 (510 US contracts, 41 categories) mapped to the 18 ClauseXplain categories. `clauses_mapped.jsonl` = 9,786 spans. Start with `FINAL_DATASET_README.txt`. |
| `phase2a_risk_schema/` | Risk schema, annotation guidelines, first version of the statutory rule table. |
| `phase2b_annotation_pilot/` | Annotation app and an AI-assisted risk-label pilot. **These AI labels are not used anywhere** (single AI annotator, no human review); kept for provenance. |
| `phase3_model/` | All model code: data loading, models, training (`train_phase3.py`, `finetune_indian.py`), evaluation, labelling tools (`make_label_sheet.py`, `recheck.py`, `adjudicate.py`), rule engine and the demo (`pipeline.py`, `app.py`). |
| `indian_set/` | Indian clause set. `documents/`, `documents_batch2/`: public contract texts; `documents_batch3_redacted/`: 13 private-sector agreements with personal numbers masked. `clauses_v5_public.csv` -> `built_v5_public/`: public v5 training set (769 clauses; v4 + batch 3 without the two leave-and-license deeds). `clauses_v4.csv` -> `built_v4/`: the frozen 334-clause test set. `labelling/`: labelling guide (v2), instructions, agreement and adjudication files. `archive/`: earlier versions. |
| `indian_law/` | Human feature-audit sheets and rule-coverage reports (the rule table is `phase3_model/indian_legal_rules.csv`). |
| `experiments/` | Result files, one folder per experiment. `experiments/README.md` lists which ones are canonical; `archive/` is superseded. |
| `notebooks/` | Kaggle notebooks and the `make_*.py` scripts that generate them. Upload the `.ipynb`, not the `make_*.py`. Old ones are in `archive/`. |
| `demo_reports/` | Three sample HTML reports from the demo. |
| `paper/` | IEEE workshop paper (`main.tex`). |
| `docs/` | `HANDOFF.md`: current status, decisions and next steps. |

## Quick start

```bash
pip install -r requirements.txt
cd phase3_model
python smoke_test.py            # CPU, ~2 min, tiny random model
python test_rules_engine.py     # rule engine tests
```

Document-held-out training on the Indian set (TF-IDF runs on a CPU in under a minute; neural runs need a GPU, see
`notebooks/indian_finetune_kaggle.ipynb`):

```bash
python finetune_indian.py run --method tfidf --tricks --indian_dir ../indian_set/built_v4 --name tfidf --out_dir runs
```

CUAD training needs a GPU and `CUAD_v1.json` (not in this repo; from the CUAD release, CC BY 4.0):

```bash
python train_phase3.py --mapped_jsonl ../phase1_cuad_mapping/clauses_mapped.jsonl --cuad_json CUAD_v1.json \
  --experiments cls_only span_only joint_cls_span --cv_folds 5 --folds 0 1 2 --resume --save_ckpt --out_dir stageB
```

## Demo

Runs on a laptop CPU, about 25 s per contract. Train the demo classifier with `notebooks/demo_model_kaggle.ipynb`
(~15 min on a Kaggle T4) and unzip `demo_model_legalbert.zip` into `demo_model/` (not in the repo, ~200 MB). It is
LegalBERT trained on CUAD plus all 334 Indian clauses.

```bash
pip install gradio pypdf
python app.py --cls_model ../demo_model/demo_model_legalbert      # http://127.0.0.1:7860
python pipeline.py --input contract.pdf --cls_model ../demo_model/demo_model_legalbert \
  --rules indian_legal_rules.csv --out report.html                  # same report, no UI
```

## Known limits

- The Indian set is small and mostly government documents; categories such as Non-Solicitation have very few clauses.
- CUAD category labels come from a mapping, so CUAD scores are partly circular.
- The rules have not been reviewed by a lawyer, and no risk model is trained.

## Not in the repo

Model checkpoints, `CUAD_v1.json`, original PDFs, large regenerable JSONL files, the unredacted batch 3 documents,
the two leave-and-license deeds and their 19 clauses, and the batch 3 annotation sheets (`indian_set/labelling/BATCH3_INSTRUCTIONS.md` describes them).

## Data notes

CUAD: Hendrycks et al., CC BY 4.0. The Indian documents in this repo are publicly available government
tenders/agreements; check each source's terms before reusing them.
