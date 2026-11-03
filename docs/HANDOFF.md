# ClauseXplain - handoff / context (paste this into a new chat or Claude Code session)

Last updated: 2026-10-29. Owner: Riya Suryawanshi (B.Tech CE, SPIT Mumbai). Repo: https://github.com/riyasuryawanshi746/ClauseXplain (main).

## How to work with Riya
Casual, direct, concise, English only. Honest and practical, no hype. Push to GitHub constantly. Use "n/a" never 0 for undefined metrics.
Never assign labels to the test clauses for her: the labels must be hers (and the second annotator's). Say plainly what is weak.

## Goal
1. IEEE-format paper (realistic target: workshop / applied-NLP venue, not a strong-model claim). 2. Placement demo.
Pitch: legal-contract NLP for the Indian context. Multi-task Longformer-base-4096: 18-way clause category head + evidence-span head,
plus a statutory rule layer (15 rules for Indian Contract Act / MSMED Act / DPDPA; NOT yet reviewed by a lawyer).
Paper framing: domain-shift study (CUAD -> Indian contracts) + cheap span localiser + small Indian benchmark.

## Data
- CUAD v1: 510 contracts, 13,823 spans, 41 -> 18 categories. 12 trainable; 6 have no CUAD data
  (Confidentiality, Force Majeure, Indemnification, Modification, Payment, Privacy/Data Protection).
  Dispute Resolution only comes from "Covenant Not To Sue" (173 spans).
- Indian set: 228 clauses from 11 public documents (mostly government agreements). 129 in seen categories (incl. Other), 99 in unseen ones.
  Single primary annotator (Riya). Second annotator on 100 random units: raw agreement 0.68, Cohen kappa 0.65 (0.73/0.71 on the 79 clause units).
  32 disagreements adjudicated blind on 2026-11-02 (see next steps).
  The old 50-unit spot check (kappa 1.00) is NOT credible; never report it.
- Batch 2 (labelled by Riya, 2026-11-02): CPWD GCC 2023 + SBM waste-to-compost draft, 107 keyword-assisted units
  (`indian_set/labelling/label_sheet_batch2_labelled.csv`; she marked none as not-a-clause, kept all labels after review;
  53% of labels equal the keyword that selected the unit -> say "keyword-assisted pool" in the paper).
  Merged set: `indian_set/clauses_v3.csv` -> `indian_set/built_v3/` = 334 clauses, 13 docs (v2's 228 unchanged + 106; 1 duplicate dropped).
  Evaluated 2026-11-02: see `experiments/indian_eval_v3/README.md`.
- Judgment slice (labelled): `indian_set/built_judgments/` (11 clauses, 5 judgments; classification only; report separately, never merge).
  Waste-to-compost is a draft full of [placeholders]; GCC text came from a PDF so check OCR noise.

## Training / eval setup
Contract-grouped 5-fold CV; folds 0-2 run. 2 epochs, lr 2e-5 (head 1e-4), seed 13, span_train_frac 0.5, window 3000 chars,
max_len_span 1024, max_len_clause 512, T4 fp16. Needs GPU (Kaggle). Code in `phase3_model/`.

## Results (3-fold means; use these in the paper)
- CUAD macro-F1: cls_only 0.839, joint 0.837, TF-IDF+LR 0.802, majority 0.030. Micro-F1 0.875 / 0.872 / 0.853.
  Joint vs cls_only: no difference (CI includes 0) - say so.
- Span token-F1: joint 0.760, span_only 0.736, lexical-centroid 0.604, whole-window 0.236, random 0.106. Exact match joint 0.520.
- Span head vs scanning every clause with the classifier (500 windows/fold): 0.756 vs 0.630 token-F1, 54 ms vs ~1314 ms per window (~24x faster).
- FINAL Indian (v4, adjudicated, 334 clauses / 192 seen; `experiments/indian_eval_v4/README.md`): accuracy Longformer 0.347,
  LegalBERT 0.356, TF-IDF 0.333; macro-F1 0.338 / 0.342 / 0.344. Original pool 0.382, batch 2 0.275, judgments 0.606 (n=11).
  Span head 0.322 vs lexical 0.302 (random 0.098). LegalBERT CUAD macro-F1 0.866 (+-0.011; first run 0.870, GPU noise).
- Older (superseded, do not quote): v3 = pre-adjudication 334; Indian v2 (228, the numbers below): Longformer acc 0.377 / macro-F1 0.382 vs TF-IDF 0.349 / 0.360 (i.e. barely better out of domain). Span F1 0.347 vs lexical 0.312.
  Folds reuse the same 129 clauses, so they are not independent. Dispute Resolution F1 0.00. Best: Penalties 0.61, Governing Law 0.53, IP 0.48.
  Unseen-category errors: Payment->Commercial Terms 17/22, Indemnification->Liability 11/15, Confidentiality->Other 28/40.
  Excluding OCR-noisy Noida doc and true "Other": acc 0.412/0.459/0.482 (n=85; TF-IDF not rescored).

## Repo map
- `phase3_model/`: train/eval code, `eval_indian.py` (writes indian_predictions.csv), `summarise_predictions.py`, `pipeline.py` (segment -> classify -> rules -> optional `--find` span head -> HTML report), `rules_engine.py`, `make_label_sheet.py`, `build_indian_testset.py`, `smoke_test.py` (16/16).
- `notebooks/`: `indian_eval_v2_kaggle.ipynb`, `demo_kaggle.ipynb`. Kaggle datasets: `riyasuryawanshi/clausexplain-indian-eval`, `riyasuryawanshi/scan-vs-span`; checkpoints `ckpt_joint_cls_span_fold{k}.pt`.
- `experiments/indian_eval/` results per fold + README table; `demo_reports/` 3 sample HTML reports.
- `paper/main.tex` IEEEtran skeleton (compiled only with a stub class; check on Overleaf). References are from memory: VERIFY every one.
- `indian_set/labelling/`: LABELLING_GUIDE, FINDING_CONTRACTS, second-annotator files.

## Known limitations (be upfront)
Small single-primary-annotator Indian set (kappa 0.65; disagreements adjudicated); classifier ~ TF-IDF out of domain; 6 categories untrained;
rules not lawyer-reviewed; folds 3-4 and extra seeds not run; no RoBERTa/LegalBERT baseline yet; Indian clauses mostly government documents.

## Next steps (in order)
1. DONE 2026-11-02: 32 disagreements adjudicated blind (final = annotator 1 in 21, annotator 2 in 11; 11 primary labels changed;
   `indian_set/labelling/adjudication_report.md`). Frozen set: `indian_set/clauses_v4.csv` -> `built_v4` (334 clauses, 192 seen).
   v4 Kaggle runs done 2026-11-02; paper uses them. Experiments are finished; remaining work is writing (step 5).
2. Find more private-sector style contracts (see FINDING_CONTRACTS.md); target 400+ clauses.
3. Re-run Indian eval on the enlarged set; train with some Indian clauses for missing categories, held-out contracts for test.
4. LegalBERT cls-only baseline: DONE 2026-11-02 (`experiments/legalbert_baseline/`): CUAD macro-F1 0.870 (> Longformer 0.839),
   Indian acc 0.349 / macro-F1 0.341 (= Longformer, = TF-IDF). In the paper. Re-run after adjudication with make_legalbert_kaggle.py.
   Folds 3-4 and extra seeds: decided to skip (limitation in paper).
5. Paper: related work, verify citations, second-annotator background, fill red TODOs, data-release licensing.
6. DONE 2026-11-03: document-held-out training on built_v4 (`experiments/indian_finetune/`, `phase3_model/finetune_indian.py`).
   Best: LegalBERT CUAD->Indian 0.566 acc / 0.528 macro-F1 over 334 (seen mF1 0.597 vs 0.342 zero-shot). Class weights/
   smoothing/LLRD did not help; legalbert_indian_tricks failed to train. In the paper (Section "Training on Indian Clauses").
   Next: LEDGAR data for weak categories (step 5 of the plan), then InLegalBERT / domain-adaptive pretraining (step 6).
7. DONE 2026-11-03: step 5 (`experiments/indian_ledgar/`): LegalBERT CUAD->Indian 3 seeds 0.565/0.524; + LEDGAR 0.600/0.512
   (accuracy up, macro-F1 flat; helps unseen categories, hurts seen). Step 6 (`experiments/indian_inlegalbert/`):
   InLegalBERT 0.537/0.450, worse than LegalBERT. Both in the paper. Demo: `phase3_model/app.py` + `--cls_model`
   (model from `notebooks/demo_model_kaggle.ipynb`, not in repo).
   LATER (decided 2026-11-03): steps 2-4 = more (private-sector) data, label consistency (more double annotation,
   guideline fixes), cleaning noisy test units. Paper lists them as future work.
8. DONE 2026-11-03: batch 3 (15 private-sector docs) double-annotated, re-checked under guide v2, adjudicated ->
   v5 = 788 clauses (`built_v5`, local only, 23 groups). TF-IDF on the same 334 v4 clauses: 0.482/0.351 -> 0.536/0.469
   with batch 3 in training (`experiments/indian_v5/`). NEXT: run `notebooks/indian_v5_kaggle.ipynb` (private), add to paper.
ON HOLD until she says so: lawyer review of rules; 12-vs-18 category decision.

## Environment notes
The cloud sandbox cannot reach gov sites/HuggingFace (only package registries + GitHub), so Riya downloads PDFs herself. GPU runs happen on Kaggle.
