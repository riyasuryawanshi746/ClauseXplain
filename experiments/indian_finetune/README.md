# Indian set: document-held-out training (steps 1 and 7)

Run 2026-11-03 on Kaggle (T4) with `notebooks/indian_finetune_kaggle.ipynb` (the 9-configuration version, before commit
`b96022f`); executed copy: `indian_finetune_kaggle_run.ipynb`. Code: `phase3_model/finetune_indian.py`.

**Protocol.** Leave-one-document-group-out over the frozen `built_v4` set (334 clauses, 13 documents, 12 groups: the
two MEA agreements share a template and are held out together). Every clause is predicted once by a model that never
saw its document. The judgment slice (11 clauses) is scored by a model trained on all 334. Settings fixed in code before
the run: 8 epochs, lr 2e-5 (head 1e-4), batch 16, max 512 tokens, seed 13, last-epoch model (no selection on held-out
data). `--tricks` = inverse-frequency class weights + label smoothing 0.1 + layer-wise LR decay 0.9 (TF-IDF: balanced
class weights). `cuad_*` neural runs start from a CUAD-trained fold-0 checkpoint (LegalBERT trained in the notebook;
Longformer from `scan-vs-span/ckpt_joint_cls_span_fold0.pt`); `tfidf_cuad_*` add all mapped CUAD spans to training.

## Results (n = 334; 192 seen-category, 142 unseen-category clauses)

| Run | acc (95% CI) | macro-F1 (95% CI) | seen acc / mF1 | unseen acc / mF1 | original / batch 2 acc | judgments acc |
|---|---|---|---|---|---|---|
| **legalbert_cuad_indian** | **0.566** (0.512-0.618) | **0.528** (0.457-0.562) | **0.536 / 0.597** | 0.606 / 0.516 | 0.623 / 0.443 | 0.727 |
| legalbert_cuad_indian_tricks | 0.512 (0.459-0.565) | 0.466 (0.420-0.528) | 0.495 / 0.501 | 0.535 / 0.510 | 0.566 / 0.396 | 0.455 |
| longformer_cuad_indian | 0.521 (0.467-0.574) | 0.469 (0.413-0.517) | 0.490 / 0.560 | 0.563 / 0.494 | 0.544 / 0.472 | 0.364 |
| longformer_cuad_indian_tricks | 0.488 (0.435-0.541) | 0.472 (0.420-0.530) | 0.479 / 0.503 | 0.500 / 0.574 | 0.509 / 0.443 | 0.364 |
| legalbert_indian | 0.467 (0.414-0.521) | 0.296 (0.257-0.345) | 0.312 / 0.231 | 0.676 / 0.508 | 0.518 / 0.358 | 0.364 |
| legalbert_indian_tricks | 0.249 (0.205-0.298) | 0.272 (0.219-0.327) | 0.182 / 0.221 | 0.338 / 0.448 | 0.316 / 0.104 | 0.182 |
| tfidf_indian | 0.422 (0.370-0.476) | 0.297 (0.261-0.345) | 0.328 / 0.269 | 0.549 / 0.439 | 0.417 / 0.434 | 0.273 |
| tfidf_indian_balanced | 0.482 (0.429-0.536) | 0.351 (0.310-0.401) | 0.396 / 0.336 | 0.599 / 0.485 | 0.474 / 0.500 | 0.182 |
| tfidf_cuad_indian | 0.284 (0.239-0.335) | 0.292 (0.215-0.322) | 0.432 / 0.472 | 0.085 / 0.095 | 0.285 / 0.283 | 0.818 |
| tfidf_cuad_indian_balanced (local CPU, see note) | 0.416 (0.365-0.470) | 0.427 (0.352-0.462) | 0.484 / 0.540 | 0.324 / 0.351 | 0.439 / 0.368 | 0.636 |

Zero-shot reference (CUAD-only, `indian_eval_v4/`, `legalbert_baseline_v4/`, same 192 seen clauses, mean of folds 0-2):
Longformer 0.347 / 0.338, LegalBERT 0.356 / 0.342, TF-IDF 0.333 / 0.344. Unseen categories cannot be predicted zero-shot.

Note on `local_cpu/`: `tfidf_cuad_indian_balanced` was not in the notebook version that was run; it was run locally
with `phase1_cuad_mapping/clauses_mapped.jsonl`. The same local setup gives 0.293 / 0.307 for `tfidf_cuad_indian`
against 0.284 / 0.292 on Kaggle, so local and Kaggle TF-IDF+CUAD numbers differ by about 0.01 (CUAD file copy or
sklearn version; not investigated). Both local runs are kept in `local_cpu/`.

## Reading

- **CUAD pretraining + Indian training is best.** LegalBERT CUAD -> Indian: seen-category macro-F1 0.342 -> 0.597 on the
  same 192 clauses, and the six CUAD-missing categories become learnable (unseen macro-F1 0.516; Force Majeure F1 0.89,
  Confidentiality 0.78, Indemnification 0.77, Payment 0.48; Modification 0.00 on 7 and Privacy 0.00 on 8).
- **Both ingredients matter.** Indian-only training (no CUAD) is weak on the seen categories (LegalBERT 0.231 mF1,
  mostly collapses to Other / Confidentiality / Payment); CUAD-only is the zero-shot baseline. The best TF-IDF
  (balanced CUAD+Indian, 0.427 mF1) stays below the best neural run (0.528), whose intervals barely overlap.
- **The tricks did not help the neural models** with these fixed settings (LegalBERT CUAD: 0.528 -> 0.466 macro-F1;
  Longformer: 0.469 -> 0.472). `legalbert_indian_tricks` failed to train: it predicts Non-Solicitation for 192 of 345
  clauses with mean confidence 0.12. Treat it as a failed run, not evidence about the tricks.
- **Still well short of 70%.** Batch 2 (0.443) and the big SBM / CPWD / DoT documents (0.38-0.45) are the weak spots;
  per-group accuracy ranges 0.38-0.84 for groups with at least 10 clauses. Main confusions: Confidentiality -> Other (8),
  Payment -> Commercial Terms (7), Other -> Payment (6), Privacy -> Confidentiality (5).
- **Caveats.** One seed; ten configurations compared on the same held-out predictions, so picking the best one is mild
  selection; groups are unequal (2 to 73 clauses); judgment slice n = 11.

Files: `<run>/metrics.json` (all breakdowns, per-category, per-group, confusions, fold log), `<run>/predictions.csv`,
`summary.md` (notebook output).
