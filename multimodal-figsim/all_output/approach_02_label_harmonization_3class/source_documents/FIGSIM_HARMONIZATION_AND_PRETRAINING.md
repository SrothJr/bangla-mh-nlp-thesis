# Claude Code Implementation Prompt

## Role and objective

You are working directly inside an existing thesis codebase for cross-lingual multimodal suicide-content classification. Inspect the repository first, preserve all working code and artifacts, and then implement a new experimental pipeline with two connected changes:

1. Harmonize the small FigSIM suicide-severity target into a reliable three-class primary task while retaining the original five-class target as auxiliary supervision.
2. Use the public CMBAN and BN-HIB Bangla meme datasets for image-text pretraining and dataset-specific auxiliary learning. Do not relabel their sentiment, sarcasm, offensiveness, hate, inflammatory, or benign labels as suicide labels.

The final deployed output must be one three-class FigSIM suicide-content prediction. External tasks and the original Bangla depression task are training aids. They are not additional final outputs unless an evaluation script explicitly requests them.

Work autonomously after inspecting the repository. Do not stop at a high-level plan. Implement the pipeline, add reproducible commands, run data checks and smoke tests, and document what was changed. If a required local dataset path or checkpoint is missing, build the code and configuration around an explicit path placeholder, identify the exact missing path, and continue with every task that does not require the missing files. Do not fabricate data, labels, scores, or successful runs.

## Project context

The completed Phase 2 system used approximately 250,000 English Reddit mental-health posts translated into Bangla for domain-adaptive pretraining of BanglaBERT. The intended translation model for the current work is Facebook NLLB-200 1.3B. Verify the checkpoint recorded in the existing translation metadata before regenerating anything. Do not replace already cached translations simply because the documentation mentions a different NLLB size. Record the checkpoint that actually produced each artifact.

The DAPT encoder was then fine-tuned on an original Bangla depression dataset with four labels:

- Minimum
- Mild
- Moderate
- Severe

The original documented dataset contains 4,897 or 4,898 records depending on whether a header or excluded record was counted. A later augmented version may contain approximately 8,000 to 9,000 records. Inspect the actual files and report exact counts. Keep original and synthetic records distinguishable. Synthetic children must remain in the same split as their source record and must never enter validation or test independently.

The multimodal extension uses FigSIM suicide memes. The leakage-safe subset contains 973 records with the fixed split:

- Train: 582
- Validation: 195
- Test: 196

The current multimodal branches are:

- Text: EasyOCR English text, NLLB translation into Bangla, then DAPT-BanglaBERT.
- Image: SigLIP-so400m.
- Fusion: gated fusion with an orthogonal image residual and a separate token-patch cross-attention model.

The original six FigSIM severity labels were already reduced to five by combining Suicide Attempt and Suicide Death. The five existing labels are:

1. None
2. Wish to be dead
3. Suicide ideation
4. Suicide planning
5. Suicide attempt or death

Known results and constraints from earlier experiments include:

- Original locked five-class gated model test macro-F1: approximately 0.4635.
- Later six-model gated plus cross-attention ensemble test macro-F1: approximately 0.4984; weighted-F1 0.5232; accuracy 0.5306; QWK 0.3857.
- Test per-class F1 for that ensemble: None 0.476, Wish 0.659, Ideation 0.603, Planning 0.348, Attempt or Death 0.406.
- Cross-attention improves ideation and planning and complements gated fusion.
- Dropout 0.4 helped the gated model.
- Ordinal label smoothing helped; focal loss, embedding mixup, supervised contrastive alignment, CLIP replacement, full or partial encoder unfreezing, and a hard two-stage hierarchical classifier did not help.
- A binary None-versus-any classifier reached validation macro-F1 around 0.7406, but hard routing into a second classifier reduced full-pipeline performance because first-stage errors became unrecoverable.
- LoRA improved a simple concatenation model from approximately 0.5139 to 0.5264 validation macro-F1. LoRA has not yet been fully integrated into the best gated and cross-attention ensemble.
- Per-class validation calibration increased one split's score but did not generalize in five-fold analysis. Do not optimize thresholds directly on the validation labels again.
- The existing test set has already been inspected in earlier experiments. Do not use it for model selection, label-scheme selection, loss tuning, early stopping, or threshold tuning. Be transparent in the final report that it is a reused evaluation set rather than a perfectly untouched blind test.

Search the repository for the exact implementations, checkpoints, cached embeddings, predictions, logs, and configuration files. Likely identifiers include `train_e6.py`, `IMPROVEMENT_PLAN.md`, `PROGRESS_LOG.md`, `figsim_leakage_safe_index.csv`, `BanglaBERT_fold5/checkpoint-735`, and `BanglaBERT_fold5_tapt`. Treat these as search hints, not guaranteed paths.

## Non-negotiable scientific rules

Follow these rules throughout the implementation.

1. Never map BN-HIB Benign to FigSIM None.
2. Never map BN-HIB Hate or Inflammatory to a suicide-positive class.
3. Never map CMBAN negative sentiment, offensiveness, or sarcasm to suicide ideation or severity.
4. Never map Minimum depression to non-suicidal or Severe depression to suicidal.
5. Never treat a missing label as a negative label.
6. Never make Invalid a suicide-content class. Invalid means a corrupt, missing, unreadable, duplicated, or unusable record and must be filtered before modeling.
7. Do not resplit the 973 FigSIM records. Preserve the existing leakage-safe train, validation, and test assignments.
8. Deduplicate external images and text before pretraining. Detect both exact duplicates and near-duplicate images. Do not allow an external validation or test sample into its own supervised training split.
9. Keep source dataset identifiers and availability masks for every label and modality.
10. Do not overwrite old checkpoints, cached embeddings, results, or reports. Write the new work under a clearly named experiment directory.
11. Do not run additional test evaluation until the complete new configuration has been selected and locked using training and validation or grouped cross-validation.
12. Do not claim clinical diagnosis, personal intent, or actual patient risk. FigSIM labels describe meme content under its annotation guidelines.
13. Do not claim that CMBAN or BN-HIB increases the number of suicide-labeled records. Only FigSIM supplies the suicide-severity labels in this plan.
14. Do not download or switch to a large new model without checking disk space and documenting the reason. Reuse the existing DAPT-BanglaBERT, SigLIP, NLLB outputs, OCR, and cached representations wherever possible.

## Primary FigSIM label harmonization

Implement the following three-class target as the main task:

| Original FigSIM label | Harmonized ID | Harmonized label |
|---|---:|---|
| None | 0 | No expressed suicide severity signal |
| Wish to be dead | 1 | Suicidal thought or desire |
| Suicide ideation | 1 | Suicidal thought or desire |
| Suicide planning | 2 | High acuity suicidal content |
| Suicide attempt | 2 | High acuity suicidal content |
| Suicide death | 2 | High acuity suicidal content |
| Suicide attempt or death | 2 | High acuity suicidal content |

Use short machine labels in code, for example:

```python
HARMONIZED_LABELS = {
    0: "no_expressed_severity",
    1: "suicidal_thought_or_desire",
    2: "high_acuity_suicidal_content",
}
```

Do not display class 0 as simply `not suicidal`. FigSIM has separate severity and suicide-related-content dimensions. A severity label of None can coexist with method depiction, harmful factors, protective factors, third-person discussion, or context that does not express one of the personal severity levels.

Retain the established five-class target as an auxiliary task:

```text
0 None
1 Wish to be dead
2 Suicide ideation
3 Suicide planning
4 Suicide attempt or death
```

Also derive an auxiliary binary target:

```text
0 No expressed severity signal
1 Any expressed severity signal
```

Implement a four-class ablation, but do not make it the default:

| Original label | Four-class label |
|---|---|
| None | No expressed severity |
| Wish to be dead | Passive death wish |
| Suicide ideation or Suicide planning | Active ideation or planning |
| Suicide attempt or Suicide death | Suicide behavior or outcome |

The four-class scheme intentionally merges the ideation-planning boundary because earlier errors concentrate there. Report it as an ablation, not a clinically validated replacement scale.

## Quick diagnostic before retraining

If saved validation probabilities from the best five-class model exist, implement a diagnostic that aggregates probabilities without retraining:

```text
P class 0 = P None
P class 1 = P Wish + P Ideation
P class 2 = P Planning + P AttemptDeath
```

Calculate three-class validation macro-F1, weighted-F1, balanced accuracy, accuracy, confusion matrix, and per-class precision, recall, F1, and support. Save the result as a diagnostic only. It is not the final three-class model.

Do the same for the four-class ablation by summing the relevant five-class probabilities. Do not inspect or aggregate test probabilities during label-scheme development.

Then retrain the classifiers for the selected three-class target. A probability collapse is not a substitute for retraining.

## External datasets

### CMBAN

Official paper: https://aclanthology.org/2025.findings-ijcnlp.135/

Official repository: https://github.com/newazbenalam/CMBAN

Expected characteristics:

- 2,641 cartoon-based Bangla or Bangla-English memes.
- Train: 2,113.
- Test: 528.
- Image, manually supplied Bangla text, and English translation.
- Labels for humor, sarcasm, offensiveness, motivation, and overall sentiment.
- Repository license reported as MIT. Preserve the license file and citation information in the project documentation.

### BN-HIB

Official dataset: https://data.mendeley.com/datasets/9vg79v65nr/1

Paper: https://arxiv.org/abs/2602.22391

Expected characteristics:

- 3,247 Bangla and code-mixed memes.
- Train: 2,272.
- Validation: 487.
- Test: 488.
- Hate: 1,158; Inflammatory: 1,106; Benign: 983 across the complete dataset.
- Image and embedded text, with the published description stating that extracted text was manually verified.
- CC BY-NC-SA 4.0 license. Preserve attribution and record the non-commercial and share-alike conditions in the data card.

Do not rely only on these expected counts. Audit the actual downloaded files, schema, missing values, image availability, and label distributions.

## Unified dataset index

Create a unified manifest without destroying the original files. Use Parquet if available and also export a human-readable CSV summary. Include at least these fields:

```text
record_id
source_dataset
source_split
image_path
image_available
text_original
text_bn
text_en
text_available
figsim_severity_original
figsim_severity_5
figsim_severity_4
figsim_severity_3
figsim_binary_any_severity
figsim_figurative_labels
figsim_suicide_related_labels
figsim_modality
figsim_context
cmban_humor
cmban_sarcasm
cmban_offensiveness
cmban_motivation
cmban_sentiment
bnhib_harm_label
depression_4
is_synthetic
parent_record_id
is_valid
exclusion_reason
sha256
phash
text_hash
```

Use null for unavailable labels. Do not encode missing labels as zero, None severity, benign, neutral, or Minimum.

Add source-aware validation that verifies:

- FigSIM counts total 973 and split 582/195/196.
- Every valid FigSIM record maps to exactly one three-class, one four-class, one five-class, and one binary target.
- CMBAN and BN-HIB rows have null FigSIM severity labels.
- Depression rows have null FigSIM severity labels unless a separate, explicit suicide annotation truly exists.
- Image paths resolve where `image_available` is true.
- Parent and synthetic child records cannot cross splits.
- Exact duplicate and near-duplicate image groups do not cross FigSIM splits.
- Duplicate external records do not cross their own supervised train, validation, or test splits.

Produce a machine-readable audit report and a short Markdown data card.

## Preprocessing rules

### FigSIM

Reuse the current leakage-safe images, EasyOCR outputs, English OCR, Bangla NLLB translations, reasoning text, DAPT embeddings, SigLIP embeddings, and split files. Regenerate an artifact only if it is absent, corrupt, or incompatible with the new loader. Log that decision.

### CMBAN

Use the supplied Bangla text directly. Do not run English OCR and NLLB when a verified Bangla transcription is already available. Preserve the supplied English translation for bilingual diagnostics. Normalize Unicode and whitespace while retaining an untouched raw field.

### BN-HIB

Use the supplied or verified Bangla/code-mixed text directly. Preserve code-mixing. Do not translate English tokens away unless an explicit ablation requires it. Normalize Unicode and whitespace while retaining the raw text.

### Images

Use the existing SigLIP processor and image size. Apply only mild training augmentation:

- modest resized crop that does not remove most meme text;
- small brightness and contrast variation;
- mild blur or JPEG compression;
- small amounts of noise.

Do not horizontally flip images because that reverses embedded text. Do not apply augmentations to validation or test images.

## Cross-dataset pretraining

The main purpose of CMBAN and BN-HIB is to improve Bangla meme image-text representations before FigSIM specialization.

Use only their official training splits for supervised pretraining by default:

- CMBAN train: approximately 2,113 pairs.
- BN-HIB train: approximately 2,272 pairs.
- Combined expected training pool: approximately 4,385 image-text pairs before deduplication and quality filtering.

Implement the following components:

1. DAPT-BanglaBERT text encoder.
2. SigLIP-so400m vision encoder.
3. Trainable projections into a shared 256-dimensional space.
4. Symmetric image-to-text and text-to-image InfoNCE contrastive loss.
5. Dataset-specific auxiliary heads:
   - BN-HIB Hate, Inflammatory, Benign.
   - CMBAN sarcasm.
   - CMBAN offensiveness.
   - CMBAN overall sentiment.
6. Optional CMBAN humor and motivation heads controlled by configuration and disabled by default.

Start with frozen base encoders and train projections plus auxiliary heads. Then add a configurable LoRA experiment because LoRA previously helped while full-layer unfreezing overfit. Do not perform full unfreezing.

Suggested initial pretraining objective:

```text
L pretrain =
1.00 L contrastive
+ 0.25 L BN-HIB harm
+ 0.15 L CMBAN sarcasm
+ 0.15 L CMBAN offensiveness
+ 0.10 L CMBAN sentiment
```

Losses must be masked by source and label availability. Normalize the contribution of each source so that one dataset does not dominate only because it has more rows.

Implement an optional shared sarcasm auxiliary label:

- CMBAN Not Sarcastic becomes 0.
- CMBAN Little Sarcastic and Very Sarcastic become 1.
- FigSIM Irony or Sarcasm present becomes 1.
- FigSIM Irony or Sarcasm absent becomes 0.

Because the annotation schemes differ, this must remain an auxiliary experiment. Do not make it part of the primary suicide label.

Save the best pretraining checkpoint using external validation loss or retrieval metrics. Never use FigSIM test performance to choose a pretraining checkpoint.

Report at least:

- image-to-text Recall at 1, 5, and 10;
- text-to-image Recall at 1, 5, and 10;
- source-specific retrieval metrics;
- auxiliary-head macro-F1 on the appropriate external validation split;
- training curves and early-stopping decision.

## FigSIM three-class model

Initialize the text and image encoders, projection layers, and any compatible fusion components from the external pretraining checkpoint. Retain two complementary fusion models:

1. Gated fusion with orthogonal image residual and dropout 0.4.
2. Token-patch cross-attention using DAPT tokens as queries over SigLIP patches.

Keep a simple concatenation model as a diversity baseline and LoRA diagnostic.

The main three-class head produces:

```text
0 No expressed suicide severity signal
1 Suicidal thought or desire
2 High acuity suicidal content
```

Add these auxiliary heads from the same shared representation:

- original five-class FigSIM severity;
- binary None versus any expressed severity;
- FigSIM modality if available;
- FigSIM context if available;
- FigSIM figurative labels if available;
- FigSIM suicide-related-content labels if available.

Use masked losses for missing auxiliary labels. A reasonable initial objective is:

```text
L FigSIM =
1.00 L severity3
+ 0.25 L severity5
+ 0.20 L any-severity binary
+ 0.10 L modality
+ 0.10 L context
+ 0.10 L figurative
+ 0.10 L suicide-related content
```

Treat the weights as configuration values. Tune only a very small predefined set on validation or grouped cross-validation. Do not perform an open-ended weight search.

Use modality dropout during FigSIM training:

- 70 percent keep both text and image;
- 15 percent drop the text representation;
- 15 percent drop the image representation.

At inference, use both modalities when available. Include explicit modality-availability masks so a missing branch is distinguishable from an all-zero natural embedding.

## Integration of the Phase 2 Bangla depression dataset

Use the original four-class Bangla depression task as auxiliary text supervision. Do not merge its labels with suicide severity.

The shared DAPT-BanglaBERT encoder receives both:

- FigSIM translated Bangla meme text for the multimodal suicide task.
- Phase 2 Bangla text for the Minimum, Mild, Moderate, Severe depression task.

Attach a separate depression head. Alternate source-balanced batches or use a configured batch mixture. Start with a depression-loss weight of 0.15. If the augmented 8,000 to 9,000 record dataset is used, ensure that synthetic variants only appear in training and cannot leak across folds through their parent records.

At final FigSIM inference, do not require or display the depression prediction. Its purpose is to preserve and regularize the mental-health language representation learned in Phase 2.

Optionally sample 10,000 to 20,000 texts from the existing DAPT corpus for a low-weight masked-language-model replay loss. Keep this disabled by default until the complete classification pipeline works. If enabled, start with weight 0.05 and document runtime and validation effect.

## Training sequence

Implement the work in this order.

### Phase A Repository and data audit

1. Inspect the repository structure, active environment, GPU, disk space, dependency versions, checkpoints, and cached artifacts.
2. Identify the current training entry points and reproduce one documented validation metric or run a deterministic loading-only audit if full reproduction is too expensive.
3. Locate or configure CMBAN and BN-HIB roots.
4. Build the unified index and all integrity reports.
5. Calculate exact harmonized class distributions for train, validation, and test without using test results for any decision.

### Phase B Label-collapse diagnostic

1. Locate saved five-class validation probabilities.
2. Aggregate them into three and four classes.
3. Save diagnostic metrics.
4. Do not change the selected label scheme based only on one split. The three-class scheme remains the planned primary task because its grouping was defined before the diagnostic.

### Phase C External meme pretraining

Run these ablations with the same initialization and seeds where feasible:

- P0 no external pretraining;
- P1 BN-HIB only;
- P2 CMBAN only;
- P3 CMBAN plus BN-HIB contrastive pretraining;
- P4 CMBAN plus BN-HIB contrastive and dataset-specific auxiliary heads;
- P5 P4 with LoRA, only if P4 is stable.

Use the external validation metrics and later FigSIM train-validation evaluation to select the transferable checkpoint. Do not use any FigSIM test metric.

### Phase D FigSIM harmonized training

Train:

- H0 three-class concatenation baseline;
- H1 three-class gated fusion;
- H2 H1 plus modality dropout;
- H3 H2 plus five-class and binary auxiliary heads;
- H4 H3 initialized from the best external pretraining checkpoint;
- H5 three-class cross-attention initialized from the same checkpoint;
- H6 optional shared DAPT multi-task training with the Phase 2 depression dataset;
- H7 final gated plus cross-attention ensemble chosen from validation or out-of-fold results.

Do not require every experiment to run if a cheaper predecessor fails clearly. Record stop decisions.

### Phase E Robust model selection

Keep the fixed validation set for direct comparability with the existing work. In addition, perform grouped five-fold cross-validation on the combined FigSIM train plus validation portion when compute permits. Preserve perceptual duplicate or template groups within folds. Never include the fixed test set in these folds.

Use out-of-fold probabilities for any stacking model. Limit stacking to a regularized logistic regression or another low-capacity linear meta-learner. Compare it with majority vote and simple probability averaging. Do not tune per-class multipliers directly on the same validation labels.

### Phase F Final evaluation

After the full configuration is locked, evaluate once on the existing 196-record test split. Produce both three-class results and the auxiliary five-class results from the same locked system. Do not change the model after viewing them.

Explicitly state in the report that the test split was used in earlier project stages and is therefore a reused held-out evaluation set, not a perfectly untouched blind test.

## Evaluation requirements

For the main three-class task, report:

- macro-F1 as the primary metric;
- per-class precision, recall, F1, and support;
- weighted-F1;
- balanced accuracy;
- overall accuracy;
- confusion matrix;
- quadratic weighted kappa;
- ordinal mean absolute error;
- expected calibration error or a reliability plot;
- mean and standard deviation across seeds;
- grouped cross-validation mean and standard deviation when available.

For five-class auxiliary results, retain the existing metrics for comparison with earlier work.

Never present the higher three-class macro-F1 as a direct improvement over the five-class macro-F1. They are different tasks. Describe the result as a granularity versus reliability comparison.

Add error slices for:

- text, image, and complementary FigSIM modality;
- context required versus not required;
- figurative versus non-figurative;
- OCR text present versus absent;
- low versus high OCR confidence;
- prediction confidence bands;
- source dataset during pretraining diagnostics.

## Required ablation table

Generate a report that includes at least:

| ID | Target | External pretraining | Fusion | Auxiliary supervision | Purpose |
|---|---|---|---|---|---|
| B0 | Five class | None | Existing best ensemble | Existing | Historical baseline |
| H0 | Three class | None | Concatenation | None | Harmonized baseline |
| H1 | Three class | None | Gated | None | Multimodal baseline |
| H2 | Three class | None | Gated | Modality dropout | Robust modality use |
| H3 | Three class | None | Gated | Five-class plus binary | Coarse-to-fine supervision |
| H4 | Three class | CMBAN plus BN-HIB | Gated | Dataset-specific pretraining heads | Cross-domain transfer |
| H5 | Three class | CMBAN plus BN-HIB | Cross-attention | Dataset-specific pretraining heads | Fine image-text interaction |
| H6 | Three class | CMBAN plus BN-HIB | Best fusion | Bangla depression auxiliary | Phase 2 integration |
| H7 | Three class | Best | Ensemble | Best validated combination | Proposed final system |

Also include four-class and binary tasks as secondary analyses if they can be produced without compromising the schedule.

## Code organization

Adapt names to the repository, but keep responsibilities separate. A suitable structure is:

```text
configs/
  data_paths.example.yaml
  pretrain_cmban_bnhib.yaml
  finetune_figsim_3class.yaml
  ablations/

src/
  data/
    build_unified_manifest.py
    harmonize_figsim_labels.py
    external_meme_datasets.py
    duplicate_audit.py
    samplers.py
  models/
    encoders.py
    projections.py
    gated_fusion.py
    cross_attention.py
    multitask_heads.py
    lora_setup.py
  training/
    losses.py
    pretrain_meme_alignment.py
    train_figsim_harmonized.py
    train_multitask.py
  evaluation/
    aggregate_saved_probabilities.py
    metrics.py
    error_slices.py
    compare_experiments.py

tests/
  test_label_mapping.py
  test_missing_label_masks.py
  test_split_integrity.py
  test_duplicate_groups.py
  test_loss_routing.py
  test_model_shapes.py

docs/
  DATA_INTEGRATION_CARD.md
  IMPLEMENTATION_LOG.md
  EXPERIMENT_PROTOCOL.md
```

Do not perform a large refactor merely to match this example. Reuse the working repository layout when possible.

## Configuration and reproducibility

All important choices must be configurable:

- data roots;
- checkpoint roots;
- cached embedding roots;
- label scheme;
- enabled datasets;
- enabled auxiliary heads;
- loss weights;
- LoRA rank and target modules;
- modality-dropout probabilities;
- seeds;
- batch size;
- learning rates;
- early stopping;
- output directory;
- whether test evaluation is allowed.

Require an explicit flag such as `--allow-test-evaluation` before any test loader returns labels or metrics. The default must refuse test evaluation.

Save:

- resolved configuration;
- package versions;
- git commit or working-tree status if available;
- random seeds;
- class mappings;
- class distributions;
- checkpoint provenance;
- preprocessing provenance;
- metrics in JSON and CSV;
- confusion matrices and plots;
- predictions with record IDs;
- run logs.

Use deterministic settings where practical and document any unavoidable nondeterminism.

## Tests and acceptance criteria

The implementation is complete only when the following checks pass:

1. Unit tests verify every FigSIM label mapping, including both original six-label spellings and the existing merged attempt/death spelling.
2. Split integrity tests reproduce 582 train, 195 validation, and 196 test FigSIM records.
3. No invalid record reaches a dataset loader.
4. Missing labels are masked and never converted to class zero.
5. A CMBAN row cannot contribute FigSIM severity loss.
6. A BN-HIB row cannot contribute FigSIM severity loss.
7. A depression row cannot contribute suicide-severity loss.
8. Each source contributes only its available auxiliary losses.
9. Contrastive batches contain correctly matched image-text positives.
10. Modality dropout never drops both modalities simultaneously.
11. Training and validation augmentations differ as intended.
12. A CPU or small-GPU smoke test completes one forward and backward pass for external pretraining and FigSIM fine-tuning.
13. A small overfit test on a tiny training subset shows that the model and losses can learn.
14. Existing historical checkpoints and results remain untouched.
15. Test evaluation remains disabled by default.

## Documentation requirements

Update or create documentation covering:

- the exact three-class and four-class mappings;
- why None is not named simply non-suicidal;
- why Invalid is filtered instead of modeled;
- why CMBAN and BN-HIB labels are not converted into suicide labels;
- how external image-text pretraining works;
- how dataset-specific heads and masks work;
- how Phase 2 depression data supports the shared DAPT encoder;
- exact dataset counts after filtering and deduplication;
- license and citation information;
- experiment commands;
- completed, failed, and skipped runs;
- limitations and test-set reuse.

Keep a chronological implementation log. Record failures as well as successful results. Never replace an old reported result without explaining why.

## Decision rules

Use these decision rules to control scope:

- If a pretraining variant does not improve mean validation or cross-validation macro-F1 and does not improve a clearly relevant error slice, do not carry it into the final ensemble.
- If LoRA becomes unstable, keep the frozen-encoder checkpoint and record the failed ablation. Do not unfreeze full encoder layers.
- If CMBAN-only or BN-HIB-only transfer hurts, test the combined contrastive checkpoint once before dropping external pretraining.
- If auxiliary heads reduce the main task, lower their weights once using the predefined alternatives, then remove them rather than performing an open-ended search.
- Prefer a stable improvement across folds and seeds over a higher single-split peak.
- Preserve the three-class primary task even if a binary model scores higher. Binary detection is a secondary task and answers a less detailed question.
- Do not add audio, video, new manual annotation, synthetic suicide-image generation, or another large architecture during this implementation.

## Expected final response from Claude Code

When implementation and available verification are complete, provide:

1. A concise description of the completed pipeline.
2. The exact files created or changed.
3. Dataset audit counts and exclusions.
4. Tests and smoke runs performed, including failures.
5. Validation or cross-validation results that were actually obtained.
6. A statement confirming whether the test set remained locked.
7. Any missing datasets, checkpoints, or paths that prevented specific runs.
8. The next exact command to continue from the current state.

Do not claim success for experiments that were not run. Do not present expected gains as measured gains.

