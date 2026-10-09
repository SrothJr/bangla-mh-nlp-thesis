# Project Handoff Brief — Bangla Multimodal Mental-Health Thesis (v2, updated)

**Purpose of this document:** self-contained context transfer for a new Claude Code session running on the device with the DAPT-BanglaBERT checkpoint, the FigSIM dataset, and the RTX 5090. Everything below is final and settled, including scope cuts made under a real deadline. Do not re-derive, re-litigate, or "improve" these decisions without a specific, stated reason — they were arrived at deliberately, under time pressure, with tradeoffs already considered.

**Real submission deadline: 26 September 2026. Internal working deadline: 20-22 September 2026** (23-26 Sept is a deliberately untouched emergency reserve — do not plan to use it, only fall back on it if something breaks).

---

## 1. Background: what already exists (Phase 2 — do not touch)

1. English Reddit mental-health posts translated to Bangla via **NLLB-200**.
2. Translated corpus (~43,278 posts) used for **Domain-Adaptive Pretraining (DAPT)** of **BanglaBERT** → **DAPT-BanglaBERT**.
3. Fine-tuned on a labelled Bangla dataset (4,898 samples) → 4-class depression severity: **Minimum, Mild, Moderate, Severe** (codes 0-3).
4. Class distribution: Minimum 2097, Mild 1540, Moderate 707, Severe 553. Original eval: 5-fold stratified CV.

**This system is complete and stays completely unchanged.** It remains its own thesis chapter/result.

**File-safety rules (non-negotiable):**
- The DAPT folder must be backed up and marked read-only before any new work begins.
- All new code and outputs go into a separate, new project folder.
- Only ever *read* the checkpoint (for inference); never write/save anything back into that folder.

### 1.1 Environment/venv — DECIDED: reuse the existing venv (disk space is tight — ~18GB free)

**Status: already done, do not redo.** A `requirements_snapshot.txt` (the output of `pip freeze` run on the existing venv in `DAPT_models`, before any new installs) has already been created and moved into the new project folder (`Multimodal_figsim`, sibling to `DAPT_models` — see Section 1.2). This is the rollback reference if anything below causes a conflict.

**Decision, final: do NOT create a second, duplicate venv.** With only ~18GB free, duplicating torch/transformers/etc. into a second environment is a real, unnecessary cost — loading model *weight files* never required duplicating them in the first place (see Section 1.2), so the only actual disk cost of a second venv would be the libraries themselves, for no real benefit.

**Instead: install the small number of genuinely new packages directly into the existing venv** (already in `DAPT_models`, activate it from there when running code, or reference its Python interpreter from the new project folder):
- An OCR library: `pip install paddleocr` (or `easyocr`)
- `pip install scikit-fuzzy` (for the fuzzy-logic layer, Section 9)
- Whatever Python client talks to the local Ollama server (typically `pip install ollama`)

None of these packages should force a version change in torch/transformers, so the risk to checkpoint-loading compatibility is low — but if anything unexpected breaks after installing them, `requirements_snapshot.txt` tells you exactly what the working state looked like before, so you can diagnose or roll back precisely.

### 1.2 Confirmed folder structure

```
(some root folder)\
  ├── DAPT_models\                     ← existing, READ-ONLY, back up, never write here
  │     ├── .venv\                     ← the ONE venv in use — new small packages installed here (1.1), done
  │     ├── data\
  │     └── results\dapt_eval\         ← per-fold checkpoints — CONFIRMED, see Section 1.3
  ├── tari_translation_v3\             ← existing, ALSO READ-ONLY — Phase-2's original translation project
  │     └── nllb_model\                ← the REAL NLLB-200-3.3B lives here, not in DAPT_models (corrected)
  └── multimodal_figsim\               ← NEW, sibling folder — ALL new code, data, and outputs go here
        ├── requirements_snapshot.txt  ← done
        ├── i-h\ (extracted figsim.zip) → multimodal_figsim\data\leakage_safe\...
        ├── code\ (e.g. run_translation.py — confirmed working)
        ├── cached embeddings
        └── outputs
```

**Model weight files are never duplicated between these folders.** Code running from `multimodal_figsim` loads the DAPT-BanglaBERT checkpoint from `DAPT_models` and NLLB from `tari_translation_v3`, both read-only — no copying needed.

### 1.3 FULLY RESOLVED via the Day-1 sanity check — correcting an earlier wrong assumption in this brief

`results/dapt_eval/` contains three base models (BanglaBERT, mBERT, sahajBERT), each with 5 folds, each fold in a plain and a `_tapt` version. Fold 5 is confirmed as the best-performing fold.

**Important correction — an earlier version of this brief incorrectly named `BanglaBERT_fold5_tapt` as the checkpoint to use. That is wrong. Do not use it.** Investigation during the Day-1 sanity check found:

- **`BanglaBERT_fold5_tapt/` is the TAPT/MLM checkpoint saved *before* fine-tuning** (`architectures: ["ElectraForMaskedLM"]`, no `id2label` — it never learned the 4-class task at all).
- **The correct checkpoint is `results/dapt_eval/BanglaBERT_fold5/checkpoint-735/`** — confirmed `architectures: ["ElectraForSequenceClassification"]`, `id2label: {0: Minimum, 1: Mild, 2: Moderate, 3: Severe}`, and `trainer_state.json` shows `best_metric: 0.8758` at `best_global_step: 735`, matching the recorded fold-5 F1 exactly.
- **Known gotcha: `checkpoint-735` has no saved tokenizer files** (the original Trainer was never given a tokenizer to auto-save). Loading `AutoTokenizer.from_pretrained` directly on `checkpoint-735` silently produces a dummy tokenizer and reproduces only ~42% accuracy — a *silent* failure, not a crash. **Fix: load the tokenizer from the sibling `BanglaBERT_fold5_tapt/` directory instead** (same tokenizer throughout the pipeline, never modified). With this fix, reproduced metrics exactly match `BanglaBERT_dapt_metrics.json` (accuracy 0.8795, F1 0.8758, precision 0.8812, recall 0.8795) — sanity check passed.

**Final, confirmed instruction for building the frozen text encoder:** load the model body from `results/dapt_eval/BanglaBERT_fold5/checkpoint-735` (strip the `ElectraForSequenceClassification` head, keep the Electra encoder), and load its tokenizer from `BanglaBERT_fold5_tapt/` (not from `checkpoint-735`, which has no tokenizer saved).

---

## 2. The research gap and objective

Phase 2 is text-only. Memes (image + embedded text) are common mental-health-relevant content where meaning often depends on image-text interaction (e.g., sarcasm — cheerful image + dark caption). **Objective:** reuse DAPT-BanglaBERT as a frozen, transferred text encoder inside a new multimodal architecture for a second, related task.

---

## 3. The dataset: FigSIM — FINAL, CONFIRMED DECISIONS

**Source:** Chen et al., "FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes," Findings of ACL 2026.

- 1,049 Reddit memes (r/SuicideMeme), English.
- **Leakage-safe cleaned subset (already built — verify these files exist before redoing anything):** 973 retained records, split **Train 582 / Validation 195 / Test 196**. Look for `figsim_leakage_safe_index.csv`, `leakage_resolution_summary.json`, `leakage_decisions.csv` under `multimodal_figsim/data/leakage_safe/`.

### CONFIRMED target label scale: 5 classes, not 6

**Decision, final:** merge "Suicide attempts" and "Suicide death" into one class — **"Suicide attempt or death."** Reason: each has only ~6 examples in the test split; a classifier cannot be reliably trained or evaluated on classes that small, and reporting confident numbers on them would be indefensible. This exactly follows the precedent set by FigSIM's own authors in their own paper.

**Final target scale (5 levels, ordinal):**
1. None
2. Wish to be dead
3. Suicide ideation
4. Suicide planning
5. Suicide attempt or death (merged)

Use this 5-level scale **consistently everywhere** — the classifier, the ablation table, and the fuzzy-logic rule table (Section 9). Do not use the original 6-level naming anywhere in this project; the classifier physically cannot produce two distinct confidence values for Attempt vs. Death given the training data, so keeping them conceptually separate anywhere downstream would be misleading.

### Other FigSIM facts already established
- Modality field (already annotated, free to use for stratified analysis): Complementary 879, Text 68, Image 26.
- Context field (already annotated): No-context-required 847, Context-required 126.
- FigSIM's own fine-tuned CLIP fusion barely beat text-only in their paper; the largest gains came only from massive frozen prompted MLLMs, not from an embedding-fusion architecture. **Do not assume gated fusion produces a large gain — measure it, don't assume it.**
- Documented weakness to target: severity is systematically **underpredicted for figurative memes** (their Table 12/Appendix F).
- **Do not** attempt to relabel FigSIM images under the Bangla 4-class depression scale — under the original clinical criteria, ~85% of FigSIM content would collapse into one class under a naive mapping. This has already been analyzed and rejected. The two tasks (depression severity, suicide severity) are two separate, honest outputs sharing one encoder — this framing has been independently confirmed by the researcher's supervisor's RA as the correct way to present it.

---

## 4. Hardware

- **Local: RTX 5090 (32GB VRAM), ~61GB+ RAM.** No Kaggle, no cloud, no paid API — everything runs on this one machine.

**Models to run locally:**
- AI reasoning: **`qwen2.5vl:7b` — CONFIRMED pulled and vision-capable (6.0GB), Day 1 complete.** Not yet used for inference — that's Day 3.
- Translation: **NLLB-200-3.3B — location CORRECTED.** It is NOT inside `DAPT_models` (that cache entry is an empty stub — do not use it). **The real, complete model is at `tari_translation_v3/nllb_model/`** (Phase-2's original translation project, a separate folder from `DAPT_models`), confirmed 3.3B via its config. Used read-only from there, same as `DAPT_models` — no redownload, no modification.
- Vision encoder: **SigLIp-so400m only** — chosen directly, no CLIP-vs-SigLIp comparison (cut for time; see Section 6). Not yet extracted — Day 2.
- OCR: **EasyOCR — CONFIRMED in use, not PaddleOCR.** PaddleOCR was abandoned: it needs a separate `paddlepaddle` engine not already installed, which risked a large download on the now-tight disk. EasyOCR reuses the existing torch/CUDA install already present, at no extra disk cost. **Day 1 complete: 973/973 images processed, 0 errors** (one fixed bug: an animated GIF crashed EasyOCR until normalized via PIL first). 18 images had no text — matches FigSIM's own known count of image-only memes, a good consistency check that the pipeline is working correctly.

**Disk space status (update as it changes): ~11GB free as of end of Day 1**, down from ~18GB — the `qwen2.5vl:7b` pull (6GB) accounts for nearly all of it; OCR and translation outputs added under 1MB. Remaining planned work (SigLIp, cached embeddings, small trained pieces) is small, but this is now a genuinely tight budget — avoid pulling any additional models beyond what's already specified in this brief without re-checking free space first.
- DAPT-BanglaBERT: **model body from `results/dapt_eval/BanglaBERT_fold5/checkpoint-735`, tokenizer from `BanglaBERT_fold5_tapt/`** (confirmed via Day-1 sanity check, see Section 1.3 — note the checkpoint and tokenizer come from two different sibling folders, this is not a typo). **Body only, classification head discarded.** (Deliberately using the fine-tuned checkpoint, not the DAPT-only-before-fine-tuning artifact — the fine-tuning task and the new task are both severity judgments, so the specialization risk is low, and this is the exact artifact already validated in Phase 2, which also makes the "one encoder, two tasks" thesis narrative cleaner.)

### 4.1 IMPORTANT — this is a shared PC, plan for interrupted access

The RTX 5090 machine is **shared with other groups/people** who may need it unpredictably, potentially costing a day or more of lost access at any point. This is a real, live risk — plan around it explicitly, don't just hope it doesn't happen:

- **Split all work into two lists and switch between them based on machine access, not the calendar:**
  - *GPU-dependent* (needs this machine): checkpoint sanity check, OCR, translation, AI reasoning pass, embedding extraction, all training (gate, alignment, classifier), the final test-set run.
  - *No-GPU* (can happen anywhere, anytime, on any device): drafting methodology/ethics/limitations text, designing/refining the fuzzy-logic rule table, planning error-analysis categories, formatting, proofreading.
  - When you have machine access, work strictly down the GPU-dependent list in priority order. When you don't, switch to the no-GPU list. Access loss should cost switching lists, not losing progress.
- **Front-load GPU-heavy work whenever access is actually available** — don't wait for a "scheduled day" if the machine is free right now; grab it and move down the priority list.
- **Save every stage's output to disk immediately**, never leave anything only in memory or mid-run — OCR results, translations, embeddings, model checkpoints, all persisted right after they're produced, so an interruption never costs recomputation.

### 4.2 IMPORTANT — build every batch-processing stage to actually resume, not just to save

"Save to disk" alone is not enough — a script also needs to check what's already done and skip it, or an interruption at item 500 of 973 means starting over from item 1. Every stage that loops over the 973 memes (OCR, translation, AI reasoning generation, embedding extraction) must follow this pattern:

1. **Write results incrementally, one item at a time** — append each meme's result to an output file (CSV/JSONL) and flush it to disk immediately after that one item finishes, not buffered in memory until the whole batch is done.
2. **At the start of every run, check what's already been written** and build the list of remaining items by excluding whatever's already in the output file. Only process what's missing.
3. **Practical effect:** if the machine is interrupted after 500 of 973 images, simply re-running the same script picks up at item 501 automatically — no manual bookkeeping, no reprocessing, no lost work.

This matters most for OCR, translation, and the AI-reasoning pass, since those loop over all 973 memes and can take real time. It matters much less for training the small pieces (alignment, gate, classifier) — those finish in minutes given the frozen encoders and cached embeddings, so if a training run gets interrupted, simply restarting it from scratch is fine and not worth extra engineering. (If it ever becomes worth it, the equivalent pattern for training is: save a checkpoint — model weights + optimizer state + current epoch — every epoch, and resume from the last saved epoch on restart, rather than epoch 0.)

### 4.3 Checkpoint retention policy — keep at most 3 files per trained component, never a full history

For every trained component (Stage A alignment projections, Stage B gate, Stage D classifier — and the stretch-goal two-model ensemble if built), **do not save a separate checkpoint file per epoch.** Keep only:
- **`best`** — overwritten only when a new best validation score is reached. This is the file actually used for evaluation and the final model.
- **`current`** — the latest epoch's weights, overwritten every epoch. This is what a restart resumes from if training was interrupted.
- **`previous`** — one epoch back from `current`, kept as a one-step-back safety net in case the latest save is corrupted or interrupted mid-write.

Never accumulate every epoch's checkpoint indefinitely — that grows without bound for no benefit here. In practice, since these trained pieces are small (a few linear/MLP layers on top of frozen encoders, not full models), this policy mostly matters as good discipline rather than solving a large disk problem — but it costs nothing to build in from the start and avoids any surprise disk creep during the E0-E7 ablation runs.

---

## 5. The architecture — FINAL scope, with what's core vs. stretch

### 5.1 Text branch (per meme) — CORE

0. **Before anything else, verify the reasoning model is actually vision-capable.** Run `ollama list` on the DAPT machine. Look at the exact tag: a vision-capable model has an explicit indicator (e.g. `qwen2.5vl`, `qwen2-vl`); a plain tag like `qwen2.5` or `qwen2` with no vision indicator is **text-only and cannot process images** — it would only ever see the OCR text, defeating the purpose of this step. If only a text-only Qwen is installed, pull a vision-capable alternative instead (`qwen2.5vl` if available, or `llama3.2-vision`, or `llava`) — the specific model matters less than "can it take an image plus a prompt and return text."
1. **OCR** extracts English text + confidence score.
2. **AI reasoning — ONE model (Qwen2.5-VL-7B-Instruct, or whichever vision-capable model was confirmed in step 0), not an ensemble.** Prompt: given image + OCR text, explain (a) likely cause-effect context, (b) any figurative/ironic/sarcastic meaning and what it means, (c) implied emotional state. Prompt must require quoted evidence and permit explicit "uncertain" answers rather than confident guessing.
   - **STRETCH GOAL, not core:** a second model (InternVL2-8B) + agreement-checking between the two, feeding a confidence signal into the gate. Only attempt this if Days 1-7 of the schedule (Section 11) finish early. Otherwise this is documented future work.
3. **Translation:** OCR text and reasoning text both translated to Bangla via NLLB-200. Capture translation log-probability as a confidence signal.
4. **Concatenation:** `[OCR_bn] [SEP] [reasoning_bn]`.
5. **Encoding:** through DAPT-BanglaBERT (frozen, body only) → one **text vector**.

### 5.2 Vision branch (per meme) — CORE

- Raw image → **SigLIp only** (no CLIP comparison — decided directly for time) → **image vector**.

### 5.3 Contrastive alignment (Stage A) — CORE, WITH A STOP-LOSS RULE

Adapted from Wang et al. ("CMAF"), ACM TOMM 2026.

- Small trainable projection layers map text vector and image vector into a shared dimension.
- Train with InfoNCE contrastive loss: a meme's own (text, image) pair = positive; other memes in the batch = negatives.
- Self-supervised, no labels needed, run on the training split.
- **This is the single highest-risk technical component in the whole plan** (contrastive training can be unstable on a small dataset). **Explicit stop-loss rule: if this is not showing sensible, decreasing loss after a reasonable debugging effort (allocated: Sept 14-15, two days), drop it entirely and proceed with Stage B/C applied directly to the unaligned vectors.** Do not let open-ended debugging here eat into the test-set evaluation day.

### 5.4 Reliability-aware gated fusion + orthogonal feature (Stage B/C) — CORE

Adapted from Yadav et al. ("RESTORE"), ACL 2023.

- **Gate:** small MLP takes text vector, image vector, OCR confidence, translation confidence (and reasoning-agreement score, only if the stretch-goal ensemble was built) → outputs alpha (sigmoid) → `gated = alpha*text + (1-alpha)*image`.
- **Orthogonal feature:** project image vector onto text vector, subtract the shared component, keep what's left. **This is NOT a trained component — just vector math, near-zero implementation cost.** RESTORE's own ablation found this the single largest lever in their system (+2.6 F1, larger than the gate itself at +1.2) — do not skip this one, it is cheap and high-value.
- Combine: LayerNorm both, concatenate with raw text vector → final representation `z`.

### 5.5 Classification (Stage D) — CORE

- `z` → small classifier head → prediction over the **5-class merged FigSIM scale** (Section 3).
- Consider ordinal loss (CORAL-style) vs. plain softmax cross-entropy as one comparison point.

### 5.6 Frozen vs. trained — summary

| Component | Status |
|---|---|
| DAPT-BanglaBERT | Frozen (body only) |
| SigLIp | Frozen |
| Qwen2.5-VL (reasoning) | Frozen, inference only |
| NLLB-200 | Frozen |
| Stage A alignment projections | Trained (self-supervised) |
| Stage B gate MLP | Trained (supervised) |
| Stage C orthogonal step | Not trained — fixed math |
| Stage D classifier | Trained (supervised) |

---

## 6. Vision encoder — DECIDED, no comparison

Use **SigLIp-so400m directly.** The CLIP-vs-SigLIp comparison experiment was cut for time. State in the methodology: "SigLIp was selected directly based on its strong recent benchmark performance; a comparative evaluation against CLIP was out of scope given project timeline constraints."

---

## 7. Reasoning-quality validation — REDUCED SCOPE

- Do an **informal spot-check**, not the full formal rated-sample study: look at ~15-20 of the AI's generated explanations, confirm they're sensible (not hallucinating wildly), write one paragraph about this in the limitations/methodology.
- The formal Likert-scale human-rating study (M3H-style) and the prompt-design ablation (simple vs. structured prompt) are both **cut for time** — document as future work.

---

## 8. Evaluation plan — FINAL ablation ladder

Cache every embedding once (OCR text, translated text, reasoning text, text vectors, image vectors) — never re-run the big frozen models per experiment; only the small pieces below get retrained per configuration, which is fast.

| Step | Configuration |
|---|---|
| E0 | Majority-class baseline |
| E1 | Image-only (SigLIp) |
| E3 | DAPT-BanglaBERT, text-only (OCR only, no reasoning) |
| E3b | DAPT-BanglaBERT, text + AI reasoning (single model) |
| E4 | Text + image, simple concatenation (no alignment, no gate) |
| E5 | E4 + gated fusion + orthogonal feature |
| E6 | E5 + contrastive alignment (Stage A) — **subject to the stop-loss rule in 5.3** |
| E7 | *(stretch goal only)* E6 + two-model reasoning ensemble |

**Protocol:** fixed 582/195/196 split throughout. Never touch the test set until the final configuration is locked (Day 7-8 of the schedule). Report macro-F1 (primary, classes are imbalanced), weighted-F1, per-class precision/recall/F1, confusion matrix, quadratic weighted kappa (classes are ordinal). Run 3 seeds where feasible. Slice test results by FigSIM's own `modality` and `context` metadata fields (free, already annotated) for the "when does the image help" analysis.

---

## 9. The fuzzy-logic risk-combination layer — CORE, MANDATORY, but implemented last

Separate, final step, applied *after* both classifiers (Phase-2 depression model and the new FigSIM model) have produced their own independent outputs. Does not retrain or touch either model.

### 9.0 IMPORTANT — when does this layer actually have two numbers to combine?

Fuzzy logic needs both a depression-severity reading AND a suicide-severity reading for the *same* piece of content. This does not happen automatically for every input type:

- **Plain text post, no image:** only the Phase-2 pipeline runs. Only one number exists (depression severity). **Fuzzy logic does not apply here** — report the single reading, nothing to combine.
- **A meme:** the FigSIM pipeline already extracts and translates the meme's embedded text (the OCR step) as part of producing the suicide-severity reading. **Reuse that same translated text and additionally run it through Phase-2's existing, untouched classifier** (treating the meme's caption as a short text post in its own right) to get a depression-severity reading for that same meme. This costs almost nothing — Phase-2's model is small, frozen, and already loaded; you're just running one more inference pass on text you already have translated and ready. Now both numbers exist for that one meme, and the fuzzy-logic layer has something real to combine.

So in practice: for text-only inputs, fuzzy logic is skipped. For meme inputs, always run the extracted OCR text through both Phase-2's classifier and the full FigSIM multimodal pipeline, and combine those two outputs via the rule table below.

**Implementation approach (simplified, decided):** use each classifier's own softmax output probabilities directly as fuzzy membership degrees — do **not** build separate triangular/trapezoidal membership functions on an abstract continuous scale; that's unnecessary extra design work given the classifiers already output per-class confidence.

**Draft rule table (send to supervisor/RA for review TODAY, before Day 9 implementation — their turnaround time is not something extra days on our end can control, so ask early):**

| Suicide severity (5-class) | Depression severity (4-class) | → Risk level |
|---|---|---|
| Attempt or Death | any | **Critical** |
| Planning | Moderate or Severe | **Critical** |
| Planning | Minimum or Mild | **Elevated** |
| Ideation | Moderate or Severe | **Elevated** |
| Ideation | Mild | **Elevated** |
| Ideation | Minimum | **Low** |
| Wish to be dead | Moderate or Severe | **Elevated** |
| Wish to be dead | Minimum or Mild | **Low** |
| None | Severe | **Elevated** |
| None | Moderate | **Low** |
| None | Minimum or Mild | **Minimal** |

Note the design logic in rule 1: any actual attempt/planning-level content triggers Critical regardless of the depression reading — acute safety signals should not be diluted by an unrelated mood-severity score. State this reasoning explicitly in the methodology.

**Implementation:** Python's `scikit-fuzzy` library (fuzzification → rule evaluation → aggregation → defuzzification).

**Validation:** qualitative only — there is no ground-truth "risk level" to score against. Run on a handful of paired example outputs, produce one illustrative table for the thesis, ideally reviewed by the mental-health experts already involved in the project.

---

## 10. Ethical framing (maintain throughout)

All predictions describe **severity of content expressed**, never a clinical diagnosis, never a claim about the poster's real intent, never a substitute for professional assessment. Applies identically to Phase-2 and this extension.

---

## 11. Schedule (internal target: finish by Sept 20-22; Sept 23-26 is untouched reserve)

**The dates below are pacing checkpoints, not strict one-task-per-day boundaries.** Most of these steps are fast on this hardware (OCR, translation, and embedding extraction can each take well under an hour). If several rows finish in one sitting, keep going immediately down the list rather than waiting for the "next day" — the goal is to move as fast as the pipeline actually allows and bank any early progress as extra buffer before Sept 20-22, not to pace deliberately slowly to match the table. Use the dates only to notice if you're falling *behind*, never as a reason to hold back when you're ahead.

| Day | Date | Task |
|---|---|---|
| 1 | Sept 10 | **COMPLETE.** Packages installed (EasyOCR, scikit-fuzzy, ollama client). `qwen2.5vl:7b` pulled (6.0GB) and confirmed vision-capable. Checkpoint sanity check passed (with the fold correction in Section 1.3). OCR: 973/973, 0 errors, 18 no-text images. Translation: 973/973 (955 actually translated, 18 empty correctly skipped), mean log-prob confidence -0.73, spot-checked fluent. Disk: ~11GB free remaining. |
| 2 | Sept 11 | **COMPLETE.** SigLIp-so400m (3.5GB) and DAPT-BanglaBERT (fold5/checkpoint-735 body, fold5_tapt tokenizer) embeddings extracted for all 973 images. E0/E1/E3 run on the VALIDATION split (test stays locked): E0 (majority-class) macro-F1 0.100, E1 (image-only, SigLIp) macro-F1 0.390, E3 (text-only, OCR-only) macro-F1 0.379. Both E1/E3 clear the majority-class floor by a wide margin; "Suicide planning" is the weakest class for both. Full detail in `PROGRESS_LOG.md` and `outputs/e0_e1_e3_results.json`. |
| 3 | Sept 12 | **COMPLETE.** Qwen2.5vl:7b reasoning pass on all 973 images (973/973, 0 errors, evidence-grounded JSON output, honest "uncertain" on ~3% of claims). Mid-day scare: `tari_translation_v3\` briefly vanished from disk (a teammate had moved it on this shared machine) then was moved back — resolved, no redownload needed, confirmed working again. Reasoning text translated via the same NLLB-200-3.3B, E3b embeddings extracted (OCR_bn + reasoning_bn via tokenizer pair-encoding). **E3b macro-F1 0.3834 vs E3's 0.3792 (val split) — only a marginal +0.004 lift**, though per-class effects are larger in both directions (ideation recall up, planning/wish-to-be-dead recall down). See `PROGRESS_LOG.md` for full detail. |
| 4 | Sept 13 | **COMPLETE.** E4 (concat) and E5 (gated + orthogonal) built and run, class-weighted CE and CORAL-ordinal loss, 3 seeds each on the validation split. **E4/E5 (weighted CE) both jump ~0.12-0.13 macro-F1 over E3b's 0.3834 (E4: 0.5038 ± 0.0057, E5: 0.4983 ± 0.0277) — fusion is real, not noise.** Gated fusion did NOT beat simple concat here (matches Section 3's own caution not to assume it would) and was noticeably less stable across seeds. CORAL underperformed weighted CE substantially for both (~0.35-0.37) — traced to a real, explicable cause (single-scalar rank-score bottleneck starving the rarest class), not a bug. Section 7 spot-check also done in parallel: reasoning is grounded and non-hallucinatory on text-bearing memes; on image-only memes it sometimes gets the objects right but the higher-level narrative wrong without flagging uncertainty (2/4 visually-verified cases) — full writeup and a ready-to-use limitations paragraph in `PROGRESS_LOG.md`. |
| 5-6 | Sept 14-15 | **COMPLETE.** CORAL dropped from further runs per Day-4 decision (weighted CE only going forward). Stage A contrastive alignment built — first attempt overfit badly (val loss diverged while train loss/retrieval kept improving), one regularization pass (plain linear projections, smaller shared dim, more weight decay/dropout) produced a modest but genuine, still-improving retrieval signal (~13x chance) — stop-loss rule watched closely but not triggered. **E6 gated+orth (aligned) = 0.5070 ± 0.0048 macro-F1, matching E4's 0.5038 ± 0.0057 within noise, but decisively fixing the gate-collapse diagnosed after Day 4** — alpha went from 64-71% collapsed near 0 (raw E5) to a genuinely balanced 0.37-0.51 range with 0% collapsed, and cross-seed variance dropped ~6x. E6 concat (aligned) added nothing over E4, as expected. Full detail in `PROGRESS_LOG.md`. |
| 7 | Sept 16 | **COMPLETE.** Locked final config: E6 gated+orth (aligned, weighted CE), justified by the mechanistic gate-fix finding. Test set (196 items) unlocked exactly once after model selection was finalized on validation alone. **Final test: macro-F1 0.4635 ± 0.0173, weighted-F1 0.4857 ± 0.0173, accuracy 0.4932 ± 0.0168, QWK 0.3654 ± 0.0153** (3-seed mean ± std). "Suicide planning" remains the weakest class (F1 0.304, primary seed), consistent with every prior experiment. Gate alpha on test confirms the Day 5-6 fix generalizes (mean 0.445, range 0.36-0.51, still fully non-collapsed). Full detail in `PROGRESS_LOG.md` and `outputs/final_test_results.json`. |
| 8 | Sept 17 | **COMPLETE.** Reasoning-quality spot-check was already done Day 4 (ahead of schedule). Error analysis (modality/context slicing) on the locked model's test predictions: Complementary (n=178) macro-F1 0.497 ≈ overall average as expected; Text (n=13) and Image (n=5) modality slices both weak (0.200, 0.100) though tiny-n; Context-required (n=25) macro-F1 0.302 vs No-context-required (n=171) 0.434 — a real, better-grounded gap. Full detail in `PROGRESS_LOG.md`. |
| 9-10 | Sept 18-19 | **COMPLETE.** Fuzzy-logic layer built: Phase-2's untouched depression classifier run on the same translated OCR text, combined with the locked model's suicide-severity output via Section 9's draft rule table (unmodified — no supervisor/RA feedback available to incorporate, flagged explicitly). Risk distribution across all 196 test memes: Minimal 13.3%, Low 32.7%, Elevated 31.1%, Critical 23.0% — no degenerate collapse. Illustrative table and one notable soft-membership finding in `PROGRESS_LOG.md`. |
| 11 | Sept 20 | **COMPLETE.** Full draft written: Methodology, Results (ablation ladder, gate-collapse finding, final test metrics, error analysis, fuzzy-logic layer), Limitations, Ethical framing — `THESIS_SECTIONS_DRAFT.md`. |
| 12 | Sept 21-22 | **COMPLETE.** Proofreading pass caught and fixed 3 real rounding/transcription errors that had propagated from the original chat reports into `PROGRESS_LOG.md` and the draft (final test macro-F1 0.464→0.463; "None" class precision 0.539→0.538; same macro-F1 fix applied to the already-published results artifact). Citations cross-checked against Section 12; added and flagged one citation (CORAL, not in the original source list) as unverified against a live source. **Internal "done" checkpoint reached.** |
| — | Sept 23-26 | **Untouched reserve.** Only use if something above genuinely slipped. |

**Writing runs in parallel with the schedule above, starting Day 1** — not stacked at the end:
- Methodology (architecture, every component and why, citations) — draft from Day 1, since the design is already fully settled.
- Ethics section — draft from Day 1.
- Limitations — draft the structure from Day 2-3; explicitly name the cut items (CLIP comparison, two-model ensemble, formal quality-validation study) as scoped decisions under deadline, not oversights.
- Results section structure/tables — build early, drop in real numbers once Day 7 produces them.

**Action for today (Day 1), in addition to the technical work:** send the supervisor/RA the fuzzy-logic rule table in Section 9 for a quick sanity check now — do not wait until Day 9.

### 11.1 Mandatory progress checkpoint — end of Day 4 (Sept 13)

At this point, honestly assess whether E0/E1/E3/E3b/E4/E5 are done or close to done. This is a decision point, not just a status update:

- **On pace →** continue into the contrastive alignment stage (Days 5-6) as planned.
- **Behind pace →** immediately drop to the lean fallback: skip contrastive alignment (E6) entirely, skip the fuzzy-logic layer's full implementation in favor of a minimal illustrative version (or document it as future work if truly tight), lock in E5 as the final model, and move straight to the final test-set run and writing. A complete thesis built on E0-E5 is the guaranteed floor — it was validated as a real, defensible result earlier in this project's design process, and falling back to it is a plan working correctly, not a failure.

Do not let enthusiasm about the fuller version push this decision past Day 4 — the whole point of a checkpoint is that it happens on schedule, not "whenever it feels done."

---

## 12. Key sources (for thesis citation)

- Chen et al., "FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes," Findings of ACL 2026.
- Yadav et al., "Towards Identifying Fine-Grained Depression Symptoms from Memes," ACL 2023, pp.8890-8905. (Gated fusion + orthogonal feature mechanism, Section 5.4.)
- Wang et al., "Multimodal Depression Estimation via Contrastive Modality Alignment and Fusion," ACM TOMM 2026, DOI 10.1145/3778172. (Contrastive alignment mechanism, Section 5.3.)
- Mazhar et al., "Figurative-cum-Commonsense Knowledge Infusion for Multimodal Mental Health Meme Classification" ("M3H"), WWW 2025, DOI 10.1145/3696410.3714778. (Reasoning-generation methodology, Section 5.1.)
- Park et al., "Iterative Large Language Model-Guided Sampling and Expert-Annotated Benchmark Corpus for Harmful Suicide Content Detection," JMIR Medical Informatics 2026;14:e73725. (Evidence that captions/descriptions outperform raw vision-token input — supports the OCR/caption-mediated text design over naive raw-pixel fusion.)
