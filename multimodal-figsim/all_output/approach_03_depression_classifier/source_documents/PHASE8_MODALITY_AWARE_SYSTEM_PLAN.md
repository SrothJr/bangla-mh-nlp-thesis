# Phase 8 — Modality-Aware System: Preserve Phase 2, Add Missing-Input Support

**Status: planned, about to start.** Written before any new code runs, per this project's established discipline. Triggered by `CLAUDE_HANDOFF_MODALITY_AWARE_MULTIMODAL_SYSTEM.md`, which was read in full before this plan was written.

## 1. Why this phase exists

Every architecture built across this whole project (concat, gated fusion, cross-attention — the entire 9-model ensemble) assumes **both** text and image are always present. There is no defined, tested behavior for a text-only post or an image with no readable text. This was flagged honestly as a known gap in `THESIS_SECTIONS_DRAFT.md`'s limitations section, and the handoff document above proposes a concrete way to close it: **add routing and fallback paths around the existing system, without touching what already works.**

This is explicitly **not** a redesign. The handoff document is direct about this: *"Do not redesign the entire project before reading this document. A complete replacement with a large vision-language model is not recommended under the current small-data and short-deadline conditions."*

## 2. What this phase modifies vs. what it protects

**Protected, untouched, verified rather than assumed safe:**
- The Phase 2 depression classifier (checkpoint, tokenizer, preprocessing, label order) — wrapped and parity-tested, never retrained or altered.
- The existing 9-model image+text ensemble — used exactly as-is when both modalities are present. No retraining, no re-tuning.
- The locked test result (0.5730 macro-F1) — not touched again in this phase.
- All existing checkpoints, embeddings, and result files — new work goes in new, separate locations, following this project's existing "never overwrite" convention.

**New, additive components this phase builds:**
- A parity-tested wrapper around the Phase 2 depression classifier (verification, not modification).
- A new, separate text-only 3-class suicide-content head (frozen DAPT + a new small classifier).
- An optional image-only fallback head (frozen SigLIP + a new small classifier), only if validation supports it.
- A modality-aware router that picks a path based on what input is actually available.
- One correction to existing documentation (Section 3 below) — not a code change, a reporting fix.

## 3. First and most urgent item: fix an invalid comparison already in our documentation

Before any new code, one thing needs correcting. `IMPROVEMENT_PLAN.md`, `THESIS_SECTIONS_DRAFT.md`, and `CURRENT_FINAL_SYSTEM_REPORT.md` all currently report **"+0.0746 improvement" (0.5730 vs. 0.4984)** as if it were a direct model comparison. The handoff document correctly flags this as **not a valid direct comparison**: 0.4984 was measured on the **5-class** task, 0.5730 on the **3-class harmonized** task — different targets, not directly subtractable.

**The valid fix, already partially done:** we do have a way to check this properly — `phase7_1_label_harmonization_diagnostic.py` already established the method (aggregate a 5-class model's own predictions into 3-class groupings, no retraining needed) and applied it on validation. **What's missing is doing the same collapse on the ORIGINAL LOCKED 5-CLASS TEST predictions** (`outputs/final_test_predictions.json`, already saved, per-example, from the very first Day-7 lock) — collapsing those into the 3-class scheme and computing macro-F1 on the exact same 196 test records, for a true apples-to-apples number.

This is Phase 8's Step 0 — cheap (no training, pure aggregation, matching the Step 1 diagnostic's original method), and it corrects the record before anything else is built on top of it.

## 4. Preservation requirements (non-negotiable, per the handoff doc)

- **Checkpoint isolation.** New work goes under new, clearly-named directories (e.g. `outputs/checkpoints/suicide_text_3class/`, `outputs/checkpoints/suicide_image_fallback/`). Never write into the Phase 2 depression checkpoint's directory.
- **Parity test before anything else touches the depression classifier's path.** A fixed sample of Phase 2's own original test records run through (a) the original inference call and (b) the new wrapper must match — 100% agreement on predicted class, probabilities equal within a measured numerical tolerance. If parity fails, stop and diagnose; do not proceed on "close enough."
- **Provenance tracking.** Every text field used anywhere in this phase must record its source (original Bangla, OCR English, NLLB-translated, Qwen-generated description, etc.) — mixing untracked sources was explicitly flagged as a risk in the handoff doc.
- **Validation-only decisions**, same protocol as every phase before this one. Test has now been touched three times; no further touches planned as part of this phase.

## 5. Workflow, step by step

### Step 0 — Correct the 5-class-vs-3-class comparison (Section 3)
Collapse `outputs/final_test_predictions.json`'s per-example 5-class test probabilities into the 3-class scheme, compute macro-F1 on the same 196 records, and report the true delta (or lack of direct comparability) alongside the existing 0.5730 test number. Update the three documents named in Section 3.

### Step 1 — Audit (per the handoff doc's Section 20 questions)
Before writing new model code, confirm directly from the repository (not assumed):
- Exact Phase 2 checkpoint path and whether it's the ~4,897-record or augmented-dataset run (`DAPT_models/results/dapt_eval/BanglaBERT_fold5/checkpoint-735`, per every script in this project to date — confirm no other candidate checkpoint exists).
- Exact NLLB identifier actually used (verify from `translate_reasoning.py` / `run_translation.py`, not assumed to be 1.3B just because the handoff doc suggests it).
- Confirm the multimodal text branch's exact input construction (OCR text and/or Qwen reasoning, and join order) from `extract_text_embeddings_e3b.py`.
- **Already resolved by inspection, recorded here so it isn't re-litigated:** the fuzzy layer's suicide-membership vector is **hard vote fractions** across the 9 models (`majority_vote_fractions()` in `phase7_11_fuzzy_logic_final_test.py` — fraction of models voting for each class), **not** averaged softmax probabilities.

### Step 2 — Phase 2 depression classifier: parity-tested wrapper
Build `code/phase8_depression_wrapper.py`: one function, `predict_depression(text_bn)`, calling the exact existing `AutoModelForSequenceClassification` + `CKPT_PATH` + `TOKENIZER_PATH` combination already used in `fuzzy_logic_layer.py` and `phase7_11_fuzzy_logic_final_test.py` — reused, not reimplemented. Run a parity test against a fixed sample of original predictions before this wrapper is considered done.

### Step 3 — New text-only 3-class suicide-content head
`code/phase8_text_only_suicide_head.py`. Frozen DAPT-BanglaBERT (identical checkpoint, same as everywhere else in this project) + a new small classifier head, trained on translated OCR text only (the input that can actually exist at text-only deployment time — a Qwen-generated description requires an image, so it's excluded from this specific model's training input, per the handoff doc's Section 12.1 caution). 3-class harmonized target, 3 seeds, same ordinal-smoothing recipe used throughout this project. Validation-only model selection.

### Step 4 — Optional image-only fallback head
`code/phase8_image_only_fallback.py`, only attempted if Step 3 completes cleanly and time permits. Frozen SigLIP + a new small classifier, trained on the 582 FigSIM training images alone. Adopted only if it clears a meaningful bar above majority-class behavior on validation — treated as a low-confidence fallback, not a peer of the full ensemble, per the handoff doc's own caution about the image-only slice's weak historical evidence (n=5, macro-F1 0.100 in the original error analysis).

### Step 5 — Modality router
`code/phase8_modality_router.py`. Routes on **input availability, not predicted labels**:
```
text only              -> Step 2 (depression) + Step 3 (text-only suicide head)
image + usable OCR text -> existing 9-model ensemble (unchanged) + Step 2 (depression)
image + no usable text, Qwen description available -> generated-text route, flagged low-confidence
image + nothing usable  -> Step 4 fallback if adopted, else "insufficient evidence"
```

### Step 6 — Fuzzy-logic integration for new routes
Reuse the existing fuzzy combination mechanism and the already-derived 3-class rule table (`PHASE7_FUZZY_LOGIC_5CLASS_VS_3CLASS_PLAN.md`) for any route that produces both a depression and a suicide-content reading. Explicitly label this output as exploratory, not clinically validated, matching the existing framing already used throughout this project.

### Step 7 — Evaluation by route
Report metrics separately per route — text-only, image+OCR-text, image+generated-text, image-only fallback — never blended into one unexplained number, per the handoff doc's explicit caution against mixing native Bangla, machine-translated text, and AI-generated descriptions into a single metric.

### Step 8 — Documentation
Update `IMPROVEMENT_PLAN.md` (this phase's results), `THESIS_SECTIONS_DRAFT.md` (methodology and limitations), and `CURRENT_FINAL_SYSTEM_REPORT.md` (the system description and pipeline diagram) to reflect the router and the corrected 5-vs-3-class comparison.

## 6. Output contract

Every prediction from the completed router should return, not just a final answer:

```json
{
  "route": "text_only | image_text | image_generated_text | image_only_fallback",
  "depression": {"model": "phase2_best_immutable", "prediction": "...", "probabilities": [...]},
  "suicide_content": {"model": "text_3class | multimodal_9model | image_fallback", "prediction": "...", "probabilities": [...]},
  "combined_risk": {"status": "exploratory_not_clinically_validated", "prediction": "..."},
  "warnings": []
}
```

This makes the route and data provenance visible in every output, per the handoff doc's Section 13 — never silently collapsing multiple possible input situations into one opaque answer.

## 7. What this phase deliberately does not attempt

Per the handoff doc's Section 19: no move to a large multilingual vision-language backbone (Gemma 3 / Qwen-VL-scale continued pretraining) under the current ~973-example dataset — flagged there as high-risk future work requiring roughly 5,000+ consistently labeled pairs, not something to attempt under the current data constraints. This phase stays scoped to routing and fallback heads around the existing, proven system.
