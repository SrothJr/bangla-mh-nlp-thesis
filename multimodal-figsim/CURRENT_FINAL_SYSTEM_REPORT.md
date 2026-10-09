# Current Final System — Full Report

**What this document is:** a single, self-contained, plain-language walkthrough of the system as it stands right now — every moving part, how they connect, why each decision was made, and every result, backed by figures. Unlike `IMPROVEMENT_PLAN.md` (a chronological experiment log of everything tried, including the failures) or `THESIS_SECTIONS_DRAFT.md` (formal academic prose), this document exists purely to explain: *here is the finished thing, here is how it works, here is what it achieves.*

All figures referenced below are copied into `final_report_figures/` alongside this document, so this folder is self-contained — you don't need to go hunting through `figures/` to follow along.

---

## 1. What the system does, in one paragraph

Given a Bangla meme (an image with embedded text) or a piece of Bangla text on its own, the system estimates two related but distinct things — how severe the *suicide-related content* in it is, and separately, how severe the *depressive tone* of the text is — and then combines both readings into a single, final risk level (Minimal / Low / Elevated / Critical) using an explicit, inspectable rule-based combination layer. It is a **screening-support research system**, not a diagnostic tool: every output describes *content expressed*, never a clinical judgment about a real person's actual state or intent.

---

## 2. The two source models, and why there are two

There are genuinely two separate models feeding into this system, each doing a different job, each with a different history:

| | Depression classifier | Suicide-severity model |
|---|---|---|
| Input | Text only | Text + image (multimodal) |
| Dataset | Original Bangla depression dataset (~4,897 records) | FigSIM leakage-safe meme dataset (973 records) |
| Output | Minimum / Mild / Moderate / Severe (4 classes) | 3-class harmonized suicide severity |
| Status | Pre-existing, from an earlier phase of this project, **never modified** | Built, tuned, and locked across this entire project |
| Reported performance | ~85%+ (its own original evaluation) | **0.5730 macro-F1, test-confirmed** |

**Why keep them separate instead of merging into one model?** Depression severity and suicide-risk severity are related but genuinely different clinical dimensions — someone can be severely depressed with no suicidal ideation at all, or express active suicidal planning during an acute crisis without a broader depressive pattern. Collapsing them onto one shared scale would assert a clinical equivalence that has never been validated by any annotator. So instead, both stay as independent readings, combined later through an explicit rule table (Section 5) rather than a merged label space — a deliberate design decision, discussed and confirmed during this project (recorded as "Plan B, deferred" in `IMPROVEMENT_PLAN.md`).

---

## 3. How text and image actually get combined for the suicide-severity model

This is the part worth being precise about, since "combining modalities" can mean different things.

### 3.1 The two frozen encoders

- **DAPT-BanglaBERT** — a Bangla language model that was further pretrained on ~250,000 mental-health-adjacent Reddit posts (domain-adaptive pretraining, DAPT) before this project began. It reads Bangla text and outputs a 768-dimensional vector describing it. It **never sees images** — it has no concept of them.
- **SigLIP** (so400m variant) — a general-purpose image model. It reads the meme image and outputs a 1152-dimensional vector describing it. It **never sees text.**

Both are used entirely **frozen** — their weights are never updated anywhere in this system. This was a deliberate, evidence-backed choice: every attempt to update them (full unfreezing, LoRA fine-tuning on external data) either hurt results or, at best, didn't clearly help once tested properly (see `IMPROVEMENT_PLAN.md` Phases 3, 6, and 7 for the specific experiments). With only 582-777 labeled training examples, there simply isn't enough data to safely adapt large pretrained models further — keeping them frozen and only training small connector modules on top turned out to be the right call, confirmed repeatedly rather than assumed.

### 3.2 Where the actual combination happens

Neither encoder alone ever "understands" both text and image — that happens in a **third, separate, small module** that sits after both, trained specifically to combine their two outputs. Three different combination styles were built and are all used together:

1. **Simple concatenation** — literally join the 768-dim text vector and 1152-dim image vector into one 1920-dim vector, feed through a small classifier (linear layer → ReLU → dropout → linear layer).
2. **Gated fusion** — a small learned "gate" computes a weight (alpha) per example, deciding how much to trust text vs. image for that specific meme, then blends them: `alpha·text + (1-alpha)·image`. This also includes an orthogonal-residual decomposition and operates on a separately-aligned version of both vectors (Section 3.3).
3. **Cross-attention** — instead of one pooled vector per modality, this lets individual text *tokens* attend over individual image *patches*, a much finer-grained way of asking "which part of the image is this word talking about."

**None of these three is used alone.** All three are trained (3 random seeds each = 9 models total), and their individual predictions are combined by **majority vote** — whichever class most of the 9 models agree on wins. This was not an arbitrary choice: it directly re-applies the single biggest lever discovered earlier in this project (ensembling different architectures beats any one of them alone, because they make different, partially-independent mistakes) — see figure `09_phase3_architecture.png` for where this was first discovered, and figure `19_end_to_end_pipeline.png` for how it's wired together in the final system.

### 3.3 Why there's an extra "alignment" step for the gated architecture

DAPT-BanglaBERT and SigLIP were trained completely independently — nothing forces their output spaces to be geometrically comparable just because a given text and image describe the same meme. Early in this project, this actually broke the gated-fusion architecture: its gate collapsed, learning to ignore text and lean on the image alone for 71% of examples (figure `03_gate_alpha_fix.png`). The fix — called "Stage A alignment" — trains two small projection layers (mapping both modalities into a shared 256-dim space) using a self-supervised contrastive objective, so that a meme's own text and image land close together in that shared space. This fixed the collapse and is still part of the gated architecture today. (A later attempt to improve this by anchoring the shared space to DAPT's own native output space instead of a freshly-learned one was tried and found to work worse — see `IMPROVEMENT_PLAN.md` item 7.7 — so the original fix remains in place.)

---

## 4. The label scheme: why 3 classes instead of 5

FigSIM's memes were originally annotated on a 5-level suicide-severity scale: *None, Wish to be dead, Suicide ideation, Suicide planning, Suicide attempt or death.* Partway through this project, these were **harmonized into 3 coarser classes**, after testing showed this was a genuine improvement, not just a simplification:

| 3-class label | Combines original label(s) |
|---|---|
| **No expressed severity** | None |
| **Suicidal thought or desire** | Wish to be dead + Suicide ideation |
| **High acuity suicidal content** | Suicide planning + Suicide attempt or death |

This is now the primary target the whole multimodal pipeline is built around. See figure `14_label_harmonization.png` for the evidence this is a real improvement (not just an artifact of having fewer classes to get right), and `IMPROVEMENT_PLAN.md` Phase 7 Steps 1 and 4 for the full derivation and testing.

---

## 5. The fuzzy-logic risk combination layer — full explanation

This is the final step: taking the suicide-severity reading and the depression-severity reading for the *same* piece of text, and combining them into one overall risk level.

### 5.1 Why "fuzzy logic" specifically

Instead of hard rules like "if class X, then risk Y," the system uses each classifier's own **probability distribution** (not just its single top prediction) as a continuous "membership degree" — how strongly does this meme belong to each severity category, not just which one it's most likely to be. This lets borderline, uncertain cases be handled more gracefully than a hard rule ever could.

- **Fuzzy AND** = take the **minimum** of two membership degrees (both conditions must hold, so the weaker one caps the combination).
- **Fuzzy OR** = take the **maximum** of two membership degrees (either condition satisfies, so the stronger one wins).
- **"Defuzzification"** (turning the fuzzy result back into one crisp answer) = simply take the risk level with the highest resulting membership after all rules have fired.

### 5.2 The rule table actually used (3-class version)

Each rule reads as: *if the suicide-severity reading is X, AND the depression-severity reading is in group Y, THEN the risk level is Z.*

| Suicide severity | Depression severity | → Risk level |
|---|---|---|
| High acuity suicidal content | any level | **Critical** |
| Suicidal thought or desire | Mild, Moderate, or Severe | **Elevated** |
| Suicidal thought or desire | Minimum | Low |
| No expressed severity | Severe | Elevated |
| No expressed severity | Moderate | Low |
| No expressed severity | Minimum or Mild | Minimal |

This table was derived from an original 11-row table built for the 5-class scheme, using a documented **"cautious merge" policy**: wherever two original rows got merged into one 3-class bucket, the *more severe* of the two original risk levels was kept, at each depression level. The reasoning: in a mental-health screening context, it's safer to over-flag risk than under-flag it when uncertain. Full derivation with worked tables is in `PHASE7_FUZZY_LOGIC_5CLASS_VS_3CLASS_PLAN.md`.

### 5.3 Fuzzy logic results — what the final system actually outputs

Run on the same 196 test memes used for the locked multimodal evaluation, using the actual locked 9-model ensemble's predictions and the same untouched depression classifier:

| Risk level | Count | Percentage |
|---|---|---|
| Minimal | 45 | 23.0% |
| Low | 49 | 25.0% |
| Elevated | 49 | 25.0% |
| Critical | 53 | 27.0% |

See figure `18_final_fuzzy_test_comparison.png` for how this compares to the original 5-class version of the system (which concentrated much more heavily in the Low/Elevated middle, 63.8% combined, vs. this new system's much more even spread across all four levels).

**Important, stated honestly:** there is no ground-truth "correct risk level" label for any meme to check this against — this is a distributional, qualitative result, not an accuracy figure. The fuzzy-logic layer's job is to combine two model outputs sensibly, not to be independently validated against ground truth (no such ground truth exists for this specific combined judgment).

---

## 6. Final locked results — the real, test-confirmed numbers

### 6.1 Suicide-severity model (the headline multimodal result)

Evaluated on FigSIM's held-out test split (196 memes never used for training or any tuning decision) — the **third and, so far, final deliberate touch** of this test set in the whole project's history.

| Metric | Value |
|---|---|
| **Macro-F1** | **0.5730** |
| Weighted-F1 | 0.6126 |
| Accuracy | 0.6071 |
| Balanced accuracy | 0.5871 |
| Quadratic weighted kappa | 0.3765 |

Per-class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No expressed severity | 0.391 | 0.563 | 0.462 | 32 |
| Suicidal thought or desire | 0.701 | 0.667 | 0.683 | 102 |
| High acuity suicidal content | 0.623 | 0.532 | 0.574 | 62 |

This is a real, verified **+0.0746 macro-F1 improvement** over the original locked model from earlier in the project (5-class, 0.4984 test macro-F1). See figure `17_final_3class_test_lock.png`.

**One thing stated plainly, not smoothed over:** this configuration showed 0.6649 on validation before being locked, and 0.5730 on test — a larger gap than typical for this project, attributable to a long sequence of decisions all made against the same 195-item validation set (see `IMPROVEMENT_PLAN.md` item 7.10 for the full honest accounting). 0.5730 is the number that reflects genuine, unbiased performance — that's the number this report and the thesis draft both lead with.

### 6.2 Depression classifier

Reported informally as above 85% on its own original evaluation set (the depression dataset it was trained and tested on). **Not independently re-verified on FigSIM's specific translated meme text** — flagged explicitly as an open limitation (Section 7).

### 6.3 The whole journey, in one number

Cumulative improvement across every kept step of this entire project, from the very first locked baseline to the current final system:

**0.4635 → 0.4984 → 0.5730** (macro-F1, all test-confirmed)

— an overall **+0.1095** improvement, achieved through (in order): multi-seed ensembling, ordinal label smoothing, cross-architecture ensembling, hyperparameter tuning, post-hoc calibration exploration, label harmonization, and a second round of cross-architecture ensembling on the harmonized target. See figure `11_cumulative_progression.png` for the earlier portion of this trajectory and figure `17_final_3class_test_lock.png` for where it currently stands.

---

## 7. What this system does *not* yet do (honest, known gaps)

Two real limitations, not hidden:

1. **No graceful handling of a missing modality.** All three fusion architectures assume both text and image are always present. Feed the multimodal side text-only or image-only right now, and there's no tested, defined behavior — it was never built, because every architecture comparison in this project needed a clean baseline with both modalities always available. A real fix (modality-dropout training, or routing to the project's earlier text-only baselines) remains future work.
2. **The depression classifier's accuracy specifically on FigSIM's translated meme text is unverified.** Its ~85%+ figure comes from its own, different evaluation set. This project has twice independently confirmed that domain shift causes real, measurable performance loss in closely analogous situations (a generic text-encoder swap cost 0.073 macro-F1; external image pretraining failed to transfer) — so this is a named, plausible risk, not a settled fact either way.

Full discussion of both is in `THESIS_SECTIONS_DRAFT.md` Section 3 (Limitations).

---

## 8. Figure index

All figures below are in `final_report_figures/` alongside this document.

| Figure | What it shows |
|---|---|
| `01_ablation_ladder.png` – `06_final_test_summary.png` | Original Day 1-7 baseline build-up, ending in the first locked model |
| `07_phase1_experiments.png` – `10_phase4_refinements.png` | The improvement round's loss-function, architecture, and refinement experiments |
| `11_cumulative_progression.png` | The full validation-score trajectory across every kept improvement |
| `12_hierarchical_diagnosis.png` | Why a two-stage hard classifier failed (error cascading) |
| `13_phase6_lora.png` | LoRA fine-tuning succeeding where full unfreezing failed |
| `14_label_harmonization.png` | Evidence the 3-class scheme is a genuine improvement |
| `15_bnhib_pretraining_transfer_test.png` | External meme pretraining: real signal on its own task, didn't transfer |
| `16_fuzzy_logic_5class_vs_3class.png` | Early validation-based fuzzy-logic comparison |
| `17_final_3class_test_lock.png` | **The headline result** — locked test score vs. prior best, and the validation/test gap |
| `18_final_fuzzy_test_comparison.png` | **Final system risk output** — old vs. new, on actual test data |
| `19_end_to_end_pipeline.png` | **The full system architecture diagram** — start here for the big picture |
