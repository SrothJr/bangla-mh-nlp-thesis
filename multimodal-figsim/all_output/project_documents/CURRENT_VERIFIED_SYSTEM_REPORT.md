# Current Verified System Report — Full Detail, Post Phase 8 Correction

**What this document is:** the complete, up-to-date picture of the system as it stands right now, including everything that changed after `CURRENT_FINAL_SYSTEM_REPORT.md` was written — the invalid-comparison correction, and every Phase 8 verification result. This document supersedes that one for accuracy; nothing here is left out or simplified away. Read this one if you want the full, current, honest picture in one place.

**The one-paragraph version:** this project built a multimodal system for Bangla mental-health and suicide-content screening research. Its two components — a text-only depression-severity classifier (Phase 2, ~93% verified accuracy on its own domain) and a multimodal suicide-content-severity classifier (built and refined across this whole project) — are combined through an explicit, inspectable fuzzy-logic rule layer. A significant self-correction was made partway through: a claimed "+0.0746 improvement" from a later relabeling of the suicide-content task turned out, on rigorous re-checking, to not be a real improvement at all. Both configurations are now reported honestly, side by side. Separately, the depression classifier's text-only behavior was formally proven correct through a parity test and verified against its full original labeled dataset.

---

## 1. Full workflow diagram

See `figures/20_current_verified_pipeline.png` (also copied to `final_report_figures/` and `supervisor_result/final_report_figures/` if those folders are kept in sync — check timestamps if unsure). This diagram reflects the system exactly as it stands today: both severity models shown honestly side by side (not one "beating" the other), and the depression classifier's text-only route marked as verified.

Plain-text summary of the same flow:

```
Bangla meme (text + image) OR text alone
        │
        ├──→ DAPT-BanglaBERT (text embedding)
        │         │
        │         ├──→ [multimodal ensemble, needs image too]
        │         │
        │         └──→ [depression classifier -- TEXT-ONLY, VERIFIED]
        │
        └──→ SigLIP (image embedding, only if image present)
                  │
                  └──→ [multimodal ensemble]

multimodal ensemble  → suicide severity (TWO locked models, reported separately)
depression classifier → depression severity (4-class)
                  │
                  ▼
        fuzzy-logic risk combination
                  │
                  ▼
     final risk level (Minimal/Low/Elevated/Critical)
```

---

## 2. The two source models

### 2.1 Depression classifier (Phase 2) — preserved, now formally verified

- **Input:** Bangla text only.
- **Model:** DAPT-BanglaBERT (`DAPT_models/results/dapt_eval/BanglaBERT_fold5/checkpoint-735`) + its own original 4-class classification head.
- **Status:** never retrained, never modified, at any point in this entire project, including everything done in Phase 8.
- **Confirmed via checkpoint metadata (Phase 8 Step 1):** trained on the **original ~4,897-record dataset** (not the 8,000-9,000-record augmented version) — verified by back-calculating training steps (735 steps × batch size 16 over 3 epochs ≈ 3,920 examples/epoch, matching a ~4/5 fold of 4,897 almost exactly; the augmented dataset would predict roughly 1,275 steps, not 735).
- **Parity-tested (Phase 8 Step 2):** a wrapper reusing the exact existing inference code was checked against 196 already-saved depression predictions from the original Days 9-10 run. **Result: 100.00% label agreement, 0.00 maximum probability difference — an exact match**, not an approximation.
- **Verified against its own full labeled dataset (Phase 8 Step 2b):** run against `dataset.xlsx` (4,897 records, user-supplied). **Result: macro-F1 0.9182, accuracy 93.2%.** Per-class: Minimum 0.963 F1, Mild 0.924 F1, Moderate 0.832 F1 (weakest, confused mostly with adjacent Mild), Severe 0.954 F1.
  - **Caveat, stated plainly:** this checkpoint was trained on ~4/5 of this exact dataset, so this number is a **behavior sanity check against its own training domain**, not an unbiased held-out estimate. It confirms the model works correctly and gives no unusual behavior — it should not be quoted as a fresh, independent accuracy figure.

### 2.2 Suicide-content severity model — two locked configurations, both real, neither superseding the other

This project built and locked **two separate configurations** at different points, on different label granularities:

| | 5-class model | 3-class model |
|---|---|---|
| Classes | None / Wish to be dead / Suicide ideation / Suicide planning / Suicide attempt or death | No expressed severity / Suicidal thought or desire / High-acuity suicidal content |
| Architecture | 6-model cross-architecture ensemble (gated + cross-attention, 3 seeds each) | 9-model cross-architecture ensemble (concat + gated + cross-attention, 3 seeds each) |
| Test macro-F1 | **0.4984** | **0.5730** |
| Test weighted-F1 | 0.5232 | 0.6126 |
| Test accuracy | 0.5306 | 0.6071 |
| Test QWK | 0.3857 | 0.3765 |
| Status | This project's **strongest validated result** | A real result on its own, differently-scoped task |

**These two numbers cannot be validly compared by direct subtraction** — they measure different-granularity tasks. This was originally reported incorrectly (see Section 3).

---

## 3. The correction (Phase 8, Step 0) — in full, nothing omitted

**What was originally claimed:** "+0.0746 macro-F1 improvement" from the 3-class model (0.5730) over the 5-class model (0.4984), and a chained trajectory "0.4635 → 0.4984 → 0.5730" presented as one continuous line of improvement.

**Why this is invalid:** 0.5730 and 0.4984 are scores on tasks with different numbers of classes. Subtracting a 3-class score from a 5-class score does not measure whether one model is actually better — a coarser task is inherently easier to score well on, independent of whether the underlying model actually improved.

**The valid check:** collapse the older model's own saved predictions into the same 3-class groupings it would have to use (no retraining involved — pure aggregation, exactly the method already established and validated earlier in this project for a similar purpose), and compare on the identical 196 test records.

| Comparison | Old model's predictions, collapsed to 3-class (no retraining) | New 3-class model | Valid delta |
|---|---|---|---|
| vs. the original single-model Day-7 lock (weaker baseline) | 0.5443 | 0.5730 | **+0.0287** |
| **vs. the actual prior-best configuration (the 5-class ensemble)** | **0.6024** | 0.5730 | **−0.0294** |

**The finding: against this project's actual strongest baseline, the 3-class model is not an improvement — it is slightly weaker, once fairly compared.** The earlier framing was not just overstated; it compared the wrong things in a way that produced an incorrect conclusion.

**What changed as a result:**
- The 5-class ensemble (0.4984) is reinstated as this project's strongest validated result.
- The 3-class model (0.5730) is retained and reported as a legitimate, real result — for its own task — not as an upgrade.
- Every document that previously claimed the invalid comparison (`IMPROVEMENT_PLAN.md`, `THESIS_SECTIONS_DRAFT.md`, `CURRENT_FINAL_SYSTEM_REPORT.md`, and their `supervisor_result/` copies) has been corrected, with the original claim kept visible alongside an explicit correction note — not silently rewritten, per this project's established practice.

Full computation: `code/phase8_0_valid_5to3_comparison.py`, `outputs/phase8_0_valid_5to3_test_comparison_results.json`.

---

## 4. How text and image combine (unchanged from before — still accurate)

Two frozen encoders (DAPT-BanglaBERT for text, SigLIP for image) never see each other's modality directly. Three separately-trained fusion architectures combine their outputs:

1. **Simple concatenation** — join the two vectors, feed a small classifier.
2. **Gated fusion** — a learned per-example weight blends aligned text and image vectors, plus an orthogonal-residual term preserving non-redundant image information. Requires a contrastive alignment step (Stage A) to keep the two independently-pretrained encoder spaces geometrically comparable — without it, this architecture's gate collapses to ignoring text almost entirely (a real failure diagnosed early in this project).
3. **Cross-attention** — individual text tokens attend over individual image patches, a finer-grained combination than one pooled vector per modality.

All three (3 seeds each) are combined by majority vote. No single architecture is used alone in either locked configuration — the combination itself, not any one fusion mechanism, is what each locked model actually is.

---

## 5. The fuzzy-logic risk combination layer — full explanation

### 5.1 Mechanism

Each classifier's own probability/vote distribution (not just its top prediction) is treated as a "membership degree" for each severity category. Fuzzy AND = minimum of two memberships; fuzzy OR = maximum; the final risk level is whichever category ends up with the highest combined membership after all rules are checked.

**Confirmed directly from code (Phase 8 Step 1), not assumed:** the suicide-severity membership vector fed into this layer is **hard vote fractions** (what fraction of the ensemble's models voted for each class), not averaged softmax probabilities. The depression side uses genuine softmax probabilities from the single depression classifier.

### 5.2 Two rule tables, one for each severity scheme

**5-class rule table (11 rows, the original, Days 9-10):**

| Suicide class | Depression class(es) | Risk |
|---|---|---|
| Attempt or death | any | Critical |
| Planning | Moderate/Severe | Critical |
| Planning | Minimum/Mild | Elevated |
| Ideation | Moderate/Severe | Elevated |
| Ideation | Mild | Elevated |
| Ideation | Minimum | Low |
| Wish to be dead | Moderate/Severe | Elevated |
| Wish to be dead | Minimum/Mild | Low |
| None | Severe | Elevated |
| None | Moderate | Low |
| None | Minimum/Mild | Minimal |

**3-class rule table (6 rows, derived via a documented "cautious merge" — for merged buckets, take the more severe of the original rules being combined at each depression level):**

| Suicide class | Depression class(es) | Risk |
|---|---|---|
| High-acuity suicidal content | any | Critical |
| Suicidal thought or desire | Mild/Moderate/Severe | Elevated |
| Suicidal thought or desire | Minimum | Low |
| No expressed severity | Severe | Elevated |
| No expressed severity | Moderate | Low |
| No expressed severity | Minimum/Mild | Minimal |

### 5.3 Results — final risk output, on the actual test set, both schemes

| Risk level | 5-class system (original) | 3-class system (harmonized) |
|---|---|---|
| Minimal | 26 (13.3%) | 45 (23.0%) |
| Low | 64 (32.7%) | 49 (25.0%) |
| Elevated | 61 (31.1%) | 49 (25.0%) |
| Critical | 45 (23.0%) | 53 (27.0%) |

Per-meme agreement between the two systems on the same 196 test memes: **69.2%** (on an earlier validation-based comparison) — the two schemes produce meaningfully different final risk distributions, which is expected given they're built on different-granularity severity readings.

**No ground-truth "risk level" label exists for any meme** — this remains a distributional/qualitative comparison, not an accuracy comparison, for both schemes.

---

## 6. Phase 8 progress — everything done so far, in full

| Step | What | Result |
|---|---|---|
| 0 | Correct the invalid 5-vs-3-class comparison | Done — see Section 3 |
| 1 | Audit: checkpoint, dataset, NLLB version, fuzzy aggregation method | Done — NLLB corrected to **3.3B** (not 1.3B as the triggering handoff document assumed); DAPT checkpoint confirmed trained on the **original** dataset; fuzzy aggregation confirmed as hard vote fractions |
| 2 | Parity-tested depression classifier wrapper | Done — **100.00% exact match** against 196 known outputs |
| 2b | Verify depression classifier against full labeled dataset | Done — **macro-F1 0.9182, accuracy 93.2%** (with training-overlap caveat) |
| 3 | New text-only 3-class suicide-content head | Not yet started |
| 4 | Optional image-only fallback head | Not yet started |
| 5 | Modality-aware router | Not yet started |
| 6 | Fuzzy-logic integration for new routes | Not yet started |
| 7 | Evaluation by route | Not yet started |
| 8 | Documentation | Ongoing (this document is part of it) |

**What Phase 8 has proven so far:** the depression classifier can be trusted to handle text-only input reliably and correctly. **What Phase 8 has not yet built:** an equivalent text-only (or image-only) path for the *suicide-content* severity side — that still requires both modalities, with no defined fallback yet. This is the next real piece of work.

---

## 7. Known limitations, complete list

1. **No defined behavior for a missing modality on the suicide-severity side.** All three fusion architectures assume both text and image are present. (The depression side no longer has this limitation — Phase 8 fixed it there.)
2. **The depression classifier's ~93% verification figure is inflated by training-data overlap** — stated explicitly, not a fresh unbiased estimate.
3. **The two suicide-severity models should never be compared by direct macro-F1 subtraction** — this was the error corrected in Section 3, and the lesson generalizes: any future label-scheme change must be checked the same way before claiming an improvement.
4. **The fuzzy-logic layer is exploratory and rule-based**, not clinically validated, and has no ground-truth combined-risk label to check against, for either severity scheme.
5. **This project's strongest validated result is the 5-class ensemble (0.4984)** — this should be the number led with in any formal reporting, not the 3-class figure, unless explicitly discussing the harmonized task on its own terms.

---

## 8. Where to find everything referenced here

- Correction computation: `code/phase8_0_valid_5to3_comparison.py`, `outputs/phase8_0_valid_5to3_test_comparison_results.json`
- Parity wrapper: `code/phase8_2_depression_parity_wrapper.py`, `outputs/phase8_2_depression_parity_report.json`
- Dataset verification: `code/phase8_3_depression_dataset_verification.py`, `outputs/phase8_3_depression_dataset_verification_results.json`
- Full phase plan: `PHASE8_MODALITY_AWARE_SYSTEM_PLAN.md`
- Triggering document: `CLAUDE_HANDOFF_MODALITY_AWARE_MULTIMODAL_SYSTEM.md`
- Complete chronological experiment log, including this correction: `IMPROVEMENT_PLAN.md`
- Formal thesis prose, corrected: `THESIS_SECTIONS_DRAFT.md`
- Workflow diagram: `figures/20_current_verified_pipeline.png`
