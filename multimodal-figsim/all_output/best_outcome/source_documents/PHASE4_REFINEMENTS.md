> **Extracted from `IMPROVEMENT_PLAN.md`.**
>
> Phases 7, 8 and 9 each wrote their own planning document. Phases 1 to 6 did
> not: they were written directly into the project's chronological experiment
> log. This file carves out that phase's section so it can sit with the
> approach it belongs to.
>
> The text is unchanged. The complete log, with every phase in sequence, is in
> `../../project_documents/IMPROVEMENT_PLAN.md`.

---

## Phase 4 narrative log

Working baseline for Phase 4 comparisons: the Phase 3 cross-architecture
ensemble, validation macro-F1 **0.5592** (test 0.4984, already locked and
reported above -- Phase 4 iterates on validation only, same protocol as
every phase before it, and does not re-touch test until/unless a new
configuration is deliberately locked).

### 4.1 — Systematic hyperparameter sweep, E6 gated+orth head (DONE, kept)

Every classifier head in this entire project used one reasonable-guess
configuration (hidden_dim=256, dropout=0.2, lr=1e-3, weight_decay=1e-4),
never swept. One-at-a-time coordinate search from that default (8 configs,
not a full grid), 3 seeds each, majority-vote macro-F1 on validation:

| Config | Majority-vote macro-F1 | Delta vs. default |
|---|---|---|
| default (256, 0.2, 1e-3, 1e-4) | 0.5330 | — |
| hidden_dim=128 | 0.5193 | −0.0137 |
| hidden_dim=512 | 0.5138 | −0.0192 |
| dropout=0.1 | 0.5261 | −0.0069 |
| **dropout=0.4** | **0.5418** | **+0.0088** |
| lr=5e-4 | 0.5157 | −0.0173 |
| lr=2e-3 | 0.5312 | −0.0018 |
| weight_decay=1e-3 | 0.5347 | +0.0017 |
| dropout=0.4 + weight_decay=1e-3 (combined) | 0.5414 | +0.0084 (no better than dropout=0.4 alone) |

**Finding: the default was mildly under-regularized.** Both directions
that increased regularization (higher dropout, higher weight decay)
helped or were neutral; both directions that increased capacity (larger
hidden_dim) or increased the learning rate hurt. This is the expected
pattern for a small dataset (582 training examples) -- consistent with
everything else observed in this document (Phase 3.1's encoder-unfreezing
failure, the mixup failure) all pointing the same direction: this
specific model+data combination is much more sensitive to overfitting
than to underfitting, so err toward more regularization, not more
capacity, when in doubt.

**Decision: kept.** `dropout=0.4` adopted as the new default for the E6
gated+orth head (weight_decay left at 1e-4, since combining with the
weight_decay increase added nothing on top). This is a genuine, if modest,
gain (+0.0088) on the gated architecture specifically -- checked next
for whether it also improves the full cross-architecture ensemble, since
the gated model is one of its two components.

**Effect on the full ensemble:** re-ran the gated+orth component with
dropout=0.4 in place of the cross-architecture ensemble's gated half
(cross-attention component and combination method unchanged):

| Configuration | Majority-vote macro-F1 |
|---|---|
| Cross-architecture ensemble, untuned gated (Phase 3 locked) | 0.5592 |
| Cross-architecture ensemble, tuned gated (dropout=0.4) | **0.5638** |
| Delta | +0.0046 |

A modest further gain, and **the per-class picture is now the most
balanced of the entire project** -- every single class above 0.50 for
the first time: None 0.510, Wish to be dead 0.575, Suicide ideation
0.650, Suicide planning 0.533, Attempt/Death 0.551. "None" specifically
jumped from 0.449 to 0.510, suggesting the extra regularization reduced
overfitting on exactly the class the previous ensemble was weakest on.

**Decision: kept, promoted to new best.** Validation macro-F1
progression: 0.5117 → 0.5330 (Phase 1) → 0.5592 (Phase 3) →
**0.5638 (Phase 4.1)**. Cumulative gain from the original locked baseline:
**+0.0521 (~10.2% relative)**.

### 4.4 — Post-hoc per-class decision calibration (DONE, kept with a caveat)

Re-weighted the 6-model ensemble's vote fractions per class (not
retraining anything -- `vote_fraction[c] = models voting for c / 6`,
multiplied by a learned per-class weight before the final argmax) via
coordinate ascent: cycle through each class, try 7 candidate multipliers
(0.7-1.3), keep whichever most improves validation macro-F1, repeat until
no class improves further.

**Important methodological caveat, unlike every other item in this
document:** this search directly optimizes 5 free weight parameters
*against the exact validation macro-F1 being reported* -- a more
leakage-prone setup than a normal train/validation split, since it's
closer to fitting parameters to the metric itself than to learning a
generalizable pattern from data. Every other kept change in this
document (ensembling, ordinal smoothing, dropout tuning, the
cross-architecture combination) was validated in the normal train-then-
evaluate sense; this one was *searched* directly against the number being
reported. Flagging this plainly rather than presenting it as equally
trustworthy -- its true benefit is only confirmed if/when this
configuration is tested on the held-out test set.

**Result:** converged after a single pass (no further improvement on a
second pass), which is a mildly reassuring sign against overfitting the
search -- a badly overfit 5-parameter search against 195 validation
examples would more likely keep finding "improvements" round after
round rather than settling immediately.

| | Macro-F1 |
|---|---|
| Uncalibrated majority vote (Phase 4.1 best) | 0.5638 |
| Calibrated | **0.5695** |
| Delta | +0.0057 |

Learned weights: None ×0.7, Wish to be dead ×1.1, all others ×1.0 (unchanged).
**This is the opposite correction from the initial hypothesis** going into
this item (which expected "Wish to be dead" to need *down*-weighting,
since it's the class over-predicted as a fallback guess in every prior
experiment) -- the search found the reverse helped instead. Worth naming
explicitly: the intuitive hypothesis about which direction to correct was
wrong, and letting the data-driven search find the actual answer instead
of confirming the assumption was the right call here.

Per-class effect: None 0.510→0.524, Attempt/Death 0.551→0.571, Wish to be
dead 0.575→0.568 (slight give-back), Ideation and Planning unchanged.

**Decision: kept as a candidate, pending test-set confirmation.**
Cumulative validation macro-F1: 0.5117 → 0.5330 → 0.5592 → 0.5638 →
**0.5695 (+0.0578 / ~11.3% relative from the original baseline)** -- but
given the caveat above, this specific increment should be treated as
provisional until confirmed (or not) by the next deliberate one-time test
evaluation, unlike the increments before it.

### 4.5 — Hierarchical decomposition (DONE, NOT kept — clean, explained negative result)

Split the problem into two stages, using the same aligned representations
and tuned E6GatedOrth architecture (dropout=0.4) for both: **Stage 1**, a
binary classifier (None vs. any suicide content); **Stage 2**, a 4-way
classifier among the non-None severity levels, trained only on non-None
examples. At inference, Stage 2's prediction is only used where Stage 1
predicted "any suicide content"; Stage 1 predicting "None" is final.

| Stage | Result |
|---|---|
| Stage 1 alone (binary, majority vote of 3 seeds) | 0.7406 macro-F1 |
| Stage 2 alone (4-way, evaluated on the TRUE non-None subset -- an oracle-routing best case) | 0.561-0.572 macro-F1 |
| **Full pipeline (Stage 1 routes into Stage 2, real end-to-end)** | **0.5203 macro-F1** |
| Flat ensemble (Phase 4.4, for comparison) | 0.5695 |
| Delta vs. flat ensemble | **−0.0492** |

**Both stages individually looked promising -- Stage 1's binary
separability (0.74) is genuinely much higher than "None" ever scores as
one of 5 classes in any flat model, confirming the hypothesis that
motivated this item.** But the **full pipeline underperforms the flat
ensemble**, and the reason is the classic hierarchical-classifier failure
mode: **Stage 1's routing mistakes become unrecoverable errors for Stage
2.** Any item Stage 1 misroutes (a true non-None case predicted as "None,"
or vice versa) is wrong regardless of how good Stage 2 is on the cases it
does receive -- there's no way for the pipeline to hedge or reconsider
once Stage 1 has committed, unlike a flat joint classifier that can
distribute probability mass across all 5 classes simultaneously and let
softer, joint evidence resolve close calls. "Suicide planning" F1
specifically collapsed to 0.333 in the full pipeline (vs. 0.557 in the
flat ensemble) -- plausibly because borderline planning/ideation cases
that a flat model can weigh against evidence from the whole label space
get force-committed early and incorrectly once routed through a hard
Stage 1 boundary.

**Decision: not kept.** The individual-stage numbers were genuinely
encouraging and the underlying hypothesis (None is more separable) was
correct, but hard hierarchical routing is not is the right way to
exploit that in this data regime -- a soft version (e.g., using Stage 1's
predicted probability as an additional input FEATURE to a flat 5-way
classifier, rather than a hard gate deciding whether Stage 2 runs at all)
would avoid the error-cascading failure mode while still giving the model
access to the same separability signal, and remains a theoretically
available follow-up if revisited, though not attempted here given time.

### 4.2 — Direct VLM prompting (DONE, NOT kept — clear, explained negative result)

Prompted Qwen2.5-VL-7B directly to classify each meme's suicide-severity
into one of the 5 categories (zero-shot, temperature=0, image + OCR text
+ class definitions in the prompt), rather than using it to generate
reasoning text for a separately-trained classifier as the rest of this
project does. Run on all 195 validation images (test stays locked). 0
errors, 0 unparseable responses -- the model reliably followed the
requested output format.

| | Macro-F1 |
|---|---|
| Current best (trained ensemble, Phase 4.4) | 0.5695 |
| Direct VLM prompting (zero-shot) | **0.2912** |
| Delta | **−0.2783** |

**This is a large, decisive negative result, and the failure pattern is
clear and explainable, not random:**

| Class | True count | Predicted count |
|---|---|---|
| None | 28 | 51 (over-predicted) |
| Wish to be dead | 32 | **6** (severely under-predicted) |
| Suicide ideation | 65 | 102 (heavily over-predicted) |
| Suicide planning | 31 | 30 |
| Suicide attempt or death | 39 | **6** (severely under-predicted) |

The model collapses toward "Suicide ideation" (the vaguer middle
category) and "None" (the safest, no-risk category), while almost never
committing to "Wish to be dead" or "Suicide attempt or death" -- the two
categories requiring the most definitive, severe judgment. This reads as
classic LLM safety-alignment hedging: a 7B instruction-tuned model
reluctant to commit to an extreme severity judgment on sensitive content,
defaulting to a moderate or negative classification instead of a
confident severe one, rather than a comprehension failure (it correctly
identified "Suicide planning" at almost the right rate, 30 vs 31 true).

**This also meaningfully revises this document's own forecast.** Item 4.2
was flagged, before it was tried, as "the most likely to actually move
the needle" -- directly citing FigSIM's own paper finding that large
prompted models gave the biggest gains on this dataset. That citation
still stands, but it evidently does not transfer down to a **locally-run
7B open model** the way it was assumed to -- FigSIM's own large-model
results were almost certainly obtained with frontier-scale models
(GPT-4V/Gemini-class), a different resource tier than what is available
in this project. The specialized, carefully-tuned, ensembled pipeline
built over Phases 1-4 (0.5695) substantially outperforms naive direct
prompting of the only large multimodal model actually available here.

**Decision: not kept.** Few-shot prompting (providing labeled examples in
the prompt) was considered as a natural next step but not attempted --
given the failure mode is about the model's alignment-driven reluctance to
commit to severe categories rather than a lack of task understanding, it
is not obvious that showing examples would fix a hedging behavior baked
in during the model's own safety training, though it remains a
theoretically available follow-up if revisited.

**4.3 (ensembling the trained classifier with a directly-prompted model)
is no longer worth pursuing as originally planned** -- it was premised on
4.2 producing a reasonably competent second opinion whose errors might be
uncorrelated with the trained ensemble's. At 0.29 macro-F1 with a strong,
systematic bias against two entire classes, 4.2 is not a reasonable
second opinion to combine with; it would very likely drag the combined
result down; not attempted given this.

---
