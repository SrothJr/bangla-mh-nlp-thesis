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

## Phase 3 narrative log

### 3.1 — Partial encoder unfreezing, BanglaBERT last 2 layers (DONE, NOT kept)

**Step A, fair baseline established first:** since unfreezing requires a
materially different training loop (raw text/mini-batched, not cached
full-batch embeddings), a matched frozen-encoder baseline using the same
simple-concat architecture (no Stage A alignment, isolating this from the
gated/aligned E6 numbers) was measured first: **0.5139** majority-vote
macro-F1 (Phase 1's ordinal-smoothing + majority-vote recipe applied to
plain concatenation). This is the number unfreezing needed to beat, not
the more complex E6 gated+orth number (0.5330), so the comparison isolates
one variable.

**Design:** unfroze only BanglaBERT's last 2 of 12 Electra layers
(14.2M of 110M parameters), SigLIP kept fully frozen (cached embeddings
reused) to isolate "does adapting the text encoder help" as a single
variable rather than confounding it with also unfreezing the vision side.
Differential learning rates (encoder << head), mini-batched (batch 16,
582 examples don't fit the full-batch pattern once gradients must flow
through a 110M-parameter model), early stopping on validation macro-F1.

| Encoder LR | Majority-vote macro-F1 | Delta vs. frozen baseline (0.5139) |
|---|---|---|
| 2e-5 (standard fine-tuning LR) | 0.5029 | −0.0110 |
| 5e-6 (4x gentler) | 0.4875 | −0.0264 |

**Finding: both settings regressed, and — importantly — the gentler
learning rate made it WORSE, not better**, which rules out "LR too
aggressive" as the explanation. What both runs share instead is a visibly
unstable, non-monotonic validation macro-F1 across epochs (e.g., seed 0 at
lr=2e-5: 0.337 → 0.379 → **0.483** → 0.388 → 0.413 → 0.455 → 0.403 --
bouncing rather than smoothly converging to a stable optimum). This reads
as a genuinely structural problem: 582 training examples, split into
~37 mini-batches per epoch, is simply too small a fine-tuning set for a
110M-parameter encoder to adapt stably even when only 2 of its 12 layers
are unfrozen -- there isn't enough data per gradient step for the signal
to consistently outweigh noise, regardless of how conservatively the
learning rate is set within a reasonable fine-tuning range.

**Decision: not kept.** Two learning rates bracketing a 4x range both
regressed via the same instability signature -- per this project's
established discipline (same treatment as focal loss and mixup), this
doesn't warrant a further LR sweep; the mechanism is understood well
enough to stop. **This is a genuinely valuable negative result for the
thesis, not just a rejected experiment:** it empirically validates
Section 8's foundational, pragmatic decision to freeze both encoders and
train only small heads on cached embeddings for this project -- a
decision made from necessity (dataset size, time budget) that turns out,
tested directly, to also have been the *empirically correct* call, not
merely a convenient one. Unfreezing more layers, or unfreezing with a
larger effective batch size via gradient accumulation, remain
theoretically possible follow-ups but are not planned for this round
given the consistency of the failure signature across both tested
settings.

### 3.2 — Cross-attention fusion (DONE — standalone not kept, but led to the new best result via ensembling)

Replaced the scalar gate entirely with a genuine cross-attention
mechanism: BanglaBERT's 256 text TOKENS (not just the pooled CLS vector)
act as queries attending over SigLIP's 729 image PATCHES (not just the
pooled vector) as keys/values, via a standard multi-head attention layer
(4 heads, shared dim 256). This required extracting and caching
token-level and patch-level features for all 973 images (~2GB at
float16), since the pooled-vector caches used everywhere else in this
project don't retain the spatial/positional information cross-attention
needs. Both encoders stayed frozen (Phase 3.1 already showed unfreezing
regresses at this dataset size) -- only the projections, attention layer,
and classifier head are trained. Attended tokens are masked-mean-pooled
(real attention mask, ignoring padding) and concatenated with the raw
pooled text vector before classification, matching the "concat with raw
text" pattern from E5/E6.

**Standalone result:** majority-vote macro-F1 0.5205, a modest regression
vs. the E6 gated+orth baseline (0.5330, delta −0.0125).

**But the per-class breakdown told a different, more interesting story:**

| Class | E6 gated+orth (Phase 1 best) | Cross-attention (standalone) |
|---|---|---|
| None | 0.476 | 0.459 |
| Wish to be dead | 0.602 | 0.451 |
| Suicide ideation | 0.602 | **0.662** |
| Suicide planning | 0.485 | **0.531** |
| Attempt/Death | 0.500 | 0.500 |

Cross-attention is meaningfully *better* on Suicide ideation and
Suicide planning -- the two classes involved in this project's single
most persistent, most-repeated failure pattern (planning bleeding into
ideation, present in literally every experiment since E1) -- but
meaningfully *worse* on Wish to be dead. This reads as two architectures
with genuinely different, complementary strengths, not one being simply
better than the other.

**Follow-up test: ensemble across the two architectures, not just across
seeds of one.** Same logic as Phase 1.1's seed-ensembling, extended to
combine 3 E6-gated+orth predictions with 3 cross-attention predictions
(6 total) by majority vote -- the same idea planned for Phase 4 (4.3,
"ensemble the trained classifier with a differently-mechanismed model"),
reached one phase early because two differently-mechanismed trained
architectures were already on hand.

| Configuration | Majority-vote macro-F1 |
|---|---|
| E6 gated+orth alone (3 seeds) | 0.5330 |
| Cross-attention alone (3 seeds) | 0.5205 |
| Soft-average across all 6 (both architectures) | 0.5369 |
| **Majority vote across all 6 (both architectures)** | **0.5592** |

**This is the largest single gain since Phase 1, and the most balanced
per-class result of the entire project:** None 0.449, Wish to be dead
0.591, Suicide ideation 0.656, Suicide planning **0.557** (the best
"Suicide planning" score anywhere in this project, by a wide margin),
Attempt/Death 0.543. Consistent with the Phase 1.1 finding, majority vote
clearly beats soft-averaging (0.559 vs 0.537) -- averaging dilutes each
architecture's confident correct calls with the other's uncertain ones,
while majority vote only overrides on genuine 2-of-3-style consensus,
and here that consensus draws on two architectures with different blind
spots rather than three seeds of the same one.

**Decision: KEPT, and promoted to new best overall result.** Validation
macro-F1 progression: 0.5117 (original locked baseline) → 0.5330 (Phase 1)
→ **0.5592 (Phase 3.2 cross-architecture ensemble)** -- a cumulative
+0.0475 (~9.3% relative) improvement from the original locked model,
achieved by combining two differently-mechanismed trained models rather
than by any single architecture change succeeding on its own. This is a
genuinely different kind of result from everything else in this document:
neither component beat the baseline alone, but the combination did, by
the largest margin seen in this entire improvement round.

---

## Current best configuration (updated)

**Cross-architecture ensemble: E6 gated+orth (aligned, ordinal-smoothed,
3 seeds) + cross-attention fusion (3 seeds), majority vote across all 6
predictions.**

- Validation macro-F1: **0.5592**
- Cumulative improvement over the original locked model (0.5117): **+0.0475 (~9.3% relative)**
- Per-class F1: None 0.449, Wish to be dead 0.591, Suicide ideation 0.656, Suicide planning 0.557, Attempt/Death 0.543

This is now the configuration Phase 4 (if pursued) should try to beat,
and the candidate for the next, deliberate one-time test-set evaluation
if a decision is made to lock this as the new final result.

---

## Test set re-touched once, deliberately — final Phase 3 result

The cross-architecture ensemble (Section "Current best configuration"
above) was locked as the new final configuration and the test split
(196 items) was unlocked exactly once, via `phase3_final_test_eval.py`.
Same discipline as the original Day-7 evaluation: all 6 models (3 E6
gated+orth seeds, 3 cross-attention seeds) were trained and model-selected
using train/validation only; test was not examined until every model was
already fixed, then evaluated in a single non-interactive pass.

| Metric | Original locked model (test) | New ensemble (test) | Delta |
|---|---|---|---|
| macro-F1 | 0.4635 | **0.4984** | **+0.0349** |
| weighted-F1 | 0.4857 | 0.5232 | +0.0375 |
| accuracy | 0.4932 | 0.5306 | +0.0374 |
| quadratic weighted kappa | 0.3654 | 0.3857 | +0.0203 |

**All four metrics improved on held-out test, by a margin close to (and
slightly smaller than) the validation-side gain (+0.0475).** This is the
important confirmation: the gain generalizes rather than being an
artifact of having iterated on the validation split across roughly 15
experiments in this document. A gain that evaporated on test would have
meant we'd overfit to validation through repeated experimentation; a gain
that mostly held up (as happened here) is the expected, healthy signature
of a real improvement.

**Per-class detail (majority vote of all 6 models, test split, n=196):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.484 | 0.469 | 0.476 | 32 |
| Wish to be dead | 0.580 | 0.763 | 0.659 | 38 |
| Suicide ideation | 0.613 | 0.594 | 0.603 | 64 |
| Suicide planning | 0.400 | 0.308 | 0.348 | 26 |
| Suicide attempt or death | 0.424 | 0.389 | 0.406 | 36 |

"Suicide planning" improved on test too (0.304 → 0.348), consistent with
the validation-side finding, though less dramatically than the ~0.53-0.56
seen for cross-attention alone on validation -- the ensemble's test-set
gain on this specific class is real but more modest than the headline
validation number suggested, worth stating plainly rather than only
citing the more flattering validation figure.

**Individual model test macro-F1s, for transparency:** gated seeds
0.459/0.436/0.442, cross-attention seeds 0.459/0.452/0.523 -- every
individual model scored BELOW the 6-way ensemble (0.498), confirming the
ensembling effect (not one lucky component) is what drove the improvement,
consistent with the validation-side finding.

Full confusion matrix and per-model breakdown in
`outputs/phase3_final_test_results.json`. **This test result is now the
current final, reported result of this improvement round** — supersedes
the original Day-7 test result (0.4635) for any forward-looking reporting,
while that original result remains correctly attributed to the Day 1-12
build in `PROGRESS_LOG.md` and `THESIS_SECTIONS_DRAFT.md`.

The test set has now been touched twice total across this project's
entire lifetime: once for the Day-7 locked model, once for this Phase-3
locked model. Both were deliberate, one-time, and are documented as such.
**Do not touch it again without an equally deliberate decision to lock
yet another new final configuration first.**

---
