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

## Phase 5 — retrospective hindsight items: a rigor upgrade and remaining score-chasing options

**Why this phase exists.** After Phase 4 closed out, a retrospective
question was asked: if Day 1 could be replayed, what structural or
workflow change (not an architecture tweak) would have mattered most?
Two things stood out as bigger levers than anything actually tried in
Phases 1-4:

1. This project has used a single fixed train(582)/val(195)/test(196)
   split throughout. Every seed-to-seed macro-F1 spread observed across
   the whole project has been roughly ±0.01-0.03 -- a single split of
   this size cannot distinguish a real 0.01-0.02 improvement from noise,
   which means some Phase 1/2 "this is slightly worse, reject it" calls
   carry real uncertainty about whether they were actually worse or just
   unlucky on this particular 195-item validation set.
2. Ensembling (multi-seed, then cross-architecture) turned out to be the
   single largest lever in the entire project (Phase 1.1 and Phase 3.2
   combined account for more of the total gain than every loss-function,
   fusion-mechanism, and hyperparameter change put together) but was only
   adopted 9 days into a 12-day schedule, rather than being a first-class
   part of the pipeline from Day 1.

Item 5.1 below retrofits a fix for (1) as a rigor/measurement upgrade,
not a score-chasing experiment -- it exists to tell us how much of
Phases 1-4's decision-making was signal vs. noise, which is valuable for
the thesis write-up regardless of whether it changes the final number.
(2) is not separately retrofittable at this point (ensembling is already
baked into the current best pipeline) but is recorded here as the
clearest single lesson of the whole project for the thesis's
methodology/limitations discussion.

**Score-chasing items** (from the earlier "what else could we try"
discussion) remain available as 5.2 onward, in descending order of
expected value:

| # | Item | Rationale |
|---|---|---|
| 5.1 | **k-fold cross-validation re-evaluation of the current best config** (rigor upgrade, not a new experiment) -- re-run the locked Phase 4.4 pipeline (tuned gated + cross-attention ensemble, calibrated) under 5-fold CV over train+val combined, holding test out untouched, to get a variance estimate around 0.5695 and confirm the Phase 1-4 decision trail was not substantially noise-driven | Directly answers the hindsight question: were our validation-based accept/reject calls trustworthy? Cheap relative to its information value -- no new modeling ideas, just re-running the existing training code under a different split scheme. |
| 5.2 | **Stacking meta-learner** in place of majority vote -- train a small logistic regression on the 6 base models' class probabilities (or vote fractions) instead of hand-tuned coordinate-ascent calibration | Majority vote + calibration is a blunt combination rule; a learned combiner can weight models per-class more precisely. Low effort, no base-model retraining needed. |
| 5.3 | **Soft multi-task head** (shared backbone, joint 5-way ordinal loss + auxiliary binary "any severity" loss) as a non-cascading way to exploit the same "None is more separable" signal that motivated Phase 4.5 | Phase 4.5's hard hierarchical gate failed via error-cascading; a soft auxiliary loss keeps the same backbone able to hedge across all 5 classes jointly while still being nudged by the binary signal -- mechanistically different from 4.5, not a rehash. |
| 5.4 | **More seeds in the ensemble** (5+5 instead of 3+3 gated/cross-attention models) | Pure variance reduction; ensembling has been the most reliable lever in the whole project. Cheap, no new ideas, diminishing but real returns expected. |
| 5.5 | **CORN loss** (proper cumulative-link ordinal regression) as a genuine ordinal-regression alternative to label smoothing | Different mechanism from both CORAL (rejected, Day 4) and ordinal smoothing (kept, Phase 1.3) -- we have never tried a true cumulative-link approach despite the task being explicitly ordinal. |

**Deprioritized:** SMOTE-style oversampling of "Suicide planning," test-time
augmentation, and larger contrastive-pretraining runs remain on the table
as lower-expected-value fallbacks if 5.1-5.5 stall, per the same reasoning
used to deprioritize items in Phase 4's original planning table.

**Protocol note (unchanged from Phases 1-4):** 5.1 uses train+val only
(test stays locked); 5.2-5.5, if pursued, are evaluated the same way as
every prior phase -- validation-only iteration, with a single deliberate
test-set check only if a new configuration is chosen to replace the
current locked best.

Status: **5.1 done. 5.2-5.5 not yet started.**

### 5.1 — 5-fold cross-validation of the locked pipeline (DONE — informative, and surfaced an honest new limitation)

Re-ran the current locked Phase 4.4 pipeline (3 tuned gated+orth seeds +
3 cross-attention seeds, majority vote, per-class calibration weights
{None: 0.7, Wish to be dead: 1.1, others: 1.0}) under 5-fold stratified
CV over train+val combined (777 examples: 621-622 train / 155-156
held-out per fold). Test set untouched. Stage A's contrastive alignment
projections were kept frozen and shared across folds (see
`phase5_1_kfold_cv.py` docstring for why); only the 6 downstream
classifier heads were retrained per fold, and confidence-feature
normalization was recomputed per fold from that fold's train subset only.

| Fold | n held-out | Uncalibrated macro-F1 | Calibrated macro-F1 |
|---|---|---|---|
| 1 | 156 | 0.5975 | 0.5904 |
| 2 | 156 | 0.5933 | 0.5850 |
| 3 | 155 | 0.5124 | 0.4689 |
| 4 | 155 | 0.5274 | 0.5197 |
| 5 | 155 | 0.6383 | 0.6448 |
| **Mean ± std** | | **0.5738 ± 0.0470** | **0.5618 ± 0.0611** |

| | Macro-F1 |
|---|---|
| Original single fixed-split locked result (Phase 4.4) | 0.5695 |
| 5-fold CV mean (calibrated) | 0.5618 |
| Delta (single-split vs. CV mean) | +0.0077 |

**Finding 1 (reassuring): the headline 0.5695 number is not a lucky
outlier.** It sits well inside the 5-fold spread (0.4689-0.6448) and only
0.0077 above the CV mean — the single validation split this project
locked its final number against was a representative, not
favorably-biased, sample.

**Finding 2 (the actual point of this check): per-fold variance is large
-- std of 0.047-0.061 macro-F1, on results whose mean is ~0.56-0.57.**
This quantitatively confirms the hindsight concern that motivated 5.1:
several Phase 1/2 accept/reject calls in this project were decided by
deltas of 0.01-0.03 macro-F1 on the single 195-item validation split
(e.g. Phase 1.2 gate-uncertainty at −0.0059, Phase 2.2 mixup at −0.0141)
-- deltas smaller than one fold-to-fold standard deviation here. Those
specific decisions are not necessarily wrong (they were also consistent
with mechanistic reasoning, not just the number), but the confidence
level attached to any single-split delta of that size should be read as
low, and this is now recorded plainly for the thesis's methodology and
limitations discussion rather than left implicit.

**Finding 3 (unexpected, and the most important new result from this
item): the Phase 4.4 calibration weights, tuned on one specific 195-item
validation split via coordinate ascent, do NOT generalize -- they help on
3 of 5 folds and actively hurt on 2 of 5** (fold 3: 0.5124 → 0.4689, a
−0.0434 drop; fold 4: 0.5274 → 0.5197, a smaller −0.0077 drop), while
folds 1, 2, and 5 see small gains. Net effect across folds is slightly
negative on average (calibrated mean 0.5618 vs. uncalibrated mean
0.5738). This is a textbook small-sample overfitting signature: a
handful of scalar weights fit by search against one 195-example set will
capture some of that set's idiosyncrasies along with the real signal.

**This revises Phase 4.4's status.** It was already flagged as
"provisional pending test-set confirmation" due to its different
methodology; this CV result adds a second, independent reason for
caution -- the calibration step should be treated as a mild,
data-set-specific adjustment rather than a robust general improvement,
and if a further one-time test-set check is done, comparing WITH and
WITHOUT the 4.4 calibration step (not just the calibrated number alone)
would be the honest way to report it, since CV suggests calibration's
true expected effect may be closer to zero, or even slightly negative,
than the +0.0057 seen on the original split.

**Decision: 5.1 is a completed rigor check, not a config change** -- it
does not replace the locked Phase 4.4 pipeline, but the finding above is
now part of this project's honest record of what is and is not
well-supported. Recommended next: 5.2 (cheapest remaining score-chasing
item) before 5.3-5.5, and if any further test-set evaluation is planned,
report both calibrated and uncalibrated ensemble numbers given Finding 3.

---
