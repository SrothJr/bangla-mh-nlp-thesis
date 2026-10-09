# Phase 9 Results — the 70% attempt, and what it actually found

**Status: complete. The target was not reached. The test set was never touched.**

This document reports what Phase 9 set out to do, every experiment it ran, and
what each one measured. It is written to the same rule the rest of this project
follows: negative results are reported with the same prominence as positive ones,
and nothing is quietly dropped.

---

## 1. The headline, up front

**The goal was ~70% test accuracy on the 3-class harmonized task. It was not
reached, and the test set was deliberately not evaluated.**

| | Value |
|---|---|
| Best configuration, cross-validated accuracy | **0.6873** (95% CI 0.6538 to 0.7207) |
| Pre-registered gate required to unlock the test set | 0.72, with lower CI bound at least 0.68 |
| Gate outcome | **STOP** |
| Test-set evaluations performed in Phase 9 | **0** |
| Best 3-class-equivalent test accuracy this project has ever achieved | **0.6582**, unchanged (Phase 8) |

Total measured gain over the honest baseline: **+0.0219 accuracy and +0.0242
macro-F1**, from ten experiments across five tracks.

**Almost all of that came from one idea**, found late and described in Section
11: feeding the depression classifier's output distribution into the
suicide-severity model as input features, rather than letting the two halves meet
only in the fuzzy-logic layer at the end. It contributed +0.0129 of the +0.0219
and was the only change that survived into the full ensemble.

**The one large effect found anywhere in Phase 9 is the learning curve**, and it
points at data this project does not have.

Note on reading this document: Sections 1 to 9 were written when the best
configuration stood at 0.6744, and their internal comparisons are all against the
Step 1 baseline of 0.6654, so they remain valid as written. Sections 10 and 11
are addenda that supersede the headline number.

---

## 2. What changed about the framing before anything ran

Two corrections shaped the whole phase, both made before any experiment.

**Accuracy and macro-F1 are far apart on this task.** The best 3-class-equivalent
test result was 0.6582 accuracy but only 0.6024 macro-F1. So "70%" is a 4.2-point
climb on accuracy and a 9.8-point climb on macro-F1. Phase 9 targeted accuracy
and said so up front, rather than switching metrics later to make a number look
reachable.

**The best available baseline was higher than the headline number suggested.**
The natively-trained 3-class model scores 0.6071 accuracy on test. The 5-class
ensemble's predictions collapsed into the same 3 groupings score 0.6582. The
second is the real bar, and it is the one Phase 9 measured against.

---

## 3. The central diagnosis, and the evidence for it

### 3.1 Most of the apparent validation performance was selection optimism

The locked configuration scored 0.6974 accuracy on the 195-example validation
split but 0.6071 on test — a nine-point drop. Step 1 measured why.

Every model in the original protocol early-stops by picking the epoch with the
best macro-F1 **on the very split it is then reported on**. That makes the
validation figure an upper bound on itself. Step 1 re-ran the identical
configuration under 5-fold CV over the combined 777 train+val examples, computing
two numbers in one pass:

| | Accuracy | Macro-F1 |
|---|---|---|
| **Unbiased** — inner split used for early stopping, outer rows never used for any selection | **0.6654** | **0.6155** |
| **Peeking** — best epoch chosen on the outer rows themselves, reproducing the original protocol | 0.6924 | 0.6557 |
| Measured selection optimism | **+0.0270** | **+0.0402** |

The peeking figure (0.6924 / 0.6557) almost exactly reproduces the old
single-split validation figure (0.6974 / 0.6649). That is direct evidence the old
number was inflated by selection rather than being a held-out estimate.

Selection optimism accounts for roughly 2.7 of the 9 accuracy points. The rest is
sampling noise on a 196-example test set and genuine split difficulty.

**0.6654 accuracy / 0.6155 macro-F1 became the baseline every later step was
measured against.** Comparing later results to the old 0.5730 test figure would
have repeated exactly the invalid-comparison error Phase 8 exists to correct.

### 3.2 Data, not architecture, is the binding constraint

Step 3 retrained all nine models on nested stratified subsets of each fold's
training rows, holding the validation and held-out rows fixed so that only
training-set size varied.

| Training examples | Accuracy |
|---|---|
| 210 | 0.6126 |
| 315 | 0.6178 |
| 421 | 0.6332 |
| 527 | 0.6654 |

The curve is **still climbing steeply at the largest size available, and
accelerating rather than flattening**: +0.0304 accuracy and +0.0395 macro-F1 per
additional 100 training examples.

This is the clearest quantitative confirmation of the diagnosis carried through
this entire project — that dataset size, not model capability, is the ceiling.
It is also, unfortunately, a lever that cannot be pulled: there are 973 labelled
memes in total and no more.

---

## 4. Every experiment, and what it measured

Each was A/B'd independently against the Step 1 baseline under identical folds,
seeds and the unbiased protocol. None were bundled, because bundled changes
cannot be attributed.

| # | Experiment | Accuracy | vs baseline | Verdict |
|---|---|---|---|---|
| 1 | Honest CV baseline | 0.6654 | — | baseline |
| 2 | Soft probability averaging | 0.6692 | +0.0039 | marginal |
| 2 | Soft averaging, gated + cross-attention only | 0.6718 | +0.0064 | marginal, P(better) 0.696 |
| 2 | **Stacking meta-learner** (item 5.2) | 0.6615 | −0.0039 | **rejected** |
| 2 | Stacking, class-balanced | 0.6384 | −0.0270 | **rejected** |
| 2 | Per-class decision weights | 0.6654 | +0.0000 | neutral on accuracy, +0.0165 macro-F1 |
| 4 | **Five seeds per architecture** (item 5.4) | 0.6718 | +0.0064 | **kept**, P(better) 0.834 |
| 4 | **CORN ordinal loss** (item 5.5) | 0.6654 | +0.0000 | **rejected**, macro-F1 worse |
| 5 | **Structured reasoning streams** (Track D) | 0.6075–0.6126 | significantly worse | **rejected** |
| 6 | **External negatives** (Track A2) | 0.6229 | −0.0039 | **rejected**, harmful |
| 7 | **Train on 5 classes, collapse to 3** | 0.6692 | +0.0039 | **not significant** |
| 8 | Best assembled configuration | **0.6744** | **+0.0090** | gate not cleared |

Three of these deserve their own account.

### 4.1 Track D — structured figurative-reasoning streams (the novelty item)

This was the contribution intended to headline a thesis chapter. The pipeline
generates three distinct analytic fields per meme — cause and effect, figurative
meaning, emotional state — and then flattens them into a single string before
BanglaBERT ever sees them. The hypothesis was that keeping them apart, and
letting the model weight them per example, should help, because memes are
figurative by nature and weighing "what this figuratively means" separately from
"what it literally says" is aligned with what makes the task hard.

All 973 memes had each field translated to Bangla separately and embedded as its
own stream, then combined by learned stream attention.

| Streams | Accuracy | Macro-F1 |
|---|---|---|
| e3b alone (the current flattened blob) | 0.6293 | 0.5937 |
| OCR + e3b | 0.6306 | 0.5908 |
| OCR + the three reasoning fields | 0.6075 | 0.5847 |
| OCR + e3b + the three reasoning fields | 0.6126 | 0.5829 |

The full versions are **significantly worse** than the flattened baseline
(P(better) 0.030 and 0.050). The learned attention came out near-uniform, between
0.176 and 0.211 across five streams, meaning the model found no structure worth
exploiting and simply spent capacity splitting attention evenly.

Rejected, exactly as the plan said it would be if it did not beat the blob. The
hypothesis was plausible, it was tested properly, and the answer is no.

### 4.2 Track A2 — external negatives made the target class worse

Class 0 was the measured weak point: F1 0.469, recall 0.419, from only 117 of 777
examples. BN-HIB supplied 1,667 screened Bangla troll memes as candidate extra
class-0 examples, capped at 300 per fold, training-only, with a three-layer
screen.

| Arm | Accuracy | Macro-F1 | Class-0 F1 |
|---|---|---|---|
| Baseline, real data only | 0.6268 | 0.5711 | 0.3860 |
| Plus external negatives | 0.6229 | 0.5302 | **0.2733** |

It made the class it was designed to fix **significantly worse**: class-0 F1
−0.1127, P(better) 0.004.

The screening output explains why. Even the *most* confidently class-0 external
memes were assigned only about 45% class-0 probability by the model. The external
memes are simply out of distribution, so 300 of them labelled class 0 taught a
confused boundary rather than a sharper one.

**This establishes a distinction worth keeping:** more in-domain data helps a
great deal (Section 3.2), but out-of-domain negatives do not substitute for it,
even when they are topically safe, visually similar and linguistically matched.

### 4.3 Phase 8's coarse-versus-fine gap does not replicate

Phase 8 observed that the old 5-class predictions, collapsed to 3 classes, beat
the native 3-class model on test by +0.0511 accuracy. That was never examined as
a technique, and if real it would have been the largest lever available at zero
cost.

Under matched folds, seeds and architectures:

| Arm | Accuracy | Macro-F1 |
|---|---|---|
| Native 3-class training | 0.6654 | 0.6155 |
| 5-class training, collapsed to 3 | 0.6692 | 0.5814 |

Accuracy difference +0.0039, P(better) 0.599 — indistinguishable from noise.
Macro-F1 is worse. **The +0.0511 test gap was an artefact** of comparing two
single-split runs with different ensemble compositions, not evidence of a better
training strategy. Phase 8's reading of that gap should be treated accordingly.

---

## 5. The gate, and why it was honoured

The plan fixed this before any experiment ran:

> proceed to the single test evaluation only if CV accuracy ≥ 0.72 **and** the
> lower bootstrap CI bound ≥ 0.68

The 0.72 margin existed because of the nine-point validation-to-test gap this
project had already been bitten by once, and because comparing several combiners
on the same 777 rows carries selection optimism of its own.

| | Required | Actual |
|---|---|---|
| CV accuracy | ≥ 0.72 | 0.6744 |
| Lower CI bound | ≥ 0.68 | 0.6396 |

**Result: STOP.** The test set was not touched.

There was a tempting argument for proceeding anyway. Track A1 — refitting the
final models on all 777 examples instead of the 582 the locked models used —
cannot show up in cross-validation, because CV must hold data out. The Step 3
learning curve projects roughly +0.06 accuracy from that refit, which would put a
final fit near 0.72.

That argument was not acted on. Adjusting a gate so that it admits the result you
wanted is the precise failure mode the gate exists to prevent, and a linear
extrapolation of a curve that is certainly concave is not evidence. The
projection is recorded here as an untested hypothesis, which is what it is.

---

## 6. What Phase 9 contributes, given it did not hit the target

1. **A quantified measurement of selection optimism in this project's own
   protocol** — +0.0270 accuracy and +0.0402 macro-F1. This explains a large part
   of a nine-point gap that had previously been unexplained, and it is a concrete
   methodological finding about how every result in Phases 1 through 8 was
   selected.
2. **The first honest cross-validated baseline for the 3-class task**, with
   bootstrap confidence intervals: 0.6654 accuracy, 0.6155 macro-F1.
3. **A measured learning curve** showing supervision is the binding constraint,
   with a slope of +0.030 accuracy per 100 examples — turning a long-standing
   qualitative claim into a number.
4. **The last three open items from the Phase 5 table are now closed**: stacking
   (5.2, rejected), more seeds (5.4, kept), CORN (5.5, rejected).
5. **A tested answer to the structured-reasoning hypothesis** — negative, with
   the attention weights showing why.
6. **Evidence that out-of-domain negatives do not substitute for in-domain data**,
   including a case where they actively harmed the class they targeted.
7. **A correction to Phase 8's coarse-versus-fine observation**, which does not
   survive cross-validation.
8. **The first use of bootstrap confidence intervals** anywhere in this project,
   and a pre-registered gate that was honoured when it returned an unwelcome
   answer.

---

## 7. Honest limitations of Phase 9 itself

1. **Track A2 was run on a weaker text feature than the locked pipeline uses.**
   The local vision model degenerated into repeated-token output on every
   external image, so reasoning text could not be generated for external memes.
   Giving them an empty reasoning segment would have handed the classifier a
   trivial "this row is external" shortcut, so both arms used the OCR-only text
   feature instead. A positive result there would not have transferred unchanged
   to the e3b pipeline.
2. **Track A2 was scoped to the concat architecture alone**, because the gated
   architecture needs reasoning-derived confidence features and cross-attention
   needs token and patch tensors, neither of which exists for external memes.
3. **Several combiners were compared on the same 777 rows**, so the assembled
   winner carries some selection optimism. This is part of why the gate margin
   was set at 0.72 rather than 0.70.
4. **Track A1 was never exercised.** Its benefit cannot appear in CV, and the
   gate that would have authorised a final full-data fit was not cleared.
5. **Stage A's contrastive alignment projections were frozen and shared across
   folds** rather than refitted per fold, following the precedent and stated
   reasoning of Phase 5.1.

---

## 8. Recommendation

**Do not pursue further architecture, loss-function or combiner work on this
dataset.** Phase 9 tried eight distinct ideas across four tracks and the total
honest gain was under one accuracy point. Combined with Phases 1 through 8, which
reached the same conclusion by different routes, the evidence is now
comprehensive.

The learning curve says what would work: roughly 300 to 500 additional labelled
in-domain examples would, on the measured slope, plausibly deliver the remaining
distance to 70% on its own — more than every architectural idea in this project
combined. Track A2 establishes that they have to be genuine in-domain examples;
borrowed negatives from an adjacent task actively hurt.

For the thesis, the defensible headline remains the 5-class ensemble at 0.4984
macro-F1 on test, with the 3-class work reported on its own terms, and Phase 9's
measurement results reported as a methodological contribution rather than a
performance one.

---

## 9. Where everything is

| Step | Code | Output |
|---|---|---|
| 1 CV baseline | `code/phase9_1_cv_baseline_3class.py` | `outputs/phase9_1_cv_baseline_results.json`, `phase9_1_oof_probs.npz` |
| 2 Combiners | `code/phase9_2_combiner_upgrades.py` | `outputs/phase9_2_combiner_results.json` |
| 3 Learning curve | `code/phase9_3_learning_curve.py` | `outputs/phase9_3_learning_curve_results.json` |
| 4 Seeds and CORN | `code/phase9_4_seeds_and_corn.py` | `outputs/phase9_4_seeds_corn_results.json` |
| 5 Track D | `code/phase9_5_translate_reasoning_fields.py`, `phase9_9_extract_field_embeddings.py`, `phase9_6_multistream.py` | `outputs/phase9_6_multistream_results.json` |
| 6 Track A2 | `code/phase9_7_external_negatives_screen.py`, `phase9_10_external_features.py`, `phase9_11_external_negatives_ab.py` | `outputs/phase9_11_external_negatives_ab_results.json` |
| 7 Coarse vs fine | `code/phase9_12_coarse_vs_fine.py` | `outputs/phase9_12_coarse_vs_fine_results.json` |
| 8 Gate | `code/phase9_13_assemble_and_gate.py` | `outputs/phase9_13_gate_results.json` |
| 9 Track A1 validation | `code/phase9_15_a1_refit_validation.py` | `outputs/phase9_15_a1_refit_validation_results.json` |

Plan document: `PHASE9_BEYOND_70_PLAN.md`.

Figures:

- `figures/21_phase9_summary.png` — the learning curve, and every experiment measured against the honest baseline.
- `figures/22_phase9_final_verdict.png` — the Track A1 test, why it fails, and the resulting expected test accuracy against the benchmarks it would have to beat.

---

## 10. Addendum — Track A1 was tested, and it does not work

Section 5 recorded the full-data refit as "an untested hypothesis". It has now
been tested, because every argument that Phase 9 might still beat the previous
best rested on it.

**The question A1 actually poses.** Step 3 showed that more training data helps a
lot. But to train on *all* the data you must give up the early-stopping holdout
and run a fixed number of epochs decided in advance. That second half was never
checked, and it could cancel the first.

**The measurement.** Same 5 folds and the same 15-model configuration from Step 8.
Within each fold, one arm used the current protocol (early stopping on an inner
split, about 528 training rows) and the other used the A1 protocol (the entire
fold-training portion, about 622 rows, for a fixed epoch count taken from the
first arm's mean best epoch). Both scored on identical outer rows.

| Arm | Accuracy | Macro-F1 |
|---|---|---|
| Current protocol — early stopping, ~528 rows | **0.6744** | **0.6291** |
| A1 protocol — fixed epochs, ~622 rows (+18% data) | 0.6641 | 0.6187 |
| Delta | **−0.0103** (CI −0.0322 to +0.0129, P(better) 0.178) | −0.0105 (P(better) 0.216) |

**A1 loses, despite 18% more training data.** The mechanism is visible in the
epoch schedules: the best epoch varies enormously between seeds — for
cross-attention in fold 1 it was 34, 41, 99, 32 and 26 — so any single fixed
schedule is a poor compromise for most of the runs. The early-stopping signal is
worth more than the extra rows it costs.

**What this settles.** The projected "+0.03 to +0.06 from the final full-data
fit" was doing nearly all the work in any estimate that Phase 9 could beat the
0.6582 benchmark. That projection is now measured as *negative* at the scale
tested. Expected test accuracy for Phase 9's configuration therefore falls back
to roughly 0.616 — the known 0.6071 plus the +0.009 of real combiner and seed
gains — which is below the existing benchmark.

**Caveat, stated plainly:** this tested a step from about 528 to 622 rows
(+18%), not the full 582 to 777 step (+34%). A larger jump could in principle
behave differently. But the failure here is caused by losing early stopping, and
that cost does not shrink as the training set grows.

**Conclusion: there is no remaining path by which Phase 9's configuration would
beat the 0.6582 benchmark, and no case for spending a fourth look at the test
set.** Section 8's recommendation stands, now on measured rather than
extrapolated grounds.

Figure for this addendum: `figures/22_phase9_final_verdict.png`.

---

## 11. Addendum 2 — the one idea that worked: depression output as input features

Everything above tested ideas internal to the suicide-severity model. This one
changed how the two halves of the system talk to each other, and it is the only
Phase 9 idea that survived into the full ensemble.

**The observation.** The depression classifier and the suicide-severity model
previously met only in the fuzzy-logic layer, after both had already committed to
a decision. The depression classifier produces a 4-class distribution per meme,
that distribution is informative about suicide severity, and the suicide model
never saw it.

**The change.** Feed those four numbers in as extra input features to the fusion
heads. The depression checkpoint is not retrained or modified in any way; it is
called through the Phase 8 parity-tested wrapper, and only the small downstream
fusion heads change, by taking four more input dimensions.

**Result, 15-model ensemble, same folds and protocol as everything else:**

| Configuration | Accuracy | Macro-F1 |
|---|---|---|
| Baseline: text + image | 0.6744 | 0.6291 |
| **+ depression distribution** | **0.6873** | **0.6397** |
| Delta | **+0.0129** (CI −0.0039 to +0.0296, P(better) 0.922) | +0.0106 (P(better) 0.853) |

Per architecture: concat +0.0129, gated +0.0013, cross-attention −0.0077. The
gain is carried by concat, but the assembled ensemble still nets +0.0129.

**Per-class, best configuration:** no expressed severity 0.505 F1, suicidal
thought or desire 0.744, high acuity 0.670. The weak class improves from 0.469
at the Step 1 baseline to 0.505.

**Why it is interesting beyond the number.** The Spearman correlation between
predicted depression level and true suicide severity is only **0.0897**. The
model is therefore not exploiting a simple monotonic relationship; the
distribution carries information that its argmax does not.

**No target leakage.** The depression classifier was trained on its own separate
4,897-record Bangla text dataset, not on these memes and not on their
suicide-severity labels. Its outputs are a legitimate input feature. This is
worth stating explicitly in the thesis, because "model B's output as model A's
input" is exactly the kind of design a reader should want reassurance about.

**Updated Phase 9 totals.** Best cross-validated configuration is now **0.6873
accuracy (95% CI 0.6538 to 0.7207), macro-F1 0.6397**. Total gain over the
honest Step 1 baseline rises from +0.0090 to **+0.0219 accuracy and +0.0242
macro-F1** — most of Phase 9's entire yield coming from this one change.

**What it does not change.** The gate required 0.72 accuracy with a lower
confidence bound of 0.68. Actual is 0.6873 with a lower bound of 0.6538, so the
outcome is still STOP and the test set stays sealed. Projected test accuracy
rises to roughly 0.629, still below the 0.6582 already on record.

**So: a real improvement over the 3-class lock, a genuinely novel integration
result worth a thesis subsection, and still not a new benchmark.**

Code: `code/phase9_18_depression_features.py`,
`code/phase9_19_depression_features_full_ensemble.py`.
