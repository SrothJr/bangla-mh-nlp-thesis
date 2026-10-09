# Phase 9 — A New Workflow Targeting ~70% Test Accuracy

**Status:** PLAN ONLY. No code has been written or run for this phase. Nothing in
this document modifies, retrains, overwrites or deletes any existing result,
checkpoint, script or document.

**What this document is:** a complete, self-contained plan for a *new, parallel*
workflow whose goal is to reach roughly **70% test accuracy on the 3-class
harmonized suicide-content task**, while adding genuine methodological novelty
worth writing up in the thesis. It is written to the same discipline as every
prior phase in this project: plan first, state the honest expected value before
running anything, and never quote a number that has not been validly measured.

---

## 0. Read this first — the target, restated honestly

The request was "at least around 70% accuracy". Before planning anything, two
distinctions have to be made explicit, because they change what is and is not
achievable.

### 0.1 Accuracy is not macro-F1, and the difference is large here

| Metric, 3-class task | Best value this project has actually achieved on test |
|---|---|
| **Accuracy** | **0.6582** (65.8%) |
| Macro-F1 | 0.6024 |

These both come from the *5-class ensemble's predictions collapsed into the
3-class grouping* (the valid-comparison method established in Phase 8, Step 0),
not from the natively-trained 3-class model, which scored lower on both
(accuracy 0.6071, macro-F1 0.5730).

So the honest gap to the target is:

- **To 70% accuracy: +4.2 points.** A real but plausible stretch.
- **To 70% macro-F1: +9.8 points.** Not realistic on this dataset, and
  `IMPROVEMENT_PLAN.md`'s closing note already explains why.

**This phase therefore targets accuracy, and will report macro-F1 alongside it
without pretending macro-F1 will reach 70%.** Any write-up must say which metric
the 70% refers to. Reporting "70%" while quietly switching metrics would repeat
exactly the error Phase 8 Step 0 was created to correct.

### 0.2 The real obstacle is generalization, not model capacity

This is the single most important finding behind this plan, and it changes the
whole strategy. The 9-model ensemble's **validation** scores, from
`outputs/phase7_9_harmonized_ensemble_results.json`:

| Configuration | Val macro-F1 | Val accuracy |
|---|---|---|
| concat alone (3 seeds) | 0.6425 | 0.6769 |
| gated alone (3 seeds) | 0.6330 | 0.6667 |
| cross-attention alone (3 seeds) | 0.5982 | 0.6308 |
| concat + cross-attention (6) | 0.6487 | 0.6872 |
| **concat + gated + cross-attention (9) — the locked config** | **0.6649** | **0.6974** |

Then the same locked 9-model configuration on test
(`outputs/phase7_final_test_results_3class.json`):

| | Validation | Test | Gap |
|---|---|---|---|
| Accuracy | 0.6974 | 0.6071 | **−0.0903** |
| Macro-F1 | 0.6649 | 0.5730 | −0.0919 |

**The model already reaches 69.7% accuracy — on validation.** It loses roughly
nine points moving to test. The target is not "build a better model"; it is
**"stop losing nine points between validation and test."**

Why that gap exists, stated plainly:

1. **Selection overfitting.** Across Phases 1-8, dozens of accept/reject
   decisions were made on the *same 195-example validation set*. That set has
   been implicitly fitted to. Phase 5.1's k-fold work already flagged this as a
   limitation; this phase acts on it.
2. **Small test set.** 196 examples. The 95% binomial confidence interval around
   60.7% accuracy is roughly ±6.8 points. Some of the gap is genuine noise, and
   **some of any future "improvement" will be noise too** — which is why this
   plan builds in confidence intervals rather than celebrating a point estimate.
3. **Models train on only 582 examples.** Confirmed from
   `code/phase7_10_final_test_eval_3class.py`: all 9 models train on the train
   split alone and early-stop on validation. The 195 validation examples are
   used for selection and then thrown away — never trained on.

Point 3 is the most actionable single fact in this entire document.

### 0.3 Where the accuracy is actually being lost

The locked 3-class test confusion matrix (rows = true, cols = predicted):

|  | Pred: None | Pred: Thought/desire | Pred: High-acuity |
|---|---|---|---|
| **True: No expressed severity** (32) | 18 | 11 | 3 |
| **True: Suicidal thought or desire** (102) | 17 | 68 | 17 |
| **True: High-acuity** (62) | 11 | 18 | 33 |

Per-class test F1: **No expressed severity 0.462** (precision 0.391, recall
0.563), Suicidal thought or desire 0.683, High-acuity 0.574.

Two clear, targetable problems:

- **Class 0 precision is 0.391** — of 46 examples predicted "no expressed
  severity", 28 were wrong. The model over-assigns class 0, and 28 of those
  errors are false negatives on genuinely suicidal content. This is both the
  largest accuracy leak *and* the most safety-relevant error direction.
- **High-acuity recall is 0.532** — nearly half of the most severe cases are
  missed.

Fixing class 0's precision alone, without touching anything else, is worth
several points of accuracy. This gives Phase 9 a specific, measurable target
rather than a vague "make it better".

---

## 1. Preservation rules — what this phase may never touch

Identical in spirit to Phase 8's rules, and non-negotiable.

**Never modified, never retrained, never overwritten:**

1. **The DAPT-BanglaBERT depression classifier** (`checkpoint-735`) and its
   4-class head. Frozen inference only, exactly as in Phase 8.
2. **Every existing locked test result**: `outputs/final_test_results.json`,
   `outputs/phase3_final_test_results.json`,
   `outputs/phase7_final_test_results_3class.json`, and every Phase 8 output.
3. **Every existing script in `code/`.** Phase 9 code imports from them; it does
   not edit them. If a change to shared code appears necessary, the function is
   copied into a `phase9_*` file instead, with a comment saying where it came
   from and what was changed.
4. **Every existing document.** `IMPROVEMENT_PLAN.md`,
   `CURRENT_VERIFIED_SYSTEM_REPORT.md`, `THESIS_SECTIONS_DRAFT.md` and the
   `supervisor_result/` package stay as they are until Phase 9 produces a
   validated result worth recording — and then they are *appended to*, with the
   existing content left visible, per this project's standing practice.
5. **The existing figures.** New figures get new numbers (21+).

**Naming convention so nothing can collide:** all new code is
`code/phase9_*.py`, all new outputs are `outputs/phase9_*.json`, all new
checkpoints are `outputs/checkpoints/phase9_*/`.

**Test-set protocol, stated in advance and binding:** the test set is touched
**exactly once** in this entire phase, at Step 7, and only if Step 6's gate is
passed. Every decision before that point is made on cross-validation over
train+val only. If Step 7's result is worse than the current best, **it is
reported as worse** and the current locked model stands. That outcome is a
legitimate, publishable finding, not a failure to be re-run away.

---

## 2. Strategy — four tracks, ordered by honest expected value

The diagnosis in Section 0 says the problem is generalization and data, not
architecture. The tracks are ordered accordingly. Architecture novelty comes
last, not first, because this project has already demonstrated repeatedly
(Phases 3.1, 4.5, 6.1, 7.2) that architectural changes yield small or negative
gains on this dataset size.

### Track A — Use the data that is already being wasted (highest expected value)

**A1. Refit the final ensemble on train + validation combined (777 examples
instead of 582).**

This is the largest safe, untried lever available. Currently 195 labeled
examples — 25% of all available non-test data — are used only for early stopping
and then discarded. Standard practice, once hyperparameters are fixed, is to
refit the final model on all non-test data.

The complication is that early stopping needs a validation signal. The solution
is the standard one: use k-fold cross-validation over the combined 777 examples
to determine the *epoch count* (and confirm the hyperparameters), then refit on
all 777 for that fixed number of epochs with no early stopping. The machinery
for k-fold over train+val already exists and is proven —
`code/phase5_1_kfold_cv.py` did exactly this for the 5-class pipeline.

Expected gain: **+1 to +3 accuracy points.** A 33% increase in training data on a
dataset this small is meaningful. This is the single highest-confidence item in
the plan.

**A2. External negative augmentation for the weakest class.**

Class 0 ("no expressed severity") has 32 test examples, the fewest of any class,
the worst F1 (0.462) and the worst precision (0.391). It is starved of training
data and the model's decision boundary around it is poor.

`BN-HIB/` is already present locally and already verified usable by Phase 7.2,
which trained a SigLIP vision head on it (2,272 train / 487 val Bangla memes,
hate-speech-labeled). Those memes are Bangla, in-domain visually and
linguistically, and — being about hate speech and general inflammatory content,
not self-harm — are overwhelmingly **true negatives for suicide content**.

The proposal is to add a screened, capped subset of BN-HIB memes as additional
class-0 training examples.

Screening is mandatory, not optional, and has three layers:
1. A Bangla + English self-harm/suicide keyword screen to drop any candidate
   that might genuinely contain suicidal content.
2. A confidence screen using the existing locked 3-class model: drop any
   candidate it assigns non-trivial probability to classes 1 or 2.
3. A manual spot-check of a random sample of at least 50 survivors before any
   are used.

Hard caps to prevent the model learning a "BN-HIB style ⇒ class 0" shortcut:
- Add at most **300** external negatives (roughly doubling class 0's training
  count, not swamping it).
- Hold the external examples **out of validation entirely** — they are training
  signal only, so that CV scores remain measured on real in-domain data.
- Run the A/B comparison with and without them under identical CV folds.

Expected gain: **+1 to +4 accuracy points**, concentrated in class-0 precision,
which is where the errors actually are. This carries the most risk of the data
track and is therefore explicitly A/B-tested, with a documented rollback if it
does not help.

**A3. (Contingent) Confidence-filtered self-training on remaining external
memes.** Only attempted if A2 works and Step 6's gate has not yet been cleared.
Pseudo-label the remaining BN-HIB pool with the A1+A2 model, keep only
high-confidence predictions, retrain. Classic semi-supervised self-training,
directly aimed at the diagnosed data bottleneck. Treated as a fallback because
self-training on a model that is only ~65% accurate risks amplifying its own
errors.

### Track B — Extract more from the nine models already trained

**B1. Soft probability averaging instead of hard vote fractions.**

Phase 8's audit confirmed the ensemble combines **hard vote fractions** — each
model contributes only its argmax. The models' actual probability distributions
are computed and then discarded. Soft averaging normally outperforms hard voting
because it preserves confidence information, and **this has never been compared
on the 3-class task.**

This is nearly free: no retraining, just a different aggregation over
predictions. It must be evaluated on CV, not test.

Expected gain: **+0.5 to +2 accuracy points.** Low cost, decent odds.

**B2. Out-of-fold stacking meta-learner (the long-deferred item 5.2).**

Replace majority vote with a small multinomial logistic regression trained on
the nine models' out-of-fold predicted probabilities. A learned combiner can
weight architectures per-class — for instance, trusting cross-attention more on
high-acuity and concat more on class 0 — which a uniform vote cannot.

Critically, it must be trained on **out-of-fold** predictions generated under
the Track A CV scheme. Training a stacker on in-fold predictions is a classic
leakage error and would inflate CV scores while making test results worse.

Expected gain: **+1 to +3 accuracy points.** Listed in `IMPROVEMENT_PLAN.md` as
item 5.2 since Phase 5 and never attempted.

**B3. More seeds — 5 per architecture instead of 3 (item 5.4).**

Pure variance reduction. Ensembling has been the most reliable lever in the
entire project. 15 models instead of 9. Cheap, no new ideas, diminishing but
real returns — and variance reduction is precisely what a 9-point val-to-test
gap calls for.

Expected gain: **+0.5 to +1.5 accuracy points.**

### Track C — Make the objective match the task

**C1. CORN loss (item 5.5, never attempted).** The task is ordinal: "no
severity" < "thought or desire" < "high-acuity". CORAL was tried and rejected on
Day 4; ordinal label smoothing was tried and kept in Phase 1.3. A proper
cumulative-link formulation has never been tried despite the task being
explicitly ordinal. Mechanistically different from both.

**C2. Class-0-targeted decision thresholds.** Given class 0's precision problem
(0.391), tune per-class decision offsets on CV folds. Phase 4.4 did per-class
calibration for the 5-class task and it helped; it has never been redone for the
3-class task, whose class balance is completely different.

Expected gain, combined: **+1 to +3 accuracy points**, with C2 the more likely
contributor since it targets the measured failure directly.

### Track D — The novelty contribution: structured figurative-reasoning streams

Every track above is sound engineering, but none is novel enough to headline a
thesis chapter. This one is.

**Current state:** `outputs/translation_reasoning_results.jsonl` contains 973
structured reasoning records, each with three distinct analytic fields —
`cause_effect`, `figurative_meaning`, `emotional_state` — each carrying a
`claim`, supporting `evidence`, and an `uncertain` flag. Confirmed from
`code/extract_text_embeddings_e3b.py`, these are currently **flattened into a
single text blob**, concatenated with the OCR text, and pushed through
BanglaBERT as one undifferentiated sequence. The structure is generated, then
immediately thrown away.

**Proposal:** embed each reasoning field as its own stream and learn attention
over the streams:

```
OCR text            ──→ BanglaBERT ──┐
cause_effect        ──→ BanglaBERT ──┤
figurative_meaning  ──→ BanglaBERT ──┼──→ learned stream-attention ──→ text vector
emotional_state     ──→ BanglaBERT ──┤                                      │
SigLIP image vector ────────────────┴──────────────────────────────────────┴──→ fusion ──→ 3-class
```

Why this is defensible rather than decorative:

- Memes are figurative by nature — that is the premise of the entire FigSIM
  dataset. A model that can weight *figurative meaning* separately from *literal
  OCR content* is directly aligned with what makes the task hard.
- The `uncertain` flags are a free, currently-unused reliability signal. A stream
  flagged uncertain can be down-weighted rather than trusted equally.
- It is genuinely new for this project, and the thesis framing writes itself:
  **structured figurative-reasoning decomposition for multimodal mental-health
  screening.**

**Honest risk:** this adds parameters to a 777-example problem. It is ordered
last precisely because of that. Mitigations: stream-attention is a single small
weight vector, not a transformer; the four streams share one frozen encoder;
and it is A/B-tested against the flattened-blob baseline under identical CV
folds. **If it does not beat the blob on CV, it is reported as a negative result
and dropped** — which is still a genuine thesis contribution, since it tests a
plausible hypothesis and answers it.

Expected gain: **−1 to +3 accuracy points.** The widest uncertainty band in the
plan, and the highest novelty.

### Expected-value summary

| Track | Item | Expected accuracy gain | Confidence | Cost |
|---|---|---|---|---|
| A1 | Refit on train+val (777) | +1 to +3 | High | Low |
| A2 | External class-0 negatives | +1 to +4 | Medium | Medium |
| B1 | Soft averaging vs hard votes | +0.5 to +2 | Medium-high | Very low |
| B2 | Out-of-fold stacking | +1 to +3 | Medium | Low |
| B3 | 5 seeds per architecture | +0.5 to +1.5 | High | Medium |
| C1 | CORN ordinal loss | 0 to +2 | Low-medium | Medium |
| C2 | Class-0 threshold calibration | +1 to +3 | Medium-high | Very low |
| D | Structured reasoning streams | −1 to +3 | Low | High |

**These gains are not additive.** They overlap heavily — B1, B2 and C2 all
operate on the same combination stage and will partly substitute for one
another. A realistic aggregate expectation is **+3 to +6 accuracy points over
the 65.8% current best, landing somewhere in 68-72%.**

**So: is 70% reachable? Plausibly yes, but not assured.** That is the honest
answer, and it is deliberately recorded here *before* running anything, so the
outcome cannot be retrofitted into a success story afterwards.

---

## 3. Workflow

```
                    EXISTING SYSTEM  (frozen, untouched)
  ┌──────────────────────────────────────────────────────────────┐
  │  DAPT-BanglaBERT (depression, verified)                      │
  │  SigLIP encoder │ Stage-A alignment │ 9 locked fusion models  │
  │  Locked test results: 5-class 0.4984 │ 3-class 0.5730        │
  └──────────────────────────────────────────────────────────────┘
                              │  read-only
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 1   Establish the honest CV baseline                    │
  │           5-fold CV over train+val (777), 3-class,            │
  │           current config. Bootstrap CIs. No test.             │
  └──────────────────────────────────────────────────────────────┘
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 2   Track B — combiner upgrades (no retraining)         │
  │           B1 soft averaging │ B2 OOF stacking │ C2 thresholds │
  └──────────────────────────────────────────────────────────────┘
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 3   Track A1 — refit on all 777, epochs fixed by CV     │
  └──────────────────────────────────────────────────────────────┘
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 4   Track A2 — screened external class-0 negatives      │
  │           3-layer screen → cap 300 → A/B under same folds     │
  └──────────────────────────────────────────────────────────────┘
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 5   Tracks B3 + C1 + D — seeds, CORN, reasoning streams │
  │           each A/B'd independently under identical folds      │
  └──────────────────────────────────────────────────────────────┘
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 6   GATE.  Assemble best CV config. Proceed only if     │
  │           CV accuracy ≥ 0.72 with lower CI bound ≥ 0.68.      │
  └──────────────────────────────────────────────────────────────┘
                     │ pass                    │ fail
                     ▼                          ▼
  ┌──────────────────────────┐   ┌────────────────────────────────┐
  │ STEP 7  ONE test         │   │ Stop. Report CV honestly.      │
  │ evaluation. Bootstrap CI.│   │ Locked model stands. Write up  │
  │ Whatever it says, stands.│   │ what was learned.              │
  └──────────────────────────┘   └────────────────────────────────┘
                     ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  STEP 8   Fuzzy-logic re-integration + documentation          │
  └──────────────────────────────────────────────────────────────┘
```

### Why the gate at Step 6 demands 0.72, not 0.70

Because of the nine-point validation-to-test gap measured in Section 0.2.
Requiring only 0.70 on CV would almost certainly produce a test result below
0.70. The margin is deliberate insurance against the exact failure this project
has already experienced once. The requirement that the **lower bootstrap
confidence bound** clear 0.68 is insurance against the second failure mode:
declaring victory on a lucky point estimate.

---

## 4. Step-by-step detail

### Step 1 — Honest CV baseline (no new modelling)

`code/phase9_1_cv_baseline_3class.py`

Re-run the current locked 3-class configuration under 5-fold stratified CV over
the combined 777 train+val examples. Test untouched. Stage-A alignment
projections stay frozen and shared across folds, exactly as
`phase5_1_kfold_cv.py` established and justified for the 5-class case.

Produces: mean and standard deviation of accuracy and macro-F1 across folds,
plus per-fold confusion matrices and bootstrap confidence intervals.

**This number, not the 0.5730 test figure, is the comparison baseline for every
subsequent step.** Comparing later CV results against an old test number would
be an invalid comparison of exactly the kind Phase 8 Step 0 exists to prevent.

Also saves each model's **out-of-fold predicted probabilities**, which Step 2
requires and which no existing output file contains.

### Step 2 — Combiner upgrades

`code/phase9_2_combiner_upgrades.py`

Using Step 1's out-of-fold probabilities, with no retraining:
- **B1:** soft probability averaging vs. the current hard vote fractions.
- **B2:** multinomial logistic regression stacker on out-of-fold probabilities.
- **C2:** per-class decision offsets tuned by coordinate ascent on CV folds,
  targeting class-0 precision specifically.

All three compared against Step 1's baseline on identical folds. The winner
carries forward; losers are recorded with their numbers, not deleted.

### Step 3 — Refit on all 777

`code/phase9_3_refit_full_trainval.py`

Determine the epoch count per architecture from Step 1's CV curves, then train
each of the 9 models on all 777 examples for that fixed count with no early
stopping. Evaluate by nested CV so the comparison against Step 1 stays fair.

### Step 4 — External negatives

`code/phase9_4_external_negatives.py`

Implements the three-layer screen from A2, writes the surviving candidate list
to `outputs/phase9_4_external_negative_candidates.json` **for manual review
before any training run uses it**, then runs the capped A/B comparison.

Privacy note: BN-HIB is CC BY-NC-SA and already correctly excluded from git by
`.gitignore`. Phase 9 outputs must save **indices and aggregate metrics only** —
never raw meme text or images — consistent with how
`phase8_3_depression_dataset_verification.py` handled `dataset.xlsx`.

### Step 5 — Seeds, ordinal loss, reasoning streams

`code/phase9_5_seeds_corn_streams.py` (or three separate scripts if runtime
demands it)

Each of B3, C1 and D is A/B-tested **independently** against the Step 3/4 best,
under identical folds. No bundling — bundled changes cannot be attributed, and
this project's log is valuable precisely because every change is individually
attributed.

### Step 6 — The gate

`code/phase9_6_assemble_and_gate.py`

Assemble the best combination of everything that individually passed, re-run CV
on that exact combination (a combination of individually-good components is not
guaranteed to be good, and must be measured as a whole), compute bootstrap CIs,
and print a clear PASS or STOP against the 0.72 / 0.68 criteria.

### Step 7 — The single test evaluation

`code/phase9_7_final_test_lock.py`

Runs only on a PASS. One non-interactive pass over the test set. Reports
accuracy, macro-F1, weighted-F1, balanced accuracy, QWK, confusion matrix,
per-class report, and **bootstrap 95% confidence intervals on accuracy and
macro-F1** — the first time this project will have reported uncertainty on a
locked test number, which is itself a rigor improvement worth noting in the
thesis.

Compared against **both** prior 3-class-equivalent results: the native 3-class
lock (0.6071 accuracy) and the collapsed 5-class ensemble (0.6582 accuracy).
Beating only the weaker of the two would be reported as such and would not
count as reaching the goal.

### Step 8 — Re-integration and documentation

Re-run the fuzzy-logic layer against the new model on test, mirroring
`phase7_11_fuzzy_logic_final_test.py`, so the end-to-end system stays coherent.
Then append results to `IMPROVEMENT_PLAN.md`, write a new report document,
generate figures 21+, and sync `supervisor_result/`.

---

## 5. What gets reported if this fails

Recorded now, before the outcome is known, so that a null result cannot be
quietly buried.

If Step 6's gate is not cleared, this phase reports:

1. The honest CV baseline with confidence intervals — **which the project has
   never had for the 3-class task**, and which is independently valuable.
2. A measured decomposition of which levers helped and by how much, each
   individually attributed.
3. Direct evidence on whether the 9-point validation-to-test gap is selection
   overfitting or intrinsic task difficulty.
4. A tested answer to the structured-reasoning-decomposition hypothesis.

Every one of those is thesis material. A phase that measures carefully and finds
the ceiling is a contribution; a phase that reaches 70% by quietly tuning on test
is not.

---

## 6. Compute estimate

| Step | Runtime estimate |
|---|---|
| 1 — CV baseline, 9 models × 5 folds | 2-4 hours |
| 2 — Combiner upgrades (no retraining) | Minutes |
| 3 — Refit on 777 | 1-2 hours |
| 4 — External negatives (screen + A/B) | 2-3 hours |
| 5 — Seeds + CORN + streams | 4-8 hours |
| 6 — Gate | 1-2 hours |
| 7 — Test lock | Under 1 hour |
| 8 — Fuzzy re-integration, figures, docs | 1-2 hours |

Total roughly **12-22 hours** of compute, resumable. All embeddings are already
extracted and cached in `outputs/embeddings/`, except any new streams Step 5's
Track D requires, which are a one-off extraction over 973 records.

Track D is the only step needing new embedding extraction, and it is last — so
Steps 1-4, the highest-expected-value work, can complete with zero new
extraction.

---

## 7. Status

**COMPLETE. The target was not reached. The gate was not cleared, so the test set
was never touched.** Full write-up: `PHASE9_RESULTS.md`. Figure:
`figures/21_phase9_summary.png`.

| | Value |
|---|---|
| Best assembled configuration, CV accuracy | **0.6744** (95% CI 0.6396–0.7066) |
| Gate required | 0.72, lower CI bound ≥ 0.68 |
| Gate outcome | **STOP** |
| Test evaluations performed | **0** |
| Total gain over the honest baseline | +0.0090 accuracy, +0.0136 macro-F1 |

**How the forecasts in Section 2 held up.** Recorded here because the whole point
of writing expected values down in advance is to check them afterwards.

| Track | Item | Forecast | Actual | Verdict |
|---|---|---|---|---|
| A1 | Refit on all 777 | +1 to +3 | not exercised | gate not cleared; cannot show in CV |
| A2 | External class-0 negatives | +1 to +4 | **−0.4, and −11 points on class-0 F1** | badly wrong; it harmed the target class |
| B1 | Soft averaging | +0.5 to +2 | +0.4 to +0.6 | low end |
| B2 | Out-of-fold stacking | +1 to +3 | **−0.4 to −2.7** | wrong sign |
| B3 | 5 seeds per architecture | +0.5 to +1.5 | +0.6 | in range |
| C1 | CORN ordinal loss | 0 to +2 | 0.0, macro-F1 worse | bottom of range |
| C2 | Class-0 threshold calibration | +1 to +3 | 0.0 accuracy, +1.7 macro-F1 | below range |
| D | Structured reasoning streams | −1 to +3 | **−1.7 to −2.2** | bottom of range, as the risk note warned |

Six of eight forecasts came in at or below the bottom of their range, and two had
the wrong sign. The plan's own aggregate estimate of "+3 to +6 points, landing in
68–72%" was **substantially too optimistic**; the real figure was +0.9 points.
That miss is itself worth recording: on a dataset this small, plausible
mechanistic reasoning about what *should* help is a poor predictor of what does.

**The one forecast that was right** was Section 0.2's diagnosis that the problem
is generalization and data rather than model capacity. Step 1 measured selection
optimism at +0.027 accuracy, and Step 3's learning curve measured +0.030 accuracy
per +100 training examples with no sign of flattening. That is the finding Phase 9
actually delivered.

**Recommendation on file:** stop pursuing architecture, loss-function and combiner
work on this dataset. See `PHASE9_RESULTS.md` Section 8.
