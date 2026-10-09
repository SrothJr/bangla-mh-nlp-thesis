# Verified results — every number you can quote

**Every metric below was recomputed from the raw confusion matrices by
`code/phase9_17_build_thesis_package.py` and checked against the stored values.
They match.** Do not hand-copy numbers from anywhere else; take them from here
or from `results_table.csv`.

Test set throughout: the same **196 held-out memes**, touched three times in the
project's history (Day 7, Phase 3, Phase 7.10) and never since.

---

## 1. Which number to lead with

**Lead with the 5-class ensemble: macro-F1 0.4984, accuracy 0.5306.**
It is the project's strongest validated result on the full-granularity task.

The 3-class work is reported **on its own terms**, never as an improvement over
the 5-class model. Section 4 explains why.

---

## 2. Suicide-severity results (multimodal, 196 test memes)

| Model | Task | Accuracy | Macro-F1 | Weighted-F1 | QWK |
|---|---|---|---|---|---|
| Day-7 single-model lock | 5-class | 0.5051 | 0.4753 | 0.4953 | 0.3732 |
| **Phase 3 six-model ensemble** | 5-class | **0.5306** | **0.4984** | 0.5232 | 0.3857 |
| Same ensemble, collapsed to 3 classes | 3-class | 0.6582 | 0.6024 | 0.6532 | — |
| Phase 7.10 nine-model lock | 3-class | 0.6071 | 0.5730 | 0.6126 | 0.3765 |

**Per-class F1**

- 5-class ensemble: None 0.476 / Wish 0.659 / Suicide 0.603 / Suicide 0.348 / Suicide 0.406
- 3-class lock: No 0.462 / Suicidal 0.683 / High 0.574

**Confusion matrices** (rows = true, columns = predicted)

5-class ensemble:
```
  [15, 5, 6, 2, 4]
  [4, 29, 3, 1, 1]
  [3, 11, 38, 4, 8]
  [1, 2, 9, 8, 6]
  [8, 3, 6, 5, 14]
```

3-class lock:
```
  [18, 11, 3]
  [17, 68, 17]
  [11, 18, 33]
```

**A point that causes confusion, so state it clearly in the thesis.** For the
3-class model, 0.5730 is the **macro-F1** and 0.6071
is the **accuracy**. Both come from the same model on the same 196 memes. The
gap exists because accuracy is carried by the large middle class (102 of 196),
while macro-F1 weights all three classes equally and is dragged down by the
small "no expressed severity" class (32 examples, F1 0.462).

---

## 3. Depression classifier (text-only, DAPT-BanglaBERT)

| Metric | Value |
|---|---|
| Accuracy | 0.9320 |
| Macro-F1 | 0.9182 |
| Weighted-F1 | 0.9308 |
| Records | 4897 |

Per-class F1: Minimum 0.963 / Mild 0.924 / Moderate 0.832 / Severe 0.954

**Mandatory caveat — always quote this alongside the number.** This checkpoint
was trained on roughly four fifths of this exact dataset. The figure is a
behaviour sanity check against its own training domain, **not** an unbiased
held-out generalization estimate. It confirms the model works correctly; it
must not be presented as a fresh accuracy result.

Separately, the classifier's text-only path was **parity-tested**: 100.00% label
agreement and 0.00 maximum probability difference against 196 previously saved
outputs. That is an exact match, and it is a clean claim to make.

---

## 4. The comparison rule you must not break

**Never subtract a 3-class score from a 5-class score.** A coarser task is
easier to score well on regardless of whether the model improved. An earlier
draft claimed a "+0.0746 improvement" by doing exactly this, and it was wrong.

The valid method is to collapse the finer model's own predictions into the
coarser grouping and compare on identical records, with no retraining:

| Comparison | Collapsed 5-class | Native 3-class | Valid delta |
|---|---|---|---|
| Macro-F1 | 0.6024 | 0.5730 | **-0.0294** |
| Accuracy | 0.6582 | 0.6071 | **-0.0510** |

**The 3-class model is not an improvement. It is slightly weaker once fairly
compared.** Report it as a differently-scoped result, which is what it is.

---

## 5. Fuzzy-logic risk layer (196 test memes)

Final risk distribution:

| Risk level | 5-class system | 3-class system |
|---|---|---|
| Minimal | 26 | 45 |
| Low | 64 | 49 |
| Elevated | 61 | 49 |
| Critical | 45 | 53 |

Mechanism: fuzzy AND = minimum, fuzzy OR = maximum, final level = argmax of the
combined memberships. The suicide-side membership vector is **hard vote
fractions** from the ensemble (confirmed from code), while the depression side
uses genuine softmax probabilities.

**No ground-truth combined risk label exists for any meme**, so this is a
distributional and qualitative comparison, never an accuracy claim.

---

## 6. Phase 9 methodological findings (strong thesis material)

These are measurements, not performance gains, and they are arguably the most
defensible content in the project.

**Selection optimism, quantified.** The original protocol early-stops by picking
the epoch with the best score *on the split it then reports*. Re-running the
identical configuration with a proper inner split for early stopping:

| | Accuracy | Macro-F1 |
|---|---|---|
| Unbiased | 0.6654 | 0.6155 |
| Peeking (the original protocol) | 0.6924 | 0.6557 |
| **Selection optimism** | **+0.0270** | **+0.0402** |

The peeking figure nearly reproduces the old single-split validation number
(0.6974 / 0.6649), which is direct evidence that number was inflated by
selection rather than being a held-out estimate.

**Learning curve — data is the binding constraint.**

| Training examples | 210 | 315 | 421 | 527 |
|---|---|---|---|---|
| Accuracy | 0.6126 | 0.6178 | 0.6332 | 0.6654 |

Still climbing and accelerating at the largest size available:
**+0.0304 accuracy and +0.0395 macro-F1 per +100
training examples.** This converts a qualitative claim carried through the whole
project into a measured number.

**A pre-registered gate, honoured.** Phase 9 fixed in advance that the test set
would only be unlocked at CV accuracy >= 0.72 with a lower confidence bound
>= 0.68. The best configuration reached 0.6873
(95% CI 0.6538 to
0.7207). The gate failed and
**the test set was not touched**.

### 6.1 The one change that worked — deeper integration of the two components

Worth its own thesis subsection, because it is a design result rather than a
tuning result.

The depression classifier and the suicide-severity model originally met only in
the fuzzy-logic layer, after both had already committed to a decision. Feeding
the depression classifier's 4-class output distribution into the suicide model
as **input features** instead gives:

| Configuration (15-model ensemble, cross-validated) | Accuracy | Macro-F1 |
|---|---|---|
| Baseline: text + image | 0.6744 | 0.6291 |
| **+ depression distribution** | **0.6873** | **0.6397** |
| Delta | +0.0129 (P(better) 0.922) | +0.0106 (P(better) 0.853) |

Per architecture: concat +0.0129, gated +0.0013, cross-attention -0.0077.

**This was the only Phase 9 idea that survived into the full ensemble**, and it
contributed most of the phase's total gain.

Two points to make in the write-up:

- **No target leakage.** The depression classifier was trained on its own
  separate 4,897-record Bangla text dataset, not on these memes and not on
  their suicide-severity labels, so its output is a legitimate input feature.
  State this explicitly; a reader should want the reassurance.
- **It is not exploiting a simple correlation.** The Spearman correlation
  between predicted depression level and true suicide severity is only
  0.0897. The full distribution
  therefore carries information that its argmax does not.

The depression checkpoint was never retrained or modified. It was called through
the Phase 8 parity-tested wrapper, and only the small downstream fusion heads
changed by taking four extra input dimensions.

---

## 7. Ideas tested and rejected (use for the "what we tried" section)

Each was A/B'd under identical folds and seeds, never bundled.

| Idea | Result | Verdict |
|---|---|---|
| Stacking meta-learner | −0.0039 accuracy | rejected |
| Stacking, class-balanced | −0.0270 | rejected |
| CORN cumulative-link ordinal loss | ±0.0000, macro-F1 worse | rejected |
| Structured figurative-reasoning streams | significantly worse | rejected |
| External out-of-domain negatives | class-0 F1 −0.1127 | rejected, harmful |
| Full-data refit with fixed epochs | −0.0103 | rejected |
| Train on 5 classes, collapse to 3 | +0.0039, P(better) 0.599 | not significant |
| Soft probability averaging | +0.0039 | marginal, kept |
| Five seeds per architecture | +0.0064, P(better) 0.834 | kept |

Two findings worth a sentence each in the discussion:

- **Out-of-domain negatives do not substitute for in-domain data.** Adding 300
  screened Bangla troll memes as extra "no severity" examples made that class
  significantly *worse*, because even the safest of them sat outside the
  model's distribution.
- **Losing the early-stopping signal costs more than 18% extra data gains.**
  The best epoch varies enormously between seeds, so a fixed schedule suits
  almost none of them.

---

## 8. Limitations to state in the thesis

1. Test set is 196 examples, so every test metric carries roughly ±7 points of
   95% confidence interval. Two results within that range are not separated.
2. The depression classifier's ~93% figure overlaps its own training data.
3. No defined behaviour when a modality is missing on the suicide-severity side.
4. The fuzzy layer is exploratory and rule-based, not clinically validated, and
   has no ground-truth label to check against.
5. Decisions in Phases 1-8 were made on one 195-example validation split, with
   selection optimism now measured at +0.0270 accuracy.
6. Stage A alignment projections were frozen and shared across folds rather
   than refitted per fold.

---

## 9. Numbers that must NOT appear in the thesis

- **"+0.0746 improvement"** and **"+0.110 improvement"** — invalid
  cross-granularity comparisons, already corrected.
- **"0.4635 → 0.4984 → 0.5730 trajectory"** — chains two different tasks into
  one line and implies progress that did not happen.
- **The 0.6974 validation accuracy** as if it were a held-out estimate. It is
  inflated by selection; use 0.6654 from cross-validation.
- **Any Phase 9 test number.** There is none. Phase 9 never evaluated the test
  set.
