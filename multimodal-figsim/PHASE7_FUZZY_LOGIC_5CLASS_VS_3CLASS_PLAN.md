# Fuzzy-logic risk combination: 5-class vs. 3-class harmonized, side by side

**Status: planned, about to start.** This document is written before any
new code runs, per this project's established discipline.

## 1. Why this comparison

The fuzzy-logic risk-combination layer (Days 9-10) is the actual final
output of the whole system — it takes the FigSIM suicide-severity
prediction and Phase 2's depression-severity prediction (both from the
same meme's translated text) and produces one overall risk level
(Minimal / Low / Elevated / Critical) via an explicit rule table. So far
it has only ever been run once, on the locked 5-class model's test
predictions. Since Phase 7 has since shown the 3-class harmonized target
is a genuinely stronger severity classifier (0.643 vs. 0.564 macro-F1,
Section 2.6), it's worth checking whether that improvement actually
carries through to more useful *final risk output*, not just a better
intermediate classification number — that's the real question this
comparison answers.

## 2. What does NOT change

- Phase 2's depression classifier: identical, untouched, same checkpoint,
  same 4 classes (Minimum/Mild/Moderate/Severe).
- The fuzzy combination mechanism itself (fuzzy AND = min, fuzzy OR =
  max, "defuzzification" = argmax over aggregated risk membership) —
  unchanged, per `fuzzy_logic_layer.py`'s existing design.
- FigSIM's locked test set: **not touched by this comparison.** The
  3-class model has never been evaluated on test (it's still a
  validation-only, not-yet-locked candidate) — running it on test just
  for this comparison would break this project's test-set discipline.
  **Both the 5-class and 3-class runs in this comparison use FigSIM's
  validation split**, so the two are compared on equal footing. (The
  original Days 9-10 fuzzy-logic run used test, since the 5-class model
  was already locked at that point — this comparison is a separate,
  validation-only analysis alongside that, not a replacement for it.)

## 3. What's new: a 3-class fuzzy rule table

The existing rule table has one row per (suicide-severity level,
depression-level-group) pair, 11 rows total across the 5 suicide
classes. A 3-class version needs a **principled way to merge** the rows
belonging to classes that get grouped together, not an arbitrary
re-guess. The method used here: **for each merged bucket, at each
depression level, take the MORE severe (higher-priority) risk level
among the original rules being merged** — a safety-first / cautious
merge, consistent with the domain (better to over-flag risk than
under-flag it when uncertain).

**Class 0 "No expressed severity" = original "None"** — maps directly,
no merge needed:

| Depression level | Risk |
|---|---|
| Minimum, Mild | Minimal |
| Moderate | Low |
| Severe | Elevated |

**Class 1 "Suicidal thought or desire" = "Wish to be dead" + "Suicide ideation"** —
merged cautiously (worked example: at "Mild" depression, Wish alone was
Low but Ideation alone was Elevated — the merged bucket takes Elevated,
the more cautious of the two):

| Depression level | Wish alone | Ideation alone | Merged (cautious) |
|---|---|---|---|
| Minimum | Low | Low | **Low** |
| Mild | Low | Elevated | **Elevated** |
| Moderate | Elevated | Elevated | **Elevated** |
| Severe | Elevated | Elevated | **Elevated** |

**Class 2 "High acuity suicidal content" = "Suicide planning" + "Suicide attempt or death"** —
"Attempt or death" was already Critical at every depression level in the
original table, so the cautious merge pulls the entire bucket to
Critical regardless of depression level:

| Depression level | Planning alone | Attempt/Death alone | Merged (cautious) |
|---|---|---|---|
| Minimum, Mild | Elevated | Critical | **Critical** |
| Moderate, Severe | Critical | Critical | **Critical** |

**Resulting 3-class rule table** (6 rows, down from 11):

```python
RULES_3CLASS = [
    (2, [0, 1, 2, 3], "Critical"),   # High-acuity suicidal content, any depression
    (1, [1, 2, 3],    "Elevated"),   # Suicidal thought/desire, Mild/Moderate/Severe
    (1, [0],          "Low"),        # Suicidal thought/desire, Minimum
    (0, [3],          "Elevated"),   # No expressed severity, Severe
    (0, [2],          "Low"),        # No expressed severity, Moderate
    (0, [0, 1],        "Minimal"),   # No expressed severity, Minimum/Mild
]
```

**Honest caveat, stated plainly:** this derivation is a reasonable,
documented, cautious-merge policy — not a clinically re-validated scale.
It inherits whatever validity the original 11-row table had (itself
described in the codebase as a "draft rule table," never independently
validated against clinical ground truth) and adds one more layer of
judgment (the merge policy) on top. This is exactly the kind of
assumption that belongs in the thesis's limitations section, stated
this precisely.

## 4. Workflow, step by step

1. **Regenerate 5-class validation predictions with per-example
   probabilities saved** (not just aggregate metrics — needed as
   fuzzy-logic input). Reuses the existing best 5-class ensemble recipe
   unchanged.
2. **Regenerate 3-class validation predictions with per-example
   probabilities saved**, using the current best 3-class config (simple
   concat, original SigLIP — Phase 7 Step 4 variant A, 0.6425).
3. **Run Phase 2's depression classifier on the validation split's**
   translated OCR text (same mechanism as the original fuzzy-logic run,
   just pointed at validation indices instead of test).
4. **Combine via fuzzy logic**, once with the existing 11-row 5-class
   table, once with the new 6-row 3-class table above.
5. **Compare**: risk-level distributions side by side, plus a direct
   per-meme agreement analysis (for the same 195 validation memes, how
   often do the two schemes land on the same final risk level, and
   where they disagree, in which direction).
6. **Generate figures** comparing the two risk-level distributions.
7. **Document** findings, honestly, including the caveat in Section 3
   above, in `IMPROVEMENT_PLAN.md`.

## 5. What this comparison can and cannot tell us

It **can** tell us whether the 3-class model's higher raw macro-F1
translates into a meaningfully different (hopefully more decisive, less
uncertain) final risk-level output. It **cannot** tell us which risk
output is more *correct*, since — as already true of the original
fuzzy-logic layer — there is no ground-truth "risk level" label to
validate against; this stays a qualitative/distributional comparison,
not an accuracy comparison.
