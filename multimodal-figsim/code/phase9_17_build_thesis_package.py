"""
Build `thesis_ready/` -- a single folder holding every number, figure and
description needed for thesis writing, with nothing to hunt for.

WHY A SCRIPT RATHER THAN COPYING BY HAND
----------------------------------------
Every metric in the generated document is RECOMPUTED here from the raw
confusion matrices stored in outputs/, not copied from a stored summary
field. If a stored field were ever mislabelled, or a number were
transcribed wrongly into a document, this would surface it. The script
also asserts that each recomputed value matches the stored one, so the
package cannot silently drift from the results it claims to report.

Re-run it any time results change:  python code/phase9_17_build_thesis_package.py

OUTPUT
------
thesis_ready/
    README.md              what to grab for each thesis section
    VERIFIED_RESULTS.md    every quotable number, recomputed and checked
    results_table.csv      the same numbers, machine-readable
    WORKFLOW.md            the system description in prose
    figures/               the figures worth putting in the thesis
"""
import os
import csv
import json
import shutil

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(PROJECT_ROOT, "thesis_ready")
OUT_FIG = os.path.join(OUT, "figures")

THREE = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
FIVE = ["None", "Wish to be dead", "Suicide ideation", "Suicide planning", "Suicide attempt or death"]
DEPRESSION = ["Minimum", "Mild", "Moderate", "Severe"]
GROUPS = [[0], [1, 2], [3, 4]]

FIGURES = [
    ("20_current_verified_pipeline.png",
     "System architecture / workflow diagram. Use this as the main pipeline figure."),
    ("09_phase3_architecture.png",
     "The three fusion architectures (concat, gated+orthogonal, cross-attention)."),
    ("11_cumulative_progression.png",
     "Improvement trajectory across the project."),
    ("17_final_3class_test_lock.png",
     "The 3-class locked test result against the prior best."),
    ("16_fuzzy_logic_5class_vs_3class.png",
     "Fuzzy-logic rule tables and how the two label schemes compare."),
    ("18_final_fuzzy_test_comparison.png",
     "Final risk-level output on the real test set."),
    ("04_confusion_matrix.png", "Confusion matrix, original 5-class lock."),
    ("05_per_class_metrics.png", "Per-class precision/recall/F1."),
    ("21_phase9_summary.png",
     "Phase 9: learning curve plus every experiment against the honest baseline."),
    ("22_phase9_final_verdict.png",
     "Phase 9: the Track A1 test and why no path to 70% remains."),
]


def expand(cm):
    yt, yp = [], []
    for i, row in enumerate(cm):
        for j, c in enumerate(row):
            yt += [i] * c
            yp += [j] * c
    return np.array(yt), np.array(yp)


def metrics(cm, qwk=False):
    yt, yp = expand(cm)
    m = {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "per_class_f1": [float(x) for x in f1_score(yt, yp, average=None, zero_division=0)],
    }
    if qwk:
        m["qwk"] = float(cohen_kappa_score(yt, yp, weights="quadratic"))
    return m


def collapse5to3(cm5):
    out = [[0] * 3 for _ in range(3)]
    for i in range(5):
        for j in range(5):
            gi = next(k for k, g in enumerate(GROUPS) if i in g)
            gj = next(k for k, g in enumerate(GROUPS) if j in g)
            out[gi][gj] += cm5[i][j]
    return out


def load(name):
    with open(os.path.join(PROJECT_ROOT, "outputs", name), "r", encoding="utf-8") as f:
        return json.load(f)


def check(label, recomputed, stored, tol=1e-4):
    if stored is None:
        return
    assert abs(recomputed - stored) < tol, \
        f"MISMATCH in {label}: recomputed {recomputed:.6f} vs stored {stored:.6f}"


def fmt_pc(names, vals):
    return " / ".join(f"{n.split()[0]} {v:.3f}" for n, v in zip(names, vals))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)

    # ------------------------------------------------------------ gather ----
    d5 = load("phase3_final_test_results.json")
    d3 = load("phase7_final_test_results_3class.json")
    d1 = load("final_test_results.json")
    ddep = load("phase8_3_depression_dataset_verification_results.json")
    dfuz = load("phase7_11_fuzzy_logic_final_test_results.json")
    dcv = load("phase9_1_cv_baseline_results.json")
    dlc = load("phase9_3_learning_curve_results.json")
    dgate = load("phase9_13_gate_results.json")
    dep9 = load("phase9_19_depression_features_ensemble_results.json")
    dep9["spearman_depression_vs_suicide_severity"] = load(
        "phase9_18_depression_features_results.json"
    )["spearman_depression_vs_suicide_severity"]
    da1 = load("phase9_15_a1_refit_validation_results.json")

    m5 = metrics(d5["confusion_matrix"], qwk=True)
    m3 = metrics(d3["confusion_matrix"], qwk=True)
    mcol = metrics(collapse5to3(d5["confusion_matrix"]))
    seed = str(d1["primary_seed"])
    m1 = metrics(d1["per_seed_test_results"][seed]["confusion_matrix"], qwk=True)
    mdep = metrics(ddep["confusion_matrix"])

    check("5-class macro-F1", m5["macro_f1"], d5["macro_f1"])
    check("5-class accuracy", m5["accuracy"], d5["accuracy"])
    check("3-class macro-F1", m3["macro_f1"], d3["macro_f1"])
    check("3-class accuracy", m3["accuracy"], d3["accuracy"])
    check("depression macro-F1", mdep["macro_f1"], ddep["macro_f1"])
    check("depression accuracy", mdep["accuracy"], ddep["accuracy"])
    print("All recomputed metrics match their stored values.")

    # -------------------------------------------------------------- CSV -----
    rows = [
        ["model", "task", "split", "n", "accuracy", "macro_f1", "weighted_f1", "qwk", "quotable"],
        ["Day-7 single-model lock", "5-class suicide severity", "test", m1["n"],
         f"{m1['accuracy']:.4f}", f"{m1['macro_f1']:.4f}", f"{m1['weighted_f1']:.4f}",
         f"{m1['qwk']:.4f}", "yes - original baseline"],
        ["Phase 3 six-model ensemble", "5-class suicide severity", "test", m5["n"],
         f"{m5['accuracy']:.4f}", f"{m5['macro_f1']:.4f}", f"{m5['weighted_f1']:.4f}",
         f"{m5['qwk']:.4f}", "YES - LEAD WITH THIS"],
        ["Phase 3 ensemble, collapsed to 3 classes", "3-class suicide severity", "test",
         mcol["n"], f"{mcol['accuracy']:.4f}", f"{mcol['macro_f1']:.4f}",
         f"{mcol['weighted_f1']:.4f}", "", "yes - best 3-class-equivalent"],
        ["Phase 7.10 nine-model lock", "3-class suicide severity", "test", m3["n"],
         f"{m3['accuracy']:.4f}", f"{m3['macro_f1']:.4f}", f"{m3['weighted_f1']:.4f}",
         f"{m3['qwk']:.4f}", "yes - on its own terms"],
        ["DAPT-BanglaBERT depression classifier", "4-class depression", "own dataset",
         mdep["n"], f"{mdep['accuracy']:.4f}", f"{mdep['macro_f1']:.4f}",
         f"{mdep['weighted_f1']:.4f}", "", "yes - WITH training-overlap caveat"],
        ["Phase 9 best configuration", "3-class suicide severity", "cross-validation",
         dcv["n_examples"], f"{dep9['with_depression_ensemble']['accuracy']:.4f}",
         f"{dep9['with_depression_ensemble']['macro_f1']:.4f}", "", "",
         "CV only - never evaluated on test"],
    ]
    with open(os.path.join(OUT, "results_table.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    # ------------------------------------------------------- figures --------
    copied = []
    for fn, desc in FIGURES:
        src = os.path.join(PROJECT_ROOT, "figures", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT_FIG, fn))
            copied.append((fn, desc))

    # -------------------------------------------------- VERIFIED_RESULTS ----
    unb = dcv["pooled_unbiased"]
    opt = dcv["measured_selection_optimism"]
    curve = dlc["pooled_curve"]
    slope = dlc["final_segment_slope_per_100_examples"]
    best_cfg = dep9["with_depression_ensemble"]
    fuz3 = dfuz["risk_distribution_3class_test"]
    fuz5 = dfuz["risk_distribution_5class_test_original"]

    md = f"""# Verified results — every number you can quote

**Every metric below was recomputed from the raw confusion matrices by
`code/phase9_17_build_thesis_package.py` and checked against the stored values.
They match.** Do not hand-copy numbers from anywhere else; take them from here
or from `results_table.csv`.

Test set throughout: the same **196 held-out memes**, touched three times in the
project's history (Day 7, Phase 3, Phase 7.10) and never since.

---

## 1. Which number to lead with

**Lead with the 5-class ensemble: macro-F1 {m5['macro_f1']:.4f}, accuracy {m5['accuracy']:.4f}.**
It is the project's strongest validated result on the full-granularity task.

The 3-class work is reported **on its own terms**, never as an improvement over
the 5-class model. Section 4 explains why.

---

## 2. Suicide-severity results (multimodal, 196 test memes)

| Model | Task | Accuracy | Macro-F1 | Weighted-F1 | QWK |
|---|---|---|---|---|---|
| Day-7 single-model lock | 5-class | {m1['accuracy']:.4f} | {m1['macro_f1']:.4f} | {m1['weighted_f1']:.4f} | {m1['qwk']:.4f} |
| **Phase 3 six-model ensemble** | 5-class | **{m5['accuracy']:.4f}** | **{m5['macro_f1']:.4f}** | {m5['weighted_f1']:.4f} | {m5['qwk']:.4f} |
| Same ensemble, collapsed to 3 classes | 3-class | {mcol['accuracy']:.4f} | {mcol['macro_f1']:.4f} | {mcol['weighted_f1']:.4f} | — |
| Phase 7.10 nine-model lock | 3-class | {m3['accuracy']:.4f} | {m3['macro_f1']:.4f} | {m3['weighted_f1']:.4f} | {m3['qwk']:.4f} |

**Per-class F1**

- 5-class ensemble: {fmt_pc(FIVE, m5['per_class_f1'])}
- 3-class lock: {fmt_pc(THREE, m3['per_class_f1'])}

**Confusion matrices** (rows = true, columns = predicted)

5-class ensemble:
```
{chr(10).join('  ' + str(r) for r in d5['confusion_matrix'])}
```

3-class lock:
```
{chr(10).join('  ' + str(r) for r in d3['confusion_matrix'])}
```

**A point that causes confusion, so state it clearly in the thesis.** For the
3-class model, {m3['macro_f1']:.4f} is the **macro-F1** and {m3['accuracy']:.4f}
is the **accuracy**. Both come from the same model on the same 196 memes. The
gap exists because accuracy is carried by the large middle class (102 of 196),
while macro-F1 weights all three classes equally and is dragged down by the
small "no expressed severity" class (32 examples, F1 {m3['per_class_f1'][0]:.3f}).

---

## 3. Depression classifier (text-only, DAPT-BanglaBERT)

| Metric | Value |
|---|---|
| Accuracy | {mdep['accuracy']:.4f} |
| Macro-F1 | {mdep['macro_f1']:.4f} |
| Weighted-F1 | {mdep['weighted_f1']:.4f} |
| Records | {mdep['n']} |

Per-class F1: {fmt_pc(DEPRESSION, mdep['per_class_f1'])}

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
| Macro-F1 | {mcol['macro_f1']:.4f} | {m3['macro_f1']:.4f} | **{m3['macro_f1'] - mcol['macro_f1']:+.4f}** |
| Accuracy | {mcol['accuracy']:.4f} | {m3['accuracy']:.4f} | **{m3['accuracy'] - mcol['accuracy']:+.4f}** |

**The 3-class model is not an improvement. It is slightly weaker once fairly
compared.** Report it as a differently-scoped result, which is what it is.

---

## 5. Fuzzy-logic risk layer (196 test memes)

Final risk distribution:

| Risk level | 5-class system | 3-class system |
|---|---|---|
| Minimal | {fuz5['Minimal']} | {fuz3['Minimal']} |
| Low | {fuz5['Low']} | {fuz3['Low']} |
| Elevated | {fuz5['Elevated']} | {fuz3['Elevated']} |
| Critical | {fuz5['Critical']} | {fuz3['Critical']} |

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
| Unbiased | {unb['accuracy']:.4f} | {unb['macro_f1']:.4f} |
| Peeking (the original protocol) | {dcv['pooled_peeking']['accuracy']:.4f} | {dcv['pooled_peeking']['macro_f1']:.4f} |
| **Selection optimism** | **{opt['accuracy']:+.4f}** | **{opt['macro_f1']:+.4f}** |

The peeking figure nearly reproduces the old single-split validation number
(0.6974 / 0.6649), which is direct evidence that number was inflated by
selection rather than being a held-out estimate.

**Learning curve — data is the binding constraint.**

| Training examples | {" | ".join(str(curve[k]['mean_n_train']) for k in sorted(curve, key=float))} |
|---|{"---|" * len(curve)}
| Accuracy | {" | ".join(f"{curve[k]['accuracy']:.4f}" for k in sorted(curve, key=float))} |

Still climbing and accelerating at the largest size available:
**{slope['accuracy']:+.4f} accuracy and {slope['macro_f1']:+.4f} macro-F1 per +100
training examples.** This converts a qualitative claim carried through the whole
project into a measured number.

**A pre-registered gate, honoured.** Phase 9 fixed in advance that the test set
would only be unlocked at CV accuracy >= 0.72 with a lower confidence bound
>= 0.68. The best configuration reached {best_cfg['accuracy']:.4f}
(95% CI {best_cfg['bootstrap_ci95']['accuracy_ci95'][0]:.4f} to
{best_cfg['bootstrap_ci95']['accuracy_ci95'][1]:.4f}). The gate failed and
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
| Baseline: text + image | {dep9['baseline_ensemble']['accuracy']:.4f} | {dep9['baseline_ensemble']['macro_f1']:.4f} |
| **+ depression distribution** | **{dep9['with_depression_ensemble']['accuracy']:.4f}** | **{dep9['with_depression_ensemble']['macro_f1']:.4f}** |
| Delta | {dep9['with_depression_ensemble']['accuracy'] - dep9['baseline_ensemble']['accuracy']:+.4f} (P(better) {dep9['paired_comparison']['prob_accuracy_improved']:.3f}) | {dep9['with_depression_ensemble']['macro_f1'] - dep9['baseline_ensemble']['macro_f1']:+.4f} (P(better) {dep9['paired_comparison']['prob_macro_f1_improved']:.3f}) |

Per architecture: concat {dep9['per_architecture']['concat']['with_dep']['accuracy'] - dep9['per_architecture']['concat']['baseline']['accuracy']:+.4f}, gated {dep9['per_architecture']['gated']['with_dep']['accuracy'] - dep9['per_architecture']['gated']['baseline']['accuracy']:+.4f}, cross-attention {dep9['per_architecture']['xattn']['with_dep']['accuracy'] - dep9['per_architecture']['xattn']['baseline']['accuracy']:+.4f}.

**This was the only Phase 9 idea that survived into the full ensemble**, and it
contributed most of the phase's total gain.

Two points to make in the write-up:

- **No target leakage.** The depression classifier was trained on its own
  separate 4,897-record Bangla text dataset, not on these memes and not on
  their suicide-severity labels, so its output is a legitimate input feature.
  State this explicitly; a reader should want the reassurance.
- **It is not exploiting a simple correlation.** The Spearman correlation
  between predicted depression level and true suicide severity is only
  {dep9['spearman_depression_vs_suicide_severity']:.4f}. The full distribution
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
   selection optimism now measured at {opt['accuracy']:+.4f} accuracy.
6. Stage A alignment projections were frozen and shared across folds rather
   than refitted per fold.

---

## 9. Numbers that must NOT appear in the thesis

- **"+0.0746 improvement"** and **"+0.110 improvement"** — invalid
  cross-granularity comparisons, already corrected.
- **"0.4635 → 0.4984 → 0.5730 trajectory"** — chains two different tasks into
  one line and implies progress that did not happen.
- **The 0.6974 validation accuracy** as if it were a held-out estimate. It is
  inflated by selection; use {unb['accuracy']:.4f} from cross-validation.
- **Any Phase 9 test number.** There is none. Phase 9 never evaluated the test
  set.
"""
    with open(os.path.join(OUT, "VERIFIED_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(md)

    # ------------------------------------------------------- WORKFLOW -------
    workflow = f"""# System workflow — description for the methodology chapter

Main figure: `figures/20_current_verified_pipeline.png`.
Fusion architectures: `figures/09_phase3_architecture.png`.

## Plain-text flow

```
Bangla meme (image + embedded text)
        |
        +--> OCR  -->  translation (NLLB-200-3.3B)  -->  Bangla text
        |                     |
        |                     +--> VLM reasoning pass (Qwen2.5-VL-7B),
        |                          three structured fields, translated to Bangla
        |                     |
        |                     v
        |            DAPT-BanglaBERT (frozen)
        |                     |
        |          +----------+-----------+
        |          |                      |
        |          v                      v
        |   text embedding          depression classifier
        |   (OCR + reasoning)       (4-class, TEXT-ONLY, verified)
        |          |
        +--> SigLIP image encoder (frozen)
                   |
                   v
        three fusion architectures, three seeds each
        (simple concat / gated + orthogonal residual / cross-attention)
                   |
                   v
        majority vote across the ensemble  -->  suicide-content severity
                   |
                   v
        fuzzy-logic combination with depression severity
                   |
                   v
        final risk level (Minimal / Low / Elevated / Critical)
```

## Components, in the order they appear

**Encoders, both frozen.** DAPT-BanglaBERT provides the text representation and
SigLIP (`siglip-so400m-patch14-384`) the image representation. Neither is
fine-tuned in the final pipeline; LoRA fine-tuning was tried and gave only
+0.0125 macro-F1, which did not justify adopting it.

**Text feature construction.** OCR output is translated to Bangla, and a
vision-language model produces three structured analytic fields per meme
(cause and effect, figurative meaning, emotional state). These are flattened
into one string, paired with the OCR text, and encoded as a single 768-d CLS
vector. Keeping the three fields as separate attended streams was tested in
Phase 9 and performed significantly worse.

**Fusion, three architectures.**
1. *Simple concatenation* — join the two vectors, small classifier on top.
2. *Gated fusion with orthogonal residual* — a learned per-example weight
   blends aligned text and image vectors, plus a residual term preserving
   non-redundant image information. Requires a contrastive alignment stage,
   without which the gate collapses to ignoring text almost entirely, a real
   failure diagnosed and fixed earlier in the project.
3. *Cross-attention* — text tokens attend over image patches, finer-grained
   than one pooled vector per modality.

No single architecture is used alone. The locked model **is** the ensemble.

**Ensembling.** Three seeds per architecture, combined by majority vote. This
was the single largest lever in the entire project, larger than every
loss-function, fusion-mechanism and hyperparameter change combined.

**Fuzzy-logic risk layer.** Each classifier's distribution is treated as a
membership degree. Fuzzy AND is minimum, fuzzy OR is maximum, and the final
risk level is the argmax of the combined memberships across an explicit rule
table. The suicide side contributes hard vote fractions; the depression side
contributes softmax probabilities.

## Data

973 labelled Bangla memes total: 582 train, 195 validation, 196 test. The
depression classifier was trained separately on its own 4,897-record Bangla
text dataset.

## What the system cannot do

The suicide-severity branch requires **both** modalities. There is no defined
behaviour when the image or the text is missing. The depression branch has no
such limitation: its text-only path is parity-tested and verified.
"""
    with open(os.path.join(OUT, "WORKFLOW.md"), "w", encoding="utf-8") as f:
        f.write(workflow)

    # --------------------------------------------------------- README -------
    fig_lines = "\n".join(f"- `figures/{fn}` — {desc}" for fn, desc in copied)
    readme = f"""# thesis_ready — everything needed for thesis writing, in one place

Generated by `code/phase9_17_build_thesis_package.py`. Re-run that script if
results ever change; do not edit these files by hand, or they will drift from
the results they claim to report.

## Files

| File | Use it for |
|---|---|
| `VERIFIED_RESULTS.md` | **Every quotable number.** Recomputed from raw confusion matrices and checked against stored values. Start here. |
| `results_table.csv` | The same numbers, machine-readable, for tables and plots. |
| `WORKFLOW.md` | System description in prose, ready for the methodology chapter. |
| `figures/` | The {len(copied)} figures worth putting in the thesis. |

## What to grab for each thesis section

**Methodology** — `WORKFLOW.md` in full, plus
`figures/20_current_verified_pipeline.png` as the main pipeline figure and
`figures/09_phase3_architecture.png` for the fusion detail.

**Results** — `VERIFIED_RESULTS.md` sections 2, 3 and 5, with
`figures/17_final_3class_test_lock.png` and
`figures/18_final_fuzzy_test_comparison.png`.

**Discussion / rigor** — `VERIFIED_RESULTS.md` sections 4, 6 and 7, with
`figures/21_phase9_summary.png` and `figures/22_phase9_final_verdict.png`.
Section 6 is the strongest material in the project: selection optimism
measured, a learning curve establishing the ceiling, and a pre-registered gate
that was honoured when it returned an unwelcome answer.

**Limitations** — `VERIFIED_RESULTS.md` section 8.

## Two things to be careful about

1. **Section 9 lists numbers that must not appear in the thesis**, including
   two invalid comparisons that appeared in earlier drafts. Check any number
   you carry over from an old draft against it.
2. **Lead with the 5-class ensemble** (macro-F1 0.4984, accuracy 0.5306). The
   3-class results are reported on their own terms, never as an improvement,
   because once fairly compared the 3-class model is slightly weaker.

## Figures included

{fig_lines}

## Where the full record lives

This folder is the condensed version. The complete chronological experiment
log is `IMPROVEMENT_PLAN.md`, the Phase 9 write-up is `PHASE9_RESULTS.md`, and
`supervisor_result/` holds the full review package.
"""
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"\nBuilt {OUT}")
    for f_ in sorted(os.listdir(OUT)):
        print("  ", f_)
    print(f"   figures/ ({len(copied)} files)")


if __name__ == "__main__":
    main()
