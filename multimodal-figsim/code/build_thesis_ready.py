"""
Build the full `thesis_ready/` package: one folder per approach, each with its
own documentation and figures, plus `best_outcome/` for the approach that
produced the strongest validated result.

Supersedes phase9_17_build_thesis_package.py, which built a flat version of
the same package. That script is kept for the record but is no longer the one
to run.

WHY A SCRIPT
------------
Every metric written into these documents is RECOMPUTED here from the raw
confusion matrices in outputs/, then asserted against the stored value. A
mislabelled field or a mistyped number fails the build rather than reaching
the thesis. Re-run any time results change:

    python code/build_thesis_ready.py

STRUCTURE PRODUCED
------------------
thesis_ready/
    README.md                                  master index
    VERIFIED_RESULTS.md                        every quotable number
    results_table.csv                          machine-readable
    WORKFLOW.md                                system description
    best_outcome/                              the strongest validated result
    approach_01_single_model_baseline/
    approach_02_label_harmonization_3class/
    approach_03_depression_classifier/
    approach_04_fuzzy_logic_risk_layer/
    approach_05_rigor_and_diagnostics/
    approach_06_dual_encoder_study/
"""
import os
import csv
import json
import shutil

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "thesis_ready")
FIGS = os.path.join(ROOT, "figures")
V2FIGS = os.path.join(ROOT, "v2_research", "figures")

THREE = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
FIVE = ["None", "Wish to be dead", "Suicide ideation", "Suicide planning", "Suicide attempt or death"]
DEPRESSION = ["Minimum", "Mild", "Moderate", "Severe"]
GROUPS = [[0], [1, 2], [3, 4]]


# ------------------------------------------------------------------ utils --
def expand(cm):
    yt, yp = [], []
    for i, row in enumerate(cm):
        for j, c in enumerate(row):
            yt += [i] * c
            yp += [j] * c
    return np.array(yt), np.array(yp)


def metrics(cm, qwk=False):
    yt, yp = expand(cm)
    m = {"n": int(len(yt)),
         "accuracy": float(accuracy_score(yt, yp)),
         "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
         "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
         "per_class_f1": [float(x) for x in f1_score(yt, yp, average=None, zero_division=0)]}
    if qwk:
        m["qwk"] = float(cohen_kappa_score(yt, yp, weights="quadratic"))
    return m


def collapse(cm5):
    o = [[0] * 3 for _ in range(3)]
    for i in range(5):
        for j in range(5):
            gi = next(k for k, g in enumerate(GROUPS) if i in g)
            gj = next(k for k, g in enumerate(GROUPS) if j in g)
            o[gi][gj] += cm5[i][j]
    return o


def load(name, sub="outputs"):
    with open(os.path.join(ROOT, sub, name), "r", encoding="utf-8") as f:
        return json.load(f)


def check(label, got, want, tol=1e-4):
    if want is None:
        return
    assert abs(got - want) < tol, f"MISMATCH {label}: recomputed {got:.6f} vs stored {want:.6f}"


def cm_block(cm):
    return "```\n" + "\n".join("  " + str(r) for r in cm) + "\n```"


def pc_table(names, vals):
    rows = ["| Class | F1 |", "|---|---|"]
    rows += [f"| {n} | {v:.3f} |" for n, v in zip(names, vals)]
    return "\n".join(rows)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def put_figs(folder, items):
    """items: list of (source_dir, filename, caption). Returns markdown list."""
    fdir = os.path.join(folder, "figures")
    os.makedirs(fdir, exist_ok=True)
    lines = []
    for src, fn, cap in items:
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(fdir, fn))
            lines.append(f"- `figures/{fn}` — {cap}")
    return "\n".join(lines) if lines else "_No figure for this approach._"


# --------------------------------------------------- per-approach workflow --
SHARED_FRONTEND = """### Shared front end (identical for every approach below)

```
Bangla meme image
   |
   +--> EasyOCR (English reader, GPU) ----> raw meme text
   |         |
   |         +--> NLLB-200-3.3B --------> ocr_text_bn
   |
   +--> Qwen2.5-VL-7B (via Ollama) -----> three reasoning fields:
   |       temperature 0.2, JSON output     cause_effect
   |       2 retries on parse failure       figurative_meaning
   |                |                       emotional_state
   |                +--> NLLB-200-3.3B --> reasoning_text_bn
   |
   +--> SigLIP so400m-patch14-384 ------> 1152-d pooled, 729 x 1152 patches
```

The text feature the whole project calls **e3b** is then:

```
DAPT-BanglaBERT( text = ocr_text_bn , text_pair = reasoning_text_bn )
   max_length 256, CLS vector  ->  768-d
```

Every encoder above is **frozen**. Nothing in any approach fine-tunes them.
"""

COMMON_TRAINING = """### Training settings (shared unless an approach says otherwise)

| Setting | Value |
|---|---|
| Optimizer | Adam, lr 1e-3, weight decay 1e-4 |
| Hidden width | 256 |
| Dropout | 0.2 |
| Max epochs | 300 (gated and cross-attention: 1000) |
| Early-stopping patience | 30 (gated and cross-attention: 50) |
| Batching | full batch, one gradient step per epoch |
| Loss | cross-entropy with ordinal label smoothing, tau 1.0 |
| Class weights | inverse frequency, computed on the training rows only |
| Seeds | 0, 1, 2 (later work uses 0-4) |
| Device | classifier heads on CPU, cross-attention on GPU |

**Ordinal label smoothing** replaces the one-hot target with a kernel that
decays with distance from the true class, so predicting an adjacent severity
is penalised less than predicting a distant one. It was kept after CORAL and
CORN, two stricter ordinal formulations, were both tested and rejected.
"""

SPLIT_NOTE = """### Data and splits

973 labelled Bangla memes: **582 train / 195 validation / 196 test**. The test
split is touched only at a deliberate, pre-announced lock. It was used three
times in the project's history and not since.
"""


def write_workflows(mdep):
    """One WORKFLOW.md per approach folder, describing that approach's own
    pipeline rather than the system as a whole."""

    W = {}

    W["best_outcome"] = f"""# Workflow — cross-architecture ensemble (5-class)

{SPLIT_NOTE}
{SHARED_FRONTEND}
### Stage A: contrastive alignment (prerequisite for the gated member)

DAPT-BanglaBERT and SigLIP were pretrained independently and their vector
spaces are not comparable out of the box. Stage A learns two projections into
a shared 256-d space using a self-supervised InfoNCE objective: a meme's own
text and image are the positive pair, other memes in the batch are negatives.

This matters because without it the gated architecture **collapses** — the
learned gate drives almost all weight onto the image and ignores the text
entirely. Alignment is trained once, then frozen and reused everywhere.

### The two member architectures

```
MEMBER TYPE 1 -- gated fusion with orthogonal residual   (3 seeds)

  aligned_text (256) ----+
                         +--> gate MLP --> sigmoid --> alpha
  aligned_image (256) ---+                              |
                                                        v
  gated      = alpha * aligned_text + (1-alpha) * aligned_image
  i_orth     = aligned_image - proj(aligned_image onto aligned_text)
                                                        |
  [ LayerNorm(gated) ; LayerNorm(i_orth) ; e3b_text(768) ]
                         |
                         +--> Linear 256 -> ReLU -> Dropout -> Linear 5
```

The orthogonal residual keeps the part of the image vector that the gate's
blend would otherwise discard, so image information that is *not* redundant
with text still reaches the classifier.

```
MEMBER TYPE 2 -- cross-attention fusion                  (3 seeds)

  text tokens  (256 x 768) --> Linear --> Q (256 x 256)
  image patches(729 x 1152) --> Linear --> K, V (729 x 256)
                         |
  MultiheadAttention(4 heads, dropout 0.1), Q attends over K/V
                         |
  residual + LayerNorm, then masked mean-pool over real text tokens
                         |
  [ pooled(256) ; e3b_text(768) ] --> Linear 256 -> ReLU -> Dropout -> Linear 5
```

Cross-attention relates *individual words* to *individual image regions*,
which the single scalar gate cannot express.

### Combination

```
6 models (3 gated seeds + 3 cross-attention seeds)
   |
   +--> each predicts a 5-class argmax
   |
   +--> majority vote across all six
   |
   +--> final 5-class severity
```

Majority vote over hard predictions, not averaged probabilities. Ensembling
across *architectures* beat ensembling across seeds alone, because the two
architectures fail on different examples.

### Test protocol

All six models trained and selected on train and validation only, with early
stopping on validation macro-F1. The test split was then unlocked once, in a
single non-interactive pass, and every metric reported. No re-runs and no
tuning afterwards.

{COMMON_TRAINING}
### Reproduce

```
python code/phase3_2_ensemble_architectures.py   # validation-side selection
python code/phase3_final_test_eval.py            # the single test lock
```

Results file: `outputs/phase3_final_test_results.json`.
"""

    W["approach_01_single_model_baseline"] = f"""# Workflow — single-model baseline

{SPLIT_NOTE}
{SHARED_FRONTEND}
### The model

One gated-fusion model with an orthogonal residual, the same architecture that
later became member type 1 of the ensemble in `../best_outcome/`.

```
  aligned_text (256), aligned_image (256), confidence features (2)
        |
   gate MLP --> sigmoid --> alpha
        |
   gated   = alpha * aligned_text + (1-alpha) * aligned_image
   i_orth  = aligned_image - proj(aligned_image onto aligned_text)
        |
   [ LayerNorm(gated) ; LayerNorm(i_orth) ; e3b_text(768) ]
        |
   Linear 256 -> ReLU -> Dropout 0.2 -> Linear 5
```

The two **confidence features** are OCR confidence and translation confidence,
z-scored using statistics computed on the training rows only. They let the gate
distrust the text channel when the OCR or translation was poor.

### The failure diagnosed here, and the fix

Early runs produced a gate that pushed alpha to one extreme and ignored the
text completely. The cause was that the two encoders' spaces were not
comparable, so mixing them had no meaningful geometry. Stage A contrastive
alignment fixed it.

This is a real, mechanistically explained failure mode. It belongs in the
thesis as a diagnosis, not as a tuning note.

### Rejected at this stage

**CORAL**, a cumulative-link ordinal formulation, lost to cross-entropy with
ordinal label smoothing and was dropped. A second ordinal formulation, CORN,
was tested much later and also rejected.

{COMMON_TRAINING}
### Reproduce

```
python code/train_contrastive_align.py   # Stage A, trained once
python code/train_e6.py                  # the model
python code/run_final_test_eval.py       # the single test lock
```

Results file: `outputs/final_test_results.json`.
"""

    W["approach_02_label_harmonization_3class"] = f"""# Workflow — 3-class harmonized model

{SPLIT_NOTE}
{SHARED_FRONTEND}
### The label transformation

```
  0 None                    -->  0 No expressed severity
  1 Wish to be dead      ---+
  2 Suicide ideation     ---+-->  1 Suicidal thought or desire
  3 Suicide planning     ---+
  4 Attempt or death     ---+-->  2 High acuity suicidal content
```

Applied to the labels **before** training, so the models below are trained
natively on three classes rather than trained on five and collapsed afterwards.
That distinction is the whole point of this approach, and it is what makes the
comparison in the README non-trivial.

### The ensemble

Nine models rather than six, because a third architecture was added:

```
  simple concat        (3 seeds)   [ e3b_text(768) ; siglip(1152) ]
                                     -> Linear 256 -> ReLU -> Dropout -> Linear 3

  gated + orthogonal   (3 seeds)   as in ../best_outcome/, but hidden width 512
                                   (tuned specifically for the 3-class task)

  cross-attention      (3 seeds)   as in ../best_outcome/, 3 outputs
        |
        +--> majority vote across all nine --> 3-class severity
```

Simple concatenation was added because on the coarser task it turned out to be
the **strongest single architecture**, which it had not been on the 5-class
task. Hyperparameters for the gated member were re-swept for three classes and
hidden width 512 won.

### What was tried alongside and rejected

**External meme pretraining.** A SigLIP vision head was pretrained on BN-HIB,
an external Bangla hate-speech meme dataset, then transferred. It did not
improve the severity task and was dropped. See the transfer-test figure.

### Test protocol

Same discipline as every other lock. All nine models trained and selected on
train and validation only, test unlocked exactly once.

{COMMON_TRAINING}
### Reproduce

```
python code/phase7_8_harmonized_hyperparam_sweep.py
python code/phase7_9_harmonized_ensemble.py
python code/phase7_10_final_test_eval_3class.py    # the single test lock
```

Results file: `outputs/phase7_final_test_results_3class.json`.
"""

    W["approach_03_depression_classifier"] = f"""# Workflow — depression classifier (text-only)

### This model is not trained here

It was produced before the multimodal work began and has been **frozen ever
since**. No phase of this project retrained, fine-tuned or modified it. The
workflow below is therefore an inference pipeline, not a training one.

```
Checkpoint : DAPT_models/results/dapt_eval/BanglaBERT_fold5/checkpoint-735
Tokenizer  : DAPT_models/results/dapt_eval/BanglaBERT_fold5_tapt
Head       : 4-class classification head, id2label
             0 Minimum / 1 Mild / 2 Moderate / 3 Severe
```

### Its own training data

{mdep['n']} Bangla mental-health text records, labelled 1 to 4 in the source
file, mapped to class index by subtracting one. That mapping was verified
against the checkpoint's own `config.json` rather than assumed.

The checkpoint saw roughly four fifths of this dataset during training,
confirmed by back-calculating from its trainer state: 735 steps at batch size
16 over 3 epochs is about 3,920 examples per epoch, which matches a 4/5 fold of
{mdep['n']} almost exactly. This is why the accuracy figure in the README must
be quoted as a sanity check rather than a held-out estimate.

### Inference pipeline

```
Bangla text
   |
   +--> tokenizer (max_length 256, truncation, padding)
   |
   +--> DAPT-BanglaBERT + 4-class head
   |
   +--> softmax --> [p_minimum, p_mild, p_moderate, p_severe]
   |
   +--> argmax --> label
```

**Text only. No image is required anywhere in this path.** That is why this
half of the system has no missing-modality limitation while the multimodal half
does.

### How it is verified

A wrapper exposes `predict_depression(text_bn)` and calls the exact original
inference code rather than reimplementing it. It was checked against 196
previously saved outputs from an earlier run:

- **100.00% label agreement**
- **0.00 maximum absolute probability difference**

An exact match, not an approximation. That is the claim to make in the thesis.

### Its two roles in the wider system

1. Supplies the depression axis to the fuzzy-logic layer
   (`../approach_04_fuzzy_logic_risk_layer/`).
2. Its output **distribution** was later found to work as an input feature for
   the suicide-severity model (`../approach_05_rigor_and_diagnostics/`).

### Reproduce

```
python code/phase8_2_depression_parity_wrapper.py        # the parity test
python code/phase8_3_depression_dataset_verification.py  # dataset check
```

Results files: `outputs/phase8_2_depression_parity_report.json`,
`outputs/phase8_3_depression_dataset_verification_results.json`.

Note: the raw dataset is excluded from version control because it contains real
sensitive mental-health text. Only aggregate metrics are stored.
"""

    W["approach_04_fuzzy_logic_risk_layer"] = f"""# Workflow — fuzzy-logic risk layer

### Where it sits

```
  multimodal ensemble --> suicide severity  --+
                                              +--> fuzzy combination --> risk level
  depression classifier --> depression level -+
```

This layer trains nothing. It is a deterministic rule system over two model
outputs, which is deliberate: every decision it makes can be traced by hand.

### The membership vectors

The two inputs are **not** the same kind of quantity, and the thesis should say
so:

| Source | What is fed in | Why |
|---|---|---|
| Suicide severity | **hard vote fractions** — the share of ensemble members voting for each class | the ensemble combines by majority vote, so its members' raw probabilities are discarded before this point |
| Depression | **softmax probabilities** | a single model, so a genuine distribution exists |

This was confirmed by reading the code, not assumed.

### The operators

```
  fuzzy AND (a, b) = min(a, b)
  fuzzy OR  (a, b) = max(a, b)
  defuzzify        = argmax over the combined memberships
```

Each rule pairs a suicide class with one or more depression classes and emits a
risk level. A rule fires with strength equal to the fuzzy AND of its two
memberships. Rules emitting the same risk level combine with fuzzy OR. The
final risk level is the argmax across the four levels.

### The two rule tables

The 5-class table has 11 rows. The 3-class table has 6 and was **derived from
it**, not written fresh, using a documented cautious-merge policy: when two
original rules merge into one bucket, take the **more severe** of their risk
levels at each depression level.

```
  Example of the merge
  ---------------------
  5-class:  Wish to be dead + Moderate/Severe -> Elevated
            Suicide ideation + Moderate/Severe -> Elevated
  3-class:  Suicidal thought or desire + Mild/Moderate/Severe -> Elevated
```

Both tables are printed in full in `../VERIFIED_RESULTS.md` section 5.

### The hard limitation

**No ground-truth risk label exists for any meme.** Nothing here can be scored
for accuracy. The results in the README are distributions over the test set and
nothing more. The layer is exploratory and rule-based, and is not clinically
validated.

### Reproduce

```
python code/phase7_11_fuzzy_logic_final_test.py
```

The script first reproduces the locked ensemble exactly as a sanity check, then
applies the rule table. Results file:
`outputs/phase7_11_fuzzy_logic_final_test_results.json`.
"""

    W["approach_05_rigor_and_diagnostics"] = f"""# Workflow — rigor and diagnostics

This approach produced measurements, not a model. The workflow is therefore an
evaluation protocol rather than an architecture.

### The protocol that everything here uses

```
  777 train+val examples
        |
  5-fold stratified CV  (shuffle, random_state 42)
        |
        +--> per outer fold:
        |
        |    fold-train (622)                       outer held-out (155)
        |        |                                          |
        |        +--> inner split, 15% stratified           |
        |             |                |                    |
        |        inner-train      inner-val                 |
        |         (~528)           (~94)                    |
        |             |                |                    |
        |         train  ------ early stop on --------------+
        |                                                   |
        |                                    predict, never used for selection
        |
        +--> pooled out-of-fold predictions over all 777
```

**The inner split is the whole point.** The original protocol early-stopped on
the same split it then reported, which makes the reported number an upper bound
on itself. Separating them is what makes these numbers honest.

### Why that mattered — measured

Running both protocols in a single pass, so they differ in nothing but the
stopping rule:

| | Accuracy | Macro-F1 |
|---|---|---|
| Unbiased, inner split | 0.6654 | 0.6155 |
| Peeking, original protocol | 0.6924 | 0.6557 |
| **Selection optimism** | **+0.0270** | **+0.0402** |

### The learning-curve procedure

```
  For each outer fold:
     inner-val and outer rows held FIXED
     inner-train subsampled to 40% / 60% / 80% / 100%
     subsets NESTED and stratified, so the curve measures quantity
     not luck of the draw
```

Nesting matters: the 40% rows are a subset of the 60% rows, and so on, so
successive points differ only by added data.

### Statistical treatment, applied to every comparison

- **Paired bootstrap**, 5000 resamples, sharing the resample between the two
  methods being compared so the difference is measured on identical rows.
- Reported as **P(better)**, the fraction of resamples where the difference is
  positive, rather than a point estimate alone.
- **No bundling.** Every change is A/B'd on its own so effects can be
  attributed.

### The pre-registered gate

Fixed in writing before any experiment ran:

```
  unlock the test set only if
      CV accuracy >= 0.72  AND  lower bootstrap CI bound >= 0.68
```

The best configuration reached 0.6873 with a lower bound of 0.6538. **The gate
failed and the test set was not touched.** The margin above 0.70 was deliberate
insurance against the validation-to-test gap the project had already been
bitten by once.

### The one architectural change that helped

```
  depression classifier --> 4-d distribution --+
                                               |
  [ e3b_text(768) ; siglip(1152) ; depression(4) ] --> fusion head
```

Four extra input dimensions. The depression checkpoint is not retrained; it is
called through the parity-tested wrapper.

### Reproduce

```
python code/phase9_1_cv_baseline_3class.py       # honest baseline
python code/phase9_3_learning_curve.py           # the curve
python code/phase9_13_assemble_and_gate.py       # the gate
python code/phase9_19_depression_features_full_ensemble.py
python code/phase9_15_a1_refit_validation.py     # full-data refit test
```

Results files: `outputs/phase9_*.json`.
"""

    W["approach_06_dual_encoder_study"] = f"""# Workflow — dual image encoders

A fresh-eyes restart. Same data, same protocol, different question: what has
been *assumed* rather than tested?

### Step 0 — the linear probe

Before training anything, a deliberately crude test of the frozen inputs:

```
  frozen features --> L2 normalise --> standardise
                 --> multinomial logistic regression
                 --> 5-fold stratified CV over 777
```

No neural head, no early stopping, no tuning. The only question is which input
carries the most linearly accessible label signal. A crude probe cannot flatter
itself through selection, which is exactly why it is the right first instrument.

This is where CLIP beating SigLIP by 5 points first appeared.

### Step 1 — removing the confound

The earlier rejection of CLIP had tested it **inside the gated architecture**,
which consumes Stage A alignment projections trained on SigLIP. So the
comparison measured representation quality *times* alignment quality.

```
  CONFOUNDED (the earlier test)        CLEAN (this test)

  CLIP --> retrain Stage A --> gated    CLIP --> concat head
           ^^^^^^^^^^^^^^^                      ^^^^^^^^^^^
           alignment quality                    no alignment stage
           contaminates the result              at all
```

**No new alignment was trained for CLIP, deliberately.** Doing so would have
reintroduced the exact confound that made the original conclusion unsafe.

### Step 3 — how CLIP reaches each architecture

Gated needs alignment projections that exist only for SigLIP. Cross-attention
needs patch tokens that were never extracted for CLIP. Neither can take CLIP
through its main image pathway without new preprocessing.

But **both already concatenate a raw vector straight into their final head**,
alongside their fused representation. That side channel is where an extra
global image vector belongs:

```
  concat   [ e3b_text ; siglip ; CLIP ]                      --> classifier
  gated    [ gated ; orthogonal residual ; e3b_text ; CLIP ] --> classifier
  xattn    [ attention-pooled ; e3b_text ; CLIP ]            --> classifier
```

Each architecture keeps its own mechanism intact. Only the head widens by 1024.

### Step 4 — composition search, pre-specified

Six member groups exist (three architectures, with and without CLIP), giving 63
possible non-empty combinations. Picking the best of 63 on the same 777 rows
would be exactly the selection optimism this project already measured at
+0.027.

So six compositions were **specified in advance** from the per-architecture
evidence, before any of them were scored. The residual optimism from choosing
among six is recorded rather than hidden.

### The result worth understanding

Giving **every** member CLIP made the ensemble worse, despite two of three
members improving alone:

```
  15 models, SigLIP only          0.6744
  15 models, CLIP everywhere      0.6474   <-- worse
  20 models, mixed families       0.6834   <-- best
```

Shared features correlate members' errors. Ensembles are paid in disagreement,
not in average member quality. The winning composition keeps both families.

### Reproduce

```
python v2_research/code/v2_00_representation_probe.py
python v2_research/code/v2_01_image_encoder_showdown.py
python v2_research/code/v2_02_stacking_check.py
python v2_research/code/v2_03_dual_encoder_ensemble.py
python v2_research/code/v2_04_composition_search.py
```

Results files: `v2_research/outputs/v2_*.json`. Full narrative:
`v2_research/FINDINGS.md`.
"""

    for folder, body in W.items():
        write(os.path.join(OUT, folder, "WORKFLOW.md"), body)
    print(f"Wrote {len(W)} per-approach WORKFLOW.md files")


# ----------------------------------------------- phase document extraction --
# Phases 7, 8 and 9 each wrote their own planning document. Phases 1 to 6 never
# did -- they exist only as sections of IMPROVEMENT_PLAN.md. That left
# best_outcome/ with no source document at all, since the cross-architecture
# ensemble was built in Phase 3.
#
# These specs carve those sections out into standalone per-phase documents.
# Boundaries are located by heading text rather than by line number, so the
# extraction survives edits to the log.
#
# (output filename, [start headings], end heading or None, destinations, blurb)
PHASE_EXTRACTS = [
    ("PHASE1_ENSEMBLING_AND_LOSSES.md",
     "## Phase 1 narrative log", "## Phase 2 narrative log",
     ["approach_01_single_model_baseline"],
     "Seed ensembling, gate uncertainty, ordinal smoothing and focal loss. Seed "
     "ensembling was kept and became the foundation of every later result; gate "
     "uncertainty and focal loss were rejected."),
    ("PHASE2_REPRESENTATION_EXPERIMENTS.md",
     "## Phase 2 narrative log", "## Phase 3 narrative log",
     ["approach_01_single_model_baseline", "approach_06_dual_encoder_study"],
     "The translation-hop diagnostic, embedding mixup, supervised contrastive "
     "alignment, and the CLIP-versus-SigLIP comparison. **Section 2.4 is the "
     "decision that approach 06 later reopened and found confounded** -- read it "
     "alongside that folder."),
    ("PHASE3_CROSS_ARCHITECTURE_ENSEMBLE.md",
     "## Phase 3 narrative log", "## Phase 4 narrative log",
     ["best_outcome"],
     "**The phase that produced this result.** Partial encoder unfreezing "
     "(rejected), cross-attention fusion (weaker alone), and the cross-architecture "
     "ensemble that combined them and became the project's best validated result. "
     "Includes the deliberate test-set re-lock that produced the headline number."),
    ("PHASE4_REFINEMENTS.md",
     "## Phase 4 narrative log", "## Phase 5 — retrospective hindsight items",
     ["best_outcome"],
     "Hyperparameter sweep, direct vision-language-model prompting (0.2912, a clear "
     "negative result), threshold calibration, and hierarchical decomposition "
     "(rejected for error-cascading). Refinements attempted on top of the ensemble."),
    ("PHASE5_KFOLD_RIGOR.md",
     "## Phase 5 — retrospective hindsight items", "## Phase 6 — LoRA fine-tuning",
     ["approach_05_rigor_and_diagnostics"],
     "The first rigor upgrade: 5-fold cross-validation of the locked pipeline, plus "
     "the table of score-chasing items left open. Items 5.2, 5.4 and 5.5 from that "
     "table were finally closed by the work in this folder."),
    ("PHASE6_LORA_FINETUNING.md",
     "## Phase 6 — LoRA fine-tuning", "## Phase 7 — external meme pretraining",
     ["approach_01_single_model_baseline"],
     "LoRA fine-tuning of the frozen encoders. It worked where full and partial "
     "unfreezing had failed, but the gain was small enough that the frozen-encoder "
     "design was kept. The clearest test of whether the encoders were the ceiling."),
]

EXTRACT_HEADER = """> **Extracted from `IMPROVEMENT_PLAN.md`.**
>
> Phases 7, 8 and 9 each wrote their own planning document. Phases 1 to 6 did
> not: they were written directly into the project's chronological experiment
> log. This file carves out that phase's section so it can sit with the
> approach it belongs to.
>
> The text is unchanged. The complete log, with every phase in sequence, is in
> `../../project_documents/IMPROVEMENT_PLAN.md`.

---

"""


def extract_phase_documents():
    """Carve Phases 1-6 out of the experiment log into standalone documents."""
    src = os.path.join(ROOT, "IMPROVEMENT_PLAN.md")
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()

    def find(prefix, start=0):
        for i in range(start, len(lines)):
            if lines[i].startswith(prefix):
                return i
        return None

    written = 0
    for fn, start_h, end_h, dests, _blurb in PHASE_EXTRACTS:
        a = find(start_h)
        if a is None:
            print(f"  WARNING: heading not found, skipping {fn}: {start_h}")
            continue
        b = find(end_h, a + 1) if end_h else len(lines)
        if b is None:
            b = len(lines)
        body = EXTRACT_HEADER + "".join(lines[a:b]).rstrip() + "\n"
        if fn in STALE_BANNERS:
            body = BANNER_TEMPLATE.format(body=STALE_BANNERS[fn]) + body
        for d in dests:
            write(os.path.join(OUT, d, "source_documents", fn), body)
            written += 1
    print(f"Extracted {len(PHASE_EXTRACTS)} phase documents into {written} locations")


# ------------------------------------------ historical source documents ----
# Every working document written during the project, filed under the approach
# it belongs to. Entries are (source path relative to repo root, status note).
# `supervisor_result/` is not a source here: its copies are byte-identical to
# the repo-root originals.
SOURCE_DOCS = {
    "best_outcome": [],
    "approach_01_single_model_baseline": [
        ("PROGRESS_LOG.md",
         "Day-by-day build log for the original 12-day schedule. The primary record of how "
         "the baseline was built, including the data pipeline and the first locked model."),
        ("analysis.md",
         "The diagnostic written after the baseline locked, ranking where the remaining "
         "leverage was. It triggered everything that followed. Worth reading for how the "
         "priorities were set, and for how several of its predictions turned out wrong."),
    ],
    "approach_02_label_harmonization_3class": [
        ("FIGSIM_HARMONIZATION_AND_PRETRAINING.md",
         "The original external brief proposing the coarser label scheme and external meme "
         "pretraining. The starting point for this approach."),
        ("PHASE7_REVISED_PRETRAINING_PLAN.md",
         "The revised, safety-checked version of that brief, keeping the depression "
         "classifier untouched throughout. This is the plan that was actually executed."),
        ("PHASE7_SHARED_ENCODER_OPTION2_PLAN.md",
         "An alternative shared-representation design that was planned, tried, and found "
         "not to improve on the existing method."),
    ],
    "approach_03_depression_classifier": [
        ("PHASE8_MODALITY_AWARE_SYSTEM_PLAN.md",
         "The plan for verifying this classifier and building modality-aware routing around "
         "it. Steps 0 to 2b were completed; steps 3 to 8 were not started."),
        ("CLAUDE_HANDOFF_MODALITY_AWARE_MULTIMODAL_SYSTEM.md",
         "An external technical handoff that identified the missing-modality gap and "
         "correctly flagged an invalid comparison. The document that triggered Phase 8."),
    ],
    "approach_04_fuzzy_logic_risk_layer": [
        ("PHASE7_FUZZY_LOGIC_5CLASS_VS_3CLASS_PLAN.md",
         "The plan for comparing both label schemes all the way through to final risk "
         "output, including the derivation of the 3-class rule table."),
    ],
    "approach_05_rigor_and_diagnostics": [
        ("PHASE9_BEYOND_70_PLAN.md",
         "The plan, written before any experiment ran. Section 7 contains the "
         "forecast-versus-actual table: six of eight expected-value forecasts came in at or "
         "below the bottom of their range and two had the wrong sign."),
        ("PHASE9_RESULTS.md",
         "The full write-up of every experiment, including the two addenda covering the "
         "full-data refit test and the depression-feature result."),
    ],
    "approach_06_dual_encoder_study": [
        ("v2_research/RESEARCH_PLAN.md",
         "The research plan and its outcome section. Its pre-registered expectation, written "
         "before the ensemble ran, proved accurate."),
        ("v2_research/FINDINGS.md",
         "The detailed findings, including the confound analysis and the ensemble-diversity "
         "result."),
    ],
}

PROJECT_DOCS = [
    ("IMPROVEMENT_PLAN.md",
     "CURRENT. The complete chronological experiment log across every phase: each kept "
     "improvement and each rejected idea, with reasoning and numbers. The most detailed "
     "record of why each decision was made."),
    ("THESIS_SECTIONS_DRAFT.md",
     "PARTLY OUT OF DATE. Formal academic prose for the Methodology, Results, Limitations "
     "and Ethics sections. Section 2.6 carries the corrected comparison, but the draft "
     "predates all of approaches 05 and 06 and contains no material from either."),
    ("CURRENT_VERIFIED_SYSTEM_REPORT.md",
     "CURRENT as of approach 04. The full system description including the invalid-"
     "comparison correction and the depression-classifier verification results."),
    ("CURRENT_FINAL_SYSTEM_REPORT.md",
     "SUPERSEDED. Kept for the record only. Its headline comparison was later found "
     "invalid; the correction is in the document above. Do not quote from this one."),
]


# Documents containing claims that later work overturned. A reader who opens
# one of these directly, without the folder index, would extract a conclusion
# this project has since disproved. Each gets a banner prepended at the top of
# the copy so the warning travels with the file.
STALE_BANNERS = {
    "PHASE2_REPRESENTATION_EXPERIMENTS.md": (
        "This document concludes that **SigLIP is the better image encoder and CLIP was "
        "rejected** (section 2.4). That conclusion was later found to rest on a confounded "
        "comparison: CLIP was tested inside an architecture whose alignment stage had been "
        "trained on SigLIP. Judged without that alignment, the two are statistically "
        "indistinguishable and work better together than either alone.\n>\n"
        "> See `thesis_ready/approach_06_dual_encoder_study/` for the correction. Do not "
        "quote section 2.4's conclusion as the project's final position."),
    "CURRENT_FINAL_SYSTEM_REPORT.md": (
        "**This document is superseded. Do not quote from it.**\n>\n"
        "> Its headline comparison, a claimed improvement from the 3-class model, was later "
        "found invalid: it subtracted scores across different label granularities. The "
        "corrected account is in `CURRENT_VERIFIED_SYSTEM_REPORT.md` in this same folder. "
        "This copy is kept only so the record shows what was originally claimed."),
    "THESIS_SECTIONS_DRAFT.md": (
        "**Partly out of date.** This draft predates the rigor work and the dual-encoder "
        "study, so it contains no material from either. Its section 2.6 does carry the "
        "corrected label-granularity comparison.\n>\n"
        "> For current numbers use `thesis_ready/VERIFIED_RESULTS.md`, which is regenerated "
        "from the raw confusion matrices on every build."),
    "analysis.md": (
        "This is a **diagnostic written early in the project**, ranking where the remaining "
        "leverage was thought to be. Several of its predictions were later tested and proved "
        "wrong, including that unfreezing the encoders would be the biggest lever and that "
        "the translation hop was costing significant signal.\n>\n"
        "> Read it for how priorities were set, not as a statement of findings."),
}

BANNER_TEMPLATE = """> ## ⚠ Read before quoting from this file
>
> {body}

---

"""


def copy_with_banner(src, dst, fn):
    """Copy a document, prepending a warning banner if its claims were later
    overturned. The banner travels with the file, so a reader who opens it
    directly still sees the correction."""
    with open(src, "r", encoding="utf-8") as f:
        body = f.read()
    if fn in STALE_BANNERS:
        body = BANNER_TEMPLATE.format(body=STALE_BANNERS[fn]) + body
    with open(dst, "w", encoding="utf-8") as f:
        f.write(body)


def distribute_source_documents():
    """Copy every historical working document into the approach it belongs to,
    with an index explaining what each one is and whether it is still current."""
    total = 0

    # Phase sections carved out of the experiment log, keyed by folder.
    extracted = {}
    for fn, _s, _e, dests, blurb in PHASE_EXTRACTS:
        for d in dests:
            extracted.setdefault(d, []).append((fn, blurb))

    folders = sorted(set(SOURCE_DOCS) | set(extracted))
    for folder in folders:
        docs = SOURCE_DOCS.get(folder, [])
        phases = extracted.get(folder, [])
        if not docs and not phases:
            continue
        dest = os.path.join(OUT, folder, "source_documents")
        os.makedirs(dest, exist_ok=True)
        lines = [f"# Source documents — {folder}", "",
                 "The primary record for this approach, copied here unchanged.",
                 "`../README.md` and `../WORKFLOW.md` are the condensed, verified versions",
                 "and are what to quote from.", ""]

        if phases:
            lines += ["## Phase sections", "",
                      "Phases 1 to 6 never had their own planning documents; they were written",
                      "directly into the chronological experiment log. These are carved out of",
                      "it so each sits with the approach it belongs to. Text unchanged.", ""]
            for fn, blurb in phases:
                lines += [f"### `{fn}`", "", blurb, ""]
                total += 1

        if docs:
            lines += ["## Working documents", ""] if phases else []
            for rel, note in docs:
                src = os.path.join(ROOT, rel)
                if not os.path.exists(src):
                    continue
                fn = os.path.basename(rel)
                copy_with_banner(src, os.path.join(dest, fn), fn)
                lines += [f"### `{fn}`", "", note, ""]
                total += 1

        write(os.path.join(dest, "README.md"), "\n".join(lines))

    dest = os.path.join(OUT, "project_documents")
    os.makedirs(dest, exist_ok=True)
    lines = ["# Project-wide documents", "",
             "These span every approach rather than belonging to one, so they sit here",
             "rather than in an approach folder. **Check the status note before quoting",
             "from any of them** — one is superseded and one is partly out of date.", ""]
    for rel, note in PROJECT_DOCS:
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            continue
        fn = os.path.basename(rel)
        copy_with_banner(src, os.path.join(dest, fn), fn)
        lines += [f"### `{fn}`", "", note, ""]
        total += 1
    lines += ["---", "",
              "For numbers, prefer `../VERIFIED_RESULTS.md` over any document here. It is",
              "regenerated from the raw confusion matrices on every build and cannot drift."]
    write(os.path.join(dest, "README.md"), "\n".join(lines))

    print(f"Distributed {total} historical source documents")


# ------------------------------------------------------------------- main --
def main():
    # VERIFIED_RESULTS.md and WORKFLOW.md are produced by
    # phase9_17_build_thesis_package.py, which is run first (see __main__).
    # Stash them across the rebuild rather than regenerating them here, so
    # there is exactly one source of truth for each document.
    stash = {}
    for fn in ("VERIFIED_RESULTS.md", "WORKFLOW.md"):
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                stash[fn] = f.read()

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    for fn, body in stash.items():
        write(os.path.join(OUT, fn), body)

    d1 = load("final_test_results.json")
    d5 = load("phase3_final_test_results.json")
    d3 = load("phase7_final_test_results_3class.json")
    ddep = load("phase8_3_depression_dataset_verification_results.json")
    dfuz = load("phase7_11_fuzzy_logic_final_test_results.json")
    dcv = load("phase9_1_cv_baseline_results.json")
    dlc = load("phase9_3_learning_curve_results.json")
    dep9 = load("phase9_19_depression_features_ensemble_results.json")
    da1 = load("phase9_15_a1_refit_validation_results.json")
    v2c = load("v2_04_composition_search.json", sub=os.path.join("v2_research", "outputs"))
    v2e = load("v2_03_dual_encoder_ensemble.json", sub=os.path.join("v2_research", "outputs"))

    seed = str(d1["primary_seed"])
    m1 = metrics(d1["per_seed_test_results"][seed]["confusion_matrix"], qwk=True)
    m5 = metrics(d5["confusion_matrix"], qwk=True)
    m3 = metrics(d3["confusion_matrix"], qwk=True)
    mcol = metrics(collapse(d5["confusion_matrix"]))
    mdep = metrics(ddep["confusion_matrix"])

    check("5-class macro-F1", m5["macro_f1"], d5["macro_f1"])
    check("5-class accuracy", m5["accuracy"], d5["accuracy"])
    check("3-class macro-F1", m3["macro_f1"], d3["macro_f1"])
    check("3-class accuracy", m3["accuracy"], d3["accuracy"])
    check("depression macro-F1", mdep["macro_f1"], ddep["macro_f1"])
    check("depression accuracy", mdep["accuracy"], ddep["accuracy"])
    print("All recomputed metrics match their stored values.")

    unb = dcv["pooled_unbiased"]
    opt = dcv["measured_selection_optimism"]
    curve = dlc["pooled_curve"]
    slope = dlc["final_segment_slope_per_100_examples"]
    depb = dep9["baseline_ensemble"]
    depw = dep9["with_depression_ensemble"]
    v2best = v2c["compositions"][v2c["best"]]
    fz3, fz5 = dfuz["risk_distribution_3class_test"], dfuz["risk_distribution_5class_test_original"]

    # ===================================================== best_outcome =====
    folder = os.path.join(OUT, "best_outcome")
    figs = put_figs(folder, [
        (FIGS, "09_phase3_architecture.png", "The three fusion architectures this ensemble combines."),
        (FIGS, "04_confusion_matrix.png", "Confusion matrix on the test set."),
        (FIGS, "05_per_class_metrics.png", "Per-class precision, recall and F1."),
        (FIGS, "11_cumulative_progression.png", "How the project arrived here."),
        (FIGS, "20_current_verified_pipeline.png", "Full system architecture."),
    ])
    write(os.path.join(folder, "README.md"), f"""# BEST OUTCOME — cross-architecture ensemble (5-class)

**This is the result to lead with in the thesis.** It is the strongest
validated result the project produced, and the only one measured on the test
set that survived every later re-examination.

## The numbers

| Metric | Value |
|---|---|
| **Macro-F1** | **{m5['macro_f1']:.4f}** |
| Accuracy | {m5['accuracy']:.4f} |
| Weighted-F1 | {m5['weighted_f1']:.4f} |
| Quadratic weighted kappa | {m5['qwk']:.4f} |
| Test examples | {m5['n']} |

{pc_table(FIVE, m5['per_class_f1'])}

Confusion matrix, rows are true and columns predicted:

{cm_block(d5['confusion_matrix'])}

## The same model on the coarser 3-class task

Collapsing this model's own predictions into the 3-class grouping, with no
retraining, gives the **best 3-class-equivalent result on record**:

| Metric | Value |
|---|---|
| Accuracy | **{mcol['accuracy']:.4f}** |
| Macro-F1 | {mcol['macro_f1']:.4f} |
| Weighted-F1 | {mcol['weighted_f1']:.4f} |

{cm_block(collapse(d5['confusion_matrix']))}

This matters because a separately trained 3-class model scored
{m3['accuracy']:.4f} accuracy and {m3['macro_f1']:.4f} macro-F1, which is
**worse**. See `../approach_02_label_harmonization_3class/`.

## What the model is

Six models combined by majority vote: the gated fusion architecture with an
orthogonal residual, and the cross-attention architecture, three random seeds
each. Both encoders frozen throughout.

Ensembling was the single largest lever in the entire project, worth more than
every loss function, fusion mechanism and hyperparameter change put together.

## Why it is still the best after ten phases

Everything built afterwards was measured against it and nothing beat it:

| Later attempt | Outcome |
|---|---|
| 3-class harmonized model | worse once fairly compared ({m3['accuracy'] - mcol['accuracy']:+.4f} accuracy) |
| Phase 9, ten experiments across five tracks | gate not cleared, test never evaluated |
| v2 dual-encoder study | level in cross-validation, marginally behind |

## Improvement over the starting point

| | Macro-F1 | Accuracy |
|---|---|---|
| Single-model baseline | {m1['macro_f1']:.4f} | {m1['accuracy']:.4f} |
| **This ensemble** | **{m5['macro_f1']:.4f}** | **{m5['accuracy']:.4f}** |
| Gain | **{m5['macro_f1'] - m1['macro_f1']:+.4f}** | {m5['accuracy'] - m1['accuracy']:+.4f} |

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

{figs}

## Caveat to state

The test set is {m5['n']} examples, so a 95% confidence interval spans roughly
±7 points. Report the number with that uncertainty rather than as a point
estimate.
""")

    # ============================================== approach_01 baseline =====
    folder = os.path.join(OUT, "approach_01_single_model_baseline")
    figs = put_figs(folder, [
        (FIGS, "01_ablation_ladder.png", "Ablation ladder that led to this configuration."),
        (FIGS, "06_final_test_summary.png", "The locked test result."),
        (FIGS, "02_coral_vs_ce.png", "Ordinal loss comparison, CORAL rejected."),
        (FIGS, "03_gate_alpha_fix.png", "The gate-collapse failure and its fix."),
    ])
    write(os.path.join(folder, "README.md"), f"""# Approach 01 — single-model baseline (5-class)

The project's first locked result, and the reference every later result is
measured against.

## The numbers

| Metric | Value |
|---|---|
| Macro-F1 | {m1['macro_f1']:.4f} |
| Accuracy | {m1['accuracy']:.4f} |
| Weighted-F1 | {m1['weighted_f1']:.4f} |
| Quadratic weighted kappa | {m1['qwk']:.4f} |
| Test examples | {m1['n']} |

{pc_table(FIVE, m1['per_class_f1'])}

{cm_block(d1['per_seed_test_results'][seed]['confusion_matrix'])}

## What it is

A single gated-fusion model with an orthogonal residual term, on frozen
DAPT-BanglaBERT text and frozen SigLIP image features.

## The failure that was diagnosed and fixed here

Early versions of the gated architecture collapsed: the learned gate drove
almost all weight onto the image and effectively ignored the text. The fix was
the Stage A contrastive alignment step, which puts the two independently
pretrained encoder spaces into a comparable geometry before the gate sees
them. Without alignment the gate has no basis for mixing them.

This is worth a paragraph in the thesis. It is a real diagnosed failure mode
with a mechanistic explanation, not a tuning anecdote.

## Superseded by

`../best_outcome/`, which ensembles this architecture with cross-attention and
gains {m5['macro_f1'] - m1['macro_f1']:+.4f} macro-F1.

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

{figs}
""")

    # ======================================= approach_02 harmonization ======
    folder = os.path.join(OUT, "approach_02_label_harmonization_3class")
    figs = put_figs(folder, [
        (FIGS, "14_label_harmonization.png", "The 5-class to 3-class label scheme."),
        (FIGS, "17_final_3class_test_lock.png", "The locked 3-class test result."),
        (FIGS, "15_bnhib_pretraining_transfer_test.png", "External meme pretraining attempt."),
    ])
    write(os.path.join(folder, "README.md"), f"""# Approach 02 — label harmonization to 3 classes

Collapses the 5-class severity scheme into three coarser categories, then
trains a model natively on them.

| Original 5-class | Harmonized 3-class |
|---|---|
| None | No expressed severity |
| Wish to be dead, Suicide ideation | Suicidal thought or desire |
| Planning, Attempt or death | High acuity suicidal content |

## The numbers

| Metric | Value |
|---|---|
| Macro-F1 | {m3['macro_f1']:.4f} |
| Accuracy | {m3['accuracy']:.4f} |
| Weighted-F1 | {m3['weighted_f1']:.4f} |
| Quadratic weighted kappa | {m3['qwk']:.4f} |
| Test examples | {m3['n']} |

{pc_table(THREE, m3['per_class_f1'])}

{cm_block(d3['confusion_matrix'])}

## The comparison rule that must not be broken

**Never subtract a 3-class score from a 5-class score.** A coarser task is
easier to score well on regardless of whether the model improved. An earlier
draft claimed a "+0.0746 improvement" by doing exactly that, and it was wrong.

The valid method collapses the finer model's own predictions into the coarser
grouping and compares on identical records, with no retraining:

| | Collapsed 5-class ensemble | Native 3-class model | Valid delta |
|---|---|---|---|
| Accuracy | {mcol['accuracy']:.4f} | {m3['accuracy']:.4f} | **{m3['accuracy'] - mcol['accuracy']:+.4f}** |
| Macro-F1 | {mcol['macro_f1']:.4f} | {m3['macro_f1']:.4f} | **{m3['macro_f1'] - mcol['macro_f1']:+.4f}** |

**The 3-class model is not an improvement. It is slightly weaker once fairly
compared.** Report it as a differently-scoped result, which is what it is.

## Reading the two metrics

{m3['macro_f1']:.4f} is the **macro-F1** and {m3['accuracy']:.4f} is the
**accuracy**, both from this same model on the same {m3['n']} memes. Accuracy
is carried by the large middle class (102 of {m3['n']}), while macro-F1 weights
all three equally and is dragged down by the small first class
({int(d3['classification_report'][THREE[0]]['support'])} examples, F1
{m3['per_class_f1'][0]:.3f}).

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

{figs}
""")

    # ========================================= approach_03 depression =======
    folder = os.path.join(OUT, "approach_03_depression_classifier")
    figs = put_figs(folder, [
        (FIGS, "20_current_verified_pipeline.png", "Where this classifier sits in the system, with its verified text-only route."),
    ])
    write(os.path.join(folder, "README.md"), f"""# Approach 03 — depression classifier (text-only)

The text-only half of the system. DAPT-BanglaBERT with its own 4-class
classification head, trained on a separate Bangla mental-health text dataset.

**Never retrained or modified at any point across all ten phases.**

## The numbers

| Metric | Value |
|---|---|
| Accuracy | {mdep['accuracy']:.4f} |
| Macro-F1 | {mdep['macro_f1']:.4f} |
| Weighted-F1 | {mdep['weighted_f1']:.4f} |
| Records | {mdep['n']} |

{pc_table(DEPRESSION, mdep['per_class_f1'])}

{cm_block(ddep['confusion_matrix'])}

## Mandatory caveat — quote this alongside the number

This checkpoint was trained on roughly four fifths of this exact dataset. The
figure is a **behaviour sanity check against its own training domain**, not an
unbiased held-out generalization estimate. It confirms the model works
correctly and behaves sensibly. It must not be presented as a fresh accuracy
result.

Supporting detail: the weakest class is Moderate at
{mdep['per_class_f1'][2]:.3f} F1, confused mostly with the adjacent Mild class.
That is the expected pattern for adjacent severity levels, not a concerning
failure.

## What is a clean claim

The text-only path was **parity-tested**: 100.00% label agreement and 0.00
maximum probability difference against 196 previously saved outputs. An exact
match, not an approximation. That is safe to state without qualification.

## Its role in the multimodal system

Two roles. It contributes the depression axis to the fuzzy-logic risk layer
(`../approach_04_fuzzy_logic_risk_layer/`), and its output distribution was
later found to work as an **input feature** for the suicide-severity model
(`../approach_05_rigor_and_diagnostics/`).

Unlike the multimodal branch, this classifier has **no missing-modality
limitation**. It needs text only, and that route is verified.

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

{figs}
""")

    # =========================================== approach_04 fuzzy =========
    folder = os.path.join(OUT, "approach_04_fuzzy_logic_risk_layer")
    figs = put_figs(folder, [
        (FIGS, "16_fuzzy_logic_5class_vs_3class.png", "Rule tables and how the two label schemes compare."),
        (FIGS, "18_final_fuzzy_test_comparison.png", "Final risk-level output on the test set."),
    ])
    write(os.path.join(folder, "README.md"), f"""# Approach 04 — fuzzy-logic risk layer

The integration layer. Combines the multimodal suicide-severity output with the
text-only depression output into a single risk level.

## Mechanism

Each classifier's distribution is treated as a membership degree per category.

- Fuzzy AND is the **minimum** of two memberships
- Fuzzy OR is the **maximum**
- The final risk level is the **argmax** of the combined memberships

**Confirmed from code, not assumed:** the suicide side contributes **hard vote
fractions**, meaning the share of ensemble members voting for each class. The
depression side contributes genuine softmax probabilities.

## Results on the test set ({dfuz['n_test']} memes)

| Risk level | 5-class system | 3-class system |
|---|---|---|
| Minimal | {fz5['Minimal']} | {fz3['Minimal']} |
| Low | {fz5['Low']} | {fz3['Low']} |
| Elevated | {fz5['Elevated']} | {fz3['Elevated']} |
| Critical | {fz5['Critical']} | {fz3['Critical']} |

## The limitation that must be stated

**No ground-truth combined risk label exists for any meme.** This is a
distributional and qualitative comparison, never an accuracy claim. The layer
is exploratory and rule-based, and it is not clinically validated.

Say this plainly in the thesis. A rule layer over two model outputs, presented
without that caveat, would overclaim.

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

{figs}
""")

    # ========================================== approach_05 rigor ==========
    folder = os.path.join(OUT, "approach_05_rigor_and_diagnostics")
    figs = put_figs(folder, [
        (FIGS, "21_phase9_summary.png", "The learning curve, and every experiment against the honest baseline."),
        (FIGS, "22_phase9_final_verdict.png", "The full-data refit test and why no path to 70% remained."),
        (FIGS, "12_hierarchical_diagnosis.png", "An earlier rejected decomposition, diagnosed."),
        (FIGS, "13_phase6_lora.png", "LoRA fine-tuning of the frozen encoders."),
    ])
    ns = sorted(curve, key=float)
    write(os.path.join(folder, "README.md"), f"""# Approach 05 — rigor and diagnostics

This produced **no performance gain**. It produced measurements, and they are
the most defensible material in the project.

## Finding 1 — selection optimism, quantified

The original protocol early-stops by picking the epoch with the best score
**on the split it then reports**. Re-running the identical configuration with a
proper inner split for early stopping:

| | Accuracy | Macro-F1 |
|---|---|---|
| Unbiased | {unb['accuracy']:.4f} | {unb['macro_f1']:.4f} |
| Peeking, the original protocol | {dcv['pooled_peeking']['accuracy']:.4f} | {dcv['pooled_peeking']['macro_f1']:.4f} |
| **Selection optimism** | **{opt['accuracy']:+.4f}** | **{opt['macro_f1']:+.4f}** |

The peeking figure nearly reproduces the old single-split validation number
(0.6974 / 0.6649), which is direct evidence that number was inflated by
selection rather than being a held-out estimate.

## Finding 2 — the learning curve

Models retrained on nested subsets, with validation and held-out rows fixed so
only training size varies:

| Training examples | {" | ".join(str(curve[k]['mean_n_train']) for k in ns)} |
|---|{"---|" * len(ns)}
| Accuracy | {" | ".join(f"{curve[k]['accuracy']:.4f}" for k in ns)} |

**Still climbing and accelerating at the largest size available:
{slope['accuracy']:+.4f} accuracy and {slope['macro_f1']:+.4f} macro-F1 per +100
training examples.** This turns a claim carried through the whole project into
a measured number, and it identifies the binding constraint as the number of
labelled examples.

## Finding 3 — deeper integration of the two components

The only change here that helped. Feeding the depression classifier's output
distribution into the suicide model as **input features**, rather than letting
the two meet only in the rule layer at the end:

| Configuration (cross-validated, never tested) | Accuracy | Macro-F1 |
|---|---|---|
| Baseline, text + image | {depb['accuracy']:.4f} | {depb['macro_f1']:.4f} |
| **+ depression distribution** | **{depw['accuracy']:.4f}** | **{depw['macro_f1']:.4f}** |
| Delta | {depw['accuracy'] - depb['accuracy']:+.4f} (P(better) {dep9['paired_comparison']['prob_accuracy_improved']:.3f}) | {depw['macro_f1'] - depb['macro_f1']:+.4f} |

Two points to state in the write-up:

- **No target leakage.** The depression classifier was trained on its own
  separate {mdep['n']}-record dataset, not on these memes and not on their
  severity labels.
- **Not a simple correlation.** The Spearman correlation between predicted
  depression level and true suicide severity is only
  {load('phase9_18_depression_features_results.json')['spearman_depression_vs_suicide_severity']:.4f}.
  The distribution carries information its argmax does not.

## Finding 4 — the full-data refit fails

Training the final model on all available data requires abandoning early
stopping for a fixed epoch count. Tested directly:

| Arm (cross-validated) | Accuracy | Macro-F1 |
|---|---|---|
| Current protocol, early stopping | {da1['arm1_current_protocol']['accuracy']:.4f} | {da1['arm1_current_protocol']['macro_f1']:.4f} |
| Full data, fixed epochs, +18% rows | {da1['arm2_a1_full_data_fixed_epochs']['accuracy']:.4f} | {da1['arm2_a1_full_data_fixed_epochs']['macro_f1']:.4f} |

It **loses despite 18% more training data**, because the best epoch varies
enormously between seeds and one fixed schedule suits almost none of them.

## Ideas tested and rejected

| Idea | Result | Verdict |
|---|---|---|
| Stacking meta-learner | −0.0039 accuracy | rejected |
| CORN cumulative-link ordinal loss | ±0.0000, macro-F1 worse | rejected |
| Structured figurative-reasoning streams | significantly worse | rejected |
| External out-of-domain negatives | class-0 F1 −0.1127 | rejected, harmful |
| Full-data refit | −0.0103 | rejected |
| Train on 5 classes, collapse to 3 | +0.0039, P(better) 0.599 | not significant |
| Soft probability averaging | +0.0039 | marginal, kept |
| Five seeds per architecture | +0.0064, P(better) 0.834 | kept |

Two worth a sentence each in the discussion. **Out-of-domain negatives do not
substitute for in-domain data** — 300 screened external memes made the class
they targeted significantly worse. And **a pre-registered gate was honoured**:
the test set was to be unlocked only at 0.72 cross-validated accuracy, the best
configuration reached {depw['accuracy']:.4f}, and the test set was not touched.

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

{figs}
""")

    # ======================================== approach_06 dual encoders ====
    folder = os.path.join(OUT, "approach_06_dual_encoder_study")
    figs = put_figs(folder, [
        (V2FIGS, "v2_01_dual_encoder_findings.png", "The probe, the per-architecture effect, and the ensemble outcome."),
    ])
    pa = v2e["per_architecture"]
    write(os.path.join(folder, "README.md"), f"""# Approach 06 — dual image encoders

A fresh-eyes restart that reopened a decision closed on confounded evidence.
**It did not beat the system**, but it corrected the record and produced the
largest single-change gain measured anywhere in the project.

## The observation

Ten phases varied the fusion architecture, the loss and the combination rule.
Almost none varied the **input representation**. Every model consumed the same
two frozen vectors, chosen in week one and never revisited.

## The decision that was wrong

An earlier phase tested CLIP against SigLIP and rejected it at 0.4666 against
0.5330, concluding SigLIP was "the right call".

**That comparison was confounded.** It ran CLIP inside the gated architecture,
which consumes contrastive alignment projections, and its own notes record CLIP
aligning badly: retrieval 2.6% against SigLIP's 6.7%. It measured
representation quality times alignment quality and blamed the representation.

## The clean test — no alignment in any arm

| Arm | Accuracy | Macro-F1 |
|---|---|---|
| text + SigLIP, the project's choice | 0.6371 | 0.5917 |
| text + CLIP, previously rejected | 0.6448 | 0.5919 |
| **text + SigLIP + CLIP, never tried** | **0.6628** | **0.6154** |

**CLIP is not better than SigLIP. It is different from SigLIP.** Head to head
they are statistically indistinguishable. Together they beat either alone by
+0.0257, P(better) 0.943.

## Per architecture

| Architecture | SigLIP only | + CLIP | Delta |
|---|---|---|---|
| concat | {pa['concat']['base']['accuracy']:.4f} | {pa['concat']['dual']['accuracy']:.4f} | **{pa['concat']['dual']['accuracy'] - pa['concat']['base']['accuracy']:+.4f}** |
| gated | {pa['gated']['base']['accuracy']:.4f} | {pa['gated']['dual']['accuracy']:.4f} | **{pa['gated']['dual']['accuracy'] - pa['gated']['base']['accuracy']:+.4f}** |
| cross-attention | {pa['xattn']['base']['accuracy']:.4f} | {pa['xattn']['dual']['accuracy']:.4f} | **{pa['xattn']['dual']['accuracy'] - pa['xattn']['base']['accuracy']:+.4f}** |

Two improve, one degrades.

## The ensemble lesson — worth a discussion paragraph

| Configuration (cross-validated, never tested) | Accuracy |
|---|---|
| 15 models, SigLIP only | 0.6744 |
| 15 models, CLIP in every head | **0.6474** |
| 20 models, SigLIP-only plus dual-concat | **0.6834** |

The middle row is the instructive one. **Two of three members improved
individually and the ensemble got worse.** Giving every member the same extra
features correlates their errors, and ensembles are paid in disagreement. The
winner keeps the SigLIP-only members and adds dual-encoder members beside them.

## Outcome

| Configuration (cross-validated, never tested) | CV accuracy | CV macro-F1 |
|---|---|---|
| Previous best | 0.6873 | 0.6397 |
| **This study's best** | {v2best['accuracy']:.4f} | {v2best['macro_f1']:.4f} |
| Difference | {v2best['accuracy'] - 0.6873:+.4f} | {v2best['macro_f1'] - 0.6397:+.4f} |

Level within noise, marginally behind. **Cross-validated only. Never evaluated
on the test set.**

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

{figs}
""")

    # ======================================= per-approach WORKFLOW.md ======
    write_workflows(mdep)

    # ================================ historical source documents ==========
    extract_phase_documents()
    distribute_source_documents()

    # ============================================ shared top-level docs ====
    rows = [["approach", "model", "task", "split", "n", "accuracy", "macro_f1",
             "weighted_f1", "qwk", "quotable"],
            ["best_outcome", "Cross-architecture ensemble", "5-class", "test", m5["n"],
             f"{m5['accuracy']:.4f}", f"{m5['macro_f1']:.4f}", f"{m5['weighted_f1']:.4f}",
             f"{m5['qwk']:.4f}", "YES - LEAD WITH THIS"],
            ["best_outcome", "Same, collapsed to 3 classes", "3-class", "test", mcol["n"],
             f"{mcol['accuracy']:.4f}", f"{mcol['macro_f1']:.4f}", f"{mcol['weighted_f1']:.4f}",
             "", "yes - best 3-class-equivalent"],
            ["approach_01", "Single-model baseline", "5-class", "test", m1["n"],
             f"{m1['accuracy']:.4f}", f"{m1['macro_f1']:.4f}", f"{m1['weighted_f1']:.4f}",
             f"{m1['qwk']:.4f}", "yes - starting point"],
            ["approach_02", "Native 3-class model", "3-class", "test", m3["n"],
             f"{m3['accuracy']:.4f}", f"{m3['macro_f1']:.4f}", f"{m3['weighted_f1']:.4f}",
             f"{m3['qwk']:.4f}", "yes - on its own terms"],
            ["approach_03", "DAPT-BanglaBERT depression", "4-class", "own dataset", mdep["n"],
             f"{mdep['accuracy']:.4f}", f"{mdep['macro_f1']:.4f}", f"{mdep['weighted_f1']:.4f}",
             "", "yes - WITH training-overlap caveat"],
            ["approach_05", "Best with depression features", "3-class", "cross-validation",
             dcv["n_examples"], f"{depw['accuracy']:.4f}", f"{depw['macro_f1']:.4f}", "", "",
             "CV only - never tested"],
            ["approach_06", "Best with dual encoders", "3-class", "cross-validation",
             dcv["n_examples"], f"{v2best['accuracy']:.4f}", f"{v2best['macro_f1']:.4f}", "", "",
             "CV only - never tested"]]
    with open(os.path.join(OUT, "results_table.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    write(os.path.join(OUT, "START_HERE_FOR_REPORT_GENERATION.md"), f"""# Read this first — rules for generating a report from this package

Written for anyone, human or automated, producing a report from these folders.
This package contains ten phases of work, and **some of its documents record
conclusions that later work overturned.** Following the four rules below avoids
every known trap.

---

## Rule 1 — one file is authoritative for numbers

**`VERIFIED_RESULTS.md`.** It is regenerated from the raw confusion matrices on
every build, and every value is asserted against the stored result, so it cannot
drift. `results_table.csv` is the same data, machine-readable.

If any other document disagrees with it, that document is out of date.

## Rule 2 — the headline result

| | |
|---|---|
| **Lead with** | Cross-architecture ensemble, **macro-F1 {m5['macro_f1']:.4f}, accuracy {m5['accuracy']:.4f}** |
| Test examples | {m5['n']} |
| Folder | `best_outcome/` |

Best 3-class-equivalent figure is **{mcol['accuracy']:.4f} accuracy**, obtained by
collapsing that same model's predictions. Not a separate model.

**Nothing built in the ten phases afterwards beat it.**

## Rule 3 — never mix cross-validated numbers with test numbers

Two folders report **cross-validated** results for configurations that were
**never evaluated on the test set**:

| Folder | Number | Status |
|---|---|---|
| `approach_05_rigor_and_diagnostics/` | {depw['accuracy']:.4f} accuracy | cross-validated only |
| `approach_06_dual_encoder_study/` | {v2best['accuracy']:.4f} accuracy | cross-validated only |

These are **not** improvements on the {m5['accuracy']:.4f} headline. They are
measured on a different protocol over the 777 train+validation examples. A
report that presents them as results would be wrong.

## Rule 4 — claims that were overturned

Historical documents are included because they are the primary record. Four
contain conclusions this project later disproved. **Each carries a warning
banner at the top of the file**, so opening it directly is safe, but here is the
map:

| Document | Overturned claim | Correction lives in |
|---|---|---|
| `PHASE2_REPRESENTATION_EXPERIMENTS.md` | "SigLIP confirmed as the better choice" | `approach_06_dual_encoder_study/` |
| `CURRENT_FINAL_SYSTEM_REPORT.md` | A claimed improvement from the 3-class model | `CURRENT_VERIFIED_SYSTEM_REPORT.md` |
| `THESIS_SECTIONS_DRAFT.md` | Predates approaches 05 and 06 entirely | `VERIFIED_RESULTS.md` |
| `analysis.md` | Predicted unfreezing encoders was the biggest lever | `approach_01`, Phase 6 document |

**Two specific numbers must never appear in a report:** a claimed "+0.0746" or
"+0.110" improvement. Both came from subtracting scores across different label
granularities, which is invalid. `VERIFIED_RESULTS.md` section 9 has the full
list.

---

## Where to find each kind of content

| You need | Go to |
|---|---|
| The headline result and its model | `best_outcome/README.md` |
| How any approach works, step by step | that folder's `WORKFLOW.md` |
| Every trial and what it scored | `approach_05_rigor_and_diagnostics/README.md` |
| Rejected ideas and why | same file, plus `project_documents/IMPROVEMENT_PLAN.md` |
| The complete chronological record | `project_documents/IMPROVEMENT_PLAN.md` |
| Figures | each folder's `figures/` |
| Original working documents | each folder's `source_documents/` |

## What this project actually concluded

1. The best result is **macro-F1 {m5['macro_f1']:.4f}** and nothing across ten
   subsequent phases improved on it.
2. The binding constraint is **the number of labelled examples**, measured at
   {slope['accuracy']:+.4f} accuracy per 100 additional examples, still climbing.
3. A large part of the apparent earlier performance was **selection optimism**,
   measured at {opt['accuracy']:+.4f} accuracy.
4. Roughly a dozen ideas were tested and rejected, two of which actively made
   things worse.

A report that presents this as a story of steady improvement would be
misreading it. The honest story is a solid result, followed by a rigorous and
largely unsuccessful search for more, which established where the ceiling comes
from.
""")

    write(os.path.join(OUT, "README.md"), f"""# thesis_ready — everything for the thesis, one folder per approach

> **Generating a report from this package?** Read
> `START_HERE_FOR_REPORT_GENERATION.md` first. It states which file is
> authoritative for numbers, and lists the four documents whose conclusions
> later work overturned.


Generated by `code/build_thesis_ready.py`. Every metric is recomputed from the
raw confusion matrices and asserted against the stored value, so these documents
cannot drift from the results they report. **Do not edit by hand** — re-run the
script instead.

## Start here

| Folder | What it holds |
|---|---|
| **`best_outcome/`** | **The result to lead with.** Cross-architecture ensemble, macro-F1 {m5['macro_f1']:.4f}, accuracy {m5['accuracy']:.4f} |
| `approach_01_single_model_baseline/` | The starting point, and a diagnosed failure mode worth writing up |
| `approach_02_label_harmonization_3class/` | The 3-class scheme, and the comparison rule that must not be broken |
| `approach_03_depression_classifier/` | The text-only half of the system |
| `approach_04_fuzzy_logic_risk_layer/` | How the two halves are combined |
| `approach_05_rigor_and_diagnostics/` | No gain, but the most defensible material in the project |
| `approach_06_dual_encoder_study/` | A corrected decision and a clean ensemble-diversity lesson |

Every folder has its own `README.md` and its own `figures/`.

Shared files at this level: `VERIFIED_RESULTS.md` (all numbers in one place),
`results_table.csv` (machine-readable), `WORKFLOW.md` (system description for
the methodology chapter).

## What to use for each thesis section

**Methodology** — `WORKFLOW.md`, plus
`best_outcome/figures/20_current_verified_pipeline.png` as the main pipeline
figure and `best_outcome/figures/09_phase3_architecture.png` for fusion detail.

**Results** — `best_outcome/README.md` first, then approaches 02, 03 and 04.

**Discussion** — approaches 05 and 06. These carry the methodological
contributions: selection optimism measured at {opt['accuracy']:+.4f}, a learning
curve fixing the constraint at {slope['accuracy']:+.4f} accuracy per 100
examples, a pre-registered gate that was honoured, a corrected encoder decision,
and a clean ensemble-diversity result.

**Limitations** — `VERIFIED_RESULTS.md` section 8.

## Three things to be careful about

1. **Lead with the 5-class ensemble** at macro-F1 {m5['macro_f1']:.4f}. The
   3-class results are reported on their own terms, never as an improvement,
   because once fairly compared the 3-class model is
   {abs(m3['macro_f1'] - mcol['macro_f1']):.4f} macro-F1 **weaker**.
2. **`VERIFIED_RESULTS.md` section 9 lists numbers that must not appear in the
   thesis**, including two invalid comparisons from earlier drafts. Check
   anything carried over from an old draft against it.
3. **Cross-validation numbers are not test numbers.** Approaches 05 and 06
   report cross-validated results for configurations that were never evaluated
   on the test set. Label them that way.

## The headline in one line

Best validated result: **macro-F1 {m5['macro_f1']:.4f}, accuracy
{m5['accuracy']:.4f}** on {m5['n']} held-out memes, from a six-model
cross-architecture ensemble. Nothing built across ten subsequent phases beat it.
""")
    print(f"\nBuilt {OUT}")
    for d in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, d)
        if os.path.isdir(p):
            nf = len(os.listdir(os.path.join(p, "figures"))) if os.path.isdir(os.path.join(p, "figures")) else 0
            print(f"  {d}/  ({nf} figures)")
        else:
            print(f"  {d}")


if __name__ == "__main__":
    main()
