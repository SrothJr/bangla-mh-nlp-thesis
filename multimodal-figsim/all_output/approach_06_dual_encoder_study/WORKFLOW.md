# Workflow — dual image encoders

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
