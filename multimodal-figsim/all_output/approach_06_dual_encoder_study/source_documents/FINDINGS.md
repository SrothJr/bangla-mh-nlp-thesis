# v2 findings — what a fresh look actually found

**Bottom line, up front: v2 found a real effect that nine phases missed, and
corrected a decision that was closed on confounded evidence. It did not produce
a better final system.** Best v2 configuration reaches 0.6834 cross-validated
accuracy against the previous effort's 0.6873. Essentially level, marginally
behind.

That is the honest summary, and everything below explains how it got there and
why the finding is still worth having.

Figure: `figures/v2_01_dual_encoder_findings.png`.

---

## 1. The question v2 started from

Nine phases produced roughly 150 experiments. Nearly all of them varied one of
three things: the fusion architecture, the loss function, or the combination
rule.

**Almost none varied the input representation.** Every model in the project
consumed the same two frozen vectors — BanglaBERT's `[CLS]` and SigLIP's pooled
output. Both choices were made in week one and never revisited, while everything
downstream was tuned exhaustively on top of them.

A researcher arriving today would look exactly there, because it is the largest
untested assumption in the project.

---

## 2. Experiment 0 — the probe that started it

A deliberately simple test: L2-normalise the frozen features, fit multinomial
logistic regression, score by 5-fold cross-validation on the 777 train+val
examples. No neural head, no early stopping, no tuning. The only question is
which frozen input carries the most linearly accessible label signal.

| Representation | Accuracy | Macro-F1 |
|---|---|---|
| TEXT `[CLS]` (e3b) — used by all nine phases | 0.5238 | 0.4689 |
| TEXT mean-pool over tokens | 0.5199 | 0.4684 |
| TEXT max-pool over tokens | 0.4955 | 0.4425 |
| TEXT `[CLS]` + mean concatenated | 0.5071 | 0.4527 |
| IMAGE SigLIP pooled — used by all nine phases | 0.5714 | 0.5103 |
| IMAGE SigLIP mean over patches | 0.5315 | 0.4418 |
| **IMAGE CLIP ViT-Large** | **0.6216** | **0.5697** |

**The text hypothesis was wrong, and that is reported as a result.** The
reasoning was that a `[CLS]` shaped by a depression-classification objective
should transfer poorly to suicide severity, and that mean-pooling would beat it.
It does not: 0.5238 against 0.5199 is a wash. Plausible mechanism, no effect.

**The image side was the surprise.** CLIP beat SigLIP by 5 accuracy points on a
choice that had been settled since week one. For scale, a plain logistic
regression on CLIP features alone (0.6216) lands within 6 points of the previous
effort's entire 15-model ensemble (0.6873).

---

## 3. Experiment 1 — why the earlier decision was wrong

CLIP was not untested. **Phase 2.4 tested it and rejected it decisively**:
0.4666 against SigLIP's 0.5330, a 6.6-point loss, written up as confirming that
SigLIP "was the right call for this specific dataset and task, not just a
reasonable guess."

That conclusion does not follow from that experiment, and the reason is visible
in Phase 2.4's own notes.

**The confound.** Phase 2.4 ran CLIP inside the gated architecture. That
architecture consumes Stage A contrastive alignment projections, so its
performance depends jointly on how good the encoder is and on how well that
encoder aligns to the text space under one particular contrastive recipe. The
notes record that CLIP aligned badly: it early-stopped at epoch 9 against
SigLIP's 457, reaching 2.6% retrieval against SigLIP's 6.7%.

So the experiment measured *representation quality times alignment quality* and
attributed the entire result to representation. An encoder that aligns poorly
under one recipe is not thereby a worse encoder.

**The clean test.** Same trained neural head, matched folds and seeds, the
unbiased protocol, and no alignment stage in any arm, so encoders are judged on
their representations alone:

| Arm | Accuracy | Macro-F1 | vs SigLIP |
|---|---|---|---|
| text + SigLIP — the project's choice | 0.6371 | 0.5917 | — |
| text + CLIP — rejected in Phase 2.4 | 0.6448 | 0.5919 | +0.0077, P(better) 0.655 |
| **text + SigLIP + CLIP — never tried** | **0.6628** | **0.6154** | **+0.0257, P(better) 0.943** |

**CLIP is not better than SigLIP. It is different from SigLIP.** Head to head
they are statistically indistinguishable. Used together they beat either alone.

The combination was never tried because Phase 2.4 framed the question as *which
one* rather than *both*. That framing cost the project a real gain for seven
phases.

---

## 4. Experiment 2 — the gains do not stack

The previous effort's one success was feeding the depression classifier's output
distribution in as input features (+0.0129). A 2x2 design tests whether the two
independent gains combine.

| | SigLIP only | SigLIP + CLIP |
|---|---|---|
| no depression features | 0.6371 | **0.6628** |
| with depression features | 0.6499 | 0.6577 |

| Effect | Value |
|---|---|
| Dual encoders alone (A) | +0.0257 |
| Depression features alone (B) | +0.0129 |
| Both together (AB) | +0.0206 |
| If perfectly additive | +0.0386 |
| **Interaction** | **−0.0180** |

**They do not stack.** Both together is worse than dual encoders alone. This is
the same pattern the previous effort hit repeatedly: independent gains in this
system overlap and cancel rather than adding, because they are different routes
to the same limited signal.

Practical consequence: dual encoders **replaces** the depression features rather
than joining them.

---

## 5. Experiment 3 — per architecture, and the diversity lesson

CLIP was given to all three architectures through the raw-vector side channel
their heads already use (Section 7 explains the design).

| Architecture | SigLIP only | + CLIP | Delta |
|---|---|---|---|
| concat | 0.6371 | 0.6628 | **+0.0257** |
| gated | 0.6088 | 0.6461 | **+0.0373** |
| cross-attention | 0.6486 | 0.6229 | **−0.0257** |

Two improve, one degrades. CLIP is not uniformly useful — cross-attention
already has fine-grained access to SigLIP patches, and a second global image
vector appears to compete with rather than complement that.

Then the ensembles, and the most instructive result in v2:

| Configuration | Accuracy | Macro-F1 |
|---|---|---|
| A — 15 models, SigLIP only | 0.6744 | 0.6291 |
| B — 15 models, CLIP in every head | **0.6474** | 0.5939 |
| C — 20 models, A plus dual-concat | **0.6834** | 0.6380 |

**B is worse than A even though two of its three members improved
individually.** That is not a contradiction, it is ensemble diversity. Giving
every member the same extra 1024 dimensions makes their errors more correlated,
and a correlated ensemble is worth less than the sum of its members. C wins
precisely because it keeps the SigLIP-only members and adds dual-encoder members
alongside them, so the two families disagree in useful ways.

This reframes the design question from "should members use CLIP" to "which
mixture of member types produces the most useful disagreement".

---

## 6. Experiment 4 — composition search

Six compositions, all specified from Experiment 3's per-architecture evidence
*before* any of them were scored, to avoid picking the best of 63 possible
combinations on the same 777 rows.

| Composition | Models | Accuracy | Macro-F1 | vs A |
|---|---|---|---|---|
| A — all base | 15 | 0.6744 | 0.6291 | — |
| **C — A + dual-concat** | 20 | **0.6834** | **0.6380** | **+0.0090** |
| D — A + dual-concat + dual-gated | 25 | 0.6744 | 0.6243 | +0.0000 |
| E — drop weakest member (base-gated) | 20 | 0.6757 | 0.6250 | +0.0013 |
| F — best variant per architecture | 15 | 0.6744 | 0.6216 | +0.0000 |
| G — all six groups | 30 | 0.6744 | 0.6230 | +0.0000 |

Best is C at **0.6834** (95% CI 0.6499 to 0.7156), macro-F1 0.6380, QWK 0.4888,
P(better than A) 0.841.

Per class: no expressed severity 0.510 F1, suicidal thought or desire 0.739,
high acuity 0.665.

**Four of six compositions land on exactly 0.6744**, identical to the baseline.
Beyond a point, adding members stops changing the vote at all. That is a
saturation signal: the ensemble has extracted what the member pool contains.

---

## 7. Design decisions, and why they were made that way

**How CLIP reaches each architecture.** The obvious shortcut would have been to
put CLIP only in concat, where it was first tested. That would have been the
weaker option.

Gated consumes Stage A alignment projections trained on SigLIP, and no CLIP
alignment exists. Cross-attention consumes SigLIP patch tokens, and CLIP patches
were never extracted. Neither can take CLIP through its main image pathway
without new preprocessing.

But both already concatenate a raw text vector directly into their final
classifier head, alongside their fused representation. That side channel is
exactly where an extra global image vector belongs: it bypasses the alignment and
attention machinery, which is what makes it usable, and it needs no new
preprocessing.

```
concat   [ text ; SigLIP ; CLIP ]                          -> classifier
gated    [ gated ; orthogonal residual ; text ; CLIP ]     -> classifier
xattn    [ attention-pooled ; text ; CLIP ]                -> classifier
```

Each architecture keeps its own mechanism intact. Only the head widens.

**Why no new alignment was trained for CLIP.** It would have reintroduced the
exact confound that made Phase 2.4's conclusion unsafe. Judging the encoders
without any alignment stage is what makes this comparison clean.

---

## 8. Workflow

```
  973 labelled Bangla memes
  582 train / 195 val / 196 test    (test sealed throughout v2)
            |
            v
  ┌─────────────────────────────────────────────────────────────┐
  │  FROZEN ENCODERS -- none fine-tuned anywhere in v2           │
  │                                                              │
  │  OCR -> NLLB-3.3B -> Bangla text ─┐                          │
  │  VLM reasoning ───────────────────┴─> DAPT-BanglaBERT        │
  │                                        -> text 768-d         │
  │  image ─┬─> SigLIP so400m  -> 1152-d pooled, 729 patches     │
  │         └─> CLIP ViT-Large -> 1024-d       <-- v2 addition   │
  └─────────────────────────────────────────────────────────────┘
            |
            v
  ┌─────────────────────────────────────────────────────────────┐
  │  MEMBER POOL -- 6 groups x 5 seeds = 30 trained models       │
  │                                                              │
  │   base_concat   base_gated   base_xattn     (SigLIP only)    │
  │   dual_concat   dual_gated   dual_xattn     (+ CLIP in head) │
  └─────────────────────────────────────────────────────────────┘
            |
            v
  ┌─────────────────────────────────────────────────────────────┐
  │  COMPOSITION -- chosen from per-architecture evidence,       │
  │  pre-specified before scoring                                │
  │                                                              │
  │  WINNER C:  base_concat + base_gated + base_xattn            │
  │             + dual_concat            = 20 models             │
  │             soft probability averaging                       │
  └─────────────────────────────────────────────────────────────┘
            |
            v
     0.6834 accuracy, 0.6380 macro-F1   (cross-validated, 777)
            |
            v
     GATE -- not yet attempted. Test set still sealed.
```

**Evaluation protocol, applied to every number above.** 5-fold stratified CV
over the 777 train+val examples. Within each fold, an inner split handles early
stopping, so the outer rows never influence any selection decision. Paired
bootstrap with 5000 resamples on every comparison, reporting P(better) rather
than point estimates alone. Identical folds and seeds to every prior measurement
in the project, so v2 numbers are directly comparable to the previous effort's.

---

## 9. Where v2 stands against the previous effort

| Configuration | CV accuracy | CV macro-F1 |
|---|---|---|
| Previous effort, 15 models, SigLIP only | 0.6744 | 0.6291 |
| Previous effort best, with depression features | **0.6873** | 0.6397 |
| **v2 best (C, 20 models with dual-concat)** | 0.6834 | 0.6380 |
| v2 minus previous best | **−0.0039** | −0.0017 |

**v2 did not beat the previous approach.** It landed 0.4 accuracy points behind,
which is well inside noise on 777 examples, so the two are best described as
level.

---

## 10. What v2 contributes anyway

1. **A corrected decision.** Phase 2.4's rejection of CLIP was based on a
   confounded comparison, and the record now says so with evidence. That matters
   independently of the score.
2. **A real, replicated component-level effect.** Dual encoders is worth +0.0257
   on concat and +0.0373 on gated, P(better) 0.943. It is the largest
   single-change gain measured anywhere in this project, roughly double the
   previous best.
3. **A clean demonstration of ensemble diversity.** Configuration B improved two
   of three members individually and still made the ensemble worse. That is a
   textbook effect shown on real data from this project, and it is a good
   discussion point.
4. **Two more honest negatives.** The text-pooling hypothesis failed, and the
   gains do not stack.
5. **A saturation signal.** Four of six compositions land on identical accuracy.
   Beyond a point the member pool contains no further disagreement to exploit.

---

## 11. Honest assessment of what is left

The previous effort's learning curve measured the binding constraint at +0.030
accuracy per +100 training examples, still climbing at 777. v2 does not change
that. What v2 shows is that even a genuinely overlooked representation
improvement — one that beats every other single change tried across ten phases —
is worth about 2.6 points at the component level and close to nothing after
assembly.

That is itself evidence about where the ceiling comes from. If a better encoder
combination cannot move the assembled system, the limit is not the encoders.

**Recommendation: stop here on modelling.** The test benchmark to beat is 0.6582
accuracy, v2's 0.6834 cross-validated projects to roughly level with it, and the
accumulated selection optimism across everything compared on these same 777 rows
argues against spending a test evaluation on a configuration that is not clearly
ahead.

The remaining lever is unchanged and is not a modelling lever: more labelled
in-domain examples.
