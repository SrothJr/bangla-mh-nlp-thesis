# v2 — starting over, as a researcher would

**What this folder is.** A clean restart. Not a continuation of the nine phases
that came before, and not bound by their decisions. The question it asks is the
one a researcher joining the project today would ask: given everything now known,
where is the best remaining opportunity, and what has been assumed rather than
tested?

Nothing here modifies the previous work. New code lives in `v2_research/code/`,
new results in `v2_research/outputs/`. The test set is not touched anywhere in
v2 before a pre-registered gate.

---

## 1. What the previous effort established

Worth stating plainly, because a restart that ignores it would waste months.

**Settled, and not worth re-litigating:**

| Finding | Evidence |
|---|---|
| Data is the binding constraint | Learning curve still climbing at +0.030 accuracy per +100 examples, no flattening |
| Combiner-stage work is exhausted | Stacking, class weights, soft averaging: under +0.01 combined |
| Ordinal losses do not help | CORAL rejected, CORN rejected, only label smoothing kept |
| Out-of-domain data actively hurts | 300 external negatives dropped the target class F1 by 0.113 |
| Ensembling is the most reliable lever | Larger than every loss and fusion change combined |
| Selection optimism was inflating results | Measured at +0.027 accuracy in the old protocol |

**Best results on record**, all on the same 196 held-out test memes:

| Model | Task | Accuracy | Macro-F1 |
|---|---|---|---|
| Phase 3 ensemble | 5-class | 0.5306 | 0.4984 |
| Same, collapsed to 3 | 3-class | **0.6582** | 0.6024 |
| Phase 7.10 lock | 3-class | 0.6071 | 0.5730 |

Best cross-validated configuration from the previous effort: **0.6873 accuracy,
0.6397 macro-F1** (15 models, three architectures, plus depression features).

---

## 2. The gap a fresh look finds

Nine phases produced roughly 150 experiments. Almost all of them varied the same
three things: the fusion architecture, the loss, and the combination rule.

**Almost none of them varied the input representation.**

Every model in the entire project consumed the same two frozen vectors:
BanglaBERT's `[CLS]` (768-d) and SigLIP's pooled output (1152-d). Those two
choices were made in the first week and never revisited, while everything
downstream was tuned exhaustively on top of them.

That is where a researcher should look, and it is what v2 does first.

---

## 3. Experiment 0 — is the representation the bottleneck?

A deliberately simple linear probe (standardize, multinomial logistic
regression, 5-fold CV on the 777 train+val examples). No neural heads, no early
stopping, no tuning. The only question is which frozen input carries the most
linearly accessible label signal.

| Representation | Accuracy | Macro-F1 |
|---|---|---|
| TEXT `[CLS]` (e3b) — what all nine phases used | 0.5238 | 0.4689 |
| TEXT mean-pool over tokens | 0.5199 | 0.4684 |
| TEXT max-pool over tokens | 0.4955 | 0.4425 |
| TEXT `[CLS]` + mean concatenated | 0.5071 | 0.4527 |
| IMAGE SigLIP pooled — what all nine phases used | 0.5714 | 0.5103 |
| IMAGE SigLIP mean over patches | 0.5315 | 0.4418 |
| **IMAGE CLIP ViT-Large** | **0.6216** | **0.5697** |

**Two results, one expected and one not.**

The text hypothesis was **wrong**. `[CLS]` versus mean-pooling is a wash
(0.5238 against 0.5199). The prior reasoning — that a `[CLS]` shaped by a
depression objective should transfer poorly — sounded good and is not supported.
Recorded as a negative result.

The image side was the surprise. **CLIP outperformed SigLIP by 5 points** on a
representation that had been treated as settled since week one. A plain logistic
regression on CLIP features alone (0.6216) comes within 6 points of the previous
effort's entire 15-model ensemble (0.6873).

---

## 4. Experiment 1 — reopening a decision that was wrongly closed

CLIP was not untested. Phase 2.4 tested it and rejected it hard: 0.4666 against
SigLIP's 0.5330, a 6.6-point loss, recorded as confirming SigLIP was "the right
call for this specific dataset and task."

**That experiment was confounded.** It ran CLIP inside the gated architecture,
which consumes Stage A contrastive alignment projections. Phase 2.4's own notes
record that CLIP aligned badly under that recipe: it early-stopped at epoch 9
against SigLIP's 457, reaching 2.6% retrieval against SigLIP's 6.7%.

So the comparison measured *representation quality times alignment quality*, and
attributed the whole result to representation. A encoder that aligns poorly under
one contrastive recipe is not therefore a worse encoder.

Re-run with a trained neural head, no alignment stage in any arm, matched folds,
seeds and the unbiased protocol:

| Arm | Accuracy | Macro-F1 | vs SigLIP |
|---|---|---|---|
| text + SigLIP — the project's choice | 0.6371 | 0.5917 | — |
| text + CLIP — rejected in Phase 2.4 | 0.6448 | 0.5919 | +0.0077, P(better) 0.655 |
| **text + SigLIP + CLIP — never tried** | **0.6628** | **0.6154** | **+0.0257, P(better) 0.943** |

**CLIP is not better than SigLIP. It is different from SigLIP.** Head to head
they are statistically indistinguishable. Used together they beat either alone by
2.6 accuracy points.

The two encoders capture complementary information, and the combination was never
tried because Phase 2.4 framed the question as *which one* rather than *both*.
That framing cost the project a real gain for seven phases.

---

## 5. Experiment 2 — do the gains stack?

The previous effort's one success was feeding the depression classifier's output
distribution in as input features (+0.0129). A 2x2 design tests whether that
combines with dual encoders.

| | SigLIP only | SigLIP + CLIP |
|---|---|---|
| **no depression features** | 0.6371 | **0.6628** |
| **with depression features** | 0.6499 | 0.6577 |

| Effect | Value |
|---|---|
| Dual encoders alone (A) | +0.0257 |
| Depression features alone (B) | +0.0129 |
| Both together (AB) | +0.0206 |
| If perfectly additive | +0.0386 |
| **Interaction** | **−0.0180** |

**They do not stack.** Adding depression features on top of dual encoders makes
things slightly worse. This matches the pattern seen throughout the project:
independent gains overlap and cancel rather than adding.

The practical consequence is that dual encoders **replaces** the depression
features rather than joining them, and it is the larger of the two effects.

---

## 6. Outcome — COMPLETE

**v2 did not beat the previous approach.** Full write-up in `FINDINGS.md`,
figure in `figures/v2_01_dual_encoder_findings.png`.

| Configuration | CV accuracy | CV macro-F1 |
|---|---|---|
| Previous effort, 15 models, SigLIP only | 0.6744 | 0.6291 |
| Previous effort best, with depression features | **0.6873** | 0.6397 |
| v2 best (20 models, dual-concat added) | 0.6834 | 0.6380 |
| Difference | **-0.0039** | -0.0017 |

Level, within noise, marginally behind.

**What happened between the component gain and the ensemble.** Dual encoders is
worth +0.0257 on concat and +0.0373 on gated, and it is the largest
single-change gain measured anywhere across ten phases. After assembly it is
worth +0.0090, and adding more dual members pushes it back to exactly the
baseline.

The instructive step was configuration B: CLIP in every member's head. Two of
three members improved individually, and the ensemble got **worse** (0.6474
against 0.6744). Shared features make members' errors correlated, and ensembles
are paid in disagreement. The winning composition keeps the SigLIP-only members
and adds dual-encoder members alongside them.

**The prediction in Section 8 was accurate.** It was written before the ensemble
ran and said the projection would land "approximately level with the existing
benchmark rather than clearly past it". That is what happened.

**Recommendation: stop modelling here.** See `FINDINGS.md` Section 11.
