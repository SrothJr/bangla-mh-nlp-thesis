# Shared-encoder alignment (Option 2): anchoring Stage A to DAPT's native space

**Status: planned, about to start.** Written before any new code runs,
per this project's established discipline. This is Option 2 from
`IMPROVEMENT_PLAN.md`'s "Phase 7 continued -- shared-encoder alignment
options" section, given its own full workflow document as requested.

## 1. The problem this addresses

DAPT-BanglaBERT and SigLIP were pretrained completely independently, on
unrelated objectives -- nothing forces their embedding spaces to be
geometrically comparable just because a given text and image describe
the same meme. This exact mismatch caused a real, diagnosed failure
early in this project (Day 4): the E5 gated-fusion architecture's gate
collapsed, learning to ignore text and lean on image alone for 71% of
validation items, because the two raw embedding spaces weren't
alignable enough for the gate to learn a sensible mixture.

**The existing fix ("Stage A," Day 5-6):** train two small projection
layers from scratch --

```
text_proj:  768  -> 256   (trainable, learned from FigSIM's 582 pairs)
image_proj: 1152 -> 256   (trainable, learned from FigSIM's 582 pairs)
```

-- with a self-supervised contrastive (InfoNCE) loss that pulls a
meme's own text and image projections together and pushes mismatched
pairs apart. This works (figure `03_gate_alpha_fix.png` documents the
gate no longer collapsing afterward), and the entire pipeline since has
been built successfully on top of it. But it asks the model to learn an
entire new 256-dimensional shared space from only 582 training
examples -- a lot to ask from very little data.

## 2. The idea: don't learn a new space, anchor to DAPT's existing one

Instead of training projections for **both** sides into an arbitrary
new space, train a projection for **only the image side**, mapping
SigLIP's embeddings directly into DAPT's own already-rich, already
domain-specialized 768-dimensional output space:

```
text:  DAPT's raw 768-dim output, used AS-IS, no projection, no training
image_proj: 1152 -> 768   (the only new trainable component)
```

The same InfoNCE contrastive objective is kept (pull matching
text/image pairs together, push mismatched pairs apart), just operating
in DAPT's native space instead of a freshly-invented one. This is
strictly **less** new capacity to learn than the current Stage A (one
trainable projection instead of two), while leveraging DAPT's already-
proven representational structure more directly -- literally making
"the already-trained text encoder" the shared space, as originally
proposed in discussion.

**DAPT-BanglaBERT itself is untouched** -- same as every other
experiment in this project since Phase 6 confirmed the value of never
disturbing its weights. Only a new image-side projection module is
introduced and trained; DAPT is loaded exactly as always (`AutoModel`,
frozen, no head, no LoRA here).

## 3. Does the fusion gate need to change?

No. The existing gate mechanism is dimension-agnostic in design intent:

```python
gate_in = torch.cat([t, i, conf], dim=-1)
alpha = torch.sigmoid(self.gate(gate_in))
gated = alpha * t + (1 - alpha) * i
```

Under the current Stage A, `t` and `i` are both 256-dim. Under this new
anchored alignment, they would both be 768-dim (DAPT's native size)
instead. The gate's own linear layers need their input/output
dimensions adjusted to match (a mechanical change, `ALIGN_SHARED_DIM`
becomes 768 instead of 256), but the mechanism itself -- compute alpha,
blend -- carries over completely unchanged in concept.

## 4. Workflow, step by step

1. **Build a new alignment module** (`DAPTAnchoredProjection` or
   similar) with only an image-side projection (1152 -> 768) and no
   text-side projection. Train it with the same InfoNCE contrastive loss
   already used for Stage A, same train/val split, same protocol
   (self-supervised, no labels involved).
2. **Save this as a NEW checkpoint**, under its own clearly-named
   directory (e.g. `outputs/checkpoints/stage_a_dapt_anchored/`) --
   never overwriting the existing Stage A checkpoint, which stays fully
   available and in use by the current best pipeline regardless of this
   experiment's outcome.
3. **Report the same retrieval-accuracy diagnostic** already built for
   the original Stage A (`top1_retrieval_accuracy` in
   `train_contrastive_align.py`) for this new anchored alignment, so we
   have a direct, comparable measure of alignment quality before even
   testing it downstream -- if this number is clearly worse than the
   existing Stage A's, that's an early, cheap signal not to bother
   continuing.
4. **Only if step 3 looks reasonable**, build the E6-style gated+orth
   classifier on top of this new alignment (same architecture, same
   training recipe, `ALIGN_SHARED_DIM=768` instead of 256), test on both
   the 3-class harmonized target and (for completeness/comparison) the
   original 5-class target.
5. **Compare against the current best reference points**:
   - 3-class: simple concat, 0.6425 (Phase 7 Step 4) -- current best
   - 3-class: gated+orth with EXISTING Stage A, 0.6144 (Phase 7.5,
     already shown worse than simple concat)
   - 5-class: gated+orth with EXISTING Stage A, 0.5638-0.5695 range
     (Phases 1-4)
6. **Decision gate, same as every other Phase 7 item:** if this new
   alignment doesn't clearly beat the relevant reference point, it is
   not adopted -- the current best pipeline (whichever configuration
   that is at the time) continues unchanged. No risk either way.
7. **Document the result honestly** in `IMPROVEMENT_PLAN.md`, whichever
   way it goes, following the same pattern as every other Phase 7 item.

## 5. What could go wrong, named in advance

- **DAPT's 768-dim space was shaped entirely by text pretraining** -- it
  may simply not have "room" or appropriate structure for image
  concepts to map into usefully, no matter how well the image
  projection is trained. This is a real, non-trivial risk; the step-3
  retrieval-accuracy check exists specifically to catch this early and
  cheaply, before investing in the full downstream classifier training.
- **Small data, again:** the image-side projection is still being
  learned from only 582 training pairs, same fundamental constraint as
  everything else in this project. This idea reduces the amount of NEW
  capacity being learned (good), but doesn't increase the amount of
  data available (unchanged constraint).
- **This is genuinely new territory**, unlike Option 1 (which reused
  fully proven components) -- appropriate to treat this result with the
  same skepticism as any other untested architectural idea in this
  project, not assume it will work just because the reasoning is sound.

## 6. Validation-only protocol, unchanged

FigSIM's locked test set is not touched at any point in this workflow.
All comparisons in step 5 are against validation-split reference
points already established elsewhere in this project.
