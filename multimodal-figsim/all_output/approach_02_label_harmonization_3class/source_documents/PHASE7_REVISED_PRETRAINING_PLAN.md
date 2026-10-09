# Phase 7 (revised): External meme pretraining + label harmonization — plain-language plan

**Status: planned only. Nothing in this document has been started.** This
revises the original `FIGSIM_HARMONIZATION_AND_PRETRAINING.md` brief based
on two problems we found while discussing it, and reorders the steps for
efficiency. Read this file, not the original, for what would actually run.

---

## 1. The two problems we found, explained simply

**Problem 1 — touching DAPT-BanglaBERT is risky.** DAPT-BanglaBERT isn't
just "a Bangla language model" — it was specially trained on this exact
kind of mental-health-adjacent text before this project even started, and
we already proved (Phase 2.1) that this specialization is worth a lot:
swapping it for a bigger, more generic model made results much worse
(−0.073 macro-F1). The original plan's pretraining step would have let
this specialized model's weights shift slightly (via LoRA) while learning
from the new external meme datasets. That's risky — we could accidentally
soften the exact specialization that's currently working.

**Problem 2 — the external datasets speak different Bangla than FigSIM
does.** CMBAN and BN-HIB are memes people wrote natively in Bangla —
real slang, real informal grammar. But FigSIM's own text doesn't start
as Bangla: it starts as English text on the meme image, which we OCR,
then machine-translate into Bangla using NLLB, and only then feed to
DAPT-BanglaBERT. Machine-translated Bangla ("translationese") reads
differently than Bangla someone actually typed. Pretraining on native
Bangla meme slang and then applying that to machine-translated Bangla
risks teaching the model a style of Bangla that FigSIM's actual text
doesn't really look like.

**The fix for both problems:** don't route the external pretraining
through DAPT-BanglaBERT or through Bangla text at all. Use the external
datasets to improve the **image side only**. Details in Section 3.

---

## 2. The new order of operations, and why it changed

The original document put "harmonize the labels" and "do external
pretraining" as somewhat separate phases without a strong reason to do
one before the other. We found a better order:

1. **Run the free diagnostic check first**, before building anything.
2. **Only if that looks promising**, invest the (large) engineering
   effort of external pretraining.
3. **Do the real harmonized retraining last**, on top of whatever
   pretraining produced (or on top of today's frozen features, if
   pretraining is skipped or doesn't pan out).

This is better because the pretraining step doesn't use FigSIM's suicide
labels at all (it only uses the external datasets' own labels, like
sarcasm or hate). That means **the label scheme decision (3-class vs.
4-class vs. 5-class) and the pretraining decision don't depend on each
other** — so there's no cost to deciding the cheap thing (labels) early
and the expensive thing (pretraining) only if it's worth it.

---

## 3. The full sequence, step by step

### Step 0 — Safety net (already done)

Current progress is already committed to git locally and backed up to
Google Drive. Nothing below can lose any of the existing, working
pipeline — all of this new work would happen in new files, under a
clearly separate experiment folder, never overwriting anything that
exists today.

### Step 1 — Free diagnostic check (minutes, no training)

We already have saved prediction probabilities from the current best
5-class model on the validation set. This step just adds those numbers
together in different groupings to simulate what a 3-class or 4-class
version would have scored, **without training anything new**:

```
3-class:  P(class0) = P(None)
          P(class1) = P(Wish to be dead) + P(Suicide ideation)
          P(class2) = P(Suicide planning) + P(Suicide attempt/death)

4-class:  same idea, but keeps ideation+planning merged into one class
          (this directly targets the ideation/planning confusion we
          found earlier)
```

**Decision gate:** if this shows a meaningfully higher, more reliable
score than the current 5-class result, harmonization is worth pursuing
further. If it shows little or no difference, we can stop here and not
invest further effort in the label-scheme idea — cheaply, before any
real cost was spent.

### Step 2 — External dataset audit (if Step 1 looks promising)

Check exactly what's inside the already-downloaded CMBAN and BN-HIB
folders: image counts, text availability, label distributions, and
remove duplicates or unusable records. Build one small index file
listing every external record and what data is available for it. No
model training yet.

### Step 3 — DAPT-safe external pretraining

This is the redesigned core step. Two independent tracks, both of which
**never load or modify DAPT-BanglaBERT**:

- **Track A — image-only.** Take the external meme images (both CMBAN and
  BN-HIB) and lightly adapt SigLIP (the image model) using each dataset's
  *own* labels as a learning signal — sarcasm, hate, offensiveness,
  sentiment. This is purely "make the image model better at recognizing
  meme-style pictures," with no text encoder involved at all.

- **Track B — image+text, but using SigLIP's own paired text half.**
  SigLIP was originally trained as a matched image+text pair of models;
  we've only ever used its image half. Its own text half can be used
  instead of DAPT, paired with CMBAN's *English* translations (CMBAN
  provides these). This keeps everything in English/SigLIP's native
  space — no Bangla, no DAPT, no translation-style mismatch.

  (BN-HIB doesn't reliably provide English translations, so it's used
  in Track A only.)

Only small LoRA adapters (or, more conservatively, just new projection
layers) get trained in either track — never full model weights, matching
what we already proved safe in Phase 6.

**Decision gate:** evaluate the resulting image representation using the
external datasets' own validation splits only (never FigSIM's test set).
If it doesn't show a clear improvement, this pretraining checkpoint is
simply not used going forward — the existing frozen-SigLIP pipeline stays
exactly as it is today. No harm done either way.

### Step 4 — Harmonized FigSIM fine-tuning

Train the actual FigSIM classifier on the chosen label scheme (3-class
primary, 5-class and binary as auxiliary signals, matching the original
document's design) — starting from the Step 3 pretraining checkpoint if
it was kept, or from today's existing frozen features if not. Reuse the
already-proven components: gated fusion, cross-attention, ordinal-style
supervision, multi-seed ensembling.

### Step 5 — Model selection

Same discipline as every phase so far: compare candidates on the
validation set (and grouped cross-validation where practical), never on
test. Keep only what shows a real, validation-confirmed improvement.

### Step 6 — One deliberate final test check (only if something new is actually chosen)

If, after all of the above, a new configuration genuinely beats the
current locked best (0.5695 validation / 0.4984 test), do exactly one
final, deliberate test-set evaluation to confirm it — the same rare,
one-time-touch discipline used throughout this whole project. If nothing
beats the current best, the test set is not touched again, and the
current locked result stands.

### Step 7 — Documentation

Whatever the outcome — success, partial success, or a clean "didn't
help" result — it gets written up honestly in `IMPROVEMENT_PLAN.md` and
folded into `THESIS_SECTIONS_DRAFT.md`, the same way every phase in this
project has been documented so far, including the ones that failed.

---

## 4. What this guarantees, in one paragraph

DAPT-BanglaBERT is never loaded, forward-passed, or modified anywhere in
this revised plan — it stays exactly as it is today throughout every
step, and only reappears, completely unchanged, at the very end for the
final FigSIM classification step, exactly like it works right now. The
cheapest, free check (Step 1) happens before anything expensive is built,
so we don't invest in the big pretraining engineering effort unless it's
already looking worthwhile. Nothing overwrites the current working
pipeline, checkpoints, or results at any point — everything happens in
new files. The test set is touched at most once more, only if something
genuinely earns it.
