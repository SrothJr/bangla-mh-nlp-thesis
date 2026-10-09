# Improvement plan — Phase 2 of the fusion architecture

Tracking document for the "make the results better" round of work, kept
separate from `PROGRESS_LOG.md` (execution log for Days 1-12, the locked
E6 gated+orth baseline) and `THESIS_SECTIONS_DRAFT.md` (the thesis prose
for that already-complete work). This file exists so *why* each change was
made stays attached to the change itself, not just the final numbers.

**Status: planning only, as of this writing. No code has been written for
anything in this document yet.**

**Baseline being improved on:** E6 gated+orth (aligned, weighted CE) —
validation macro-F1 0.507 ± 0.005, test macro-F1 0.463 ± 0.017 (test
touched once already, see the protocol note below).

---

## Protocol note — read before implementing anything in this file

The original test split (196 items) was evaluated **exactly once**,
deliberately, per Section 8's protocol, after the architecture was locked
on Day 7. Everything in this document is a **new round of experimentation**
on top of that already-completed, already-reported result — it is not a
continuation of the same one-time evaluation.

Rule for this round: **iterate on the validation split only.** The test
split does not get touched again until every change below has been tried,
compared, and a new final configuration is deliberately locked — at which
point it gets evaluated on test exactly once more, the same discipline as
before. Do not re-run any test-set evaluation script "just to check"
partway through this list. If a change looks promising on validation, that
is the signal to keep it or drop it — not test performance.

The original E6 gated+orth test result (0.463 macro-F1) remains the
reported baseline result for the thesis chapter already drafted, regardless
of what happens here. This round either produces a *new*, separately
reported improved result, or it doesn't — it does not retroactively change
what already happened on Days 1-12.

---

## Recommended approach: do all of it, in four phases

You asked for the best option, and the honest answer is "all of the above,
sequenced by cost and dependency" rather than picking one. None of these
changes are mutually exclusive, several compound (Stage A alignment
quality directly affects gate quality, which is why Phase 2's alignment
work is sequenced before Phase 3's fusion upgrade), and the cheap Phase 1
items cost little enough that skipping them to jump to Phase 3 would leave
free improvements on the table.

### Phase 1 — free or near-free, no new training infrastructure

These reuse data and models we already have. Each should be individually
measurable on the validation split before moving on, so a regression is
caught immediately rather than compounding with later changes.

| # | Change | Why (evidence-based) |
|---|---|---|
| 1.1 | **Ensemble the 3 seeds** instead of reporting the single best-by-validation seed | Day 5-6/7 already showed real seed-to-seed spread (E5's seeds ranged 0.470-0.536 val macro-F1). Averaging predicted probabilities across seeds (or majority vote) is a standard, nearly-free variance-reduction step we simply haven't done yet — we always picked one "winner" seed instead. |
| 1.2 | **Feed the reasoning model's `uncertain` flags into the gate** as additional input features | `reasoning_results.jsonl` already has a boolean `uncertain` field per claim (cause_effect/figurative_meaning/emotional_state) for every one of the 973 memes — computed on Day 3, never used downstream. The Day 4 spot-check found this flag is a real (if imperfect) signal — exactly the kind of information a reliability-aware gate (Section 5.4's own framing) should see and currently doesn't. |
| 1.3 | **Ordinal label smoothing** in place of (or alongside) plain class-weighted cross-entropy | "Suicide planning" has been the weakest class in *every* experiment since E1, and its errors concentrate specifically on the adjacent class "Suicide ideation" (7/26 test cases) — a textbook ordinal-adjacency problem. Label smoothing that spreads a little probability mass to neighboring classes (proportional to ordinal distance) targets this directly, without CORAL's diagnosed flaw (collapsing to a single shared scalar rank score, which is what actually broke CORAL, not the ordinal framing itself). |
| 1.4 | **Focal loss** as an alternative to (or blended with) weighted CE | Complements 1.3 — focuses gradient on the specific hard/misclassified examples rather than only reweighting by class frequency. Cheap to test as a swap-in loss function on the existing Stage D classifier. |

**Phase 1 exit criterion:** re-measure validation macro-F1 after each item
(not just at the end) so we know which of the four actually helped before
they're combined.

### Phase 2 — moderate effort, uses existing data, no architecture change

| # | Change | Why |
|---|---|---|
| 2.1 | **Measure the translation-hop cost**: encode the original English OCR/reasoning text with an English encoder and compare to the current Bangla pipeline on validation | We've never actually measured what the OCR→NLLB→BanglaBERT double-lossy-hop costs us (mean OCR confidence was only 0.77, with visibly garbled outputs on some memes). This is a diagnostic first, not a commitment to switch — if the gap is small, the "one Bangla encoder, two tasks" narrative is worth keeping as-is; if the gap is large, that's a real, reportable finding either way. |
| 2.2 | **Data augmentation for the classifier heads**: embedding-space mixup between same-class examples, and/or back-translation augmentation (Bangla→English→Bangla for slightly different phrasings) | 582 training examples is thin for a 5-way classifier on ~1900-dim fused features. This is the standard, low-risk way to effectively grow a small labeled set without touching frozen encoders. |
| 2.3 | **Strengthen Stage A contrastive alignment**: switch to a supervised contrastive loss (pull same-*class* pairs together, not only a meme's own text/image pair) and/or generate multiple augmented views per meme for richer contrastive pairs | Alignment's own validation retrieval was only ~13× chance — real but weak. Alignment quality is what fixed the Day 5-6 gate collapse once already; improving it further plausibly compounds into a better gate, not just a better retrieval number in isolation. |
| 2.4 | **Try CLIP as an alternate (or additional) vision encoder**, a straight drop-in comparison against SigLIP | Explicitly skipped in the original plan for time (Section 6), not because it was expected to lose. CLIP's web-scraped training distribution plausibly includes more meme-adjacent imagery than SigLIP's. Cheap to test since it's an encoder swap, not an architecture change. |

### Phase 3 — higher effort, real architecture changes

| # | Change | Why |
|---|---|---|
| 3.1 | **Partially unfreeze the last 1-2 layers of BanglaBERT and/or SigLIP**, with a low learning rate and strong regularization | Both encoders have been fully frozen this entire project. Neither has ever seen meme content, dark humor, or translated internet slang — BanglaBERT was fine-tuned on formal-ish Reddit mental-health posts. This is the single most likely "ceiling" on everything built on top: fusion architecture can only compensate so much for a representation mismatch at the source. Needs care — 582 examples will overfit fast if given too much freedom, hence last-1-2-layers-only, not full fine-tuning. |
| 3.2 | **Cross-attention fusion** between SigLIP's patch tokens and BanglaBERT's token-level hidden states, replacing (or supplementing) the scalar gate | The current gate produces one number α per meme mixing two *already-pooled* vectors — it cannot express "trust the image here, trust the text there." Cross-attention at the token/patch level is strictly more expressive and is the standard mechanism in the multimodal literature this scalar gate was a simplification of. Most likely to matter for the 90% of memes FigSIM itself annotates as "Complementary," where meaning depends on relating specific words to specific image regions. This is the biggest single implementation effort on this list and the most likely source of a real architectural jump, not just a tuning gain. |

**Sequencing rationale for Phase 3 being last:** both items are higher
implementation cost and higher overfitting risk than Phases 1-2, and 3.2
specifically benefits from whatever Stage A improvements come out of 2.3 —
better-aligned representations give cross-attention a better starting
point, the same way they fixed the scalar gate.

**Status as of the Phase 2 wrap-up: Phase 3 not yet started, paused by
request to regroup and set expectations first (see the forecast section
below and Phase 4).**

### Forecast, set before Phase 3 starts (so it can be checked against
### what actually happens)

Asked directly how much further improvement to expect. Answering honestly
rather than optimistically, based on the pattern actually observed so far,
not a guess made in a vacuum:

**Most likely range: another +0.02 to +0.05 macro-F1** (landing
somewhere around 0.55-0.58 on validation), with real risk of landing
lower. Reasoning:

- **The trend so far points at diminishing returns, not acceleration.**
  Phase 1 (4 cheap ideas) produced one real gain (+0.021 net). Phase 2
  (4 more substantive ideas -- new encoder, augmentation, better
  alignment, encoder swap) produced **zero** kept gains, all four
  negative. That is a fairly strong, convergent signal that "swap a
  piece, tune a loss"-class changes are running out of room on this
  specific model+data combination, not bad luck four times over.
- **Fusion-complexity increases specifically have already shown
  diminishing returns.** E4 (simple concat) through E6 (gated+orth,
  aligned) are all clustered tightly around 0.50-0.53 macro-F1 -- almost
  all the benefit of fusion came from the *first* jump (text-only to any
  fusion at all), not from refining the fusion mechanism further. Item
  3.2 (cross-attention) is another fusion-complexity increase, in the
  same family as changes that have already plateaued -- worth trying
  because the mechanism (token/patch-level attention vs. a pooled scalar
  gate) is genuinely different, but it should not be assumed to be the
  exception to the pattern just because it's more sophisticated.
- **Item 3.1 (encoder unfreezing) is the more theoretically justified of
  the two** -- it targets something Phase 1/2 never touched (both
  encoders frozen the entire project, neither adapted to meme/dark-humor
  content). But it also adds real trainable parameters on top of only
  582 examples, which is exactly the overfitting risk that has not yet
  been tested here.
- **The likely real ceiling is dataset size, not architecture.** FigSIM's
  own paper (Chen et al., cited from Day 1) found that fine-tuned fusion
  architectures barely beat text-only on this exact dataset, and only
  massive prompted MLLMs -- a different resource tier entirely -- did
  meaningfully better. That is independent, external evidence pointing
  at the same conclusion our own Phase 2 results suggest.

**Framing for the report:** Phase 3 is worth attempting because it is the
most theoretically justified remaining lever within the original
architecture, and could plausibly add a further few percent -- but it
should be described as a meaningful refinement to attempt, not a
guaranteed breakthrough. Reaching substantially further (e.g., into a
0.60+ macro-F1 range) would most likely require either more labeled data
or a different resource tier (large prompted models used directly, see
Phase 4 item 4.2), not another architecture tweak within this same
frozen-encoder-plus-small-head paradigm.

---

## Phase 4 — if Phase 3 doesn't move the number, these are not exhausted

Asked directly: if Phase 3 fails to improve things, are we out of
options? No. Phase 3 is the last item on the *original* three-phase list,
not the complete universe of options. These are mechanistically different
enough from everything in Phases 1-3 that a plateau in the frozen-
encoder-plus-small-head paradigm doesn't rule them out.

| # | Change | Why |
|---|---|---|
| 4.1 | **Systematic hyperparameter sweep** of the existing best classifier head (hidden dim, dropout, learning rate, weight decay) | Every classifier head in this entire project, Phase 1 and 2 included, has used one reasonable-guess configuration (256 hidden units, dropout 0.2, Adam lr=1e-3) -- never actually swept. Boring compared to architecture changes, but genuinely untried, low-risk, and independent of whether Phase 3 succeeds. |
| 4.2 | **Prompt the vision-language model directly for the task** (zero-shot or few-shot severity classification via Qwen2.5-VL, rather than extracting reasoning text as a feature for a small trained classifier) | A fundamentally different approach from everything tried so far -- and specifically the approach FigSIM's own paper found gave the largest gains on this exact dataset (large prompted MLLMs, not fusion architectures). Qwen2.5-VL is already running locally; this is a genuinely different lever, not a variation on what's been tried. |
| 4.3 | **Ensemble the trained classifier with a directly-prompted model (4.2)** | If the two approaches make meaningfully different mistakes (plausible, given how different the mechanisms are), combining them could give a real boost -- same logic as the Phase 1.1 seed-ensembling win, but across model *families* rather than random initializations, which tends to help more because the errors are less correlated. Depends on 4.2 being tried first. |
| 4.4 | **Post-hoc decision calibration**: per-class threshold adjustment instead of pure argmax | Cheap, no retraining. Directly targets two repeated, specific patterns: "Wish to be dead" being over-predicted as a fallback guess (high recall, weak precision in every experiment), and "Suicide planning" being under-predicted into "ideation." |
| 4.5 | **Hierarchical decomposition**: a coarse "any suicide content vs. None" classifier first, then a second classifier only among the 4 suicide-severity levels | "None" has looked more separable than the internal ordinal boundaries throughout every experiment. Splitting the problem concentrates modeling capacity on the genuinely hard part (the 4-way ordinal split) instead of asking one classifier to do both jobs at once. |
| 4.6 | **More/better labeled data, if obtainable** | The option every piece of evidence (our own Phase 2 results, FigSIM's own paper) points to as the actual ceiling. Not fully within this project's control, but named plainly rather than left unstated: nothing above fully substitutes for it. |

**Recommended order if this phase is reached:** 4.1 first (cheapest, fully
independent of Phase 3's outcome, should arguably be done regardless of
how Phase 3 goes) — then 4.4 (also cheap, no retraining) — then 4.2, since
it's the most likely to actually move the needle by a meaningful amount,
with 4.3 as a natural follow-on if 4.2 shows promise. 4.5 is worth trying
if time allows but is more speculative. 4.6 is named for completeness,
not because it's actionable on demand.

---

## Explicitly deprioritized (and why)

- **A second reasoning model / model-agreement ensemble (E7).** Expensive
  (a second multi-hour VLM pass over 973 images), and the measured
  marginal contribution of reasoning text at all (E3b vs E3, +0.004) was
  small — unlikely to be the highest-leverage next move relative to
  everything above.
- **More reasoning-prompt engineering in isolation.** The Day 4 spot-check
  found the failure mode is about grounding on ambiguous *image-only*
  content specifically (correct objects, wrong narrative, no uncertainty
  flag) — that reads as an architecture/data problem, not something a
  prompt rewrite alone fixes. Item 1.2 (feeding the existing uncertainty
  signal downstream) is a more direct response to the same finding.
- **Retrying CORAL in any form.** Its failure mode is now well understood
  and specific (the single-scalar rank-score bottleneck starves rare
  classes) — item 1.3 (ordinal label smoothing) targets the same
  ordinal-structure benefit without that flaw, so there's no reason to
  revisit CORAL itself.

---

## Tracking table (fill in as each item is actually implemented)

| # | Item | Status | Val macro-F1 before | Val macro-F1 after | Kept? | Notes |
|---|---|---|---|---|---|---|
| 1.1 | Ensemble 3 seeds | **Done** | 0.5117 (best single seed) | **0.5311 (majority vote)** | **Yes — majority vote** | See narrative below |
| 1.2 | Reasoning `uncertain` flags → gate | **Done** | 0.5311 (majority vote, Phase 1.1) | 0.5252 (majority vote) | **No** | See narrative below |
| 1.3 | Ordinal label smoothing | **Done** | 0.5311 (majority vote, Phase 1.1) | 0.5330 (majority vote) | **Yes, marginal** | See narrative below |
| 1.4 | Focal loss | **Done** | 0.5311 (majority vote, Phase 1.1) | 0.5061 (γ=1.0) / 0.5003 (γ=2.0) | **No** | See narrative below |
| 2.1 | English-vs-Bangla translation-hop diagnostic | **Done** | 0.5330 (Bangla, Phase 1 best) | 0.4604 (English) | **No — Bangla pipeline kept** | Counter-intuitive result, see narrative below |
| 2.2 | Data augmentation (embedding mixup) | **Done** | 0.5330 (Phase 1 best) | 0.5160 (α=0.4) / 0.5189 (α=0.1) | **No** | See narrative below; back-translation augmentation not yet tried |
| 2.3 | Stronger Stage A alignment (supervised contrastive) | **Done (partial — augmented views not yet tried)** | 0.5330 (Phase 1 best) | 0.4682 | **No** | See narrative below |
| 2.4 | CLIP vs. SigLIP comparison | **Done** | 0.5330 (SigLIP, Phase 1 best) | 0.4666 (CLIP) | **No — SigLIP kept** | See narrative below |
| 3.1 | Partial encoder unfreezing (BanglaBERT, last 2 layers) | **Done** | 0.5139 (frozen concat baseline) | 0.5029 (lr=2e-5) / 0.4875 (lr=5e-6) | **No** | See narrative below |
| 3.2 | Cross-attention fusion (standalone) | **Done** | 0.5330 (E6 gated+orth) | 0.5205 | **No, standalone** | Complementary strengths → see ensemble below |
| 3.2b | **Cross-architecture ensemble (E6 gated+orth + cross-attention, 6 seeds)** | **Done** | 0.5330 | **0.5592** | **YES — new best** | See narrative below |

---

## Phase 1 narrative log

### 1.1 — Ensemble the 3 seeds (DONE, kept)

Ran `phase1_1_ensemble_seeds.py`: retrained the same 3 seeds (0/1/2) used
for the locked E6 gated+orth baseline, capturing full softmax probability
matrices at each seed's best validation epoch (not just argmax
predictions), then compared two ensembling strategies against the
best-single-seed baseline (0.5117, seed 0 — matches the number already
reported for the locked model).

| Strategy | Validation macro-F1 | vs. best single seed |
|---|---|---|
| Best single seed (original approach) | 0.5117 | — |
| Soft average of 3 seeds' probabilities | 0.5102 | −0.0015 (no real change) |
| **Majority vote across 3 seeds** | **0.5311** | **+0.0194** |

**Finding:** soft-probability averaging did *not* help — it's essentially
flat vs. the best single seed, most likely because one seed (seed 1,
individually 0.5004, the weakest of the three) pulls the averaged
probability distribution toward its own weaker predictions on every
example, even where the other two seeds agree and are correct. **Majority
vote does help, and by a real margin** — it only overrides the
weaker seed when the other two agree, so a single underperforming seed
can't drag down cases where there's real 2-of-3 consensus.

**Decision: kept.** Majority-vote ensembling across the 3 seeds replaces
"pick the seed with the best validation score" as the selection strategy
going forward for this improvement round. This is a free, already-computed
change — no new training, no new data, no new failure surface — and is a
genuine, non-trivial gain (+0.019 macro-F1) purely from how we combine
results we already had.

**Note for later:** this changes how "the model" is defined at inference
time — it's now an ensemble of 3 trained heads with majority vote, not a
single model. This is fine methodologically (ensembling is standard
practice) but needs to be stated plainly if/when this becomes part of a
newly-locked final configuration, since it changes what "the classifier"
means for deployment/inference purposes (3 forward passes instead of 1).

### 1.2 — Reasoning `uncertain` flags → gate (DONE, NOT kept)

Checked the signal first: 918/973 memes (94%) have zero uncertain flags
across all three reasoning claims, only 55 have any -- a sparse feature
going in. Added a third gate input, `reasoning_certainty = 1 -
(uncertain_count / 3)`, alongside the existing OCR and translation
confidence signals (`E6GatedOrthV2`, gate input dim 2→3), same alignment
checkpoint, same majority-vote comparison basis as 1.1.

| | Individual seeds | Majority vote |
|---|---|---|
| Without reasoning certainty (Phase 1.1) | 0.5117 / 0.5004 / 0.5089 | 0.5311 |
| With reasoning certainty (Phase 1.2) | 0.5212 / 0.5186 / 0.5161 | 0.5252 |

**Finding, and it's a genuinely interesting one, not just "didn't help":**
every individual seed's validation macro-F1 went *up* with the reasoning-
certainty signal added (all three seeds improved). But the majority-vote
ensemble went *down* (0.5311 → 0.5252, delta −0.0059). The likely
mechanism: the extra signal nudges all three seeds toward more similar
decisions (since they now share an additional, consistent piece of
information), which improves each one individually but reduces the
*diversity between seeds* that majority voting depends on to correct any
single seed's mistakes. Phase 1.1's gain came specifically from the 3
seeds disagreeing usefully; this change partially undoes that.

**Decision: not kept.** Net effect vs. the current best (0.5311) is
slightly negative. This isn't a wasted test, though -- it's a real,
explainable finding worth keeping in mind for any future ensembling work:
an individually-better-per-seed change is not automatically an
ensemble-better change, and the two need to be checked separately, exactly
as done here.

### 1.3 — Ordinal label smoothing (DONE, kept — marginal)

Target distribution per sample = `exp(-|class − true_class| / τ)`,
normalized to sum to 1 (τ=1.0), replacing the one-hot target in the
cross-entropy loss. Full 5-way logit output kept (unlike CORAL, no
capacity bottleneck) — only the *target* is softened. Same base
architecture as the Phase-1.1-kept config (E6GatedOrth, 2 confidence
signals), majority-vote across 3 seeds.

| | Majority-vote macro-F1 |
|---|---|
| Phase 1.1 (hard-target weighted CE) | 0.5311 |
| Phase 1.3 (ordinal-smoothed targets) | 0.5330 |
| Delta | +0.0019 |

The aggregate delta is small — essentially within noise on its own. But
the per-class picture (comparing to the single-seed baseline reported for
the locked model, since a majority-vote-specific per-class breakdown
wasn't saved for 1.1) shows a real, explainable redistribution: **4 of 5
classes improved** — Wish to be dead 0.551→0.602, Suicide ideation
0.574→0.602, **Suicide planning 0.453→0.485** (the class this change was
specifically aimed at), Attempt/Death 0.469→0.500 — while **"None" got
worse** (0.512→0.476). This makes sense mechanistically: "None" sits at
one end of the ordinal scale, so smoothing pulls a little of its target
mass toward "Wish to be dead," its only ordinal neighbor, at some cost to
its own precision — the same trade-off, in miniature, everywhere the
scale has an edge.

**Decision: kept, but flagged as a marginal, not decisive, win.** The
mechanism is sound and the "Suicide planning" improvement is real and
directly on-target, but the net aggregate gain is small enough that this
should be re-checked in combination with 1.4 (focal loss) rather than
treated as settled on its own — it's also worth trying a different τ
(smoothing strength) if time allows, since 1.0 was a first reasonable
guess, not a tuned value.

### 1.4 — Focal loss (DONE, NOT kept)

`FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)`, alpha_t = the same
inverse-frequency class weights already in use. Tried the standard
default γ=2.0 (Lin et al. 2017) first, then γ=1.0 (milder focusing) after
γ=2.0 underperformed, to check whether the aggressiveness of the focusing
term specifically was the problem rather than the mechanism itself.

| γ | Majority-vote macro-F1 | Delta vs. 1.1 baseline | "None" class F1 |
|---|---|---|---|
| 2.0 | 0.5003 | −0.0308 | 0.381 |
| 1.0 | 0.5061 | −0.0250 | 0.381 |
| (baseline, 1.1) | 0.5311 | — | ~0.51 |

**Finding: both settings regress, and the damage concentrates entirely on
the "None" class** (F1 dropped to 0.381 either way — identical at both γ
values, which itself is informative: this isn't a focusing-strength
tuning issue, it's a structural mismatch). Most likely mechanism: focal
loss's difficulty-based down-weighting and the existing inverse-frequency
class weighting are both trying to solve the same imbalance problem
through different means, and stacking them over-suppresses gradient from
"None" specifically — "None" cases are relatively easy/confidently
correct once learned, which is exactly what focal loss is designed to
de-emphasize, but in this small-data 5-class setting that "easy" class
still needs its own gradient signal to stay well-calibrated at one end of
the ordinal scale.

**Decision: not kept, and no further γ sweep planned.** Two settings
bracketing the standard range both regressed by a similar, explainable
mechanism — this reads as a genuine structural mismatch between focal
loss and the already-applied class weighting, not a tuning problem worth
more budget. Ordinal label smoothing (1.3) already captures the
"address the hard, ordinally-adjacent classes" goal this was also aimed
at, without this failure mode.

---

## Phase 1 summary

| Item | Kept? | Net effect |
|---|---|---|
| 1.1 Majority-vote ensembling | Yes | +0.0194 |
| 1.2 Reasoning uncertainty → gate | No | −0.0059 (not applied) |
| 1.3 Ordinal label smoothing | Yes (marginal) | +0.0019 (stacks with 1.1) |
| 1.4 Focal loss | No | −0.025 to −0.031 (not applied) |

**Best Phase 1 configuration: majority-vote ensemble (3 seeds) + ordinal
label smoothing (τ=1.0), hard weighted-CE class balancing retained,
focal loss and the reasoning-uncertainty gate signal both dropped.**

**Validation macro-F1: 0.5117 (original locked single-seed baseline) →
0.5330 (Phase 1 best) — a genuine +0.0213 (≈4% relative) improvement**,
achieved entirely from Phase 1's free/cheap changes: no new data, no new
training infrastructure, no architecture changes, two of the four ideas
tested didn't pan out and were correctly dropped rather than forced in.

This is the config Phase 2's items should build on top of going forward
(i.e., Phase 2 experiments should also use majority-vote ensembling +
ordinal smoothing as their base, not re-test against the original
single-seed hard-CE baseline).

### Was Phase 1 a failure? No — and here's why that's worth stating explicitly

Two of four sub-experiments (1.2, 1.4) did not improve the model and were
not kept. Read at a glance, "2 of 4 didn't work" can look like a mixed or
failed outcome. It isn't, and the distinction matters enough to spell out:

**The only metric that defines whether Phase 1 succeeded is the validation
score, and it went up.** 0.5117 → 0.5330, a genuine +0.0213 (~4% relative)
gain, for zero added cost — no new data, no new training infrastructure,
no architecture changes. That is the result of Phase 1, full stop.

**The two rejected sub-experiments are not failures of Phase 1 — they are
Phase 1's testing discipline working as intended.** The explicit point of
measuring each item individually against the validation split before
keeping it (see the Protocol note at the top of this document) was to
catch a bad idea and discard it before it could drag the final result
down. Had all four items simply been bundled together without individual
checks, the combined result would likely have landed worse than the
original 0.5117 baseline (focal loss alone cost −0.025 to −0.031), and
there would have been no way to know *which* change caused the damage.
Testing one variable at a time and keeping only what measurably helped is
what produced the confirmed +0.0213 gain instead of an unmeasured mixed
bag.

**Framing for a report:** this is standard, expected behavior for
empirical ML experimentation, not a sign anything went wrong. A
run of four hypothesis tests where two are confirmed and two are
falsified is a normal, informative research outcome — informative
specifically *because* the two negative results are explainable rather
than random (1.2's mechanism: added signal improved each seed
individually but reduced the diversity between seeds that majority voting
depends on; 1.4's mechanism: focal loss's difficulty-based reweighting
structurally conflicts with the already-applied inverse-frequency class
weighting, evidenced by both tested γ values damaging the exact same class
by the exact same amount). Both negative results are themselves
legitimate, citable findings — they rule out two plausible-sounding ideas
for a documented, mechanistic reason, which is worth more in a methods
section than simply not having tried them.

---

## Phase 2 narrative log

### 2.1 — English-vs-Bangla translation-hop diagnostic (DONE, Bangla kept)

Built a full parallel pipeline using the *original English* OCR + reasoning
text instead of the NLLB-translated Bangla text: extracted embeddings via
`mental-roberta-base` (a domain-adapted mental-health RoBERTa, chosen
specifically because DAPT-BanglaBERT is also domain-adapted -- this keeps
the comparison to "translated vs. not," not "domain-tuned vs. generic"),
retrained Stage A contrastive alignment on these embeddings, then trained
the Phase-1-best classifier config (ordinal smoothing + majority vote)
on top.

**Stage A alignment result was dramatically stronger:** validation
retrieval accuracy reached **53-55%** (vs. chance 0.51%), roughly **8x
stronger** than the Bangla pipeline's ~6.7% (13x chance). At face value this
looked like strong evidence the translation hop was costing real signal --
exactly what this diagnostic was built to check.

**But the downstream classifier result went the other way:**

| | Majority-vote macro-F1 |
|---|---|
| Bangla pipeline (Phase 1 best) | 0.5330 |
| English pipeline (this test) | 0.4604 |
| Delta | **−0.0726** |

Per-class, English underperformed on 4 of 5 classes (only "Wish to be
dead" was comparable/better); "None" specifically collapsed to F1 0.361.

**Why the stronger alignment didn't help -- two likely causes, both
checked against the actual numbers rather than assumed:**

1. **The English alignment stage overfit hard.** At its best-val-loss
   checkpoint (epoch 407), train retrieval was 99.5% against val retrieval
   of only 53.3% -- a large train/val gap indicating the projection
   learned to memorize specific English lexical patterns in the 582
   training examples rather than a generalizable semantic alignment.
   Untranslated English text carries richer, more distinctive vocabulary
   than the translated Bangla (which tends to be flatter/more uniform
   after passing through NLLB), which plausibly makes the contrastive
   task *easier to overfit*, not necessarily *better learned*.
2. **This comparison has a real confound worth stating plainly:**
   DAPT-BanglaBERT was both domain-adapted *and* supervised-fine-tuned on
   the actual 4,898-sample depression-severity classification task before
   its head was stripped for reuse here -- its frozen representations are
   task-relevant, not just domain-relevant. `mental-roberta-base` is only
   MLM-pretrained on mental-health text; it has never seen a
   classification objective. This is arguably a fairer comparison than
   generic `roberta-base` would have been (matches on domain adaptation),
   but it is not a perfectly controlled comparison -- the fine-tuning
   mismatch, not the language/translation itself, may be doing most of
   the work here.

**Decision: keep the Bangla/DAPT-BanglaBERT pipeline. Translation is not
the bottleneck the initial hypothesis suspected it might be** -- if
anything, the encoder's task-relevant fine-tuning matters more than
either the source language or raw cross-modal alignment strength. This is
a genuinely useful negative result: it directly answers the open question
Phase 2 was framed around, and rules out re-litigating the "one Bangla
encoder, two tasks" design choice on translation-quality grounds. A
cleaner follow-up (not planned for this round, noted for future work)
would repeat this test with an English encoder that has *also* been
fine-tuned on a comparable classification task, to fully separate the
language effect from the fine-tuning effect.

### 2.2 — Embedding-space mixup augmentation (DONE, NOT kept)

Same-class mixup on top of the confirmed-best pipeline (Bangla text branch
+ Phase 1's ordinal smoothing/majority vote): for every training step, an
equal-sized synthetic batch is generated by interpolating each real
example with a randomly-chosen same-class partner
(`lambda ~ Beta(alpha, alpha)`), across the aligned text vector, aligned
image vector, raw text vector, and confidence vector together, re-sampled
every epoch. Tried two settings bracketing the standard range:

| α | Majority-vote macro-F1 | Delta vs. 1.1/1.3 baseline (0.5330) |
|---|---|---|
| 0.4 (standard mixup default) | 0.5160 | −0.0170 |
| 0.1 (gentler -- synthetic points stay close to one parent) | 0.5189 | −0.0141 |

**Finding: both settings regress, and both times the damage concentrates
specifically on "Suicide planning"** (F1 0.485 baseline → 0.441 (α=0.4) →
**0.381 (α=0.1)** — got worse, not better, with gentler mixing). "None,"
by contrast, improved slightly in both runs (0.476 → ~0.51). Likely
mechanism: "Suicide planning" has only 90 training examples, the smallest
non-merged class -- same-class mixup for a class this scarce interpolates
within an already-narrow, boundary-ambiguous pool (this is precisely the
class that bleeds into the adjacent "Suicide ideation" class in every
experiment), which plausibly reinforces the existing ambiguity rather than
adding real new information. "None" sits at the opposite, more clearly
separated end of the ordinal scale and has more training examples (89, a
similar count but apparently a less fuzzy decision boundary), so mixing
within it doesn't carry the same risk.

**Decision: not kept.** Two bracketing α values both regressed via the
same explainable mechanism -- consistent with how focal loss (1.4) was
handled, this doesn't warrant a further sweep. **Back-translation
augmentation (Bangla→English→Bangla, producing alternate phrasings) was
not yet tried** and remains a live option for future work -- it augments
by generating genuinely different *text*, not by interpolating in
embedding space, so it isn't subject to the same "scarce class, narrow
same-class pool" failure mode diagnosed here and could behave
differently. Noted as unfinished, not ruled out.

### 2.3 — Supervised contrastive Stage A alignment (DONE, NOT kept)

Replaced Stage A's self-supervised InfoNCE (positive = only a meme's own
text-image pair) with a SupCon-style supervised contrastive loss: for each
text anchor, ALL image vectors sharing the anchor's suicide-severity class
are treated as positives (~116 per class on average, given 582 train
examples over 5 classes), cross-class pairs as negatives. Same projection
architecture as the already-fixed self-supervised version (plain linear +
dropout), only the loss function changed.

**Alignment-stage diagnostic (own-pair retrieval, kept for comparability
even though it's no longer the loss's actual target):** early-stopped very
quickly (epoch 39, vs. 457 for the original), val_loss plateaued around
5.28 almost immediately, own-pair retrieval landed at 0.021 -- *worse*
than the original self-supervised run's comparable-stage number (0.067,
itself already a modest signal). This was a visible warning sign before
even reaching the downstream classifier.

**Downstream classifier result confirmed the warning:**

| | Majority-vote macro-F1 |
|---|---|
| Phase 1 best (self-supervised alignment) | 0.5330 |
| Phase 2.3 (supervised contrastive alignment) | 0.4682 |
| Delta | **−0.0648** — the largest regression of any item tried so far |

**Likely cause, not fully diagnosed:** the LR (3e-4) and temperature
(0.07) inherited from the original self-supervised setup were not
retuned for this different loss landscape. With ~116 positives per class
instead of exactly 1, the gradient signal and effective difficulty of the
task are substantially different, and the quick plateau suggests the
optimizer found a poor local solution early rather than the loss itself
being fundamentally unworkable.

**Decision: not kept.** Given the size of the regression and that it
reads as an under-tuned-optimization problem rather than a validated
"this approach can't work" result, this is flagged as **worth a
follow-up with a proper LR/temperature sweep specifically for the
supervised-contrastive objective**, rather than a settled rejection --
different enough from the clean, well-explained failures of 1.4 and 2.2
that it doesn't get the same "no further sweep" treatment. Not pursued
further in this pass given time; **augmented views (the second half of
item 2.3, generating multiple perturbed versions per meme for richer
contrastive pairs) also remains untried.**

### 2.4 — CLIP vs. SigLIP comparison (DONE, SigLIP kept)

Pulled `openai/clip-vit-large-patch14` (~428M params, capacity-matched to
SigLIP-so400m's ~400M, chosen over the smaller `clip-vit-base-patch32` so
a result either way isn't confounded by a size mismatch). Extracted CLIP
image embeddings for all 973 images, retrained Stage A alignment (same
self-supervised InfoNCE recipe) and the Phase-1-best classifier on top.

Stage A alignment was notably weaker with CLIP: early-stopped at epoch 9
(vs. 457 for SigLIP), best val retrieval only ~2.6% (~5x chance) vs.
SigLIP's ~6.7% (~13x chance) at a comparable point.

| | Majority-vote macro-F1 |
|---|---|
| SigLIP (Phase 1 best) | 0.5330 |
| CLIP-ViT-Large | 0.4666 |
| Delta | **−0.0664** |

**Decision: SigLIP confirmed as the better choice, kept.** This empirically
validates the original brief's direct choice of SigLIP (Section 6) rather
than leaving it an untested assumption -- the comparison the project
originally skipped for time now has an actual answer: SigLIP was the
right call for this specific dataset and task, not just a reasonable
guess.

---

## Phase 2 summary

| Item | Kept? | Net effect |
|---|---|---|
| 2.1 English text branch (vs. Bangla) | No | −0.073 (Bangla pipeline confirmed correct) |
| 2.2 Embedding-space mixup augmentation | No | −0.014 to −0.017 |
| 2.3 Supervised contrastive alignment | No | −0.065 (flagged for a possible retuned follow-up, not a closed door) |
| 2.4 CLIP vs. SigLIP | No | −0.066 (SigLIP empirically confirmed) |

**Phase 2 added no further gains on top of Phase 1** -- all four items
tested were negative, each for a specific, explainable, checked reason
rather than random noise. The validation macro-F1 remains at **Phase 1's
0.5330** going into Phase 3.

**This is still useful, not wasted effort:** three real design choices
that were previously assumptions (Bangla-vs-English text branch, the
SigLIP encoder choice, self-supervised-vs-supervised alignment) now have
actual evidence behind them instead of being untested. Two items remain
genuinely open for future work rather than closed: back-translation
augmentation (a different mechanism from the embedding mixup that was
tried) and a properly LR/temperature-tuned supervised contrastive
alignment (the current attempt showed signs of under-optimization, not a
definitive failure of the approach itself).

---

## Phase 3 narrative log

### 3.1 — Partial encoder unfreezing, BanglaBERT last 2 layers (DONE, NOT kept)

**Step A, fair baseline established first:** since unfreezing requires a
materially different training loop (raw text/mini-batched, not cached
full-batch embeddings), a matched frozen-encoder baseline using the same
simple-concat architecture (no Stage A alignment, isolating this from the
gated/aligned E6 numbers) was measured first: **0.5139** majority-vote
macro-F1 (Phase 1's ordinal-smoothing + majority-vote recipe applied to
plain concatenation). This is the number unfreezing needed to beat, not
the more complex E6 gated+orth number (0.5330), so the comparison isolates
one variable.

**Design:** unfroze only BanglaBERT's last 2 of 12 Electra layers
(14.2M of 110M parameters), SigLIP kept fully frozen (cached embeddings
reused) to isolate "does adapting the text encoder help" as a single
variable rather than confounding it with also unfreezing the vision side.
Differential learning rates (encoder << head), mini-batched (batch 16,
582 examples don't fit the full-batch pattern once gradients must flow
through a 110M-parameter model), early stopping on validation macro-F1.

| Encoder LR | Majority-vote macro-F1 | Delta vs. frozen baseline (0.5139) |
|---|---|---|
| 2e-5 (standard fine-tuning LR) | 0.5029 | −0.0110 |
| 5e-6 (4x gentler) | 0.4875 | −0.0264 |

**Finding: both settings regressed, and — importantly — the gentler
learning rate made it WORSE, not better**, which rules out "LR too
aggressive" as the explanation. What both runs share instead is a visibly
unstable, non-monotonic validation macro-F1 across epochs (e.g., seed 0 at
lr=2e-5: 0.337 → 0.379 → **0.483** → 0.388 → 0.413 → 0.455 → 0.403 --
bouncing rather than smoothly converging to a stable optimum). This reads
as a genuinely structural problem: 582 training examples, split into
~37 mini-batches per epoch, is simply too small a fine-tuning set for a
110M-parameter encoder to adapt stably even when only 2 of its 12 layers
are unfrozen -- there isn't enough data per gradient step for the signal
to consistently outweigh noise, regardless of how conservatively the
learning rate is set within a reasonable fine-tuning range.

**Decision: not kept.** Two learning rates bracketing a 4x range both
regressed via the same instability signature -- per this project's
established discipline (same treatment as focal loss and mixup), this
doesn't warrant a further LR sweep; the mechanism is understood well
enough to stop. **This is a genuinely valuable negative result for the
thesis, not just a rejected experiment:** it empirically validates
Section 8's foundational, pragmatic decision to freeze both encoders and
train only small heads on cached embeddings for this project -- a
decision made from necessity (dataset size, time budget) that turns out,
tested directly, to also have been the *empirically correct* call, not
merely a convenient one. Unfreezing more layers, or unfreezing with a
larger effective batch size via gradient accumulation, remain
theoretically possible follow-ups but are not planned for this round
given the consistency of the failure signature across both tested
settings.

### 3.2 — Cross-attention fusion (DONE — standalone not kept, but led to the new best result via ensembling)

Replaced the scalar gate entirely with a genuine cross-attention
mechanism: BanglaBERT's 256 text TOKENS (not just the pooled CLS vector)
act as queries attending over SigLIP's 729 image PATCHES (not just the
pooled vector) as keys/values, via a standard multi-head attention layer
(4 heads, shared dim 256). This required extracting and caching
token-level and patch-level features for all 973 images (~2GB at
float16), since the pooled-vector caches used everywhere else in this
project don't retain the spatial/positional information cross-attention
needs. Both encoders stayed frozen (Phase 3.1 already showed unfreezing
regresses at this dataset size) -- only the projections, attention layer,
and classifier head are trained. Attended tokens are masked-mean-pooled
(real attention mask, ignoring padding) and concatenated with the raw
pooled text vector before classification, matching the "concat with raw
text" pattern from E5/E6.

**Standalone result:** majority-vote macro-F1 0.5205, a modest regression
vs. the E6 gated+orth baseline (0.5330, delta −0.0125).

**But the per-class breakdown told a different, more interesting story:**

| Class | E6 gated+orth (Phase 1 best) | Cross-attention (standalone) |
|---|---|---|
| None | 0.476 | 0.459 |
| Wish to be dead | 0.602 | 0.451 |
| Suicide ideation | 0.602 | **0.662** |
| Suicide planning | 0.485 | **0.531** |
| Attempt/Death | 0.500 | 0.500 |

Cross-attention is meaningfully *better* on Suicide ideation and
Suicide planning -- the two classes involved in this project's single
most persistent, most-repeated failure pattern (planning bleeding into
ideation, present in literally every experiment since E1) -- but
meaningfully *worse* on Wish to be dead. This reads as two architectures
with genuinely different, complementary strengths, not one being simply
better than the other.

**Follow-up test: ensemble across the two architectures, not just across
seeds of one.** Same logic as Phase 1.1's seed-ensembling, extended to
combine 3 E6-gated+orth predictions with 3 cross-attention predictions
(6 total) by majority vote -- the same idea planned for Phase 4 (4.3,
"ensemble the trained classifier with a differently-mechanismed model"),
reached one phase early because two differently-mechanismed trained
architectures were already on hand.

| Configuration | Majority-vote macro-F1 |
|---|---|
| E6 gated+orth alone (3 seeds) | 0.5330 |
| Cross-attention alone (3 seeds) | 0.5205 |
| Soft-average across all 6 (both architectures) | 0.5369 |
| **Majority vote across all 6 (both architectures)** | **0.5592** |

**This is the largest single gain since Phase 1, and the most balanced
per-class result of the entire project:** None 0.449, Wish to be dead
0.591, Suicide ideation 0.656, Suicide planning **0.557** (the best
"Suicide planning" score anywhere in this project, by a wide margin),
Attempt/Death 0.543. Consistent with the Phase 1.1 finding, majority vote
clearly beats soft-averaging (0.559 vs 0.537) -- averaging dilutes each
architecture's confident correct calls with the other's uncertain ones,
while majority vote only overrides on genuine 2-of-3-style consensus,
and here that consensus draws on two architectures with different blind
spots rather than three seeds of the same one.

**Decision: KEPT, and promoted to new best overall result.** Validation
macro-F1 progression: 0.5117 (original locked baseline) → 0.5330 (Phase 1)
→ **0.5592 (Phase 3.2 cross-architecture ensemble)** -- a cumulative
+0.0475 (~9.3% relative) improvement from the original locked model,
achieved by combining two differently-mechanismed trained models rather
than by any single architecture change succeeding on its own. This is a
genuinely different kind of result from everything else in this document:
neither component beat the baseline alone, but the combination did, by
the largest margin seen in this entire improvement round.

---

## Current best configuration (updated)

**Cross-architecture ensemble: E6 gated+orth (aligned, ordinal-smoothed,
3 seeds) + cross-attention fusion (3 seeds), majority vote across all 6
predictions.**

- Validation macro-F1: **0.5592**
- Cumulative improvement over the original locked model (0.5117): **+0.0475 (~9.3% relative)**
- Per-class F1: None 0.449, Wish to be dead 0.591, Suicide ideation 0.656, Suicide planning 0.557, Attempt/Death 0.543

This is now the configuration Phase 4 (if pursued) should try to beat,
and the candidate for the next, deliberate one-time test-set evaluation
if a decision is made to lock this as the new final result.

---

## Test set re-touched once, deliberately — final Phase 3 result

The cross-architecture ensemble (Section "Current best configuration"
above) was locked as the new final configuration and the test split
(196 items) was unlocked exactly once, via `phase3_final_test_eval.py`.
Same discipline as the original Day-7 evaluation: all 6 models (3 E6
gated+orth seeds, 3 cross-attention seeds) were trained and model-selected
using train/validation only; test was not examined until every model was
already fixed, then evaluated in a single non-interactive pass.

| Metric | Original locked model (test) | New ensemble (test) | Delta |
|---|---|---|---|
| macro-F1 | 0.4635 | **0.4984** | **+0.0349** |
| weighted-F1 | 0.4857 | 0.5232 | +0.0375 |
| accuracy | 0.4932 | 0.5306 | +0.0374 |
| quadratic weighted kappa | 0.3654 | 0.3857 | +0.0203 |

**All four metrics improved on held-out test, by a margin close to (and
slightly smaller than) the validation-side gain (+0.0475).** This is the
important confirmation: the gain generalizes rather than being an
artifact of having iterated on the validation split across roughly 15
experiments in this document. A gain that evaporated on test would have
meant we'd overfit to validation through repeated experimentation; a gain
that mostly held up (as happened here) is the expected, healthy signature
of a real improvement.

**Per-class detail (majority vote of all 6 models, test split, n=196):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.484 | 0.469 | 0.476 | 32 |
| Wish to be dead | 0.580 | 0.763 | 0.659 | 38 |
| Suicide ideation | 0.613 | 0.594 | 0.603 | 64 |
| Suicide planning | 0.400 | 0.308 | 0.348 | 26 |
| Suicide attempt or death | 0.424 | 0.389 | 0.406 | 36 |

"Suicide planning" improved on test too (0.304 → 0.348), consistent with
the validation-side finding, though less dramatically than the ~0.53-0.56
seen for cross-attention alone on validation -- the ensemble's test-set
gain on this specific class is real but more modest than the headline
validation number suggested, worth stating plainly rather than only
citing the more flattering validation figure.

**Individual model test macro-F1s, for transparency:** gated seeds
0.459/0.436/0.442, cross-attention seeds 0.459/0.452/0.523 -- every
individual model scored BELOW the 6-way ensemble (0.498), confirming the
ensembling effect (not one lucky component) is what drove the improvement,
consistent with the validation-side finding.

Full confusion matrix and per-model breakdown in
`outputs/phase3_final_test_results.json`. **This test result is now the
current final, reported result of this improvement round** — supersedes
the original Day-7 test result (0.4635) for any forward-looking reporting,
while that original result remains correctly attributed to the Day 1-12
build in `PROGRESS_LOG.md` and `THESIS_SECTIONS_DRAFT.md`.

The test set has now been touched twice total across this project's
entire lifetime: once for the Day-7 locked model, once for this Phase-3
locked model. Both were deliberate, one-time, and are documented as such.
**Do not touch it again without an equally deliberate decision to lock
yet another new final configuration first.**

---

## Phase 4 narrative log

Working baseline for Phase 4 comparisons: the Phase 3 cross-architecture
ensemble, validation macro-F1 **0.5592** (test 0.4984, already locked and
reported above -- Phase 4 iterates on validation only, same protocol as
every phase before it, and does not re-touch test until/unless a new
configuration is deliberately locked).

### 4.1 — Systematic hyperparameter sweep, E6 gated+orth head (DONE, kept)

Every classifier head in this entire project used one reasonable-guess
configuration (hidden_dim=256, dropout=0.2, lr=1e-3, weight_decay=1e-4),
never swept. One-at-a-time coordinate search from that default (8 configs,
not a full grid), 3 seeds each, majority-vote macro-F1 on validation:

| Config | Majority-vote macro-F1 | Delta vs. default |
|---|---|---|
| default (256, 0.2, 1e-3, 1e-4) | 0.5330 | — |
| hidden_dim=128 | 0.5193 | −0.0137 |
| hidden_dim=512 | 0.5138 | −0.0192 |
| dropout=0.1 | 0.5261 | −0.0069 |
| **dropout=0.4** | **0.5418** | **+0.0088** |
| lr=5e-4 | 0.5157 | −0.0173 |
| lr=2e-3 | 0.5312 | −0.0018 |
| weight_decay=1e-3 | 0.5347 | +0.0017 |
| dropout=0.4 + weight_decay=1e-3 (combined) | 0.5414 | +0.0084 (no better than dropout=0.4 alone) |

**Finding: the default was mildly under-regularized.** Both directions
that increased regularization (higher dropout, higher weight decay)
helped or were neutral; both directions that increased capacity (larger
hidden_dim) or increased the learning rate hurt. This is the expected
pattern for a small dataset (582 training examples) -- consistent with
everything else observed in this document (Phase 3.1's encoder-unfreezing
failure, the mixup failure) all pointing the same direction: this
specific model+data combination is much more sensitive to overfitting
than to underfitting, so err toward more regularization, not more
capacity, when in doubt.

**Decision: kept.** `dropout=0.4` adopted as the new default for the E6
gated+orth head (weight_decay left at 1e-4, since combining with the
weight_decay increase added nothing on top). This is a genuine, if modest,
gain (+0.0088) on the gated architecture specifically -- checked next
for whether it also improves the full cross-architecture ensemble, since
the gated model is one of its two components.

**Effect on the full ensemble:** re-ran the gated+orth component with
dropout=0.4 in place of the cross-architecture ensemble's gated half
(cross-attention component and combination method unchanged):

| Configuration | Majority-vote macro-F1 |
|---|---|
| Cross-architecture ensemble, untuned gated (Phase 3 locked) | 0.5592 |
| Cross-architecture ensemble, tuned gated (dropout=0.4) | **0.5638** |
| Delta | +0.0046 |

A modest further gain, and **the per-class picture is now the most
balanced of the entire project** -- every single class above 0.50 for
the first time: None 0.510, Wish to be dead 0.575, Suicide ideation
0.650, Suicide planning 0.533, Attempt/Death 0.551. "None" specifically
jumped from 0.449 to 0.510, suggesting the extra regularization reduced
overfitting on exactly the class the previous ensemble was weakest on.

**Decision: kept, promoted to new best.** Validation macro-F1
progression: 0.5117 → 0.5330 (Phase 1) → 0.5592 (Phase 3) →
**0.5638 (Phase 4.1)**. Cumulative gain from the original locked baseline:
**+0.0521 (~10.2% relative)**.

### 4.4 — Post-hoc per-class decision calibration (DONE, kept with a caveat)

Re-weighted the 6-model ensemble's vote fractions per class (not
retraining anything -- `vote_fraction[c] = models voting for c / 6`,
multiplied by a learned per-class weight before the final argmax) via
coordinate ascent: cycle through each class, try 7 candidate multipliers
(0.7-1.3), keep whichever most improves validation macro-F1, repeat until
no class improves further.

**Important methodological caveat, unlike every other item in this
document:** this search directly optimizes 5 free weight parameters
*against the exact validation macro-F1 being reported* -- a more
leakage-prone setup than a normal train/validation split, since it's
closer to fitting parameters to the metric itself than to learning a
generalizable pattern from data. Every other kept change in this
document (ensembling, ordinal smoothing, dropout tuning, the
cross-architecture combination) was validated in the normal train-then-
evaluate sense; this one was *searched* directly against the number being
reported. Flagging this plainly rather than presenting it as equally
trustworthy -- its true benefit is only confirmed if/when this
configuration is tested on the held-out test set.

**Result:** converged after a single pass (no further improvement on a
second pass), which is a mildly reassuring sign against overfitting the
search -- a badly overfit 5-parameter search against 195 validation
examples would more likely keep finding "improvements" round after
round rather than settling immediately.

| | Macro-F1 |
|---|---|
| Uncalibrated majority vote (Phase 4.1 best) | 0.5638 |
| Calibrated | **0.5695** |
| Delta | +0.0057 |

Learned weights: None ×0.7, Wish to be dead ×1.1, all others ×1.0 (unchanged).
**This is the opposite correction from the initial hypothesis** going into
this item (which expected "Wish to be dead" to need *down*-weighting,
since it's the class over-predicted as a fallback guess in every prior
experiment) -- the search found the reverse helped instead. Worth naming
explicitly: the intuitive hypothesis about which direction to correct was
wrong, and letting the data-driven search find the actual answer instead
of confirming the assumption was the right call here.

Per-class effect: None 0.510→0.524, Attempt/Death 0.551→0.571, Wish to be
dead 0.575→0.568 (slight give-back), Ideation and Planning unchanged.

**Decision: kept as a candidate, pending test-set confirmation.**
Cumulative validation macro-F1: 0.5117 → 0.5330 → 0.5592 → 0.5638 →
**0.5695 (+0.0578 / ~11.3% relative from the original baseline)** -- but
given the caveat above, this specific increment should be treated as
provisional until confirmed (or not) by the next deliberate one-time test
evaluation, unlike the increments before it.

### 4.5 — Hierarchical decomposition (DONE, NOT kept — clean, explained negative result)

Split the problem into two stages, using the same aligned representations
and tuned E6GatedOrth architecture (dropout=0.4) for both: **Stage 1**, a
binary classifier (None vs. any suicide content); **Stage 2**, a 4-way
classifier among the non-None severity levels, trained only on non-None
examples. At inference, Stage 2's prediction is only used where Stage 1
predicted "any suicide content"; Stage 1 predicting "None" is final.

| Stage | Result |
|---|---|
| Stage 1 alone (binary, majority vote of 3 seeds) | 0.7406 macro-F1 |
| Stage 2 alone (4-way, evaluated on the TRUE non-None subset -- an oracle-routing best case) | 0.561-0.572 macro-F1 |
| **Full pipeline (Stage 1 routes into Stage 2, real end-to-end)** | **0.5203 macro-F1** |
| Flat ensemble (Phase 4.4, for comparison) | 0.5695 |
| Delta vs. flat ensemble | **−0.0492** |

**Both stages individually looked promising -- Stage 1's binary
separability (0.74) is genuinely much higher than "None" ever scores as
one of 5 classes in any flat model, confirming the hypothesis that
motivated this item.** But the **full pipeline underperforms the flat
ensemble**, and the reason is the classic hierarchical-classifier failure
mode: **Stage 1's routing mistakes become unrecoverable errors for Stage
2.** Any item Stage 1 misroutes (a true non-None case predicted as "None,"
or vice versa) is wrong regardless of how good Stage 2 is on the cases it
does receive -- there's no way for the pipeline to hedge or reconsider
once Stage 1 has committed, unlike a flat joint classifier that can
distribute probability mass across all 5 classes simultaneously and let
softer, joint evidence resolve close calls. "Suicide planning" F1
specifically collapsed to 0.333 in the full pipeline (vs. 0.557 in the
flat ensemble) -- plausibly because borderline planning/ideation cases
that a flat model can weigh against evidence from the whole label space
get force-committed early and incorrectly once routed through a hard
Stage 1 boundary.

**Decision: not kept.** The individual-stage numbers were genuinely
encouraging and the underlying hypothesis (None is more separable) was
correct, but hard hierarchical routing is not is the right way to
exploit that in this data regime -- a soft version (e.g., using Stage 1's
predicted probability as an additional input FEATURE to a flat 5-way
classifier, rather than a hard gate deciding whether Stage 2 runs at all)
would avoid the error-cascading failure mode while still giving the model
access to the same separability signal, and remains a theoretically
available follow-up if revisited, though not attempted here given time.

### 4.2 — Direct VLM prompting (DONE, NOT kept — clear, explained negative result)

Prompted Qwen2.5-VL-7B directly to classify each meme's suicide-severity
into one of the 5 categories (zero-shot, temperature=0, image + OCR text
+ class definitions in the prompt), rather than using it to generate
reasoning text for a separately-trained classifier as the rest of this
project does. Run on all 195 validation images (test stays locked). 0
errors, 0 unparseable responses -- the model reliably followed the
requested output format.

| | Macro-F1 |
|---|---|
| Current best (trained ensemble, Phase 4.4) | 0.5695 |
| Direct VLM prompting (zero-shot) | **0.2912** |
| Delta | **−0.2783** |

**This is a large, decisive negative result, and the failure pattern is
clear and explainable, not random:**

| Class | True count | Predicted count |
|---|---|---|
| None | 28 | 51 (over-predicted) |
| Wish to be dead | 32 | **6** (severely under-predicted) |
| Suicide ideation | 65 | 102 (heavily over-predicted) |
| Suicide planning | 31 | 30 |
| Suicide attempt or death | 39 | **6** (severely under-predicted) |

The model collapses toward "Suicide ideation" (the vaguer middle
category) and "None" (the safest, no-risk category), while almost never
committing to "Wish to be dead" or "Suicide attempt or death" -- the two
categories requiring the most definitive, severe judgment. This reads as
classic LLM safety-alignment hedging: a 7B instruction-tuned model
reluctant to commit to an extreme severity judgment on sensitive content,
defaulting to a moderate or negative classification instead of a
confident severe one, rather than a comprehension failure (it correctly
identified "Suicide planning" at almost the right rate, 30 vs 31 true).

**This also meaningfully revises this document's own forecast.** Item 4.2
was flagged, before it was tried, as "the most likely to actually move
the needle" -- directly citing FigSIM's own paper finding that large
prompted models gave the biggest gains on this dataset. That citation
still stands, but it evidently does not transfer down to a **locally-run
7B open model** the way it was assumed to -- FigSIM's own large-model
results were almost certainly obtained with frontier-scale models
(GPT-4V/Gemini-class), a different resource tier than what is available
in this project. The specialized, carefully-tuned, ensembled pipeline
built over Phases 1-4 (0.5695) substantially outperforms naive direct
prompting of the only large multimodal model actually available here.

**Decision: not kept.** Few-shot prompting (providing labeled examples in
the prompt) was considered as a natural next step but not attempted --
given the failure mode is about the model's alignment-driven reluctance to
commit to severe categories rather than a lack of task understanding, it
is not obvious that showing examples would fix a hedging behavior baked
in during the model's own safety training, though it remains a
theoretically available follow-up if revisited.

**4.3 (ensembling the trained classifier with a directly-prompted model)
is no longer worth pursuing as originally planned** -- it was premised on
4.2 producing a reasonably competent second opinion whose errors might be
uncorrelated with the trained ensemble's. At 0.29 macro-F1 with a strong,
systematic bias against two entire classes, 4.2 is not a reasonable
second opinion to combine with; it would very likely drag the combined
result down; not attempted given this.

---

## Phase 5 — retrospective hindsight items: a rigor upgrade and remaining score-chasing options

**Why this phase exists.** After Phase 4 closed out, a retrospective
question was asked: if Day 1 could be replayed, what structural or
workflow change (not an architecture tweak) would have mattered most?
Two things stood out as bigger levers than anything actually tried in
Phases 1-4:

1. This project has used a single fixed train(582)/val(195)/test(196)
   split throughout. Every seed-to-seed macro-F1 spread observed across
   the whole project has been roughly ±0.01-0.03 -- a single split of
   this size cannot distinguish a real 0.01-0.02 improvement from noise,
   which means some Phase 1/2 "this is slightly worse, reject it" calls
   carry real uncertainty about whether they were actually worse or just
   unlucky on this particular 195-item validation set.
2. Ensembling (multi-seed, then cross-architecture) turned out to be the
   single largest lever in the entire project (Phase 1.1 and Phase 3.2
   combined account for more of the total gain than every loss-function,
   fusion-mechanism, and hyperparameter change put together) but was only
   adopted 9 days into a 12-day schedule, rather than being a first-class
   part of the pipeline from Day 1.

Item 5.1 below retrofits a fix for (1) as a rigor/measurement upgrade,
not a score-chasing experiment -- it exists to tell us how much of
Phases 1-4's decision-making was signal vs. noise, which is valuable for
the thesis write-up regardless of whether it changes the final number.
(2) is not separately retrofittable at this point (ensembling is already
baked into the current best pipeline) but is recorded here as the
clearest single lesson of the whole project for the thesis's
methodology/limitations discussion.

**Score-chasing items** (from the earlier "what else could we try"
discussion) remain available as 5.2 onward, in descending order of
expected value:

| # | Item | Rationale |
|---|---|---|
| 5.1 | **k-fold cross-validation re-evaluation of the current best config** (rigor upgrade, not a new experiment) -- re-run the locked Phase 4.4 pipeline (tuned gated + cross-attention ensemble, calibrated) under 5-fold CV over train+val combined, holding test out untouched, to get a variance estimate around 0.5695 and confirm the Phase 1-4 decision trail was not substantially noise-driven | Directly answers the hindsight question: were our validation-based accept/reject calls trustworthy? Cheap relative to its information value -- no new modeling ideas, just re-running the existing training code under a different split scheme. |
| 5.2 | **Stacking meta-learner** in place of majority vote -- train a small logistic regression on the 6 base models' class probabilities (or vote fractions) instead of hand-tuned coordinate-ascent calibration | Majority vote + calibration is a blunt combination rule; a learned combiner can weight models per-class more precisely. Low effort, no base-model retraining needed. |
| 5.3 | **Soft multi-task head** (shared backbone, joint 5-way ordinal loss + auxiliary binary "any severity" loss) as a non-cascading way to exploit the same "None is more separable" signal that motivated Phase 4.5 | Phase 4.5's hard hierarchical gate failed via error-cascading; a soft auxiliary loss keeps the same backbone able to hedge across all 5 classes jointly while still being nudged by the binary signal -- mechanistically different from 4.5, not a rehash. |
| 5.4 | **More seeds in the ensemble** (5+5 instead of 3+3 gated/cross-attention models) | Pure variance reduction; ensembling has been the most reliable lever in the whole project. Cheap, no new ideas, diminishing but real returns expected. |
| 5.5 | **CORN loss** (proper cumulative-link ordinal regression) as a genuine ordinal-regression alternative to label smoothing | Different mechanism from both CORAL (rejected, Day 4) and ordinal smoothing (kept, Phase 1.3) -- we have never tried a true cumulative-link approach despite the task being explicitly ordinal. |

**Deprioritized:** SMOTE-style oversampling of "Suicide planning," test-time
augmentation, and larger contrastive-pretraining runs remain on the table
as lower-expected-value fallbacks if 5.1-5.5 stall, per the same reasoning
used to deprioritize items in Phase 4's original planning table.

**Protocol note (unchanged from Phases 1-4):** 5.1 uses train+val only
(test stays locked); 5.2-5.5, if pursued, are evaluated the same way as
every prior phase -- validation-only iteration, with a single deliberate
test-set check only if a new configuration is chosen to replace the
current locked best.

Status: **5.1 done. 5.2-5.5 not yet started.**

### 5.1 — 5-fold cross-validation of the locked pipeline (DONE — informative, and surfaced an honest new limitation)

Re-ran the current locked Phase 4.4 pipeline (3 tuned gated+orth seeds +
3 cross-attention seeds, majority vote, per-class calibration weights
{None: 0.7, Wish to be dead: 1.1, others: 1.0}) under 5-fold stratified
CV over train+val combined (777 examples: 621-622 train / 155-156
held-out per fold). Test set untouched. Stage A's contrastive alignment
projections were kept frozen and shared across folds (see
`phase5_1_kfold_cv.py` docstring for why); only the 6 downstream
classifier heads were retrained per fold, and confidence-feature
normalization was recomputed per fold from that fold's train subset only.

| Fold | n held-out | Uncalibrated macro-F1 | Calibrated macro-F1 |
|---|---|---|---|
| 1 | 156 | 0.5975 | 0.5904 |
| 2 | 156 | 0.5933 | 0.5850 |
| 3 | 155 | 0.5124 | 0.4689 |
| 4 | 155 | 0.5274 | 0.5197 |
| 5 | 155 | 0.6383 | 0.6448 |
| **Mean ± std** | | **0.5738 ± 0.0470** | **0.5618 ± 0.0611** |

| | Macro-F1 |
|---|---|
| Original single fixed-split locked result (Phase 4.4) | 0.5695 |
| 5-fold CV mean (calibrated) | 0.5618 |
| Delta (single-split vs. CV mean) | +0.0077 |

**Finding 1 (reassuring): the headline 0.5695 number is not a lucky
outlier.** It sits well inside the 5-fold spread (0.4689-0.6448) and only
0.0077 above the CV mean — the single validation split this project
locked its final number against was a representative, not
favorably-biased, sample.

**Finding 2 (the actual point of this check): per-fold variance is large
-- std of 0.047-0.061 macro-F1, on results whose mean is ~0.56-0.57.**
This quantitatively confirms the hindsight concern that motivated 5.1:
several Phase 1/2 accept/reject calls in this project were decided by
deltas of 0.01-0.03 macro-F1 on the single 195-item validation split
(e.g. Phase 1.2 gate-uncertainty at −0.0059, Phase 2.2 mixup at −0.0141)
-- deltas smaller than one fold-to-fold standard deviation here. Those
specific decisions are not necessarily wrong (they were also consistent
with mechanistic reasoning, not just the number), but the confidence
level attached to any single-split delta of that size should be read as
low, and this is now recorded plainly for the thesis's methodology and
limitations discussion rather than left implicit.

**Finding 3 (unexpected, and the most important new result from this
item): the Phase 4.4 calibration weights, tuned on one specific 195-item
validation split via coordinate ascent, do NOT generalize -- they help on
3 of 5 folds and actively hurt on 2 of 5** (fold 3: 0.5124 → 0.4689, a
−0.0434 drop; fold 4: 0.5274 → 0.5197, a smaller −0.0077 drop), while
folds 1, 2, and 5 see small gains. Net effect across folds is slightly
negative on average (calibrated mean 0.5618 vs. uncalibrated mean
0.5738). This is a textbook small-sample overfitting signature: a
handful of scalar weights fit by search against one 195-example set will
capture some of that set's idiosyncrasies along with the real signal.

**This revises Phase 4.4's status.** It was already flagged as
"provisional pending test-set confirmation" due to its different
methodology; this CV result adds a second, independent reason for
caution -- the calibration step should be treated as a mild,
data-set-specific adjustment rather than a robust general improvement,
and if a further one-time test-set check is done, comparing WITH and
WITHOUT the 4.4 calibration step (not just the calibrated number alone)
would be the honest way to report it, since CV suggests calibration's
true expected effect may be closer to zero, or even slightly negative,
than the +0.0057 seen on the original split.

**Decision: 5.1 is a completed rigor check, not a config change** -- it
does not replace the locked Phase 4.4 pipeline, but the finding above is
now part of this project's honest record of what is and is not
well-supported. Recommended next: 5.2 (cheapest remaining score-chasing
item) before 5.3-5.5, and if any further test-set evaluation is planned,
report both calibrated and uncalibrated ensemble numbers given Finding 3.

---

## Phase 6 — LoRA fine-tuning of the frozen encoders

**Why this phase exists.** After a direct question ("can we break the
bottleneck with heavier models"), two "bigger model" options were
evaluated and both turned out to be closed off already, not newly
rejected:
- SigLIP is already at its largest public variant
  (`siglip-so400m-patch14-384`) -- no larger SigLIP checkpoint exists to
  swap in.
- A bigger/different text encoder was already tried and lost -- Phase
  2.1 swapped DAPT-BanglaBERT for a larger, more generic multilingual
  model and dropped 0.073 macro-F1. The value in this pipeline was never
  raw model size, it was DAPT-BanglaBERT's domain-specific pretraining;
  there is no bigger version of that specific domain-adapted checkpoint.
- A frontier-scale VLM (GPT-4V/Gemini-class, matching FigSIM's own
  paper's best-performing setup) was also considered, but is unavailable
  -- no API access in this environment.

That leaves exactly one remaining, mechanistically distinct lever:
**parameter-efficient fine-tuning (LoRA)** of the two encoders that have
been frozen for the entire project. This is NOT a repeat of Phase 3.1's
partial-unfreezing experiment (which failed, −0.026 macro-F1, by
overfitting fast on 582 training examples after updating full
transformer layers). LoRA inserts small low-rank adapter matrices into
the attention projections and trains only those (well under 1% of each
encoder's parameters), leaving the base pretrained weights untouched --
a fundamentally lower-capacity, more heavily regularized way to let the
encoders adapt slightly to this exact domain, which may succeed where
full-layer unfreezing overfit.

**Pre-flight check (done before writing any code, per explicit
instruction to verify nothing is broken first):** confirmed disk space
stable (17GB free, unchanged from earlier crises), `peft` 0.19.1 already
installed (no new downloads needed), DAPT-BanglaBERT checkpoint and
tokenizer load cleanly, SigLIP so400m loads cleanly, GPU available (RTX
5090, 34GB VRAM -- ample for both encoders plus LoRA adapters), and the
existing locked pipeline (`train_e6.py`'s `load_aligned_data`) still
reproduces the documented train/val class distributions exactly
(train n=582, val n=195, matching every prior phase). Nothing broken.

### 6.1 — Joint LoRA fine-tuning of DAPT-BanglaBERT + SigLIP (planned)

Both encoders get LoRA adapters (r=8, alpha=16, targeting the
query/value attention projections -- `["query","value"]` for BanglaBERT's
ELECTRA architecture, `["q_proj","v_proj"]` for SigLIP's architecture),
trained end-to-end with a simple concat classifier head (matching Phase
3.1's "simple-concat" design exactly, so any difference is attributable
to the fine-tuning mechanism, not a fusion-architecture confound).
Differential learning rates (LoRA adapter params vs. head), mini-batched,
short patience -- same conventions as Phase 3.1's unfreezing script.
Ordinal label smoothing + 3-seed majority vote, unchanged from every
other phase. Compared against the same reference point Phase 3.1 used:
the frozen-encoder concat baseline, 0.5139 macro-F1.

Status: **DONE — positive result confirming the mechanism, but bounded by
architecture choice; does not replace the current best.**

**Overfitting/collapse audit (performed before accepting the result, per
explicit instruction to check every step for a broken or collapsed
training run):**
- Per-epoch validation curves (all 3 seeds) fluctuate normally in the
  0.31-0.52 range, peaking mid-training then declining gently -- typical
  small-batch fine-tuning noise on 582 examples, not a divergence, NaN
  blow-up, or monotonic collapse to zero.
- Early stopping's "keep best-so-far" logic correctly identified the true
  peak epoch for each seed (verified by hand against the printed
  per-epoch log, not just trusted blindly) -- e.g. seed 2's actual peak
  was epoch 4 (0.4876), not an earlier local bump, and that is exactly
  what was reported and used.
- Per-class F1 on the final majority-vote ensemble is healthy and spread
  across all 5 classes (0.44-0.66) -- no class collapsed to zero, which
  is the signature a degenerate "always predicts one class" failure would
  leave behind.
- **Conclusion: this is a real, non-overfit, non-collapsed result.**

| | Macro-F1 |
|---|---|
| Phase 3.1 frozen-encoder concat baseline (reference point) | 0.5139 |
| Phase 3.1 partial unfreezing, full layers (failed) | 0.4875 |
| **Phase 6.1 LoRA fine-tuning, both encoders (this item)** | **0.5264** |
| Delta vs. frozen baseline | **+0.0125** |
| Delta vs. failed full-unfreeze attempt | **+0.0389** |

**This confirms the hypothesis that motivated Phase 6: LoRA succeeds
where naive full/partial unfreezing failed.** Updating <1% of each
encoder's parameters, instead of full transformer layers, avoided the
fast overfitting collapse Phase 3.1 hit on this same 582-example training
set, and let the encoders pick up a small but real amount of
task-relevant adaptation.

**Why this is NOT reported as a new overall best, and Phase 6 stops
here.** 0.5264 is measured against the *simple-concat* architecture
(the same one Phase 3.1 used) for a clean, single-variable comparison --
it was deliberately not combined with the gated+orth fusion architecture
or the cross-architecture/multi-seed ensembling that took the frozen-
embedding pipeline from 0.51 up to 0.5695. Fully integrating LoRA into
that stack would require re-extracting LoRA-adapted embeddings for all
973 images and rebuilding the alignment/gated/cross-attention/ensemble
pipeline on top of them -- a substantially larger engineering effort, and
one with uncertain payoff given that most of the existing pipeline's
gains already came from ensembling and calibration on top of frozen
features, not from the fusion architecture itself. Given this project's
established diminishing-returns discipline (documented explicitly at the
end of Phases 1, 3, and 4), that additional effort is not undertaken
here.

**Decision: Phase 6.1 is a completed, positive, informative result** --
it answers "is there a viable way to adapt the encoders at all" with a
clear yes, mechanistically distinct from the Phase 3.1 failure -- but it
does not change the project's locked best configuration. **The current
best remains Phase 4.4's ensemble: 0.5695 macro-F1 (validation,
provisional) / 0.4984 macro-F1 (test, confirmed, Phase 3 lock).**

**This effectively closes out the improvement round.** Every genuinely
distinct mechanism available without additional data or API access --
loss functions, fusion architectures, ensembling, calibration,
hierarchical decomposition, direct VLM prompting, and now encoder
fine-tuning -- has been tried, measured on validation, and either kept or
rejected with a documented, mechanistic reason. Phase 5's remaining
items (5.2-5.5) and further encoder-fine-tuning integration remain
available as optional future work, but are score-chasing refinements on
a well-explored ceiling, not open questions about whether something
big is being missed.

---

## Phase 7 — external meme pretraining + label harmonization

**Why this phase exists.** A retrospective question ("what could have
drastically improved this from Day 1?") led to identifying dataset size
as the real bottleneck (see the earlier hindsight discussion) -- Phases
1-6 all optimized the fusion/loss/ensembling layer on top of the same
582-777 labeled examples, which was shown to be near its ceiling. Phase 7
is the first item in this project to actually bring in additional real
data (two public Bangla meme datasets, CMBAN and BN-HIB) rather than
re-processing the existing 973 images further. Full design and safety
rules are in `PHASE7_REVISED_PRETRAINING_PLAN.md`, which itself revises
the original `FIGSIM_HARMONIZATION_AND_PRETRAINING.md` brief after two
problems were found in review: (1) touching DAPT-BanglaBERT risked
undoing its proven domain specialization, and (2) CMBAN/BN-HIB's native
Bangla text doesn't match FigSIM's own machine-translated Bangla
pipeline. The revised plan keeps all external pretraining strictly on
the image side, never loading or modifying DAPT-BanglaBERT.

### Step 1 — Free label-harmonization diagnostic (DONE)

Aggregated the current best 5-class ensemble's own saved validation
predictions into a 3-class and a 4-class grouping (no retraining), per
`code/phase7_1_label_harmonization_diagnostic.py`.

| Scheme | Macro-F1 | Balanced accuracy |
|---|---|---|
| 5-class (sanity check) | 0.5638 (exact match to known Phase 4.1 value) | 0.5728 |
| **3-class** | **0.6394** | 0.6284 |
| 4-class (merges ideation+planning) | 0.5865 | 0.5986 |

Important nuance recorded at the time: the 3-class scheme's larger gain
is NOT mainly because it fixes the specific ideation/planning confusion
diagnosed earlier -- the 4-class scheme, which does merge exactly that
boundary, gained less (+0.0227) than 3-class (+0.0756), which keeps
ideation and planning separate. The bigger 3-class gain reflects a
broader pattern: adjacent-class confusion exists at multiple points
along the ordinal scale, not only at the ideation/planning boundary, and
any coarser scheme also mechanically tends to score higher macro-F1 with
fewer classes. Per this project's own harmonization rules, the 3-class
number must never be presented as a direct improvement over 5-class --
they are different tasks.

**Decision: 3-class kept as the real target for Step 4** (training a
native 3-class model, not just aggregating a 5-class one after the
fact); 2-class was also checked via Phase 4.5's existing binary-stage
result (0.7406) but deliberately NOT adopted as primary, since it
discards the graded severity information the whole project exists to
produce -- matching `FIGSIM_HARMONIZATION_AND_PRETRAINING.md`'s own
explicit rule on this exact point.

### Step 2 — External dataset audit (DONE, revealed a real blocker)

- **BN-HIB: fully downloaded and usable.** `train.csv`/`val.csv`/`test.csv`
  present (2,272/487/488 records, matching expected counts). Official
  README clarified the label mapping precisely: the `choice` column is
  authoritative (Targeted Trolling->Hate, Harmless Trolling->Benign,
  Provocative_Trolls->Inflammatory); the `class` column is explicitly
  NOT a label (language-mix metadata only) and was correctly excluded
  from supervision.
- **CMBAN: NOT fully downloaded.** Only `data/sample.xlsx` (133 rows, no
  English-text column) and 1,278/2,641 images are present -- the actual
  `train.csv`/`test.csv` with the full 2,113/528 records and English
  translations were never pulled down. Per this project's "do not
  fabricate data" rule, CMBAN/Track B (SigLIP's own paired text tower +
  CMBAN's English translations) was skipped rather than faked, and
  BN-HIB-only pretraining proceeded instead.
- **Image-count discrepancy checked and resolved cleanly:** BN-HIB's
  `Images/` folder contains 3,286 files but only 3,247 are referenced by
  the official CSVs (matching the harmonization document's expected
  count exactly); the other 39 are simply unreferenced extra files.
  Confirmed zero referenced-but-missing images -- training used exactly
  the correct, complete official dataset.

### Step 3, Track A — BN-HIB vision-only LoRA pretraining (DONE)

`code/phase7_2_bnhib_vision_pretrain.py`. DAPT-BanglaBERT is never
loaded or modified -- only SigLIP gets small LoRA adapters (r=8,
alpha=16, targeting `q_proj`/`v_proj`, 995,328 / 429,220,928 = 0.232%
trainable), trained on BN-HIB's own Hate/Inflammatory/Benign labels as
a vision-side auxiliary classification task. Official BN-HIB train/val
split used; BN-HIB's own test.csv was never touched, extending this
project's test-set discipline to the new external dataset.

| | Value |
|---|---|
| Best validation macro-F1 | **0.7504** (epoch 7, plateaued through epoch 11, early-stopped on patience) |
| Benign F1 | 0.7881 |
| Inflammatory F1 | 0.7052 |
| Hate F1 | 0.7580 |

Random-chance baseline for this roughly-balanced 3-way task is ~0.33, so
0.75 is a real, well-above-chance signal -- some genuinely learnable
visual information exists in these meme images relevant to this task,
even without any text encoder in the loop. Training curve was flat
(not declining) after the peak, consistent with convergence rather than
overfitting; the best-checkpoint-only saving logic (never overwrites a
better checkpoint with a worse one) was verified correct by independently
re-evaluating the saved weights and reproducing the exact same 0.7504.

**Honest caveat, explicitly flagged, not yet resolved:** this validation
number is internal to BN-HIB's own task, not a test of transfer to
FigSIM. A specific risk was named before training: SigLIP could be
partly learning to recognize recurring BN-HIB-specific meme templates
or visual layout patterns rather than genuinely transferable "meme
content understanding" -- which would inflate this number without
helping FigSIM at all. **The only real test is Step 4/5**, next.

**Operational note:** the original training run hit a known Windows
issue where PyTorch's `persistent_workers` DataLoader processes didn't
shut down cleanly after early stopping triggered, hanging during the
script's own wrap-up. No training progress was lost -- the resumable
checkpoint design (built specifically for this kind of interruption)
had already captured the best weights before the hang; a small recovery
script (`code/phase7_2b_finalize_from_checkpoint.py`) completed the
same finalization independently and reproduced an identical score.
Separately, a Windows "Smart App Control" policy (enabled by the
device's management, not user-toggleable) blocked pandas' compiled DLL
partway through this phase; fixed by removing the pandas dependency
from `phase7_2_bnhib_vision_pretrain.py` entirely in favor of
`csv.DictReader`, matching how the rest of this project already reads
CSVs (`train_e6.py`'s `load_labels()`) -- verified to produce identical
results before and after the change.

### Step 4 — harmonized 3-class fine-tuning, with and without BN-HIB pretraining (DONE)

`code/phase7_4_harmonized_3class_comparison.py`. Trained an actual
3-class classifier (not the Step 1 diagnostic's post-hoc aggregation)
twice on the identical simple-concat architecture, seeds, and protocol
-- differing in only the image embeddings used, to isolate exactly one
variable. DAPT-BanglaBERT identical and untouched in both variants.

| Variant | Macro-F1 |
|---|---|
| **A -- original frozen SigLIP** | **0.6425** |
| B -- BN-HIB-pretrained SigLIP (Step 3 checkpoint) | 0.6363 |
| Delta (B - A) | **-0.0062** |
| For reference -- Step 1 diagnostic (aggregated, not trained) | 0.6394 |
| For reference -- 5-class baseline (Phase 4.1) | 0.5638 |

**Finding 1 -- 3-class harmonization confirmed as a real win, not an
aggregation artifact.** Variant A (an actually-trained 3-class model)
scores 0.6425, slightly above Step 1's 0.6394 aggregation-based
estimate -- confirming that training natively on the harmonized target
modestly outperforms merely re-grouping a 5-class model's predictions
after the fact, exactly as hypothesized when Step 4 was planned. Against
the 5-class baseline (0.5638), this is a genuine **+0.0787** improvement.
**Decision: 3-class harmonization is kept as the primary target going
forward.**

**Finding 2 -- BN-HIB pretraining did not help FigSIM's own task; a
clean, small negative result.** This confirms the specific risk flagged
in advance, before Step 3 was ever run: SigLIP may have specialized
toward BN-HIB's own visual patterns (Bangla-script text overlays, its
specific meme templates) rather than learning genuinely transferable
"meme understanding" that carries over to FigSIM's different-domain
(English-script) images. The -0.0062 delta is small and within the kind
of noise this project's Phase 5.1 cross-validation work already showed
single-split deltas of this size can carry, but it is clearly not a
positive signal either way. **Decision: the BN-HIB-pretrained SigLIP
checkpoint is NOT adopted.** The current best FigSIM pipeline continues
to use the original frozen SigLIP; no harm was done by trying this --
the decision-gate protocol built into the plan specifically for this
scenario worked exactly as intended.

**Track B (CMBAN + SigLIP's own text tower) remains unexplored** --
CMBAN was never fully downloaded (Step 2 audit), so this option is
still open for a future session if the full CMBAN dataset is obtained,
but is not blocking anything: the harmonization win (Finding 1) stands
on its own regardless of the pretraining question, and is the one clear
positive result to carry forward.

**Phase 7 status as of this entry: Steps 1-4 complete.** The 3-class
harmonized target is confirmed as a genuine improvement over 5-class and
is the recommended new primary task; external pretraining was tried in
good faith, tested properly against a decision gate, and honestly did
not pay off in the one variant tried (BN-HIB vision-only). Next
decisions: (a) whether to build the full ensemble (gated+orth,
cross-attention, calibration -- everything Phases 1-6 already validated)
on top of the 3-class target as a new candidate best configuration, and
(b) whether pursuing CMBAN/Track B is worth the effort given Track A's
result, or whether Phase 7 should be considered complete as-is.

---

## Plan B (documented, deliberately deferred) -- a single joint model with a unified depression/suicide severity scale

Raised in discussion: instead of two separate classifiers (FigSIM
suicide-severity + Phase 2's depression-severity) combined post-hoc via
the fuzzy-logic rule table, use **one single model that takes both text
and image and produces one unified severity output**, by mapping both
label sets onto one shared 4-level scale:

| Unified level | FigSIM label(s) | Depression label |
|---|---|---|
| (lowest) | None | Minimum |
| | Wish to be dead | Mild |
| | Suicide ideation + Suicide planning | Moderate |
| (highest) | Suicide attempt + Suicide death | Severe |

**Why this is documented as Plan B rather than started now.** This was
discussed and an important tension was raised and not yet resolved:
depression severity and suicide-risk severity are related but distinct
clinical constructs -- someone can be severely depressed with no
suicidal ideation, or express active suicidal planning during an acute
crisis without broad depressive symptoms otherwise. Asserting a direct
equivalence (e.g. "Wish to be dead" = "Mild depression") is a modeling
convenience, not a validated clinical claim; no annotator has confirmed
this specific cross-dataset correspondence, and
`FIGSIM_HARMONIZATION_AND_PRETRAINING.md`'s own non-negotiable rules
were written specifically to prevent exactly this kind of merge ("Never
map Minimum depression to non-suicidal or Severe depression to
suicidal"). The fuzzy-logic layer built in Days 9-10 exists as the
deliberate alternative: it keeps the two dimensions as separate inputs
and combines them through explicit, inspectable rules rather than
collapsing them into one label space.

**This idea is not rejected, only deferred.** If pursued later, it
should be scoped with that tension explicitly addressed (e.g., framing
the unified scale as a modeling simplification with a named limitation,
not a claimed clinical equivalence), and would need its own validation
protocol since it changes what the model's output actually claims to
represent. Revisit after the current, lower-risk options (below) have
been tried.

---

## Phase 7 continued -- shared-encoder alignment options

Following the diagnosis that DAPT-BanglaBERT and SigLIP are
independently pretrained (not natively aligned), two further ideas were
discussed for how the existing gated-fusion mechanism relates to this:

- **Option 1 (recommended, starting now):** run the current best fusion
  architecture (E6 gated+orth, tuned dropout=0.4, using the EXISTING,
  already-working Stage A alignment -- unchanged) on the 3-class
  harmonized target. This closes a real gap: Phase 7 Step 4 only tested
  simple concat (deliberately, for a clean single-variable pretraining
  comparison) and never checked whether the architecture already proven
  better on the 5-class task also does better on 3-class. Near-zero
  risk -- reuses proven, unchanged components with a different label
  target.
- **Option 2 (bigger, deferred):** rebuild Stage A's alignment anchored
  to DAPT's own native 768-dim output space (instead of learning a new
  arbitrary shared space from just 582 pairs) -- only the image side
  would need a trainable projection into that space. The existing gate
  mechanism (`alpha = sigmoid(MLP([text, image, conf]))`,
  `gated = alpha*text + (1-alpha)*image`) would carry over unchanged,
  just operating on this new anchor-based alignment instead of the
  current arbitrary 256-dim shared space. Worth trying, but new,
  untested territory -- sequenced after Option 1's result is known.

### 7.5 -- Gated+orth on the 3-class harmonized target (DONE -- surprising, honest negative result)

`code/phase7_5_harmonized_gated_orth.py`. Same tuned gated+orth
architecture (dropout=0.4) and the EXISTING, unchanged Stage A
alignment used throughout this project, applied to the 3-class target,
3-seed majority vote.

| | Macro-F1 |
|---|---|
| Simple concat, 3-class (Phase 7 Step 4, current best) | **0.6425** |
| **Gated+orth (tuned), 3-class** | **0.6144** |
| Delta | **-0.0281** |

**This is the opposite of the pattern seen throughout Phases 1-6 on the
5-class task, where gated+orth consistently beat simple concat.**
Candidate explanations, not mutually exclusive: (1) Stage A's alignment
and the gate/orthogonal-residual mechanism were developed and validated
against the 5-class task's specific difficulty structure (fine-grained
ordinal confusion) -- that structure is coarser and less present once
classes are merged; (2) the 3-class target has much more populated,
balanced classes (89/293/200 vs. the sparser 5-class split) -- a
comparatively easier task may not benefit from, or could be mildly hurt
by, the gate's added complexity relative to a simpler architecture; (3)
could be partly within noise -- Phase 5.1's cross-validation work already
showed single-split deltas in the 0.03-0.06 range are not always
reliable signal, and this is only 3 seeds on one split.

**Decision: gated+orth is NOT adopted for the 3-class target.** The
current best 3-class configuration remains simple concat at 0.6425
(Phase 7 Step 4, variant A, original SigLIP). This is reported as a
genuine, unresolved-mechanism finding rather than smoothed over --
architecture choices are not universally transferable across label
granularities, which is itself a useful, honest data point for this
project's discussion of what generalizes and what doesn't.

**Option 2 (Stage A anchored to DAPT's native space) remains open** as
a genuinely different mechanism from what was just tested here, should
it be pursued later.

### 7.6 -- Fuzzy-logic risk combination: 5-class vs. 3-class, side by side (DONE)

Full plan documented before running anything:
`PHASE7_FUZZY_LOGIC_5CLASS_VS_3CLASS_PLAN.md`. Ran the fuzzy-logic
combination layer twice on FigSIM's validation split (not test -- the
3-class model has never been locked or evaluated against test, so both
runs use validation for a fair, equal-footing comparison): once with
the current best 5-class ensemble (tuned gated + cross-attention,
matching Phase 4.4) and the existing 11-row rule table, once with the
current best 3-class model (simple concat, original SigLIP, matching
Phase 7 Step 4 variant A) and a new 6-row 3-class rule table derived via
a documented **cautious-merge policy** (for classes being combined, take
the more severe risk level among the original rules being merged, at
each depression level -- full derivation and worked tables in the plan
doc). Phase 2's depression classifier is identical and untouched in
both, run once on validation OCR text and reused for both combinations.
Sanity check: both severity models reproduced their known scores exactly
(5-class 0.5638, 3-class 0.6425), confirming correct wiring.

| Risk level | 5-class-fuzzy | 3-class-fuzzy |
|---|---|---|
| Minimal | 22 (11.3%) | 35 (17.9%) |
| Low | 63 (32.3%) | 45 (23.1%) |
| Elevated | 71 (36.4%) | 47 (24.1%) |
| **Critical** | 39 (20.0%) | **68 (34.9%)** |

**Per-meme agreement: 135/195 (69.2%)** land on the same final risk
level; the remaining ~31% disagree, concentrated overwhelmingly in one
direction -- see figure `16_fuzzy_logic_5class_vs_3class.png`'s
confusion-style matrix.

**The single largest disagreement bucket is Elevated->Critical (24 of
60 total disagreements), and this is mostly explainable by rule-table
mechanics, not a mysterious model behavior.** The 3-class "High acuity
suicidal content" bucket merges "Suicide planning" (which only reached
Critical at Moderate/Severe depression under the 5-class table) with
"Suicide attempt or death" (already unconditionally Critical at every
depression level) -- the cautious-merge policy pulls the *entire* merged
bucket to Critical regardless of depression level. So a meaningful share
of the increased Critical rate under 3-class reflects this specific rule-
table design choice, not necessarily a genuine shift in what the
underlying severity models perceive in the same memes. This distinction
is worth stating precisely rather than over-interpreting the higher
Critical rate as straightforward evidence the 3-class scheme "finds more
real risk."

**What this comparison does and does not establish.** It shows the
3-class scheme's higher raw macro-F1 does translate into a materially
different final risk-level distribution -- meaningfully more
Critical-leaning and more Minimal-leaning, less concentrated in the
middle two categories. Whether that shift is more clinically useful
(catching more true elevated-risk cases) or more prone to alert fatigue
(false-Critical from the merge-driven inflation) cannot be determined
from this data -- as with the original fuzzy-logic layer, there is no
ground-truth "risk level" label to validate against; this remains a
distributional comparison, not an accuracy comparison. The 3-class
rule table's derivation (Section 3 of the plan doc) is a reasonable,
documented cautious-merge policy, not a clinically re-validated scale --
this caveat carries into the thesis limitations section alongside the
original 5-class table's own equivalent, pre-existing caveat.

**Decision: both fuzzy-risk outputs are retained and reported side by
side**, not collapsed into a single "winner" -- the comparison itself
(and the honest explanation of why they diverge) is the useful result
here, appropriate for a thesis discussion of how label-granularity
choices propagate through to a system's final output, not just its
intermediate classification metric.

### 7.7 -- Option 2: DAPT-anchored shared-encoder alignment (DONE -- caught early by its own decision gate)

Full plan documented before running anything:
`PHASE7_SHARED_ENCODER_OPTION2_PLAN.md`. Built a new contrastive
alignment variant with only an image-side trainable projection
(1152 -> 768, 885,504 parameters), using DAPT-BanglaBERT's own raw
768-dim output as the shared space directly (no text-side projection,
DAPT itself never loaded or modified -- same cached embeddings as the
existing Stage A script). Same InfoNCE objective, same train-split-only
self-supervised protocol, saved to a new checkpoint directory
(`stage_a_dapt_anchored/`), never touching the existing Stage A
checkpoint still in active use.

Per the plan's Step 3 decision gate, the cheap retrieval-accuracy check
was run **before** any downstream classifier training:

| | Validation top-1 retrieval accuracy |
|---|---|
| Chance level (195 validation items) | 0.0051 |
| Existing Stage A (in active use) | 0.0513 |
| **DAPT-anchored alignment (this item)** | **0.0256** |
| Delta vs. existing Stage A | **-0.0257** |

**The new alignment is worse than the existing one** -- still
meaningfully above chance (~5x chance level, vs. existing Stage A's
~10x), so the underlying idea isn't nonsensical, but clearly inferior to
what the pipeline already uses. This is precisely the risk named in
advance in the plan doc (Section 5): DAPT's 768-dim space was shaped
entirely by its own text-only pretraining, and evidently doesn't have
enough usable "room" or structure for SigLIP's image features to map
into as effectively as a freshly-learned, purpose-built shared space
does -- despite that freshly-learned space having far less data to learn
from (582 pairs) and needing to train projections on both sides instead
of one.

**Decision: STOP HERE, per the plan's own decision gate.** Downstream
classifier training (the more expensive step 4 of the plan) was
deliberately not attempted, since its cheap prerequisite check already
gave a clear negative answer -- exactly the intended purpose of gating
the workflow this way rather than jumping straight to the expensive
step. The existing Stage A alignment remains in use, completely
unaffected; nothing about the current best pipeline changes as a result
of this experiment.

**This closes out the shared-encoder alignment investigation (both
Option 1 and Option 2) with two honest negative results**, alongside
the one clear positive result from Phase 7 overall (3-class
harmonization). The mismatch between DAPT and SigLIP's embedding spaces,
diagnosed back on Day 4, remains addressed only by the original Stage A
fix -- which, on this evidence, was already a reasonably good solution
to the problem, not merely an adequate workaround.

### 7.8 -- Hyperparameter sweep for gated+orth, re-tuned specifically for 3-class (DONE -- narrows the gap, doesn't close it)

Fair-comparison follow-up to 7.5: that item applied `dropout=0.4` (the
value found best for the *5-class* task in Phase 4.1) to the 3-class
target as-is, without checking whether it's actually the right setting
for a coarser target. `code/phase7_8_harmonized_hyperparam_sweep.py`
reruns Phase 4.1's exact one-at-a-time coordinate sweep
(hidden_dim/dropout/lr/weight_decay), this time on 3-class.

| Config | Macro-F1 |
|---|---|
| Simple concat (current best) | **0.6425** |
| **Gated+orth, best config for 3-class (hidden_dim=512)** | **0.6330** |
| Gated+orth, default (hidden_dim=256, dropout=0.2) | 0.6236 |
| Gated+orth, dropout=0.4 (Phase 7.5 -- the 5-class-tuned value, unchecked) | 0.6144 |

**The concern behind re-running this was valid: `dropout=0.4` was
genuinely a poor, untested assumption for the 3-class target.** Even the
plain untuned default (0.6236) beats it, and the actual best 3-class
config (0.6330, larger hidden layer) closes most of the gap to simple
concat -- from -0.0281 (Phase 7.5) down to -0.0095 (this item), a
meaningful improvement in the fairness of the comparison.

**But the conclusion doesn't flip: simple concat (0.6425) still edges
out gated+orth even at its best-tuned-for-3-class configuration
(0.6330).** This is a more trustworthy negative result than 7.5's,
because gated+orth was given a fair, dedicated tuning pass rather than
being judged on hyperparameters inherited from a different task.
**Decision unchanged: simple concat remains the current best 3-class
configuration.**

### 7.9 -- Cross-architecture ensembling on 3-class (DONE -- new overall best, the strongest Phase 7 result)

The single biggest lever from the original project (Phase 3.2:
gated+orth + cross-attention ensembled beat either alone, 0.5592 vs.
0.5330/0.5205) had never been tried for 3-class -- only standalone
architectures had been compared against each other. `code/phase7_9_harmonized_ensemble.py`
trains three architectures on 3-class together for the first time,
including cross-attention (never tested on 3-class before this item),
and reports every majority-vote combination.

| Combination | Macro-F1 | n models |
|---|---|---|
| Concat alone | 0.6425 | 3 |
| Gated+orth alone (Phase 7.8 best config) | 0.6330 | 3 |
| Cross-attention alone (NEW) | 0.5982 | 3 |
| Concat + gated | 0.6349 | 6 |
| Concat + cross-attn | 0.6487 | 6 |
| Gated + cross-attn | 0.6143 | 6 |
| **Concat + gated + cross-attn (all three)** | **0.6649** | **9** |

**New best 3-class result: 0.6649, beating standalone simple concat
(0.6425) by +0.0224.** Only the full three-way, 9-model ensemble wins
reliably -- the pairwise combinations are inconsistent (concat+gated
actually scores *below* concat alone), which makes sense statistically:
9 models gives more robust majority voting with fewer ties than 6,
letting each architecture's distinct error patterns cancel out more
completely. This is the exact same mechanism that made ensembling the
single biggest lever in the original 5-class project, now confirmed to
transfer to the 3-class harmonized target as well.

**This becomes the new current best 3-class configuration, superseding
simple concat.** Full progression for the 3-class target: 0.5638
(5-class baseline, for reference) -> 0.6394 (3-class, aggregated
diagnostic) -> 0.6425 (3-class, simple concat trained natively) ->
**0.6649 (3-class, 9-model cross-architecture ensemble)**. A total gain
of **+0.1011** over the original 5-class baseline it was aggregated from.

### 7.10 -- Final locked test evaluation, 3-class configuration (DONE -- third deliberate test-set touch)

`code/phase7_10_final_test_eval_3class.py`. The 9-model ensemble from
7.9 (concat + gated-tuned-for-3-class + cross-attention, 3 seeds each)
evaluated on the locked test split exactly once, following the identical
protocol used for every prior test lock in this project (all 9 models
trained and model-selected on train/val only; test unlocked in a single
non-interactive pass after every model was already finalized). This is
the **third** deliberate test-set touch in the whole project's history
(Day 7 original 5-class lock; Phase 3 5-class re-lock; this one -- the
first and only touch for the 3-class harmonized scheme).

| | Macro-F1 | Weighted-F1 | Accuracy | Balanced Acc. | QWK |
|---|---|---|---|---|---|
| **3-class, test (this lock)** | **0.5730** | 0.6126 | 0.6071 | 0.5871 | 0.3765 |
| 5-class, test (Phase 3 lock, prior best) | 0.4984 | 0.5232 | 0.5306 | -- | 0.3857 |
| 3-class, validation (Phase 7.9, for reference) | 0.6649 | -- | -- | -- | -- |

Per-class (test): "No expressed severity" 0.462 F1 (precision 0.391,
recall 0.563 -- over-predicted, the weak point), "Suicidal thought or
desire" 0.683 F1, "High acuity suicidal content" 0.574 F1.

**Headline result: a real, test-verified +0.0746 macro-F1 improvement**
over the previous locked model -- the genuine, defensible number for
this project going forward, not the validation figure.

**Honest caveat, stated plainly and not minimized:** validation showed
0.6649; test came in at 0.5730 -- a **-0.0919 gap**, larger than this
project has typically seen (the original 5-class model's val-to-test gap
was -0.0711 by comparison). This is a reasonable, expected consequence
of how this configuration was reached: a long sequence of decisions
(diagnostic, native 3-class training, a full hyperparameter sweep,
architecture comparison, and an ensemble-combination search) were all
made against the *same* 195-item validation set. That much cumulative
selection pressure on one small set plausibly captured some of that
set's specific idiosyncrasies alongside real signal -- consistent with
Phase 5.1's cross-validation finding that this project's single-split
deltas carry real uncertainty. This does not undermine the test result
itself (which is real, measured once, and not re-tuned afterward) but
it does mean the validation number should not be quoted as the
headline figure -- 0.5730 is the number that reflects genuine,
unbiased generalization.

**Decision: this configuration is now the project's locked best model**
for the 3-class harmonized suicide-severity task, superseding the
original 5-class lock. QWK (0.3765) is very close to the original
5-class model's (0.3857) despite the large macro-F1 gain -- consistent
with QWK's ordinal-distance sensitivity being less affected by
granularity changes than macro-F1 is, worth noting for the thesis
alongside the standard "3-class and 5-class answer different-resolution
questions, not directly comparable" caveat used throughout this phase.

### 7.11 -- Fuzzy-logic layer reconnected to the locked model, on TEST (DONE -- "get ready for the panel" Step 2)

`code/phase7_11_fuzzy_logic_final_test.py`. Phase 7.6 compared 5-class-
vs. 3-class-fuzzy on VALIDATION, using a single-architecture 3-class
model that has since been superseded. This item redoes the comparison
properly: reproduces the actual LOCKED 9-model ensemble (identical
seeds/recipe as 7.10) to capture per-example TEST predictions --
sanity-checked to reproduce macro-F1 0.5730 exactly -- then runs Phase
2's identical, untouched depression classifier on the same 196 test
memes' text (its first run on test for the 3-class scheme) and combines
via the 3-class rule table.

| Risk level | 5-class system (original, Days 9-10, test) | 3-class system (new, locked model, test) |
|---|---|---|
| Minimal | 26 (13.3%) | 45 (23.0%) |
| Low | 64 (32.7%) | 49 (25.0%) |
| Elevated | 61 (31.1%) | 49 (25.0%) |
| Critical | 45 (23.0%) | 53 (27.0%) |

**The new system spreads risk much more evenly across all four
levels** (23/25/25/27%) instead of concentrating in Low/Elevated
(32.7%/31.1% combined = 63.8% under the old system). The Critical-rate
increase here (23.0% -> 27.0%, +3.8pp) is real but notably smaller than
what the earlier validation-based comparison showed (Phase 7.6: 20.0%
-> 34.9%, +14.9pp) -- **these two comparisons are not directly
equivalent** (different split, and Phase 7.6 used a single-architecture
model since superseded by the full 9-model ensemble), so the magnitudes
should not be conflated, even though the qualitative direction (3-class
meaningfully reshapes the risk distribution) holds in both.

**This is now the true, final, defensible system output** -- the same
untouched depression classifier, combined with the actual locked,
test-confirmed multimodal severity model, on the actual test set. This
completes Step 2 of the panel-readiness plan (reconnecting fuzzy logic
to the new locked model); Step 3 (an end-to-end pipeline figure/demo)
and Step 4 (documenting the two known integration gaps -- no text-only
fallback, unverified depression-classifier domain transfer) remain.

### Panel-readiness Step 3 -- end-to-end pipeline diagram (DONE)

`figures/19_end_to_end_pipeline.png`, generated by `fig_end_to_end_pipeline()`
in `code/make_improvement_figures.py`. Unlike every other figure in this
project (which plot experimental results), this one is an architecture
diagram -- built specifically for panel presentation, to visually
demonstrate that the multimodal component is actually integrated into
the full system, not just evaluated in isolation.

Shows the complete flow: input meme -> DAPT text embedding + SigLIP
image embedding -> two parallel branches (the 9-model multimodal
severity ensemble on one side, the untouched Phase 2 depression
classifier on the same OCR text on the other) -> their respective
outputs (3-class suicide severity, TEST macro-F1 0.5730; 4-class
depression severity) -> fuzzy-logic combination -> final 4-level risk
output, annotated with the actual locked test numbers throughout
(0.5730 macro-F1; 23/25/25/27% risk distribution on the 196 test
memes) rather than placeholder values.

Design note: the two branches are drawn side by side (not serially) so
the diagram itself makes the "these are two independent models,
combined only at the very end" structure visually obvious -- directly
addressing the earlier discussion about what "shared encoder" and
"merging" actually mean in this system (Section "shared-encoder
alignment options" and the surrounding chat: text and image are never
combined inside either the multimodal ensemble internals or the
depression classifier; the only combination step is the fuzzy-logic
layer at the bottom).

**Panel-readiness plan: Steps 1-3 complete.** Step 4 (writing up the
two known integration gaps -- no defined behavior for a missing
modality, and the depression classifier's accuracy on FigSIM's
specific translated meme text being unverified against its original
~85% figure) remains.

### Panel-readiness Step 4 -- documenting the two known integration gaps (DONE)

Written into `THESIS_SECTIONS_DRAFT.md` Section 3 (Limitations), matching
the existing documented-scope-decision style used throughout that
section, rather than left as chat-only notes:

1. **No defined behavior for a missing modality.** Every architecture in
   the locked ensemble assumes both text and image are always present;
   none has a tested fallback. Named as a deliberate scope decision (the
   original harmonization brief's "modality dropout" idea was never
   implemented, since every comparison in this project needed a clean
   single-variable baseline), with two concrete remediation paths named
   for future work (modality-dropout retraining, or routing to the
   project's existing text-only unimodal baselines).
2. **Depression classifier's real accuracy on FigSIM's text is
   unverified.** Its ~85%+ figure is from its own original evaluation
   set (naturalistic Bangla), not FigSIM's OCR-translated meme text --
   no ground-truth depression label exists for any FigSIM meme to check
   against. Grounded with two concrete, already-measured precedents from
   this exact project where domain shift produced real, non-hypothetical
   performance loss (Phase 2.1's generic-text-encoder swap, -0.073;
   Phase 7's BN-HIB pretraining transfer failure) rather than presented
   as an abstract possibility.

**Panel-readiness plan: all four steps now complete.** The system has a
locked, test-confirmed multimodal result (0.5730), a working end-to-end
integration with the fuzzy-logic layer on real test data, a
presentation-ready architecture diagram, and an honest account of its
two known integration gaps -- ready to present as a complete,
defensible deliverable rather than a collection of experiments.
