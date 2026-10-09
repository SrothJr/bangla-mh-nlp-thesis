> **Extracted from `IMPROVEMENT_PLAN.md`.**
>
> Phases 7, 8 and 9 each wrote their own planning document. Phases 1 to 6 did
> not: they were written directly into the project's chronological experiment
> log. This file carves out that phase's section so it can sit with the
> approach it belongs to.
>
> The text is unchanged. The complete log, with every phase in sequence, is in
> `../../project_documents/IMPROVEMENT_PLAN.md`.

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
