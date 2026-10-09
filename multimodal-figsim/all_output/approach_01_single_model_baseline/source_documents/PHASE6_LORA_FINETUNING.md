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
