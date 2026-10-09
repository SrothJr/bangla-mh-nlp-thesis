# Source documents — approach_01_single_model_baseline

The primary record for this approach, copied here unchanged.
`../README.md` and `../WORKFLOW.md` are the condensed, verified versions
and are what to quote from.

## Phase sections

Phases 1 to 6 never had their own planning documents; they were written
directly into the chronological experiment log. These are carved out of
it so each sits with the approach it belongs to. Text unchanged.

### `PHASE1_ENSEMBLING_AND_LOSSES.md`

Seed ensembling, gate uncertainty, ordinal smoothing and focal loss. Seed ensembling was kept and became the foundation of every later result; gate uncertainty and focal loss were rejected.

### `PHASE2_REPRESENTATION_EXPERIMENTS.md`

The translation-hop diagnostic, embedding mixup, supervised contrastive alignment, and the CLIP-versus-SigLIP comparison. **Section 2.4 is the decision that approach 06 later reopened and found confounded** -- read it alongside that folder.

### `PHASE6_LORA_FINETUNING.md`

LoRA fine-tuning of the frozen encoders. It worked where full and partial unfreezing had failed, but the gain was small enough that the frozen-encoder design was kept. The clearest test of whether the encoders were the ceiling.

## Working documents

### `PROGRESS_LOG.md`

Day-by-day build log for the original 12-day schedule. The primary record of how the baseline was built, including the data pipeline and the first locked model.

### `analysis.md`

The diagnostic written after the baseline locked, ranking where the remaining leverage was. It triggered everything that followed. Worth reading for how the priorities were set, and for how several of its predictions turned out wrong.
