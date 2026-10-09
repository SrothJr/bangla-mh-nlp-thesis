# Approach 01 — single-model baseline (5-class)

The project's first locked result, and the reference every later result is
measured against.

## The numbers

| Metric | Value |
|---|---|
| Macro-F1 | 0.4753 |
| Accuracy | 0.5051 |
| Weighted-F1 | 0.4953 |
| Quadratic weighted kappa | 0.3732 |
| Test examples | 196 |

| Class | F1 |
|---|---|
| None | 0.483 |
| Wish to be dead | 0.640 |
| Suicide ideation | 0.544 |
| Suicide planning | 0.304 |
| Suicide attempt or death | 0.405 |

```
  [14, 7, 6, 1, 4]
  [2, 32, 2, 0, 2]
  [2, 13, 31, 8, 10]
  [0, 5, 7, 7, 7]
  [8, 5, 4, 4, 15]
```

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
gains +0.0231 macro-F1.

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

- `figures/01_ablation_ladder.png` — Ablation ladder that led to this configuration.
- `figures/06_final_test_summary.png` — The locked test result.
- `figures/02_coral_vs_ce.png` — Ordinal loss comparison, CORAL rejected.
- `figures/03_gate_alpha_fix.png` — The gate-collapse failure and its fix.
