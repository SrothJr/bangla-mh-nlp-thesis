# Approach 06 — dual image encoders

A fresh-eyes restart that reopened a decision closed on confounded evidence.
**It did not beat the system**, but it corrected the record and produced the
largest single-change gain measured anywhere in the project.

## The observation

Ten phases varied the fusion architecture, the loss and the combination rule.
Almost none varied the **input representation**. Every model consumed the same
two frozen vectors, chosen in week one and never revisited.

## The decision that was wrong

An earlier phase tested CLIP against SigLIP and rejected it at 0.4666 against
0.5330, concluding SigLIP was "the right call".

**That comparison was confounded.** It ran CLIP inside the gated architecture,
which consumes contrastive alignment projections, and its own notes record CLIP
aligning badly: retrieval 2.6% against SigLIP's 6.7%. It measured
representation quality times alignment quality and blamed the representation.

## The clean test — no alignment in any arm

| Arm | Accuracy | Macro-F1 |
|---|---|---|
| text + SigLIP, the project's choice | 0.6371 | 0.5917 |
| text + CLIP, previously rejected | 0.6448 | 0.5919 |
| **text + SigLIP + CLIP, never tried** | **0.6628** | **0.6154** |

**CLIP is not better than SigLIP. It is different from SigLIP.** Head to head
they are statistically indistinguishable. Together they beat either alone by
+0.0257, P(better) 0.943.

## Per architecture

| Architecture | SigLIP only | + CLIP | Delta |
|---|---|---|---|
| concat | 0.6371 | 0.6628 | **+0.0257** |
| gated | 0.6088 | 0.6461 | **+0.0373** |
| cross-attention | 0.6486 | 0.6229 | **-0.0257** |

Two improve, one degrades.

## The ensemble lesson — worth a discussion paragraph

| Configuration (cross-validated, never tested) | Accuracy |
|---|---|
| 15 models, SigLIP only | 0.6744 |
| 15 models, CLIP in every head | **0.6474** |
| 20 models, SigLIP-only plus dual-concat | **0.6834** |

The middle row is the instructive one. **Two of three members improved
individually and the ensemble got worse.** Giving every member the same extra
features correlates their errors, and ensembles are paid in disagreement. The
winner keeps the SigLIP-only members and adds dual-encoder members beside them.

## Outcome

| Configuration (cross-validated, never tested) | CV accuracy | CV macro-F1 |
|---|---|---|
| Previous best | 0.6873 | 0.6397 |
| **This study's best** | 0.6834 | 0.6380 |
| Difference | -0.0039 | -0.0017 |

Level within noise, marginally behind. **Cross-validated only. Never evaluated
on the test set.**

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

- `figures/v2_01_dual_encoder_findings.png` — The probe, the per-architecture effect, and the ensemble outcome.
