# Approach 03 — depression classifier (text-only)

The text-only half of the system. DAPT-BanglaBERT with its own 4-class
classification head, trained on a separate Bangla mental-health text dataset.

**Never retrained or modified at any point across all ten phases.**

## The numbers

| Metric | Value |
|---|---|
| Accuracy | 0.9320 |
| Macro-F1 | 0.9182 |
| Weighted-F1 | 0.9308 |
| Records | 4897 |

| Class | F1 |
|---|---|
| Minimum | 0.963 |
| Mild | 0.924 |
| Moderate | 0.832 |
| Severe | 0.954 |

```
  [2024, 62, 9, 2]
  [34, 1484, 19, 3]
  [36, 123, 536, 12]
  [12, 3, 18, 520]
```

## Mandatory caveat — quote this alongside the number

This checkpoint was trained on roughly four fifths of this exact dataset. The
figure is a **behaviour sanity check against its own training domain**, not an
unbiased held-out generalization estimate. It confirms the model works
correctly and behaves sensibly. It must not be presented as a fresh accuracy
result.

Supporting detail: the weakest class is Moderate at
0.832 F1, confused mostly with the adjacent Mild class.
That is the expected pattern for adjacent severity levels, not a concerning
failure.

## What is a clean claim

The text-only path was **parity-tested**: 100.00% label agreement and 0.00
maximum probability difference against 196 previously saved outputs. An exact
match, not an approximation. That is safe to state without qualification.

## Its role in the multimodal system

Two roles. It contributes the depression axis to the fuzzy-logic risk layer
(`../approach_04_fuzzy_logic_risk_layer/`), and its output distribution was
later found to work as an **input feature** for the suicide-severity model
(`../approach_05_rigor_and_diagnostics/`).

Unlike the multimodal branch, this classifier has **no missing-modality
limitation**. It needs text only, and that route is verified.

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

- `figures/20_current_verified_pipeline.png` — Where this classifier sits in the system, with its verified text-only route.
