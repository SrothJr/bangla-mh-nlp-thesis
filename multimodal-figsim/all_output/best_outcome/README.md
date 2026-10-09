# BEST OUTCOME — cross-architecture ensemble (5-class)

**This is the result to lead with in the thesis.** It is the strongest
validated result the project produced, and the only one measured on the test
set that survived every later re-examination.

## The numbers

| Metric | Value |
|---|---|
| **Macro-F1** | **0.4984** |
| Accuracy | 0.5306 |
| Weighted-F1 | 0.5232 |
| Quadratic weighted kappa | 0.3857 |
| Test examples | 196 |

| Class | F1 |
|---|---|
| None | 0.476 |
| Wish to be dead | 0.659 |
| Suicide ideation | 0.603 |
| Suicide planning | 0.348 |
| Suicide attempt or death | 0.406 |

Confusion matrix, rows are true and columns predicted:

```
  [15, 5, 6, 2, 4]
  [4, 29, 3, 1, 1]
  [3, 11, 38, 4, 8]
  [1, 2, 9, 8, 6]
  [8, 3, 6, 5, 14]
```

## The same model on the coarser 3-class task

Collapsing this model's own predictions into the 3-class grouping, with no
retraining, gives the **best 3-class-equivalent result on record**:

| Metric | Value |
|---|---|
| Accuracy | **0.6582** |
| Macro-F1 | 0.6024 |
| Weighted-F1 | 0.6532 |

```
  [15, 11, 6]
  [7, 81, 14]
  [9, 20, 33]
```

This matters because a separately trained 3-class model scored
0.6071 accuracy and 0.5730 macro-F1, which is
**worse**. See `../approach_02_label_harmonization_3class/`.

## What the model is

Six models combined by majority vote: the gated fusion architecture with an
orthogonal residual, and the cross-attention architecture, three random seeds
each. Both encoders frozen throughout.

Ensembling was the single largest lever in the entire project, worth more than
every loss function, fusion mechanism and hyperparameter change put together.

## Why it is still the best after ten phases

Everything built afterwards was measured against it and nothing beat it:

| Later attempt | Outcome |
|---|---|
| 3-class harmonized model | worse once fairly compared (-0.0510 accuracy) |
| Phase 9, ten experiments across five tracks | gate not cleared, test never evaluated |
| v2 dual-encoder study | level in cross-validation, marginally behind |

## Improvement over the starting point

| | Macro-F1 | Accuracy |
|---|---|---|
| Single-model baseline | 0.4753 | 0.5051 |
| **This ensemble** | **0.4984** | **0.5306** |
| Gain | **+0.0231** | +0.0255 |

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

- `figures/09_phase3_architecture.png` — The three fusion architectures this ensemble combines.
- `figures/04_confusion_matrix.png` — Confusion matrix on the test set.
- `figures/05_per_class_metrics.png` — Per-class precision, recall and F1.
- `figures/11_cumulative_progression.png` — How the project arrived here.
- `figures/20_current_verified_pipeline.png` — Full system architecture.

## Caveat to state

The test set is 196 examples, so a 95% confidence interval spans roughly
±7 points. Report the number with that uncertainty rather than as a point
estimate.
