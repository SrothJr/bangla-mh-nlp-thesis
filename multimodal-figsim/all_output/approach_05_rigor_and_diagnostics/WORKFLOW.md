# Workflow — rigor and diagnostics

This approach produced measurements, not a model. The workflow is therefore an
evaluation protocol rather than an architecture.

### The protocol that everything here uses

```
  777 train+val examples
        |
  5-fold stratified CV  (shuffle, random_state 42)
        |
        +--> per outer fold:
        |
        |    fold-train (622)                       outer held-out (155)
        |        |                                          |
        |        +--> inner split, 15% stratified           |
        |             |                |                    |
        |        inner-train      inner-val                 |
        |         (~528)           (~94)                    |
        |             |                |                    |
        |         train  ------ early stop on --------------+
        |                                                   |
        |                                    predict, never used for selection
        |
        +--> pooled out-of-fold predictions over all 777
```

**The inner split is the whole point.** The original protocol early-stopped on
the same split it then reported, which makes the reported number an upper bound
on itself. Separating them is what makes these numbers honest.

### Why that mattered — measured

Running both protocols in a single pass, so they differ in nothing but the
stopping rule:

| | Accuracy | Macro-F1 |
|---|---|---|
| Unbiased, inner split | 0.6654 | 0.6155 |
| Peeking, original protocol | 0.6924 | 0.6557 |
| **Selection optimism** | **+0.0270** | **+0.0402** |

### The learning-curve procedure

```
  For each outer fold:
     inner-val and outer rows held FIXED
     inner-train subsampled to 40% / 60% / 80% / 100%
     subsets NESTED and stratified, so the curve measures quantity
     not luck of the draw
```

Nesting matters: the 40% rows are a subset of the 60% rows, and so on, so
successive points differ only by added data.

### Statistical treatment, applied to every comparison

- **Paired bootstrap**, 5000 resamples, sharing the resample between the two
  methods being compared so the difference is measured on identical rows.
- Reported as **P(better)**, the fraction of resamples where the difference is
  positive, rather than a point estimate alone.
- **No bundling.** Every change is A/B'd on its own so effects can be
  attributed.

### The pre-registered gate

Fixed in writing before any experiment ran:

```
  unlock the test set only if
      CV accuracy >= 0.72  AND  lower bootstrap CI bound >= 0.68
```

The best configuration reached 0.6873 with a lower bound of 0.6538. **The gate
failed and the test set was not touched.** The margin above 0.70 was deliberate
insurance against the validation-to-test gap the project had already been
bitten by once.

### The one architectural change that helped

```
  depression classifier --> 4-d distribution --+
                                               |
  [ e3b_text(768) ; siglip(1152) ; depression(4) ] --> fusion head
```

Four extra input dimensions. The depression checkpoint is not retrained; it is
called through the parity-tested wrapper.

### Reproduce

```
python code/phase9_1_cv_baseline_3class.py       # honest baseline
python code/phase9_3_learning_curve.py           # the curve
python code/phase9_13_assemble_and_gate.py       # the gate
python code/phase9_19_depression_features_full_ensemble.py
python code/phase9_15_a1_refit_validation.py     # full-data refit test
```

Results files: `outputs/phase9_*.json`.
