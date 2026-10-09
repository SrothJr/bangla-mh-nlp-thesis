# Approach 05 — rigor and diagnostics

This produced **no performance gain**. It produced measurements, and they are
the most defensible material in the project.

## Finding 1 — selection optimism, quantified

The original protocol early-stops by picking the epoch with the best score
**on the split it then reports**. Re-running the identical configuration with a
proper inner split for early stopping:

| | Accuracy | Macro-F1 |
|---|---|---|
| Unbiased | 0.6654 | 0.6155 |
| Peeking, the original protocol | 0.6924 | 0.6557 |
| **Selection optimism** | **+0.0270** | **+0.0402** |

The peeking figure nearly reproduces the old single-split validation number
(0.6974 / 0.6649), which is direct evidence that number was inflated by
selection rather than being a held-out estimate.

## Finding 2 — the learning curve

Models retrained on nested subsets, with validation and held-out rows fixed so
only training size varies:

| Training examples | 210 | 315 | 421 | 527 |
|---|---|---|---|---|
| Accuracy | 0.6126 | 0.6178 | 0.6332 | 0.6654 |

**Still climbing and accelerating at the largest size available:
+0.0304 accuracy and +0.0395 macro-F1 per +100
training examples.** This turns a claim carried through the whole project into
a measured number, and it identifies the binding constraint as the number of
labelled examples.

## Finding 3 — deeper integration of the two components

The only change here that helped. Feeding the depression classifier's output
distribution into the suicide model as **input features**, rather than letting
the two meet only in the rule layer at the end:

| Configuration (cross-validated, never tested) | Accuracy | Macro-F1 |
|---|---|---|
| Baseline, text + image | 0.6744 | 0.6291 |
| **+ depression distribution** | **0.6873** | **0.6397** |
| Delta | +0.0129 (P(better) 0.922) | +0.0106 |

Two points to state in the write-up:

- **No target leakage.** The depression classifier was trained on its own
  separate 4897-record dataset, not on these memes and not on their
  severity labels.
- **Not a simple correlation.** The Spearman correlation between predicted
  depression level and true suicide severity is only
  0.0897.
  The distribution carries information its argmax does not.

## Finding 4 — the full-data refit fails

Training the final model on all available data requires abandoning early
stopping for a fixed epoch count. Tested directly:

| Arm (cross-validated) | Accuracy | Macro-F1 |
|---|---|---|
| Current protocol, early stopping | 0.6744 | 0.6291 |
| Full data, fixed epochs, +18% rows | 0.6641 | 0.6187 |

It **loses despite 18% more training data**, because the best epoch varies
enormously between seeds and one fixed schedule suits almost none of them.

## Ideas tested and rejected

| Idea | Result | Verdict |
|---|---|---|
| Stacking meta-learner | −0.0039 accuracy | rejected |
| CORN cumulative-link ordinal loss | ±0.0000, macro-F1 worse | rejected |
| Structured figurative-reasoning streams | significantly worse | rejected |
| External out-of-domain negatives | class-0 F1 −0.1127 | rejected, harmful |
| Full-data refit | −0.0103 | rejected |
| Train on 5 classes, collapse to 3 | +0.0039, P(better) 0.599 | not significant |
| Soft probability averaging | +0.0039 | marginal, kept |
| Five seeds per architecture | +0.0064, P(better) 0.834 | kept |

Two worth a sentence each in the discussion. **Out-of-domain negatives do not
substitute for in-domain data** — 300 screened external memes made the class
they targeted significantly worse. And **a pre-registered gate was honoured**:
the test set was to be unlocked only at 0.72 cross-validated accuracy, the best
configuration reached 0.6873, and the test set was not touched.

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

- `figures/21_phase9_summary.png` — The learning curve, and every experiment against the honest baseline.
- `figures/22_phase9_final_verdict.png` — The full-data refit test and why no path to 70% remained.
- `figures/12_hierarchical_diagnosis.png` — An earlier rejected decomposition, diagnosed.
- `figures/13_phase6_lora.png` — LoRA fine-tuning of the frozen encoders.
