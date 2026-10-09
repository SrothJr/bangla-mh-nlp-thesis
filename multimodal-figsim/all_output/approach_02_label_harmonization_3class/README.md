# Approach 02 — label harmonization to 3 classes

Collapses the 5-class severity scheme into three coarser categories, then
trains a model natively on them.

| Original 5-class | Harmonized 3-class |
|---|---|
| None | No expressed severity |
| Wish to be dead, Suicide ideation | Suicidal thought or desire |
| Planning, Attempt or death | High acuity suicidal content |

## The numbers

| Metric | Value |
|---|---|
| Macro-F1 | 0.5730 |
| Accuracy | 0.6071 |
| Weighted-F1 | 0.6126 |
| Quadratic weighted kappa | 0.3765 |
| Test examples | 196 |

| Class | F1 |
|---|---|
| No expressed severity | 0.462 |
| Suicidal thought or desire | 0.683 |
| High acuity suicidal content | 0.574 |

```
  [18, 11, 3]
  [17, 68, 17]
  [11, 18, 33]
```

## The comparison rule that must not be broken

**Never subtract a 3-class score from a 5-class score.** A coarser task is
easier to score well on regardless of whether the model improved. An earlier
draft claimed a "+0.0746 improvement" by doing exactly that, and it was wrong.

The valid method collapses the finer model's own predictions into the coarser
grouping and compares on identical records, with no retraining:

| | Collapsed 5-class ensemble | Native 3-class model | Valid delta |
|---|---|---|---|
| Accuracy | 0.6582 | 0.6071 | **-0.0510** |
| Macro-F1 | 0.6024 | 0.5730 | **-0.0294** |

**The 3-class model is not an improvement. It is slightly weaker once fairly
compared.** Report it as a differently-scoped result, which is what it is.

## Reading the two metrics

0.5730 is the **macro-F1** and 0.6071 is the
**accuracy**, both from this same model on the same 196 memes. Accuracy
is carried by the large middle class (102 of 196), while macro-F1 weights
all three equally and is dragged down by the small first class
(32 examples, F1
0.462).

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

- `figures/14_label_harmonization.png` — The 5-class to 3-class label scheme.
- `figures/17_final_3class_test_lock.png` — The locked 3-class test result.
- `figures/15_bnhib_pretraining_transfer_test.png` — External meme pretraining attempt.
