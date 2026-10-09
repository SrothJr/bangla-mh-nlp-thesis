# Approach 04 — fuzzy-logic risk layer

The integration layer. Combines the multimodal suicide-severity output with the
text-only depression output into a single risk level.

## Mechanism

Each classifier's distribution is treated as a membership degree per category.

- Fuzzy AND is the **minimum** of two memberships
- Fuzzy OR is the **maximum**
- The final risk level is the **argmax** of the combined memberships

**Confirmed from code, not assumed:** the suicide side contributes **hard vote
fractions**, meaning the share of ensemble members voting for each class. The
depression side contributes genuine softmax probabilities.

## Results on the test set (196 memes)

| Risk level | 5-class system | 3-class system |
|---|---|---|
| Minimal | 26 | 45 |
| Low | 64 | 49 |
| Elevated | 61 | 49 |
| Critical | 45 | 53 |

## The limitation that must be stated

**No ground-truth combined risk label exists for any meme.** This is a
distributional and qualitative comparison, never an accuracy claim. The layer
is exploratory and rule-based, and it is not clinically validated.

Say this plainly in the thesis. A rule layer over two model outputs, presented
without that caveat, would overclaim.

## How it works

See `WORKFLOW.md` in this folder for the full pipeline, architecture
diagrams, hyperparameters and reproduction commands.

## Figures

- `figures/16_fuzzy_logic_5class_vs_3class.png` — Rule tables and how the two label schemes compare.
- `figures/18_final_fuzzy_test_comparison.png` — Final risk-level output on the test set.
