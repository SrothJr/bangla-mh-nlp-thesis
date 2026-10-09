# Workflow — fuzzy-logic risk layer

### Where it sits

```
  multimodal ensemble --> suicide severity  --+
                                              +--> fuzzy combination --> risk level
  depression classifier --> depression level -+
```

This layer trains nothing. It is a deterministic rule system over two model
outputs, which is deliberate: every decision it makes can be traced by hand.

### The membership vectors

The two inputs are **not** the same kind of quantity, and the thesis should say
so:

| Source | What is fed in | Why |
|---|---|---|
| Suicide severity | **hard vote fractions** — the share of ensemble members voting for each class | the ensemble combines by majority vote, so its members' raw probabilities are discarded before this point |
| Depression | **softmax probabilities** | a single model, so a genuine distribution exists |

This was confirmed by reading the code, not assumed.

### The operators

```
  fuzzy AND (a, b) = min(a, b)
  fuzzy OR  (a, b) = max(a, b)
  defuzzify        = argmax over the combined memberships
```

Each rule pairs a suicide class with one or more depression classes and emits a
risk level. A rule fires with strength equal to the fuzzy AND of its two
memberships. Rules emitting the same risk level combine with fuzzy OR. The
final risk level is the argmax across the four levels.

### The two rule tables

The 5-class table has 11 rows. The 3-class table has 6 and was **derived from
it**, not written fresh, using a documented cautious-merge policy: when two
original rules merge into one bucket, take the **more severe** of their risk
levels at each depression level.

```
  Example of the merge
  ---------------------
  5-class:  Wish to be dead + Moderate/Severe -> Elevated
            Suicide ideation + Moderate/Severe -> Elevated
  3-class:  Suicidal thought or desire + Mild/Moderate/Severe -> Elevated
```

Both tables are printed in full in `../VERIFIED_RESULTS.md` section 5.

### The hard limitation

**No ground-truth risk label exists for any meme.** Nothing here can be scored
for accuracy. The results in the README are distributions over the test set and
nothing more. The layer is exploratory and rule-based, and is not clinically
validated.

### Reproduce

```
python code/phase7_11_fuzzy_logic_final_test.py
```

The script first reproduces the locked ensemble exactly as a sanity check, then
applies the rule table. Results file:
`outputs/phase7_11_fuzzy_logic_final_test_results.json`.
