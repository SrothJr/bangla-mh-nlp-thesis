# Read this first — rules for generating a report from this package

Written for anyone, human or automated, producing a report from these folders.
This package contains ten phases of work, and **some of its documents record
conclusions that later work overturned.** Following the four rules below avoids
every known trap.

---

## Rule 1 — one file is authoritative for numbers

**`VERIFIED_RESULTS.md`.** It is regenerated from the raw confusion matrices on
every build, and every value is asserted against the stored result, so it cannot
drift. `results_table.csv` is the same data, machine-readable.

If any other document disagrees with it, that document is out of date.

## Rule 2 — the headline result

| | |
|---|---|
| **Lead with** | Cross-architecture ensemble, **macro-F1 0.4984, accuracy 0.5306** |
| Test examples | 196 |
| Folder | `best_outcome/` |

Best 3-class-equivalent figure is **0.6582 accuracy**, obtained by
collapsing that same model's predictions. Not a separate model.

**Nothing built in the ten phases afterwards beat it.**

## Rule 3 — never mix cross-validated numbers with test numbers

Two folders report **cross-validated** results for configurations that were
**never evaluated on the test set**:

| Folder | Number | Status |
|---|---|---|
| `approach_05_rigor_and_diagnostics/` | 0.6873 accuracy | cross-validated only |
| `approach_06_dual_encoder_study/` | 0.6834 accuracy | cross-validated only |

These are **not** improvements on the 0.5306 headline. They are
measured on a different protocol over the 777 train+validation examples. A
report that presents them as results would be wrong.

## Rule 4 — claims that were overturned

Historical documents are included because they are the primary record. Four
contain conclusions this project later disproved. **Each carries a warning
banner at the top of the file**, so opening it directly is safe, but here is the
map:

| Document | Overturned claim | Correction lives in |
|---|---|---|
| `PHASE2_REPRESENTATION_EXPERIMENTS.md` | "SigLIP confirmed as the better choice" | `approach_06_dual_encoder_study/` |
| `CURRENT_FINAL_SYSTEM_REPORT.md` | A claimed improvement from the 3-class model | `CURRENT_VERIFIED_SYSTEM_REPORT.md` |
| `THESIS_SECTIONS_DRAFT.md` | Predates approaches 05 and 06 entirely | `VERIFIED_RESULTS.md` |
| `analysis.md` | Predicted unfreezing encoders was the biggest lever | `approach_01`, Phase 6 document |

**Two specific numbers must never appear in a report:** a claimed "+0.0746" or
"+0.110" improvement. Both came from subtracting scores across different label
granularities, which is invalid. `VERIFIED_RESULTS.md` section 9 has the full
list.

---

## Where to find each kind of content

| You need | Go to |
|---|---|
| The headline result and its model | `best_outcome/README.md` |
| How any approach works, step by step | that folder's `WORKFLOW.md` |
| Every trial and what it scored | `approach_05_rigor_and_diagnostics/README.md` |
| Rejected ideas and why | same file, plus `project_documents/IMPROVEMENT_PLAN.md` |
| The complete chronological record | `project_documents/IMPROVEMENT_PLAN.md` |
| Figures | each folder's `figures/` |
| Original working documents | each folder's `source_documents/` |

## What this project actually concluded

1. The best result is **macro-F1 0.4984** and nothing across ten
   subsequent phases improved on it.
2. The binding constraint is **the number of labelled examples**, measured at
   +0.0304 accuracy per 100 additional examples, still climbing.
3. A large part of the apparent earlier performance was **selection optimism**,
   measured at +0.0270 accuracy.
4. Roughly a dozen ideas were tested and rejected, two of which actively made
   things worse.

A report that presents this as a story of steady improvement would be
misreading it. The honest story is a solid result, followed by a rigorous and
largely unsuccessful search for more, which established where the ceiling comes
from.
