# Progress log

Running log of what actually happened each day, including bugs hit and fixed.
Kept separate from PROJEC~1_UPDATED.MD (the plan/brief) — this is the record
of execution.

## Day 1 — Sept 10

- Installed `scikit-fuzzy`, `ollama` client into the existing `DAPT_models\.venv`.
  `paddleocr` installed but abandoned (see OCR note below).
- `ollama list` showed no vision-capable model installed. Pulled `qwen2.5vl:7b`
  (6.0GB). First pull attempt was accidentally killed by a `kill %1` meant to
  cancel a size-preview; caught via disk space not changing, restarted cleanly.
- **Checkpoint sanity check — found and fixed a real bug in the brief's own
  instructions.** `BanglaBERT_fold5_tapt/` is the TAPT/MLM checkpoint saved
  *before* fine-tuning (`ElectraForMaskedLM`, no classification head) — not the
  fine-tuned classifier the brief assumed. The actual fine-tuned checkpoint is
  the sibling `BanglaBERT_fold5/checkpoint-735/` (`ElectraForSequenceClassification`,
  confirmed via `trainer_state.json`: `best_metric: 0.8758` at `best_global_step: 735`,
  matching `BanglaBERT_dapt_metrics.json` exactly). That checkpoint also has no
  saved tokenizer (Trainer was never given one to auto-save) — loading
  `AutoTokenizer` directly on it silently produces a dummy vocab and ~42%
  accuracy. Fix: load the tokenizer from `BanglaBERT_fold5_tapt/` instead
  (same tokenizer throughout the pipeline, never modified). With both fixes,
  reproduced fold-5 metrics exactly: accuracy 0.8795, F1 0.8758, precision
  0.8812, recall 0.8795. Brief corrected to `PROJEC~1_UPDATED.MD` accordingly.
- **OCR: switched from PaddleOCR to EasyOCR.** PaddleOCR needs a separate
  `paddlepaddle` backend not installed here, risky to add given disk
  constraints; EasyOCR reuses the existing torch/CUDA install and is the
  brief's own listed fallback.
- OCR run over all 973 leakage-safe images → `outputs/ocr_results.jsonl`.
  Hit one real bug: image 0563.gif (animated GIF) crashed EasyOCR with a
  shape mismatch. Fixed by loading via PIL and converting to a single RGB
  frame before handing a plain array to EasyOCR. Final: 973/973, 0 errors,
  18 images with no detected text.
- **NLLB-200-3.3B correction.** The brief assumed it was inside `DAPT_models`;
  that HF-cache entry there is an empty stub (no weight files). The real,
  complete 3.3B model (confirmed via config: d_model=2048, 24 layers) is at
  `tari_translation_v3/nllb_model/` — Phase-2's original translation project
  folder. Used from there, read-only, no redownload.
- Translation run over all 973 OCR results → `outputs/translation_results.jsonl`.
  973/973, 0 errors, 955 actually translated (18 empty-OCR images correctly
  skipped), mean translation log-prob -0.73.
- Disk space: started ~17-18GB free, 11GB after the Qwen pull, later observed
  back up to ~30GB free (unrelated system cleanup, not our doing).

## Day 2 — Sept 11 (in progress)

- SigLIP-so400m (3.5GB) and DAPT-BanglaBERT (fold5/checkpoint-735 body,
  fold5_tapt tokenizer) embedding extraction running over all 973 images.
- Hit and fixed a bug in both extraction scripts: `np.save()` silently
  appends `.npy` to any filename that doesn't already end in `.npy`, so an
  atomic-write tempfile named `N.npy.tmp` actually got written as
  `N.npy.tmp.npy`, breaking the subsequent `os.replace()` rename. Fixed by
  passing an open file handle to `np.save()` instead of a bare path (numpy
  skips the auto-extension behavior for file handles).
- E0/E1/E3 baselines run on the VALIDATION split (test stays locked until Day 7,
  per Section 8's protocol). Hit and fixed one more bug: newer sklearn removed
  `LogisticRegression`'s `multi_class` argument (multinomial is now the lbfgs
  default) — dropped the argument.

  | Config | macro-F1 (val) | weighted-F1 (val) | accuracy (val) |
  |---|---|---|---|
  | E0 majority-class ("Suicide ideation" always) | 0.100 | 0.167 | 0.333 |
  | E1 image-only (SigLIP + logistic regression) | 0.390 | 0.418 | 0.426 |
  | E3 text-only (DAPT-BanglaBERT, OCR-only) | 0.379 | 0.409 | 0.420 |

  Both E1 and E3 clear the majority-class floor by a wide margin (~0.39 vs 0.10
  macro-F1), and are close to each other — image-only slightly ahead of
  text-only OCR-only here, though "Suicide planning" is the weakest class for
  both (E1 recall 0.29, E3 recall 0.13). Full per-class report and confusion
  matrices in `outputs/e0_e1_e3_results.json`. Day 3 (AI reasoning + E3b) should
  clarify whether OCR-only text is the ceiling for the text branch or whether
  reasoning text closes the gap with image-only.

## Day 3 — Sept 12 (in progress)

- Confirmed picking up from `PROJEC~1_UPDATED.MD` (Days 1-2 fully in sync);
  also updated that file's Day 2 schedule row with the actual E0/E1/E3 numbers
  (it previously still had the plan text, not results).
- Qwen2.5vl:7b reasoning pass over all 973 images. Prompt requires a JSON
  object with `cause_effect`, `figurative_meaning`, `emotional_state`, each
  with `claim`/`evidence`/`uncertain` — evidence must be a specific quote or
  visual detail, or the literal string "uncertain" rather than a fabricated
  citation. Ran under `format="json"` for reliable parsing, `temperature=0.2`.
  Result: 973/973, 0 errors, 0 empty. Model used "uncertain" honestly on 87 of
  2919 possible claims (~3%) rather than always asserting confidence — spot
  checked a few, reasoning reads as sensible and grounded, not hallucinated.
  Throughput ~0.80 img/s (~20 min total) — much faster than the single-image
  smoke test's 13s suggested (that included cold-start/model-load overhead).
- **BLOCKED: `tari_translation_v3\` (the entire folder, not just `nllb_model\`)
  has disappeared from disk.** Discovered when `translate_reasoning.py`
  crashed treating the local path as an invalid HF hub repo id --
  `os.path.isdir()` on the path returns `False`. This folder existed and
  worked correctly earlier today (Day 1's OCR-text translation completed
  successfully from it, ~16:01). Very likely explanation: someone else on
  this shared machine (Section 4.1's known risk) deleted or moved it --
  this lines up with the otherwise-unexplained free-disk-space jump observed
  earlier today (11GB -> 30GB -> 27GB, roughly matching the model's ~17.5GB).
  Verified `DAPT_models` is untouched and intact, and our own
  `outputs/translation_results.jsonl` (973 lines, already computed and saved
  Day 1) is safe -- the loss only blocks *new* NLLB inference (reasoning-text
  translation going forward), not anything already computed. Stopped here
  for a decision rather than silently redownloading ~17.5GB.
- **Resolved: false alarm.** `tari_translation_v3\` was moved by a teammate
  (not deleted) and has since been moved back to its original path. Verified
  `os.path.isdir()` now returns `True`, `config.json` present, and the
  tokenizer loads correctly (`NllbTokenizer`, tokenizes a test sentence as
  expected). No redownload, no fallback model, no path change needed.
  Resuming `translate_reasoning.py` from the original NLLB_PATH.
- Reasoning-text translation (retry): 973/973, 0 errors, mean translation
  log-prob -0.41 (higher confidence than OCR's -0.73 -- expected, reasoning
  text is grammatical model-generated English vs. noisy OCR output).
- E3b embeddings: 973/973 extracted (OCR_bn + reasoning_bn via tokenizer
  sequence-pair encoding), no errors, no leftover temp files.
- **E3b classifier result (identical `LogisticRegression(max_iter=2000)` as
  E1/E3, validation split n=195):**

  | Config | macro-F1 (val) | weighted-F1 (val) | accuracy (val) |
  |---|---|---|---|
  | E3  (OCR only) | 0.3792 | 0.4087 | 0.4205 |
  | E3b (OCR + AI reasoning) | 0.3834 | 0.4170 | 0.4359 |
  | Delta | +0.0042 | +0.0083 | +0.0154 |

  Reasoning text gives only a marginal macro-F1 lift (+0.004) over OCR-only --
  essentially within noise for n=195. Per-class detail is more interesting
  than the aggregate: "Suicide ideation" recall jumps 0.60->0.69 and "Suicide
  attempt or death" precision improves 0.34->0.46, but "Wish to be dead"
  recall drops 0.53->0.34 and "Suicide planning" recall drops further
  (0.13->0.16, still the weakest class by far). Net: AI reasoning reshuffles
  where the classifier's errors land more than it reduces them overall, at
  least with a linear classifier and no fusion with the image branch yet.
  This is consistent with the brief's own note (Section 3) that FigSIM's
  original paper also saw fusion architectures barely beat text-only, with
  real gains only from massive frozen prompted MLLMs -- worth remembering
  when interpreting E4/E5/E6 later, and worth stating plainly in results
  rather than oversold. Full detail in `outputs/e3b_results.json`.

## Day 4 — Sept 13 (in progress)

- **Design decision, documented in `train_e4_e5.py`'s docstring too:** E4/E5
  use the E3b (OCR + reasoning) text embeddings, not E3 (OCR-only). Section
  5.1's text branch treats AI reasoning as CORE (step 2), not a stretch goal
  -- E3/E3b were an ablation testing reasoning's value in isolation; E4
  onward builds the actual architecture, which uses the full text branch.
- **Design decision:** E5's gate (`alpha*text + (1-alpha)*image`) and
  orthogonal-feature step both need text/image vectors in the same dimension,
  but BanglaBERT (768) and SigLIP (1152) differ. Section 5.3's contrastive
  alignment (Stage A) would normally provide that shared space, but Stage A
  is E6, not yet run. So E5 uses its own small supervised linear projections
  (trained jointly with the gate + classifier), which E6 can later
  replace/augment with Stage A's self-supervised ones. Flagging this as an
  engineering necessity, not something the brief specified explicitly.
- Gate inputs: OCR confidence + OCR-translation confidence (as named in
  Section 5.4), standardized with train-split statistics. Reasoning-agreement
  is explicitly stretch-goal-only (needs the unbuilt two-model ensemble), so
  not included.
- CORAL-style ordinal loss implemented per Cao et al. (2020): single shared
  rank score + K-1=4 monotonically-ordered thresholds (enforced via
  cumulative softplus deltas), trained with per-threshold weighted BCE.
- Ran 3 seeds (0/1/2) x {E4, E5} x {class-weighted CE, CORAL} = 12 runs, all
  on cached embeddings (CPU, seconds each). Class weights = inverse frequency
  from train split.

  | Config | macro-F1 (val, mean +/- std over 3 seeds) |
  |---|---|
  | E3b (single run, no seeds, for reference) | 0.3834 |
  | E4 concat, weighted CE | **0.5038 +/- 0.0057** |
  | E4 concat, CORAL | 0.3716 +/- 0.0036 |
  | E5 gated+orth, weighted CE | 0.4983 +/- 0.0277 |
  | E5 gated+orth, CORAL | 0.3514 +/- 0.0007 |

  **Fusion clearly earns its place:** both E4 and E5 (CE variants) jump
  ~0.12-0.13 macro-F1 over E3b's text-only 0.3834, well outside their small
  seed-to-seed std -- this is a real effect, not noise. **Gated fusion does
  NOT beat simple concatenation here** (E5 0.498 vs E4 0.504, and E5 has
  4-5x higher variance across seeds) -- matches the brief's own explicit
  caution (Section 3) not to assume gated fusion helps; measured, and for
  this dataset size it doesn't yet. **CORAL substantially underperforms
  weighted CE for both fusion types** (~0.35-0.37 vs ~0.50). Verified this
  isn't a training bug: inspected E5_coral's confusion matrix and found a
  real, explicable cause -- the rarest class ("Suicide attempt or death")
  collapsed to 0 precision/recall/F1 for that seed. CORAL's rank-consistency
  guarantee requires collapsing the whole representation to a single scalar
  score (shared across all K-1 thresholds by construction), which is a much
  smaller-capacity bottleneck than CE's 5 independent class logits --
  plausible and expected to hurt more in a small (582-train), imbalanced,
  5-class setting like this one, not evidence of a bug. Full detail
  (confusion matrices, per-class reports, all 12 seed runs) in
  `outputs/e4_e5_results.json`.

### Section 7 spot-check (informal, in parallel with E4/E5)

Reviewed 20 AI-reasoning outputs, stratified across all 6 original
suicide_scale labels plus the 4 no-OCR-text (image-only) cases in the
sample. For 4 of those image-only cases (0116, 0157, 0518, 0682) I also
viewed the actual image directly to check visual grounding, not just text
plausibility.

**Findings:**
- When OCR text is present and legible, the reasoning tracks the quoted
  text closely and rarely goes far afield -- e.g. 0205's "I COULDN'T FACE
  LIFE... STICK MY HEAD IN THIS OVEN" is read correctly and literally; 0287's
  atoms-never-touch joke is correctly read as a dark simile for
  proximity-to-suicidal-action, not over- or under-stated.
- Image-only cases are where the real weaknesses showed up, and it's a
  specific, nameable failure mode: **correct low-level object recognition,
  wrong high-level narrative, without flagging uncertainty.** 0116 (Nike
  swoosh + gun + distressed face) was read correctly. But 0157 (children's
  clothes hung from a shop ceiling, visually reading as hanged bodies -- a
  dark hanging pun, `suicide_scale: Suicide death`) was reinterpreted as
  "pressure to conform to societal beauty standards," missing the much more
  direct hanging/suicide visual pun -- and reported `uncertain: False`
  throughout. 0682 (anime character's whiteboard equation "stool + noose =
  heart," `suicide_scale: Suicide planning`) correctly identified the
  stool/noose objects but then read the heart as being about "the dangers of
  romantic relationships" rather than the much darker ironic "peace/love as
  euphemism for death" reading the meme is actually making -- also
  `uncertain: False`.
- By contrast, the single hardest/most stylized image in the sample (0518,
  a 4-panel Lego minifigure execution-by-hanging comic) correctly triggered
  `uncertain: True` on all three fields with an appropriately hedged
  interpretation -- so the uncertainty mechanism does work, just not
  consistently across similarly-ambiguous image-only content.
- No evidence of wild fabrication (inventing people, on-image text, or
  events not actually present). The errors found are consistently "plausible
  over-interpretation of real objects," not hallucination from nothing.
- One OCR-propagation observation, not a reasoning-model failure per se:
  0149's OCR was truncated ("...9LIKE YOUR C...", almost certainly "I like
  your cuts," a callous self-harm/wrist-scarring reference) and the
  reasoning model's read stayed at a more generic "violent act" level as a
  result -- a case where reasoning quality is downstream-bounded by OCR
  quality, worth noting separately from reasoning-model accuracy itself.

**One paragraph for the limitations/methodology section (ready to use):**
"An informal review of 20 stratified AI-reasoning outputs (Section 7) found
the model to be generally sensible and non-hallucinatory: when OCR-extracted
text is present, its cause-effect, figurative-meaning, and emotional-state
claims track the quoted text closely, and no instance of fabricated
on-image content (invented text, people, or events) was observed. The
identified weakness is specific to image-only memes with no legible text:
in 2 of 4 visually-verified image-only cases, the model correctly identified
the meme's concrete visual elements (e.g., a noose, a stool, a hanging
motif) but constructed an incorrect higher-level narrative around them
(e.g., reading a hanging-clothes visual pun as commentary on beauty
standards rather than a dark reference to suicide), while reporting full
confidence (`uncertain: False`) rather than flagging the ambiguity. A
separately reviewed, more visually ambiguous case did correctly trigger the
model's uncertainty flag, indicating the mechanism is functional but not
reliably triggered on this class of error. This is a scoped, documented
limitation of the reasoning branch rather than a blocking defect, and is
consistent with this project's decision (Section 3) not to assume large
gains from any single architectural component without measuring them."

### E5 gate alpha diagnostic (requested after Day 4 results)

Retrained E5 (weighted CE) for each of the 3 Day-4 seeds, capturing the
gate's alpha on the validation set at each seed's best epoch.

| Seed | val macro-F1 | alpha mean | alpha std | alpha max | % alpha < 0.1 |
|---|---|---|---|---|---|
| 0 | 0.5357 | 0.079 | 0.079 | 0.371 | 70.8% |
| 1 | 0.4897 | 0.103 | 0.113 | 0.409 | 63.6% |
| 2 | 0.4696 | 0.097 | 0.090 | 0.457 | 68.2% |

**Finding: alpha is collapsed, not genuinely varying per-meme.** Across all
3 seeds, alpha never exceeds ~0.46 and 64-71% of validation items sit below
0.1 -- the gate has learned to almost always weight the image branch
dominant (`gated = alpha*text + (1-alpha)*image`, so alpha near 0 means
image, near 1 means text) regardless of per-meme content, rather than
adaptively mixing. Not a single-point degenerate constant, but functionally
close: it never once crosses into text-dominant territory. This directly
explains the Day 4 result -- E5 is effectively training on a near-image-only
signal most of the time, discarding most of the text branch's real,
independently-verified value (E3b's OCR+reasoning-only macro-F1 was 0.383,
not nothing), which is consistent with it landing close to E1's image-only
0.390 rather than clearly beating E4's balanced concatenation, and its
higher cross-seed variance likely reflects how close different random inits
land to slightly different degenerate solutions along this same collapsed
axis rather than finding a genuinely adaptive gate.

## Day 5-6 — Sept 14-15 (in progress)

Per the user's direction: dropped CORAL from all further runs (Day 4's
finding was clear and CE-only halves the config matrix). Built contrastive
alignment (Stage A, Section 5.3) on top of E4 (concat) as primary, on top of
E5 (gated+orth) as secondary -- testing specifically whether pre-alignment
fixes the gate-collapse just diagnosed.

**Stage A training -- stop-loss rule watched closely:**
- First attempt (2-layer MLP projections, 512-dim shared space, weight_decay
  1e-4): train loss decreased sensibly and train retrieval climbed to 70% by
  epoch 60, but VAL loss diverged after ~epoch 20 (5.47 -> 7.96 -> 8.03) while
  val retrieval stayed near chance (best-val-loss checkpoint: only 1.5%
  retrieval vs 0.5% chance) -- classic overfitting on 582 contrastive pairs,
  a real early warning sign the stop-loss rule exists to catch.
- One round of "reasonable debugging effort" (Section 5.3's own phrase):
  simplified to plain single linear projections (no hidden layer -- matches
  the brief's literal "small trainable projection layers"), shared dim
  512->256, weight_decay 1e-4->1e-2, added dropout(0.3). Result: healthier,
  still-improving retrieval trend at the point early stopping triggered
  (6.7% val retrieval at epoch 60, ~13x chance, vs ~7x chance in the first
  attempt) though val_loss still isn't cleanly monotonic. Modest but real
  signal, not degenerate -- proceeded to the actual test (does it help
  downstream classification?) rather than iterating further on retrieval
  alone, given the Day-6 time budget.
- Used the "current" checkpoint (epoch 63, closest to peak retrieval before
  early stopping) as Stage A's frozen projections for E6.

**E6 result (weighted CE only, 3 seeds, validation split):**

| Config | macro-F1 (mean +/- std) |
|---|---|
| E4 (raw concat) | 0.5038 +/- 0.0057 |
| E5 (raw gated+orth) | 0.4983 +/- 0.0277 |
| E6 concat (aligned) | 0.5013 +/- 0.0064 |
| **E6 gated+orth (aligned)** | **0.5070 +/- 0.0048** |

**Alignment on top of simple concat (E6_concat) adds essentially nothing**
over E4 (0.5013 vs 0.5038, well within noise) -- concatenation doesn't need
a shared space to begin with, so this null result is expected, not
concerning.

**Alignment on top of the gate (E6_gated_orth) is the interesting result --
and it confirms the user's hypothesis precisely.** Gate alpha, re-measured
the same way as the Day-4 diagnostic, across all 3 seeds:

| | raw E5 (Day 4) | aligned E6_gated_orth (Day 5-6) |
|---|---|---|
| alpha mean | 0.079-0.103 | **0.443** |
| alpha range | 0.013-0.457 | **0.367-0.511** |
| % alpha < 0.1 | 64-71% | **0%** |
| % alpha > 0.9 | 0% | 0% |
| macro-F1 std (seeds) | 0.0277 | **0.0048** |

Pre-aligning text/image into a shared contrastive space before the gate
sees them completely eliminated the collapse -- alpha now sits in a
genuinely balanced, per-meme-varying band instead of being stuck
image-dominant on 2/3 of validation items. The gate is doing real adaptive
mixing for the first time. Cross-seed variance also dropped ~6x (0.0048 vs
0.0277), making this the most stable strong configuration found so far.

**Being honest about the macro-F1 number itself:** the absolute gain over
E4 is small (+0.0032) and the two configs' std bands overlap at 3 seeds, so
this is not a statistically decisive win on the headline metric alone. The
real result is qualitative and mechanistic: the architecture's fusion
machinery (gate + orthogonal feature) now functions as actually designed,
rather than degenerating to a near-single-branch shortcut -- which is the
well-motivated, defensible story for why E6_gated+orth is the strongest
candidate for the locked final configuration, not just its marginally
higher mean. Full detail (confusion matrices, per-seed alpha, Stage A
training history) in `outputs/e6_results.json` and
`outputs/stage_a_training_history.json`.

Stop-loss rule was not triggered -- alignment converged to a genuine,
useful (if modest) signal within the allotted debugging effort, and
produced a positive downstream result. No need to drop it and fall back to
E4 alone.

## Day 7 — Sept 16

**Locked final configuration (user confirmed): E6 gated+orth, Stage A
aligned representations, weighted CE.** Justified by the mechanistic
finding (non-collapsed gate, ~6x lower cross-seed variance), not the raw
val macro-F1 delta over E4, which was correctly flagged as not decisive on
its own.

**Protocol discipline:** trained 3 seeds (0/1/2), model-selected the best
epoch for each purely on VALIDATION macro-F1 (test untouched during this
step). The "primary" seed for the detailed report was chosen the same way
-- by validation performance (seed 0, val macro-F1 0.5117) -- BEFORE the
test set was looked at even once, to avoid any test-set leakage via
multiple-comparison seed-picking. Test set (196 items) was then unlocked
exactly once, all 3 seeds' already-trained/already-selected models
evaluated in a single non-interactive pass, no re-runs or tuning based on
what test showed.

**Final test metrics (3-seed mean +/- std):**

| Metric | Value |
|---|---|
| macro-F1 | 0.4635 +/- 0.0173 |
| weighted-F1 | 0.4857 +/- 0.0173 |
| accuracy | 0.4932 +/- 0.0168 |
| quadratic weighted kappa | 0.3654 +/- 0.0153 |

Test macro-F1 (0.463) is ~0.04 below validation's 0.507 -- an expected,
modest generalization gap given validation was used for model selection
across 3 seeds x many epochs; not a red flag on its own.

**Detailed breakdown, primary seed 0 (chosen by val, not test):**
macro-F1 0.4753, weighted-F1 0.4953, accuracy 0.5051, QWK 0.3732.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.538 | 0.438 | 0.483 | 32 |
| Wish to be dead | 0.516 | 0.842 | 0.640 | 38 |
| Suicide ideation | 0.620 | 0.484 | 0.544 | 64 |
| Suicide planning | 0.350 | 0.269 | 0.304 | 26 |
| Suicide attempt or death | 0.395 | 0.417 | 0.405 | 36 |

Confusion matrix (rows=true, cols=pred; order = None, Wish, Ideation,
Planning, Attempt/Death):
```
[14  7  6  1  4]
[ 2 32  2  0  2]
[ 2 13 31  8 10]
[ 0  5  7  7  7]
[ 8  5  4  4 15]
```

**"Suicide planning" remains the weakest class on test too** (F1 0.304,
lowest of all 5, same pattern seen in every prior experiment from E1
onward) -- it's most often confused with "Suicide ideation" (7/26 true
planning cases predicted as ideation) and roughly evenly scattered
elsewhere, consistent with it being a genuinely hard, ordinal-adjacent
boundary rather than a data or pipeline artifact. "Wish to be dead" has
the strongest recall (0.842) but the model over-predicts it broadly (it's
the most common second-choice prediction for every other true class in the
confusion matrix), which is why its precision (0.516) lags its recall.

**Gate alpha on test:** mean 0.445, range 0.361-0.507 -- confirms the
Day-6 fix generalizes to test, not just validation; still fully
non-collapsed (0% near 0 or 1).

Full per-seed test results, all classification reports, and confusion
matrices in `outputs/final_test_results.json`. Test set has now been
touched exactly once, as required by Section 8's protocol -- do not
re-run `run_final_test_eval.py` again without a deliberate, explicit
decision to do so, since re-running it would break the "only once"
guarantee even though the script itself is technically re-runnable.

## Day 8 — Sept 17

**Error analysis (modality/context slicing, Section 8).** `run_final_test_eval.py`
only saved aggregate metrics, not per-item predictions, so reconstructed
them: retrained seed 0 deterministically (fixed seed, fixed data, no
stochastic augmentation) and asserted the reproduced test macro-F1 matched
the originally reported 0.475274 to 1e-6 before doing anything else --
confirming this is a re-read of the same one-time result, not a new test-set
peek that could change any decision. Passed exactly.

| Slice | n | macro-F1 | accuracy |
|---|---|---|---|
| Complementary (image+text both needed) | 178 | 0.497 | 0.523 |
| Text (meaning mostly in text) | 13 | 0.200 | 0.385 |
| Image (meaning mostly in image) | 5 | 0.100 | 0.200 |
| No-context-required | 171 | 0.434 | 0.497 |
| Context-required (needs outside knowledge) | 25 | 0.302 | 0.560 |

**Caveat first:** Text (n=13) and Image (n=5) modality slices are tiny --
individual misclassifications swing these numbers by 0.1-0.2 macro-F1, so
treat them as suggestive, not conclusive.

**Findings:**
- The model performs best on **Complementary** memes (macro-F1 0.497,
  matching the overall test average closely) -- expected, since this is
  878/973 (90%) of the whole dataset and what the fusion architecture was
  built and tuned for.
- **Image-only memes are near floor** (macro-F1 0.100, matching E0's
  majority-class baseline exactly) and **Text-only memes are also weak**
  (0.200) despite E1 (image-only branch alone) scoring 0.390 and E3/E3b
  (text-only branches alone) scoring ~0.38 in earlier ablations. This is a
  real, if small-sample, signal that the **fused** model does not simply
  inherit the best of both unimodal branches on their own best-suited
  subsets -- it may be relying on cross-modal agreement that these
  single-modality-driven memes don't provide by construction, or the small
  n itself is just noisy. Worth flagging as a limitation rather than
  resolving definitively given n=13/n=5.
- **Context-required memes score notably worse** (0.302 vs 0.434 macro-F1)
  than No-context-required ones, with 25 test items -- more statistically
  grounded than the modality slices. This tracks with expectation: memes
  that need outside/cultural knowledge the model never had access to should
  be harder, and this is exactly that pattern.

Per-item predictions (image_index, y_true, y_pred, softmax probs) saved to
`outputs/final_test_predictions.json` -- reused by the fuzzy-logic layer
(Section 9) so the suicide-severity side of that combination doesn't need
another test-set pass. Slice results in `outputs/error_analysis_results.json`.

## Days 9-10 — Sept 18-19

**Fuzzy-logic risk-combination layer (Section 9).** Built exactly per
Section 9.0's data-availability rule: every test meme already has OCR text
extracted and translated for the suicide-severity pipeline, so that same
translated text is reused as input to Phase-2's existing, untouched 4-class
depression classifier (loaded as `AutoModelForSequenceClassification` this
time, WITH its head -- same `BanglaBERT_fold5/checkpoint-735` checkpoint
already validated in the Day-1 sanity check, tokenizer from the `_tapt`
sibling as established).

**Note on scikit-fuzzy usage:** `fuzzy_and`/`fuzzy_or` operate on continuous
membership *functions* sampled over a shared universe (they interpolate two
arrays) -- this doesn't fit singleton softmax-probability membership
degrees, and building a continuous universe just to hold single numbers is
exactly the extra design work Section 9 said to skip ("do NOT build
separate triangular/trapezoidal membership functions"). Implemented fuzzy
AND (min) / OR (max) directly via Python's builtins on the scalar degrees,
mathematically identical to what `fuzzy_and`/`fuzzy_or` reduce to on
singleton sets. `skfuzzy` remains installed and this is the stage that
motivated installing it (Day 1), even though the actual min/max calls here
are plain Python rather than the library's array-oriented API.

**"Defuzzification" note:** the consequent (risk level) is an inherently
discrete 4-way label, not a continuous quantity -- so the discrete analogue
of centroid defuzzification used here is argmax over the aggregated
risk-level membership degrees.

Ran on all 196 test memes (not just "a handful" -- cheap given both
classifiers were already loaded/available, and gives a real distribution
to sanity-check against, not just cherry-picked illustrative rows):

| Risk level | Count | % |
|---|---|---|
| Minimal | 26 | 13.3% |
| Low | 64 | 32.7% |
| Elevated | 61 | 31.1% |
| Critical | 45 | 23.0% |

No degenerate collapse to one category -- all four risk levels are
substantively populated, a first sign the rule table is doing real
combination work rather than being dominated by one input.

**Illustrative table (3 examples per risk level, qualitative validation only
-- Section 9 is explicit there's no ground-truth "risk level" to score
against):**

| Suicide pred (true) | Depression pred | Risk membership (Minimal/Low/Elevated/**Critical**) | Crisp risk |
|---|---|---|---|
| Attempt/Death (Ideation) | Minimum | 0.08 / 0.23 / 0.24 / **0.39** | Critical |
| Attempt/Death (Attempt/Death) | Severe | 0.02 / 0.12 / 0.32 / **0.44** | Critical |
| Wish to be dead (Wish to be dead) | Severe | 0.00 / 0.03 / **0.93** / 0.01 | Elevated |
| Suicide ideation (Suicide ideation) | Severe | 0.01 / 0.01 / **0.55** / 0.37 | Elevated |
| Wish to be dead (Wish to be dead) | Minimum | 0.09 / **0.71** / 0.04 / 0.09 | Low |
| Wish to be dead (Wish to be dead) | Minimum | 0.01 / **0.88** / 0.01 / 0.02 | Low |
| None (Suicide attempt or death) | Minimum | **0.60** / 0.04 / 0.01 / 0.34 | Minimal |
| None (Suicide ideation) | Minimum | **0.54** / 0.20 / 0.09 / 0.11 | Minimal |

**One finding worth flagging:** the "None (true: Suicide attempt or death)"
row is a real suicide-severity misclassification -- the crisp prediction
missed it entirely -- yet the risk membership still shows Critical as the
clear second-highest value (0.34), not near-zero. This is the soft-membership
design earning its keep: because the fuzzy layer combines full probability
distributions rather than only the argmax class, residual uncertainty in a
wrong crisp prediction can still surface as a non-trivial risk signal rather
than being silently discarded. Worth noting explicitly in the thesis as a
reason to prefer this design over combining crisp labels alone.

**Open item, not resolved by me:** Section 9's rule table was meant to go to
the supervisor/RA for review "TODAY" (Day 1) per the original schedule. I
have no channel to actually contact them -- flagging this explicitly rather
than silently treating the draft table as final. The rule table implemented
here is exactly Section 9's draft, unmodified; if feedback comes back
changing it, only `RULES` in `fuzzy_logic_layer.py` needs updating and a
rerun (cheap -- both classifiers are already validated and cached
predictions exist for the suicide side).

Full per-item results (all 196, with full membership breakdowns and OCR
text) in `outputs/fuzzy_risk_results.json`.

## Day 11 — Sept 20

Wrote `THESIS_SECTIONS_DRAFT.md`: Methodology (dataset, architecture,
citations), Results (ablation ladder through final test + error analysis +
fuzzy-logic layer), Limitations (every scoped-out item named explicitly,
including things discovered mid-project like the gate collapse and CORAL's
underperformance, not just the originally-planned cuts), Ethical framing.

## Day 12 — Sept 21-22

Proofreading pass: cross-checked every numeric claim in the draft against
the actual saved JSON result files, not against my own prior chat messages
or memory. Found and fixed three real errors that had propagated forward
from earlier reporting:

- Final test macro-F1: reported as "0.464" in an earlier chat message and
  copied from there into `PROGRESS_LOG.md` and the draft; the actual value
  (0.46349...) rounds to **0.463**. Fixed in both files, and in the
  already-published results artifact (which had the same bug for the same
  reason -- pre-rounding to 4 decimals before formatting to 3, a
  double-rounding error. Republished with the fix.).
- "None" class precision on the final test set: reported as "0.539" in the
  original Day-7 chat report and copied forward into the log and draft;
  the actual value (0.53846...) rounds to **0.538**. Fixed in both files.
  The PNG figures (`figures/05_per_class_metrics.png`) were unaffected --
  they format directly from the source JSON at render time rather than
  from a manually retyped number, so this class of error can't occur there.

Also cross-checked every citation in the draft against Section 12 of
`PROJEC~1_UPDATED.MD` and added inline attribution for the three mechanisms
(Wang et al. for Stage A alignment, Yadav et al. for Stage B/C gated
fusion, Mazhar et al. for the reasoning-generation methodology, Park et al.
for the OCR/caption-mediated design choice) that were described in prose
but not explicitly cited. One new citation was needed and flagged as such:
CORAL (Cao et al. 2020) is not in the original brief's source list and was
reconstructed from memory rather than checked against a live source --
noted explicitly in the draft's References section as needing verification
before submission.

**Internal "done" checkpoint reached, on the internal Sept 20-22 target.**
The Sept 23-26 reserve remains untouched. Two genuinely open items outside
this session's ability to resolve: (1) the fuzzy-logic rule table has not
been reviewed by the supervisor/RA as Section 9 intended; (2) the CORAL
citation needs a live source check. Both are flagged explicitly in
`THESIS_SECTIONS_DRAFT.md` rather than silently treated as settled.

## Post-checkpoint follow-up — Sept 16 (real-world date; continuing the
## same session)

Asked to check for anything left. Two real items found:

**1. CORAL citation verified.** Web search confirmed the exact venue:
Cao, Mirjalili, Raschka, *Pattern Recognition Letters*, Vol. 140,
pp. 325-331, 2020, DOI 10.1016/j.patrec.2020.11.008 (also on arXiv as
1901.07884). Updated in `THESIS_SECTIONS_DRAFT.md`'s References section
with the verified detail, no longer flagged as unchecked.

**2. Disk-space emergency, found and partly resolved.** Free space had
dropped to 2.7GB / 931GB (100% full) -- discovered while checking
Section 1's file-safety rule ("DAPT folder must be backed up and marked
read-only," never actually done). Investigated before touching anything:
the two largest unexplained items were `tari_translation_v3 - Copy`
(17GB, an apparent duplicate left over from the Day-3 "folder moved then
moved back" incident) and `eval_augmentation` (23GB, not ours). Flagged
both to the user rather than deleting either myself. **The user cleared
space themselves** (from 2.7GB to 20GB free) -- verified afterward that
`tari_translation_v3 - Copy` specifically was what got removed, and
confirmed both `DAPT_models` and the real `tari_translation_v3/nllb_model`
are untouched and still load correctly.

**3. DAPT_models backup/read-only, resolved as far as currently possible.**
A full backup (25GB) still does not fit in 20GB free -- flagged as
infeasible rather than attempted partially. With the user's explicit
approval, marked every file under `DAPT_models/results` and
`DAPT_models/data` read-only via PowerShell (`.IsReadOnly = $true`) --
`.venv` deliberately left writable since pip/Python need it. A first
attempt via `attrib +R /S /D` silently failed to actually set the flag
(verified via a direct `attrib` check afterward showing only the Archive
flag, not ReadOnly) -- the PowerShell property-based approach worked and
was verified on both a config.json and a model.safetensors file. Confirmed
reads still succeed after the change (loaded the fold5_tapt tokenizer
successfully). This satisfies the read-only half of Section 1's rule; the
backup half remains genuinely outstanding due to persistent disk
constraints and is recorded here as an accepted, documented gap rather
than a silently-skipped requirement.
