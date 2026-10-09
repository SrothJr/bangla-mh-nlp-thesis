> ## ⚠ Read before quoting from this file
>
> **Partly out of date.** This draft predates the rigor work and the dual-encoder study, so it contains no material from either. Its section 2.6 does carry the corrected label-granularity comparison.
>
> For current numbers use `thesis_ready/VERIFIED_RESULTS.md`, which is regenerated from the raw confusion matrices on every build.

---

# Thesis sections — draft (Day 11)

Draft prose for the multimodal FigSIM extension chapter, built from
`PROJEC~1_UPDATED.MD` (the settled plan) and `PROGRESS_LOG.md` (what
actually happened, including every bug found and fixed). Numbers here are
pulled from the saved result files, not retyped from memory — cross-check
against `outputs/*.json` before submission. This is a first draft: expect
to tighten prose, not re-derive numbers.

---

## 1. Methodology

### 1.1 Objective and framing

Phase 2 of this project produced DAPT-BanglaBERT, a domain-adapted Bangla
encoder fine-tuned for 4-class depression-severity classification
(Minimum/Mild/Moderate/Severe) on a labelled dataset of 4,898 samples,
achieving pooled 5-fold cross-validation accuracy/F1 of 86.9%/86.9%. Fold 5
specifically — the fold whose checkpoint this chapter reuses, confirmed as
the best-performing fold — scored accuracy 87.95%/F1 87.58%, independently
reproduced exactly during this chapter's Day-1 checkpoint sanity check.
That system is text-only and remains complete and unmodified — it is
reused here strictly as a frozen, transferred encoder, never retrained.

This chapter extends that work to a second, related task: **fine-grained
suicide-severity classification on memes** — image + embedded text content
where meaning frequently depends on image-text interaction (e.g., a
cheerful image paired with a dark caption). The two tasks — depression
severity and suicide severity — are treated as two separate, honest outputs
sharing one text encoder, not one relabelled task. An earlier analysis
considered relabelling FigSIM content under the Bangla 4-class depression
scale and rejected it: under the original clinical criteria, roughly 85% of
FigSIM content would collapse into a single class under a naive mapping,
which would misrepresent both the data and the model's actual
discriminative power.

### 1.2 Dataset

FigSIM [1] consists of 1,049 Reddit memes
from r/SuicideMeme, English, each annotated for a 6-level suicide-severity
scale, a figurative-language flag, a modality field (Complementary /
Text / Image), and a context field (No-context-required /
Context-required). A leakage-safe cleaned subset of 973 records is used
throughout (train 582 / validation 195 / test 196), with exact-duplicate
and cross-split perceptual-hash conflicts resolved before splitting.

The two rarest classes, "Suicide attempts" (156 records) and "Suicide
death" (29 records), are merged into a single "Suicide attempt or death"
class, following the precedent set by FigSIM's own authors for classes too
small to train or evaluate reliably. The final target scale is 5 levels,
ordinal:

1. None
2. Wish to be dead
3. Suicide ideation
4. Suicide planning
5. Suicide attempt or death (merged)

### 1.3 Architecture

**Text branch.** The OCR/caption-mediated design below (extracting and
translating text rather than fusing raw image patches directly) follows
Park et al.'s finding that captions/descriptions outperform raw
vision-token input for this kind of content [5]. For each meme: (1) OCR
(EasyOCR) extracts embedded English text and a per-detection confidence
score; (2) a single vision-language model (Qwen2.5-VL-7B-Instruct, served
locally via Ollama) generates a structured reasoning explanation, following
Mazhar et al.'s reasoning-generation methodology [4] — cause-effect context, figurative/
ironic/sarcastic meaning, and implied emotional state — each claim required
to cite specific quoted evidence or explicitly report `uncertain` rather
than invent a confident reading; (3) both the OCR text and the reasoning
text are translated to Bangla via NLLB-200-3.3B, with translation
log-probability captured as a confidence signal; (4) the two Bangla texts
are concatenated via the tokenizer's own sequence-pair encoding
(`[CLS] OCR_bn [SEP] reasoning_bn [SEP]`); (5) the concatenated sequence is
encoded through DAPT-BanglaBERT's frozen Electra body (the fine-tuned
classification head is discarded; only the encoder is kept), taking the
first-token (CLS-equivalent) hidden state as a 768-dimensional text vector.

**Vision branch.** The raw image is encoded through a frozen SigLIP-so400m
vision tower, taking the pooled output as a 1152-dimensional image vector.
SigLIP was selected directly on its strong recent benchmark performance; a
comparative evaluation against CLIP was out of scope given project
timeline constraints.

**Contrastive alignment (Stage A).** Adapted from Wang et al.'s contrastive
modality alignment mechanism [3]. Two small linear projections map the
text and image vectors into a shared 256-dimensional space, trained
self-supervised via symmetric InfoNCE (CLIP-style) on the training split
only — a meme's own (text, image) pair as the positive, every other meme
in the batch as negatives. An initial two-layer-MLP projection with a
512-dimensional shared space overfit badly (validation InfoNCE loss
diverged while training loss and retrieval accuracy kept improving); a
leaner, more regularized single-linear-layer configuration (weight decay
1e-2, dropout 0.3) produced a modest but genuine, still-improving
validation retrieval signal (~13× chance at the point training was
stopped) and was used for all downstream stages.

**Gated fusion + orthogonal feature (Stage B/C).** Adapted from Yadav et
al.'s reliability-aware fusion mechanism [2]. A small MLP gate takes
the aligned text vector, aligned image vector, and two confidence signals
(OCR confidence, OCR-translation confidence) and outputs a scalar α
(sigmoid) that mixes the two modalities: `gated = α·text + (1−α)·image`.
Separately, the component of the image vector that overlaps with the text
vector is projected out (`image − proj(image, text)`), keeping only the
image's non-redundant residual — a fixed vector-math operation, not a
trained step. Both the gated combination and the orthogonal residual are
layer-normalized and concatenated with the raw (unaligned) 768-dimensional
text vector to form the final representation `z`.

**Classification (Stage D).** `z` passes through a small MLP classifier
head (256 hidden units, dropout 0.2) predicting over the 5-class merged
suicide-severity scale, trained with class-weighted (inverse-frequency)
softmax cross-entropy.

An ordinal loss variant (CORAL [6]) was also tested given the
scale's ordinal structure, and substantially underperformed weighted
cross-entropy (macro-F1 ~0.35-0.37 vs ~0.50). This is attributable to
CORAL's rank-consistency guarantee, which requires collapsing the entire
learned representation to a single shared scalar rank score — a much
smaller-capacity bottleneck than five independent class logits, which
visibly starved the rarest class (0 precision/recall for "Suicide attempt
or death" on one seed of the gated-fusion configuration). Weighted
cross-entropy was used for all subsequent experiments.

### 1.4 Ablation protocol

All embeddings (OCR text, translated text, reasoning text, text vectors,
image vectors) are cached once; only the small trained components (Stage A
projections, Stage B gate, Stage D classifier) are retrained per ablation
configuration. The fixed 582/195/196 split is used throughout; the test
split was not examined until the final architecture was locked, and was
then evaluated exactly once. Results are reported as macro-F1 (primary
metric, given class imbalance), weighted-F1, per-class precision/recall/
F1, confusion matrix, and quadratic weighted kappa (given the ordinal
scale), with 3 random seeds run wherever a trained component was involved.

---

## 2. Results

### 2.1 Ablation ladder (validation split)

| Step | Configuration | macro-F1 |
|---|---|---|
| E0 | Majority-class baseline | 0.100 |
| E1 | Image-only (SigLIP) | 0.390 |
| E3 | Text-only, DAPT-BanglaBERT, OCR only | 0.379 |
| E3b | Text-only, DAPT-BanglaBERT, OCR + AI reasoning | 0.383 |
| E4 | Text + image, simple concatenation | 0.504 ± 0.006 |
| E5 | E4 + gated fusion + orthogonal feature | 0.498 ± 0.028 |
| E6 (concat) | E4 on Stage-A-aligned representations | 0.501 ± 0.006 |
| **E6 (gated+orth)** | **E5 on Stage-A-aligned representations** | **0.507 ± 0.005** |

Adding the image branch produces the architecture's single largest gain:
E4/E5/E6 all improve on the strongest text-only configuration (E3b, 0.383)
by roughly 0.12-0.13 macro-F1, well outside seed-to-seed variance —
multimodal fusion is a real, load-bearing part of this architecture, not
an incremental refinement. AI reasoning's own marginal contribution over
OCR-only text (E3b vs E3, +0.004) is small and within noise at this sample
size (n=195), though the reasoning branch's per-class effects are larger in
both directions (ideation recall improved, "wish to be dead" and "planning"
recall both declined) — the aggregate delta understates how much the
reasoning text changes individual predictions.

### 2.2 A diagnosed and fixed failure mode: gate collapse

The gated-fusion gate's learned mixing weight α — nominally meant to vary
per meme based on how reliable each modality's signal is for that
specific item — instead collapsed to a narrow, effectively fixed,
image-dominant range when trained on raw (unaligned) text/image vectors:
mean α 0.08-0.10 across 3 seeds, never exceeding 0.46, with 64-71% of
validation items falling below α=0.1. The gate had learned a near-single-
branch shortcut rather than adaptive mixing, which explains both why E5
failed to clearly outperform simple concatenation (E4) and why its
cross-seed variance (std 0.028) was roughly 5× larger than E4's (std
0.006).

Pre-aligning the text and image representations via Stage A's contrastive
projections before the gate sees them resolved this: α shifted to a
genuinely balanced 0.37-0.51 range with 0% of items below 0.1 or above
0.9, and cross-seed macro-F1 variance dropped roughly 6× (0.028 → 0.005).
**E6 (gated+orth) was locked as the final architecture on this mechanistic
finding — a gate that measurably performs its intended function — rather
than on its raw macro-F1 margin over E4, which at 3 seeds is not on its own
statistically decisive** (0.507 ± 0.005 vs 0.504 ± 0.006).

### 2.3 Final test-set results

The locked configuration (E6 gated+orth, aligned representations, weighted
cross-entropy) was trained with 3 seeds, model-selected entirely on the
validation split, then evaluated on the test split (n=196) exactly once:

| Metric | Value (3-seed mean ± std) |
|---|---|
| macro-F1 | 0.463 ± 0.017 |
| weighted-F1 | 0.486 ± 0.017 |
| accuracy | 0.493 ± 0.017 |
| quadratic weighted kappa | 0.365 ± 0.015 |

Per-class detail (primary seed, selected by validation performance before
test was examined):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.538 | 0.438 | 0.483 | 32 |
| Wish to be dead | 0.516 | 0.842 | 0.640 | 38 |
| Suicide ideation | 0.620 | 0.484 | 0.544 | 64 |
| Suicide planning | 0.350 | 0.269 | 0.304 | 26 |
| Suicide attempt or death | 0.395 | 0.417 | 0.405 | 36 |

"Suicide planning" is the weakest class throughout every experiment in
this chapter, from the earliest unimodal ablations through the final test
result — its errors concentrate on the adjacent class "Suicide ideation"
(7 of 26 true planning cases), consistent with a genuinely hard,
ordinal-adjacent decision boundary rather than a pipeline artifact.
"Wish to be dead" shows the opposite pattern: high recall (0.842) but
comparatively low precision (0.516) — the model's most common fallback
guess when uncertain, visible as the densest off-diagonal column in the
confusion matrix.

### 2.4 Error analysis by modality and context

Slicing the same test predictions by FigSIM's own annotated modality and
context fields:

| Slice | n | macro-F1 | accuracy |
|---|---|---|---|
| Complementary | 178 | 0.497 | 0.523 |
| Text | 13 | 0.200 | 0.385 |
| Image | 5 | 0.100 | 0.200 |
| No-context-required | 171 | 0.434 | 0.497 |
| Context-required | 25 | 0.302 | 0.560 |

The Text (n=13) and Image (n=5) modality slices are too small for firm
conclusions, but the pattern is worth reporting: despite unimodal ablations
showing the image branch alone scores 0.390 and the text branch alone
scores ~0.38, the fused model performs near the majority-class floor on
memes where FigSIM's own annotation says meaning is concentrated in one
modality. This suggests the fused architecture may rely on cross-modal
agreement signal that single-modality-driven memes do not provide by
construction — a hypothesis, not a settled conclusion, given the sample
sizes. The context split rests on firmer ground (n=25 vs n=171):
context-required memes — those needing outside/cultural knowledge the
model never had access to — score meaningfully worse (0.302 vs 0.434
macro-F1), matching the expected direction of that effect.

### 2.5 Fuzzy-logic risk combination

A final risk-combination layer (Section 9 of the project plan) combines
this chapter's suicide-severity output with Phase 2's depression-severity
output for the same meme — reusing the meme's already-translated OCR text
as input to Phase 2's existing, untouched classifier, so both readings
exist for every meme without retraining either model. Both classifiers'
softmax outputs are used directly as fuzzy membership degrees (no separate
membership functions constructed); rule firing strength is the minimum of
the relevant suicide- and depression-severity memberships (fuzzy AND), and
each risk level's membership is the maximum firing strength among rules
mapping to it (fuzzy OR aggregation). The output risk level is inherently
categorical, so the discrete analogue of centroid defuzzification —
argmax over the aggregated risk-level memberships — is used to produce a
single risk label per meme.

Applied to all 196 test memes (not merely a hand-picked illustrative
subset), the resulting risk distribution is: Minimal 13.3%, Low 32.7%,
Elevated 31.1%, Critical 23.0% — populated across all four levels, with no
degenerate collapse to a single category. One qualitative finding is worth
foregrounding: because the layer combines full probability distributions
rather than only each classifier's single crisp prediction, a meme whose
suicide-severity prediction was wrong (predicted "None," true label
"Suicide attempt or death") still produced a Critical risk membership of
0.34 — the second-highest of the four levels — because the underlying
probability distribution had not collapsed entirely onto the wrong class.
This is a concrete illustration of why the layer combines soft
probabilities rather than crisp labels: residual uncertainty in a wrong
prediction remains visible downstream rather than being silently
discarded.

There is no ground-truth "risk level" to score this combination against;
validation is qualitative, per Section 9's own design.

### 2.6 Post-hoc extension: label harmonization and external meme pretraining

After the results above were fixed, a supplementary investigation asked
two further questions: whether grouping FigSIM's five ordinal severity
levels into a coarser, more reliably-classified scale is worthwhile, and
whether external, publicly available Bangla meme datasets could improve
the image encoder's representation before fine-tuning on FigSIM's own
973 images. Both were investigated under this project's established
validation-only protocol; the locked test set (Section 2.3) was not
touched by either.

**Label harmonization.** FigSIM's five severity levels were regrouped
into three coarser classes: *No expressed severity* (originally "None"),
*Suicidal thought or desire* (originally "Wish to be dead" and "Suicide
ideation" combined), and *High-acuity suicidal content* (originally
"Suicide planning" and "Suicide attempt or death" combined). A cheap
diagnostic first aggregated the existing best 5-class ensemble's own
validation predictions into this grouping with no retraining, then a
classifier was trained natively on the 3-class target for a fair test
(Figure 14):

| Scheme | Macro-F1 (validation) |
|---|---|
| 5-class (existing best) | 0.564 |
| 3-class, aggregated diagnostic (no retraining) | 0.639 |
| **3-class, trained natively** | **0.643** |
| 4-class, aggregated (merges the ideation/planning boundary specifically) | 0.587 |

The natively-trained 3-class model slightly outperforms even the
aggregation-only diagnostic, confirming the gain is a genuine property
of the coarser target rather than an artifact of re-grouping predictions
after the fact — a **+0.079** improvement over the 5-class result. Note
that the 4-class scheme, which specifically merges the "Suicide
ideation" / "Suicide planning" boundary previously identified as this
project's single most consistent confusion (Section 2.3), gains
*less* than the 3-class scheme, which does not merge that particular
boundary. This indicates the underlying confusion is distributed across
multiple adjacent ordinal boundaries, not concentrated uniquely at that
one pair — a more precise characterization than the original diagnosis.
As with any granularity change, the 3-class result should be read as a
reliability/granularity trade-off against the 5-class task, not
presented as a direct improvement on the same measurement.

**External meme pretraining.** Two publicly available Bangla meme
datasets, CMBAN and BN-HIB, were investigated as additional pretraining
signal for the image encoder, motivated by evidence (developed in a
parallel improvement round beyond this chapter's original scope) that
architecture and loss-function changes alone had reached a data-limited
ceiling on FigSIM's 973 images. To avoid disturbing DAPT-BanglaBERT's
proven domain specialization (Section 1.3; a generic substitute cost
0.073 macro-F1 in a separate ablation) and to avoid a register mismatch
between these datasets' natively-written Bangla and FigSIM's own
machine-translated Bangla pipeline, pretraining was restricted entirely
to the image encoder, using low-rank adapters (LoRA) updating under
0.3% of its parameters; DAPT-BanglaBERT was never loaded or modified.
CMBAN's full labeled release could not be obtained at the time of this
work (only an incomplete sample was available); BN-HIB (2,272 training
images, three-way hate/inflammatory/benign labels) supplied the
pretraining signal.

The resulting image encoder reached 0.750 macro-F1 on BN-HIB's own
held-out validation task — a genuine, well-above-chance signal for this
roughly-balanced three-way task (random guessing: ~0.33). Substituted
into the FigSIM 3-class classifier in place of the original image
encoder, however, validation performance did not improve (Figure 15):

| Image encoder | FigSIM 3-class macro-F1 (validation) |
|---|---|
| Original (no external pretraining) | 0.643 |
| BN-HIB-pretrained | 0.636 |
| Delta | -0.006 |

This small negative delta is consistent with the image encoder
specializing toward BN-HIB's own visual conventions — its recurring
meme templates and Bangla-script text rendering — rather than learning
features that generalize to FigSIM's substantially different visual
domain (predominantly English-script memes). The pretrained checkpoint
was accordingly not adopted for the final pipeline. This negative
result is reported deliberately: it directly tests, and does not
support, the otherwise-plausible hypothesis that adjacent, topically
similar external data would straightforwardly improve this pipeline —
a useful, honest data point for characterizing this dataset's genuine
ceiling (Section 3).

**A further architectural question — whether a fusion mechanism beyond
simple concatenation could push the 3-class result higher — was tested
systematically rather than assumed.** The gated+orthogonal-residual
architecture (Section 1.3) was re-applied to the 3-class target, first
with its original 5-class-tuned hyperparameters (0.614 macro-F1,
*below* simple concatenation's 0.643), then with a dedicated
hyperparameter sweep re-run specifically for 3-class (best: 0.633,
narrowing but not closing the gap). Cross-architecture ensembling —
the single largest lever in the original 5-class results (Section 2.1)
— was then tried for the first time on 3-class: concatenation, gated
fusion, and cross-attention (the last never previously tested on this
target), three seeds each, majority-voted across all nine models. This
reached **0.665 macro-F1 on validation**, again confirming that
combining architectures with different inductive biases outperforms any
single one, consistent with the original project's finding.

**This nine-model ensemble was then locked and evaluated on the test
split exactly once** — the third deliberate test-set touch in this
project's history (after the original Day-7 lock and the Phase 3
re-lock, both on the 5-class target). Final result:

| Metric | 3-class (this lock) | 5-class (prior best, Section 2.3) |
|---|---|---|
| macro-F1 | **0.573** | 0.463 |
| weighted-F1 | 0.613 | 0.486 |
| accuracy | 0.607 | 0.493 |
| quadratic weighted kappa | 0.377 | 0.365 |

The raw macro-F1 gap shown in the table above (0.573 vs. 0.463) is **not
a valid direct improvement claim**, and is not presented as one: the two
figures come from different-granularity tasks (3-class vs. 5-class), and
subtracting scores across a change in label granularity does not
measure a real improvement by itself. A valid check requires the older
model's own predictions to be collapsed into the identical 3-class
target on the identical test records, which was subsequently done: the
original single-model lock's saved test predictions, aggregated into
3-class groupings with no retraining, score 0.544 on that basis — making
the true, apples-to-apples delta **+0.029**, a real but modest
improvement, not +0.110.

**A second, more demanding check changes the picture further.** This
project's strongest 5-class configuration is not the original single
lock shown above but a later six-model cross-architecture ensemble
(macro-F1 0.498, from a parallel improvement round outside this
chapter's original scope). Collapsing *that* ensemble's saved
confusion matrix into 3-class terms — the same zero-retraining method —
scores **0.602**, higher than this section's 3-class model. Against
this stronger, more realistic prior-best baseline, **the 3-class model
is not an improvement; it is modestly weaker (−0.029)** once fairly
compared. The 3-class configuration is reported here as a legitimate
result for its own, differently-scoped task, not as a demonstrated
improvement over this project's best 5-class result.

A separate caveat, stated plainly rather than smoothed over: validation
showed 0.665 for the 3-class model, while test returned 0.573 — a
larger validation-to-test gap (−0.092) than this project typically
observes. This configuration was reached through a long sequence of
decisions (label-scheme diagnostic, native 3-class training, a full
hyperparameter sweep, an architecture comparison, and an ensemble-
combination search) all evaluated against the same 195-item validation
split — a degree of cumulative selection pressure on one small set that
plausibly captured some of that set's particular characteristics
alongside genuine signal. The test figure (0.573) is reported as the
governing result for this configuration precisely because of this — it
reflects a single, unbiased measurement taken after every modeling
decision was already finalized, not a number selected for being
favorable.

Notably, quadratic weighted kappa is nearly unchanged between the two
locked models (0.377 vs. 0.365), consistent with the corrected picture
above rather than contradicting it — two configurations that are
genuinely comparable in strength, not one clearly outperforming the
other, are exactly what a stable QWK alongside a misleading raw
macro-F1 gap would look like. This is a useful complementary reading:
when two metrics disagree this sharply in apparent direction, it is
worth treating as a signal to check the comparison's validity, as
turned out to be warranted here, rather than reporting the more
favorable-looking number.

---

## 3. Limitations

Several scope decisions were made deliberately, under a real deadline, and
are documented here as such rather than as oversights discovered later:

- **No CLIP-vs-SigLIP vision-encoder comparison.** SigLIP was selected
  directly on recent benchmark performance; time did not allow a
  controlled comparison.
- **No second reasoning model / model-agreement ensemble (E7).** The
  architecture uses one vision-language model (Qwen2.5-VL-7B-Instruct) for
  reasoning generation. A stretch-goal design using a second model
  (InternVL2-8B) with agreement-based confidence scoring was scoped but not
  built, as the schedule did not finish early enough to reach it.
- **No formal reasoning-quality validation study.** A structured Likert-
  scale human rating protocol (in the style of Mazhar et al.'s M3H) and a
  prompt-design ablation (simple vs. structured prompting) were both out of
  scope. In their place, an informal review of 20 stratified reasoning
  outputs was conducted, with 4 image-only cases additionally checked
  against the actual image. The reasoning module was found to be
  non-hallucinatory — no fabricated on-image content was observed — but
  showed a specific, nameable weakness on image-only memes: in 2 of 4
  visually-verified cases, it correctly identified concrete visual
  elements (a noose, hanging clothes) but built an incorrect higher-level
  narrative around them, reporting full confidence rather than flagging
  the ambiguity. A separately reviewed, comparably ambiguous case did
  correctly trigger the model's uncertainty flag, indicating the mechanism
  functions but is not reliably triggered by this class of error.
- **CORAL ordinal loss underperformed and was dropped.** Tested as
  Section 5.5 suggested, given the scale's ordinal structure; substantially
  underperformed weighted cross-entropy for a specific, verified reason
  (the single-scalar rank-score bottleneck starves rare classes in this
  small, imbalanced setting) rather than an implementation defect. Future
  work could revisit ordinal-aware losses with a higher-capacity shared
  representation.
- **Gated fusion required a diagnosed fix to function as designed.**
  Without contrastive pre-alignment, the gate collapsed to a near-fixed,
  image-dominant weighting rather than adaptive per-meme mixing. The fix
  (Stage A alignment) is now part of the locked architecture, but this
  illustrates that gated fusion is not "free" — it requires the inputs to
  already share comparable structure to behave as intended.
- **Contrastive alignment (Stage A) shows a modest, not strong, signal.**
  Validation top-1 text-to-image retrieval reached roughly 13× chance
  level at the point training was stopped (early stopping on validation
  InfoNCE loss), well below what would indicate strong, generalizable
  cross-modal alignment. It nonetheless measurably fixed the gate-collapse
  problem and slightly improved the locked configuration's stability, so it
  was kept — but the underlying alignment itself likely has headroom that
  a larger training set could realize.
- **Small-sample modality/context slices.** The Text (n=13) and Image
  (n=5) modality error-analysis slices are too small to support firm
  conclusions on their own; reported as a suggestive pattern, not a
  finding.
- **Fuzzy-logic rule table is the original draft, not expert-reviewed.**
  The rule table (Section 9) was intended for supervisor/RA sanity-check
  review before implementation; that review did not happen within this
  project's execution window. The table implemented is Section 9's
  original draft, unmodified. Updating it, if feedback arrives, requires
  changing only the `RULES` table in `fuzzy_logic_layer.py` and a rerun —
  both classifiers' predictions are already cached.
- **No CLIP/gated-fusion alternative fully explored beyond what's reported
  here** given the internal Sept 20-22 target and the Sept 23-26 reserve
  being deliberately left untouched barring something breaking.
- **The system has no defined behavior for a missing modality.** Every
  architecture in the locked multimodal ensemble (concatenation, gated
  fusion, cross-attention) was built and trained assuming both text and
  image are always present — none has a tested fallback for a text-only
  or image-only input. This was a deliberate scope decision, not an
  oversight discovered late: the original external-pretraining design
  document (Appendix, harmonization brief) specified a "modality
  dropout" training scheme precisely for this purpose (randomly omitting
  one modality during training so the model learns to cope), but it was
  never implemented, since every architecture comparison in this project
  needed a clean, single-variable baseline where both modalities are
  always available. A production deployment of this system would need
  either a modality-dropout retraining pass or a dedicated fallback path
  (e.g., the project's own earlier text-only unimodal baselines) before
  it could handle a genuinely missing modality.
- **The depression classifier's accuracy on this project's specific text
  has never been independently verified.** Phase 2's depression
  classifier (Minimum/Mild/Moderate/Severe) reports strong performance
  (reported informally as above 85%) on its own original evaluation set
  — naturalistic Bangla text from the depression dataset it was trained
  and tested on. In this system, it is applied instead to FigSIM meme
  text: OCR-extracted English, then machine-translated into Bangla — a
  meaningfully different domain and register (informal internet meme
  phrasing and translationese, not the naturalistic text the classifier
  was validated against). There is no ground-truth depression label for
  any FigSIM meme, so this domain shift's actual effect on the
  classifier's real accuracy here is unmeasured. This is not a
  hypothetical concern: this project independently confirmed domain
  shift produces real, measurable performance loss twice elsewhere under
  closely analogous conditions — swapping DAPT-BanglaBERT for a more
  generic multilingual text encoder cost 0.073 macro-F1 (Section 2,
  Phase 2.1 of the improvement round), and pretraining the image encoder
  on visually different external memes failed to transfer and was not
  adopted (Section 2.6). The depression-classifier output feeding the
  fuzzy-logic layer should accordingly be read as a plausible, not a
  verified, signal until a labeled evaluation on FigSIM-style text is
  conducted.

---

## 4. Ethical framing

All predictions in this chapter, as in Phase 2, describe the **severity of
content expressed** in a given piece of text or a meme — never a clinical
diagnosis, never a claim about a specific poster's real intent or actual
risk, and never a substitute for professional assessment. This applies
identically whether the underlying model is the Phase-2 depression
classifier, this chapter's suicide-severity classifier, or the combined
fuzzy-logic risk layer.

The risk-combination layer's design encodes one explicit ethical judgment,
stated so it can be scrutinized rather than left implicit: rule 1 assigns
Critical risk to any "Suicide attempt or death" or high-confidence
"Suicide planning" + moderate/severe depression combination *regardless* of
the accompanying depression-severity reading. Acute safety-relevant content
is not diluted by an otherwise-unrelated mood-severity score. This is a
deliberate asymmetry — a system built for a research/screening-support
context should err toward flagging acute risk signals rather than
averaging them away — and is stated here explicitly so it can be evaluated
as a design choice rather than discovered as an implicit one.

The reasoning-generation module (Qwen2.5-VL) is prompted to require quoted
evidence for its claims and to report `uncertain` rather than invent a
confident interpretation; the informal review in Section 3 found this
mechanism functions, though not with perfect reliability on ambiguous
image-only content. Given the subject matter, any deployment beyond this
research context would need a substantially more rigorous, expert-reviewed
validation process than the scope of this chapter allowed.

---

## References

[1] Chen et al., "FigSIM: A Dataset for Fine-grained Suicide Severity and
Figurative Language in Suicide Memes," Findings of ACL 2026.

[2] Yadav et al., "Towards Identifying Fine-Grained Depression Symptoms
from Memes," ACL 2023, pp. 8890-8905.

[3] Wang et al., "Multimodal Depression Estimation via Contrastive
Modality Alignment and Fusion," ACM TOMM 2026, DOI 10.1145/3778172.

[4] Mazhar et al., "Figurative-cum-Commonsense Knowledge Infusion for
Multimodal Mental Health Meme Classification" ("M3H"), WWW 2025,
DOI 10.1145/3696410.3714778.

[5] Park et al., "Iterative Large Language Model-Guided Sampling and
Expert-Annotated Benchmark Corpus for Harmful Suicide Content Detection,"
JMIR Medical Informatics 2026;14:e73725.

[6] Cao, Mirjalili, Raschka, "Rank Consistent Ordinal Regression for
Neural Networks with Application to Age Estimation," Pattern Recognition
Letters, Vol. 140, pp. 325-331, 2020, DOI 10.1016/j.patrec.2020.11.008.
Not in the original project brief's source list (Section 12) — added here
because this chapter tested CORAL. Verified via web search 2026-09-16;
volume/pages/DOI confirmed against the ScienceDirect listing (also
available as arXiv:1901.07884).
