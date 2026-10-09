# Workflow — cross-architecture ensemble (5-class)

### Data and splits

973 labelled Bangla memes: **582 train / 195 validation / 196 test**. The test
split is touched only at a deliberate, pre-announced lock. It was used three
times in the project's history and not since.

### Shared front end (identical for every approach below)

```
Bangla meme image
   |
   +--> EasyOCR (English reader, GPU) ----> raw meme text
   |         |
   |         +--> NLLB-200-3.3B --------> ocr_text_bn
   |
   +--> Qwen2.5-VL-7B (via Ollama) -----> three reasoning fields:
   |       temperature 0.2, JSON output     cause_effect
   |       2 retries on parse failure       figurative_meaning
   |                |                       emotional_state
   |                +--> NLLB-200-3.3B --> reasoning_text_bn
   |
   +--> SigLIP so400m-patch14-384 ------> 1152-d pooled, 729 x 1152 patches
```

The text feature the whole project calls **e3b** is then:

```
DAPT-BanglaBERT( text = ocr_text_bn , text_pair = reasoning_text_bn )
   max_length 256, CLS vector  ->  768-d
```

Every encoder above is **frozen**. Nothing in any approach fine-tunes them.

### Stage A: contrastive alignment (prerequisite for the gated member)

DAPT-BanglaBERT and SigLIP were pretrained independently and their vector
spaces are not comparable out of the box. Stage A learns two projections into
a shared 256-d space using a self-supervised InfoNCE objective: a meme's own
text and image are the positive pair, other memes in the batch are negatives.

This matters because without it the gated architecture **collapses** — the
learned gate drives almost all weight onto the image and ignores the text
entirely. Alignment is trained once, then frozen and reused everywhere.

### The two member architectures

```
MEMBER TYPE 1 -- gated fusion with orthogonal residual   (3 seeds)

  aligned_text (256) ----+
                         +--> gate MLP --> sigmoid --> alpha
  aligned_image (256) ---+                              |
                                                        v
  gated      = alpha * aligned_text + (1-alpha) * aligned_image
  i_orth     = aligned_image - proj(aligned_image onto aligned_text)
                                                        |
  [ LayerNorm(gated) ; LayerNorm(i_orth) ; e3b_text(768) ]
                         |
                         +--> Linear 256 -> ReLU -> Dropout -> Linear 5
```

The orthogonal residual keeps the part of the image vector that the gate's
blend would otherwise discard, so image information that is *not* redundant
with text still reaches the classifier.

```
MEMBER TYPE 2 -- cross-attention fusion                  (3 seeds)

  text tokens  (256 x 768) --> Linear --> Q (256 x 256)
  image patches(729 x 1152) --> Linear --> K, V (729 x 256)
                         |
  MultiheadAttention(4 heads, dropout 0.1), Q attends over K/V
                         |
  residual + LayerNorm, then masked mean-pool over real text tokens
                         |
  [ pooled(256) ; e3b_text(768) ] --> Linear 256 -> ReLU -> Dropout -> Linear 5
```

Cross-attention relates *individual words* to *individual image regions*,
which the single scalar gate cannot express.

### Combination

```
6 models (3 gated seeds + 3 cross-attention seeds)
   |
   +--> each predicts a 5-class argmax
   |
   +--> majority vote across all six
   |
   +--> final 5-class severity
```

Majority vote over hard predictions, not averaged probabilities. Ensembling
across *architectures* beat ensembling across seeds alone, because the two
architectures fail on different examples.

### Test protocol

All six models trained and selected on train and validation only, with early
stopping on validation macro-F1. The test split was then unlocked once, in a
single non-interactive pass, and every metric reported. No re-runs and no
tuning afterwards.

### Training settings (shared unless an approach says otherwise)

| Setting | Value |
|---|---|
| Optimizer | Adam, lr 1e-3, weight decay 1e-4 |
| Hidden width | 256 |
| Dropout | 0.2 |
| Max epochs | 300 (gated and cross-attention: 1000) |
| Early-stopping patience | 30 (gated and cross-attention: 50) |
| Batching | full batch, one gradient step per epoch |
| Loss | cross-entropy with ordinal label smoothing, tau 1.0 |
| Class weights | inverse frequency, computed on the training rows only |
| Seeds | 0, 1, 2 (later work uses 0-4) |
| Device | classifier heads on CPU, cross-attention on GPU |

**Ordinal label smoothing** replaces the one-hot target with a kernel that
decays with distance from the true class, so predicting an adjacent severity
is penalised less than predicting a distant one. It was kept after CORAL and
CORN, two stricter ordinal formulations, were both tested and rejected.

### Reproduce

```
python code/phase3_2_ensemble_architectures.py   # validation-side selection
python code/phase3_final_test_eval.py            # the single test lock
```

Results file: `outputs/phase3_final_test_results.json`.
