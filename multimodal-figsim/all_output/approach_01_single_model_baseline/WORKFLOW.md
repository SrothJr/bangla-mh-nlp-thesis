# Workflow — single-model baseline

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

### The model

One gated-fusion model with an orthogonal residual, the same architecture that
later became member type 1 of the ensemble in `../best_outcome/`.

```
  aligned_text (256), aligned_image (256), confidence features (2)
        |
   gate MLP --> sigmoid --> alpha
        |
   gated   = alpha * aligned_text + (1-alpha) * aligned_image
   i_orth  = aligned_image - proj(aligned_image onto aligned_text)
        |
   [ LayerNorm(gated) ; LayerNorm(i_orth) ; e3b_text(768) ]
        |
   Linear 256 -> ReLU -> Dropout 0.2 -> Linear 5
```

The two **confidence features** are OCR confidence and translation confidence,
z-scored using statistics computed on the training rows only. They let the gate
distrust the text channel when the OCR or translation was poor.

### The failure diagnosed here, and the fix

Early runs produced a gate that pushed alpha to one extreme and ignored the
text completely. The cause was that the two encoders' spaces were not
comparable, so mixing them had no meaningful geometry. Stage A contrastive
alignment fixed it.

This is a real, mechanistically explained failure mode. It belongs in the
thesis as a diagnosis, not as a tuning note.

### Rejected at this stage

**CORAL**, a cumulative-link ordinal formulation, lost to cross-entropy with
ordinal label smoothing and was dropped. A second ordinal formulation, CORN,
was tested much later and also rejected.

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
python code/train_contrastive_align.py   # Stage A, trained once
python code/train_e6.py                  # the model
python code/run_final_test_eval.py       # the single test lock
```

Results file: `outputs/final_test_results.json`.
