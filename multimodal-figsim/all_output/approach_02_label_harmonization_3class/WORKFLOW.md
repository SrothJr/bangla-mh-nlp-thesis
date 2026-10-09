# Workflow — 3-class harmonized model

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

### The label transformation

```
  0 None                    -->  0 No expressed severity
  1 Wish to be dead      ---+
  2 Suicide ideation     ---+-->  1 Suicidal thought or desire
  3 Suicide planning     ---+
  4 Attempt or death     ---+-->  2 High acuity suicidal content
```

Applied to the labels **before** training, so the models below are trained
natively on three classes rather than trained on five and collapsed afterwards.
That distinction is the whole point of this approach, and it is what makes the
comparison in the README non-trivial.

### The ensemble

Nine models rather than six, because a third architecture was added:

```
  simple concat        (3 seeds)   [ e3b_text(768) ; siglip(1152) ]
                                     -> Linear 256 -> ReLU -> Dropout -> Linear 3

  gated + orthogonal   (3 seeds)   as in ../best_outcome/, but hidden width 512
                                   (tuned specifically for the 3-class task)

  cross-attention      (3 seeds)   as in ../best_outcome/, 3 outputs
        |
        +--> majority vote across all nine --> 3-class severity
```

Simple concatenation was added because on the coarser task it turned out to be
the **strongest single architecture**, which it had not been on the 5-class
task. Hyperparameters for the gated member were re-swept for three classes and
hidden width 512 won.

### What was tried alongside and rejected

**External meme pretraining.** A SigLIP vision head was pretrained on BN-HIB,
an external Bangla hate-speech meme dataset, then transferred. It did not
improve the severity task and was dropped. See the transfer-test figure.

### Test protocol

Same discipline as every other lock. All nine models trained and selected on
train and validation only, test unlocked exactly once.

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
python code/phase7_8_harmonized_hyperparam_sweep.py
python code/phase7_9_harmonized_ensemble.py
python code/phase7_10_final_test_eval_3class.py    # the single test lock
```

Results file: `outputs/phase7_final_test_results_3class.json`.
