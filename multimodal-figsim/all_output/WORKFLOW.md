# System workflow — description for the methodology chapter

Main figure: `figures/20_current_verified_pipeline.png`.
Fusion architectures: `figures/09_phase3_architecture.png`.

## Plain-text flow

```
Bangla meme (image + embedded text)
        |
        +--> OCR  -->  translation (NLLB-200-3.3B)  -->  Bangla text
        |                     |
        |                     +--> VLM reasoning pass (Qwen2.5-VL-7B),
        |                          three structured fields, translated to Bangla
        |                     |
        |                     v
        |            DAPT-BanglaBERT (frozen)
        |                     |
        |          +----------+-----------+
        |          |                      |
        |          v                      v
        |   text embedding          depression classifier
        |   (OCR + reasoning)       (4-class, TEXT-ONLY, verified)
        |          |
        +--> SigLIP image encoder (frozen)
                   |
                   v
        three fusion architectures, three seeds each
        (simple concat / gated + orthogonal residual / cross-attention)
                   |
                   v
        majority vote across the ensemble  -->  suicide-content severity
                   |
                   v
        fuzzy-logic combination with depression severity
                   |
                   v
        final risk level (Minimal / Low / Elevated / Critical)
```

## Components, in the order they appear

**Encoders, both frozen.** DAPT-BanglaBERT provides the text representation and
SigLIP (`siglip-so400m-patch14-384`) the image representation. Neither is
fine-tuned in the final pipeline; LoRA fine-tuning was tried and gave only
+0.0125 macro-F1, which did not justify adopting it.

**Text feature construction.** OCR output is translated to Bangla, and a
vision-language model produces three structured analytic fields per meme
(cause and effect, figurative meaning, emotional state). These are flattened
into one string, paired with the OCR text, and encoded as a single 768-d CLS
vector. Keeping the three fields as separate attended streams was tested in
Phase 9 and performed significantly worse.

**Fusion, three architectures.**
1. *Simple concatenation* — join the two vectors, small classifier on top.
2. *Gated fusion with orthogonal residual* — a learned per-example weight
   blends aligned text and image vectors, plus a residual term preserving
   non-redundant image information. Requires a contrastive alignment stage,
   without which the gate collapses to ignoring text almost entirely, a real
   failure diagnosed and fixed earlier in the project.
3. *Cross-attention* — text tokens attend over image patches, finer-grained
   than one pooled vector per modality.

No single architecture is used alone. The locked model **is** the ensemble.

**Ensembling.** Three seeds per architecture, combined by majority vote. This
was the single largest lever in the entire project, larger than every
loss-function, fusion-mechanism and hyperparameter change combined.

**Fuzzy-logic risk layer.** Each classifier's distribution is treated as a
membership degree. Fuzzy AND is minimum, fuzzy OR is maximum, and the final
risk level is the argmax of the combined memberships across an explicit rule
table. The suicide side contributes hard vote fractions; the depression side
contributes softmax probabilities.

## Data

973 labelled Bangla memes total: 582 train, 195 validation, 196 test. The
depression classifier was trained separately on its own 4,897-record Bangla
text dataset.

## What the system cannot do

The suicide-severity branch requires **both** modalities. There is no defined
behaviour when the image or the text is missing. The depression branch has no
such limitation: its text-only path is parity-tested and verified.
