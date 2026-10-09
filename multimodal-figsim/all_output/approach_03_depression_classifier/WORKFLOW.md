# Workflow — depression classifier (text-only)

### This model is not trained here

It was produced before the multimodal work began and has been **frozen ever
since**. No phase of this project retrained, fine-tuned or modified it. The
workflow below is therefore an inference pipeline, not a training one.

```
Checkpoint : DAPT_models/results/dapt_eval/BanglaBERT_fold5/checkpoint-735
Tokenizer  : DAPT_models/results/dapt_eval/BanglaBERT_fold5_tapt
Head       : 4-class classification head, id2label
             0 Minimum / 1 Mild / 2 Moderate / 3 Severe
```

### Its own training data

4897 Bangla mental-health text records, labelled 1 to 4 in the source
file, mapped to class index by subtracting one. That mapping was verified
against the checkpoint's own `config.json` rather than assumed.

The checkpoint saw roughly four fifths of this dataset during training,
confirmed by back-calculating from its trainer state: 735 steps at batch size
16 over 3 epochs is about 3,920 examples per epoch, which matches a 4/5 fold of
4897 almost exactly. This is why the accuracy figure in the README must
be quoted as a sanity check rather than a held-out estimate.

### Inference pipeline

```
Bangla text
   |
   +--> tokenizer (max_length 256, truncation, padding)
   |
   +--> DAPT-BanglaBERT + 4-class head
   |
   +--> softmax --> [p_minimum, p_mild, p_moderate, p_severe]
   |
   +--> argmax --> label
```

**Text only. No image is required anywhere in this path.** That is why this
half of the system has no missing-modality limitation while the multimodal half
does.

### How it is verified

A wrapper exposes `predict_depression(text_bn)` and calls the exact original
inference code rather than reimplementing it. It was checked against 196
previously saved outputs from an earlier run:

- **100.00% label agreement**
- **0.00 maximum absolute probability difference**

An exact match, not an approximation. That is the claim to make in the thesis.

### Its two roles in the wider system

1. Supplies the depression axis to the fuzzy-logic layer
   (`../approach_04_fuzzy_logic_risk_layer/`).
2. Its output **distribution** was later found to work as an input feature for
   the suicide-severity model (`../approach_05_rigor_and_diagnostics/`).

### Reproduce

```
python code/phase8_2_depression_parity_wrapper.py        # the parity test
python code/phase8_3_depression_dataset_verification.py  # dataset check
```

Results files: `outputs/phase8_2_depression_parity_report.json`,
`outputs/phase8_3_depression_dataset_verification_results.json`.

Note: the raw dataset is excluded from version control because it contains real
sensitive mental-health text. Only aggregate metrics are stored.
