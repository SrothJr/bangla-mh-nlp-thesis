"""
Day 2, Step 2: DAPT-BanglaBERT text embeddings, OCR-only (no AI reasoning yet
-- that is Day 3 / E3b). Corresponds to Section 8's E3 configuration.

Per the corrected Day-1 sanity check (PROJEC~1_UPDATED.MD Section 1.3/4):
  - model body  : results/dapt_eval/BanglaBERT_fold5/checkpoint-735 (head discarded)
  - tokenizer   : results/dapt_eval/BanglaBERT_fold5_tapt/ (checkpoint-735 has none saved)
Fold 5 only -- confirmed best-performing fold. No other fold, no _tapt weights.

Pooling: first-token (CLS-equivalent) hidden state, matching the original
fine-tuned classifier's own summary_type="first" pooling strategy, so the
frozen representation is consistent with how this checkpoint was trained
to summarize a sequence.

Resumable per Section 4.2: one .npy file per image under
outputs/embeddings/banglabert_ocr/, restart skips files that already exist.
"""
import os
import json

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAPT_ROOT = r"C:\Users\user6\T2520814\DAPT_models"
CKPT_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5", "checkpoint-735")
TOKENIZER_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5_tapt")
TRANSLATION_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
EMB_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_ocr")

MAX_LEN = 256
PROGRESS_EVERY = 50


def load_records():
    records = []
    with open(TRANSLATION_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    records = load_records()
    total = len(records)
    print(f"Loaded {total} translation records.")

    os.makedirs(EMB_DIR, exist_ok=True)
    done = {
        int(fn[:-4]) for fn in os.listdir(EMB_DIR) if fn.endswith(".npy")
    }
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [r for r in records if r["image_index"] not in done]
    if not remaining:
        print("Nothing left to do. BanglaBERT OCR-only embeddings already complete.")
        return

    print(f"Loading DAPT-BanglaBERT body from {CKPT_PATH}")
    print(f"Loading tokenizer from {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    # AutoModel (not AutoModelForSequenceClassification) on a checkpoint whose
    # config says ElectraForSequenceClassification loads only the shared Electra
    # encoder body and discards the classification head weights -- exactly the
    # "body only, head stripped" instruction from Section 4.
    model = AutoModel.from_pretrained(CKPT_PATH).to("cuda")
    model.eval()
    print("Model loaded. Starting embedding extraction.")

    processed_this_run = 0
    with torch.no_grad():
        for rec in remaining:
            image_index = rec["image_index"]
            text = rec["ocr_text_bn"]  # empty string for the 18 no-OCR-text images
            out_path = os.path.join(EMB_DIR, f"{image_index}.npy")

            enc = tokenizer(
                text, truncation=True, padding="max_length",
                max_length=MAX_LEN, return_tensors="pt"
            ).to("cuda")
            out = model(**enc)
            cls_vec = out.last_hidden_state[0, 0, :].float().cpu().numpy()  # (768,)

            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as tmp_f:  # np.save() appends .npy to bare filenames;
                np.save(tmp_f, cls_vec)          # passing a file handle avoids that.
            os.replace(tmp_path, out_path)

            processed_this_run += 1
            done_total = len(done) + processed_this_run
            if processed_this_run % PROGRESS_EVERY == 0 or done_total == total:
                print(f"processed {done_total}/{total}", flush=True)

    print(f"BanglaBERT OCR-only embeddings complete. {len(done) + processed_this_run}/{total} images done.")


if __name__ == "__main__":
    main()
