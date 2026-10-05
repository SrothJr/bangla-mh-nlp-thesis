"""
Day 3, Step 3: DAPT-BanglaBERT text embeddings for E3b -- OCR text + AI
reasoning text concatenated (Section 5.1 step 4: "[OCR_bn] [SEP]
[reasoning_bn]").

Same confirmed checkpoint as E3 (PROJEC~1_UPDATED.MD Section 1.3/4):
  - model body  : results/dapt_eval/BanglaBERT_fold5/checkpoint-735 (head discarded)
  - tokenizer   : results/dapt_eval/BanglaBERT_fold5_tapt/

The "[SEP]" concatenation is implemented via the tokenizer's own sequence-pair
encoding (text=ocr_bn, text_pair=reasoning_bn), which produces exactly
[CLS] OCR_bn [SEP] reasoning_bn [SEP] with correct token_type_ids -- the
standard, correct way to do this rather than string-concatenating a literal
"[SEP]" token, which would not be tokenized/embedded the same way.

Same first-token (CLS) pooling as extract_text_embeddings_ocr.py, for a
clean, apples-to-apples E3 vs E3b comparison (same classifier will be used
on both).

Resumable per Section 4.2: one .npy file per image under
outputs/embeddings/banglabert_e3b/, restart skips files that already exist.
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
TRANSLATION_OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
TRANSLATION_REASONING_JSONL = os.path.join(
    PROJECT_ROOT, "outputs", "translation_reasoning_results.jsonl"
)
EMB_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_e3b")

MAX_LEN = 256
PROGRESS_EVERY = 50


def load_jsonl(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["image_index"]] = r
    return records


def main():
    ocr_records = load_jsonl(TRANSLATION_OCR_JSONL)
    reasoning_records = load_jsonl(TRANSLATION_REASONING_JSONL)
    common_indices = sorted(set(ocr_records) & set(reasoning_records))
    total = len(common_indices)
    print(f"Loaded {len(ocr_records)} OCR translations, {len(reasoning_records)} reasoning "
          f"translations, {total} in common.")

    os.makedirs(EMB_DIR, exist_ok=True)
    done = {int(fn[:-4]) for fn in os.listdir(EMB_DIR) if fn.endswith(".npy")}
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [i for i in common_indices if i not in done]
    if not remaining:
        print("Nothing left to do. E3b embeddings already complete.")
        return

    print(f"Loading DAPT-BanglaBERT body from {CKPT_PATH}")
    print(f"Loading tokenizer from {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModel.from_pretrained(CKPT_PATH).to("cuda")
    model.eval()
    print("Model loaded. Starting embedding extraction.")

    processed_this_run = 0
    with torch.no_grad():
        for image_index in remaining:
            ocr_bn = ocr_records[image_index].get("ocr_text_bn", "") or ""
            reasoning_bn = reasoning_records[image_index].get("reasoning_text_bn", "") or ""
            out_path = os.path.join(EMB_DIR, f"{image_index}.npy")

            enc = tokenizer(
                text=ocr_bn, text_pair=reasoning_bn,
                truncation=True, padding="max_length", max_length=MAX_LEN,
                return_tensors="pt",
            ).to("cuda")
            out = model(**enc)
            cls_vec = out.last_hidden_state[0, 0, :].float().cpu().numpy()

            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as tmp_f:
                np.save(tmp_f, cls_vec)
            os.replace(tmp_path, out_path)

            processed_this_run += 1
            done_total = len(done) + processed_this_run
            if processed_this_run % PROGRESS_EVERY == 0 or done_total == total:
                print(f"processed {done_total}/{total}", flush=True)

    print(f"E3b embeddings complete. {len(done) + processed_this_run}/{total} images done.")


if __name__ == "__main__":
    main()
