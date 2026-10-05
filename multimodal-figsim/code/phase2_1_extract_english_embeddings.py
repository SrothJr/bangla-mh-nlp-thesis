"""
Phase 2, item 2.1 (IMPROVEMENT_PLAN.md): extract text embeddings from the
ORIGINAL English OCR + reasoning text (no NLLB translation hop), using
mental-roberta-base -- a domain-adapted (mental-health) English RoBERTa,
chosen specifically because DAPT-BanglaBERT is *also* domain-adapted, so
this comparison isolates "translation hop, yes/no" rather than conflating
it with "domain-tuned encoder vs. generic encoder."

Resumable per the project's established pattern: one .npy file per image,
restart skips files that already exist.
"""
import os
import json

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "ocr_results.jsonl")
REASONING_JSONL = os.path.join(PROJECT_ROOT, "outputs", "reasoning_results.jsonl")
EMB_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "mentalroberta_english")

MODEL_NAME = "mental/mental-roberta-base"
MAX_LEN = 256
PROGRESS_EVERY = 100


def load_jsonl(path, text_key):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[r["image_index"]] = r.get(text_key, "") or ""
    return records


def main():
    ocr_texts = load_jsonl(OCR_JSONL, "ocr_text")
    reasoning_texts = load_jsonl(REASONING_JSONL, "reasoning_text")
    common_indices = sorted(set(ocr_texts) & set(reasoning_texts))
    total = len(common_indices)
    print(f"Loaded {len(ocr_texts)} OCR records, {len(reasoning_texts)} reasoning records, "
          f"{total} in common.")

    os.makedirs(EMB_DIR, exist_ok=True)
    done = {int(fn[:-4]) for fn in os.listdir(EMB_DIR) if fn.endswith(".npy")}
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [i for i in common_indices if i not in done]
    if not remaining:
        print("Nothing left to do.")
        return

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded. Starting embedding extraction.")

    processed_this_run = 0
    with torch.no_grad():
        for image_index in remaining:
            ocr_en = ocr_texts[image_index]
            reasoning_en = reasoning_texts[image_index]
            out_path = os.path.join(EMB_DIR, f"{image_index}.npy")

            enc = tokenizer(
                text=ocr_en, text_pair=reasoning_en,
                truncation=True, padding="max_length", max_length=MAX_LEN,
                return_tensors="pt",
            )
            out = model(**enc)
            cls_vec = out.last_hidden_state[0, 0, :].float().numpy()

            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as tmp_f:
                np.save(tmp_f, cls_vec)
            os.replace(tmp_path, out_path)

            processed_this_run += 1
            done_total = len(done) + processed_this_run
            if processed_this_run % PROGRESS_EVERY == 0 or done_total == total:
                print(f"processed {done_total}/{total}", flush=True)

    print(f"English embeddings complete. {len(done) + processed_this_run}/{total} images done.")


if __name__ == "__main__":
    main()
