"""
Phase 3, item 3.2: extract PATCH-level SigLIP features (729 x 1152, not
just the single pooled vector used everywhere else in this project) and
TOKEN-level BanglaBERT features (256 x 768, not just the CLS token), for
the cross-attention fusion experiment. Cross-attention needs to relate
specific words to specific image regions, which a single pooled vector per
modality cannot represent.

Saved as float16 to keep disk footprint down given this project's history
of tight disk space (~4GB at float32, ~2GB at float16 for both together).

Resumable per the project's established pattern.
"""
import os
import csv
import json

import numpy as np
import torch
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
INDEX_CSV = os.path.join(FIGSIM_ROOT, "data", "leakage_safe", "figsim_leakage_safe_index.csv")
PATCH_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip_patches")
TOKEN_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_tokens")

DAPT_ROOT = r"C:\Users\user6\T2520814\DAPT_models"
CKPT_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5", "checkpoint-735")
TOKENIZER_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5_tapt")
TRANSLATION_OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
TRANSLATION_REASONING_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_reasoning_results.jsonl")

SIGLIP_MODEL = "google/siglip-so400m-patch14-384"
MAX_LEN = 256
PROGRESS_EVERY = 100


def load_records():
    records = []
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["include_in_leakage_safe_dataset"] == "True":
                records.append({"image_index": int(row["image_index"]), "relative_path": row["relative_path"]})
    return records


def load_jsonl(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[r["image_index"]] = r
    return records


def extract_patches(records):
    os.makedirs(PATCH_DIR, exist_ok=True)
    done = {int(fn[:-4]) for fn in os.listdir(PATCH_DIR) if fn.endswith(".npy")}
    remaining = [r for r in records if r["image_index"] not in done]
    print(f"[patches] Already done: {len(done)}/{len(records)}. Resuming {len(remaining)}.")
    if not remaining:
        return

    processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL)
    model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to("cuda")
    model.eval()

    for n, rec in enumerate(remaining, 1):
        image_index = rec["image_index"]
        image_path = os.path.join(FIGSIM_ROOT, rec["relative_path"])
        out_path = os.path.join(PATCH_DIR, f"{image_index}.npy")

        with Image.open(image_path) as im:
            im.seek(0)
            img = im.convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            out = model(**inputs)
        patches = out.last_hidden_state[0].cpu().numpy().astype(np.float16)  # (729, 1152)

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as tmp_f:
            np.save(tmp_f, patches)
        os.replace(tmp_path, out_path)

        if n % PROGRESS_EVERY == 0 or n == len(remaining):
            print(f"[patches] processed {len(done) + n}/{len(records)}", flush=True)

    del model
    torch.cuda.empty_cache()


def extract_tokens(indices, ocr_records, reasoning_records):
    os.makedirs(TOKEN_DIR, exist_ok=True)
    existing = os.listdir(TOKEN_DIR)
    done = {
        int(fn[:-4]) for fn in existing
        if fn.endswith(".npy") and not fn.endswith("_mask.npy") and fn[:-4].isdigit()
    }
    remaining = [i for i in indices if i not in done]
    print(f"[tokens] Already done: {len(done)}/{len(indices)}. Resuming {len(remaining)}.")
    if not remaining:
        return

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModel.from_pretrained(CKPT_PATH).to("cuda")
    model.eval()

    with torch.no_grad():
        for n, image_index in enumerate(remaining, 1):
            ocr_bn = ocr_records.get(image_index, {}).get("ocr_text_bn", "") or ""
            reasoning_bn = reasoning_records.get(image_index, {}).get("reasoning_text_bn", "") or ""
            tokens_path = os.path.join(TOKEN_DIR, f"{image_index}.npy")
            mask_path = os.path.join(TOKEN_DIR, f"{image_index}_mask.npy")

            enc = tokenizer(
                text=ocr_bn, text_pair=reasoning_bn,
                truncation=True, padding="max_length", max_length=MAX_LEN,
                return_tensors="pt",
            ).to("cuda")
            out = model(**enc)
            tokens = out.last_hidden_state[0].cpu().numpy().astype(np.float16)  # (256, 768)
            attn_mask = enc["attention_mask"][0].cpu().numpy().astype(np.float16)  # (256,)

            for arr, path in ((tokens, tokens_path), (attn_mask, mask_path)):
                tmp_path = path + ".tmp"
                with open(tmp_path, "wb") as tmp_f:
                    np.save(tmp_f, arr)
                os.replace(tmp_path, path)

            if n % PROGRESS_EVERY == 0 or n == len(remaining):
                print(f"[tokens] processed {len(done) + n}/{len(indices)}", flush=True)

    del model
    torch.cuda.empty_cache()


def main():
    records = load_records()
    print(f"Loaded {len(records)} leakage-safe records.")
    extract_patches(records)

    ocr_records = load_jsonl(TRANSLATION_OCR_JSONL)
    reasoning_records = load_jsonl(TRANSLATION_REASONING_JSONL)
    indices = [r["image_index"] for r in records]
    extract_tokens(indices, ocr_records, reasoning_records)

    print("\nDone.")


if __name__ == "__main__":
    main()
