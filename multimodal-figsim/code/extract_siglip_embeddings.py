"""
Day 2, Step 1: SigLIP-so400m image embeddings for all 973 leakage-safe images
(Section 5.2 / Section 6: SigLIp only, no CLIP comparison).

Resumable per Section 4.2: one .npy file per image under
outputs/embeddings/siglip/, so a restart just skips files that already exist.
"""
import os
import csv

import numpy as np
import torch
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
INDEX_CSV = os.path.join(FIGSIM_ROOT, "data", "leakage_safe", "figsim_leakage_safe_index.csv")
EMB_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip")

MODEL_NAME = "google/siglip-so400m-patch14-384"
PROGRESS_EVERY = 50


def load_records():
    records = []
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["include_in_leakage_safe_dataset"] == "True":
                records.append({
                    "image_index": int(row["image_index"]),
                    "relative_path": row["relative_path"],
                })
    return records


def main():
    records = load_records()
    total = len(records)
    print(f"Loaded {total} leakage-safe records.")

    os.makedirs(EMB_DIR, exist_ok=True)
    done = {
        int(fn[:-4]) for fn in os.listdir(EMB_DIR) if fn.endswith(".npy")
    }
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [r for r in records if r["image_index"] not in done]
    if not remaining:
        print("Nothing left to do. SigLIP embeddings already complete.")
        return

    print(f"Loading {MODEL_NAME}...")
    processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipVisionModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda")
    model.eval()
    print("Model loaded. Starting embedding extraction.")

    processed_this_run = 0
    for rec in remaining:
        image_index = rec["image_index"]
        image_path = os.path.join(FIGSIM_ROOT, rec["relative_path"])
        out_path = os.path.join(EMB_DIR, f"{image_index}.npy")

        with Image.open(image_path) as im:
            im.seek(0)
            img = im.convert("RGB")

        inputs = processor(images=img, return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            out = model(**inputs)
        pooled = out.pooler_output[0].float().cpu().numpy()  # (1152,)

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as tmp_f:  # np.save() appends .npy to bare filenames;
            np.save(tmp_f, pooled)           # passing a file handle avoids that.
        os.replace(tmp_path, out_path)  # atomic — never leaves a half-written file

        processed_this_run += 1
        done_total = len(done) + processed_this_run
        if processed_this_run % PROGRESS_EVERY == 0 or done_total == total:
            print(f"processed {done_total}/{total}", flush=True)

    print(f"SigLIP embeddings complete. {len(done) + processed_this_run}/{total} images done.")


if __name__ == "__main__":
    main()
