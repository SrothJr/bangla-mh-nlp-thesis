"""
Phase 2, item 2.4 (IMPROVEMENT_PLAN.md): extract CLIP-ViT-Large-Patch14
image embeddings for all 973 images, as a straight drop-in comparison
against the SigLIP-so400m embeddings used everywhere else in this project.
Capacity-matched choice (~428M params vs. SigLIP-so400m's ~400M), picked
over the smaller clip-vit-base-patch32 (~150M) so a result either way
isn't confounded by a capacity mismatch.

Resumable per the project's established pattern.
"""
import os
import csv

import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
INDEX_CSV = os.path.join(FIGSIM_ROOT, "data", "leakage_safe", "figsim_leakage_safe_index.csv")
EMB_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "clip_vit_large")

MODEL_NAME = "openai/clip-vit-large-patch14"
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
    done = {int(fn[:-4]) for fn in os.listdir(EMB_DIR) if fn.endswith(".npy")}
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [r for r in records if r["image_index"] not in done]
    if not remaining:
        print("Nothing left to do.")
        return

    print(f"Loading {MODEL_NAME}...")
    processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    model = CLIPVisionModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda")
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
        pooled = out.pooler_output[0].float().cpu().numpy()

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as tmp_f:
            np.save(tmp_f, pooled)
        os.replace(tmp_path, out_path)

        processed_this_run += 1
        done_total = len(done) + processed_this_run
        if processed_this_run % PROGRESS_EVERY == 0 or done_total == total:
            print(f"processed {done_total}/{total}", flush=True)

    print(f"CLIP embeddings complete. {len(done) + processed_this_run}/{total} images done.")


if __name__ == "__main__":
    main()
