"""
Phase 7, Step 4 prep: extract FigSIM image embeddings using the
BN-HIB-pretrained LoRA-adapted SigLIP checkpoint (Phase 7 Step 3,
Track A), so they can be compared against the existing frozen-SigLIP
embeddings in a real, trained 3-class FigSIM classifier.

Saved to a NEW directory (outputs/embeddings/siglip_bnhib_pretrained/)
-- the original outputs/embeddings/siglip/ is never touched, per this
project's "never overwrite existing embeddings/checkpoints" rule.

Resumable, same pattern as extract_siglip_embeddings.py: one .npy file
per image, atomic write via a temp file + os.replace.
"""
import os
import csv

import numpy as np
import torch
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
INDEX_CSV = os.path.join(FIGSIM_ROOT, "data", "leakage_safe", "figsim_leakage_safe_index.csv")
EMB_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip_bnhib_pretrained")
LORA_CKPT_PATH = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "phase7_bnhib_siglip_lora", "best")

SIGLIP_MODEL = "google/siglip-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROGRESS_EVERY = 50


def load_records():
    records = []
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["include_in_leakage_safe_dataset"] == "True":
                records.append({
                    "image_index": int(row["image_index"]),
                    "relative_path": row["relative_path"],
                })
    return records


def main():
    records = load_records()
    total = len(records)
    print(f"Loaded {total} leakage-safe FigSIM records.")

    os.makedirs(EMB_DIR, exist_ok=True)
    done = {int(fn[:-4]) for fn in os.listdir(EMB_DIR) if fn.endswith(".npy")}
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [r for r in records if r["image_index"] not in done]
    if not remaining:
        print("Nothing left to do.")
        return

    print(f"Loading base SigLIP + BN-HIB LoRA adapter from {LORA_CKPT_PATH}...")
    processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL)
    base = SiglipVisionModel.from_pretrained(SIGLIP_MODEL)
    model = PeftModel.from_pretrained(base, LORA_CKPT_PATH).to(DEVICE)
    model.eval()
    print("Model loaded. Starting embedding extraction.")

    processed = 0
    for rec in remaining:
        image_index = rec["image_index"]
        image_path = os.path.join(FIGSIM_ROOT, rec["relative_path"])
        out_path = os.path.join(EMB_DIR, f"{image_index}.npy")

        with Image.open(image_path) as im:
            im.seek(0)
            img = im.convert("RGB")

        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        pooled = out.pooler_output[0].float().cpu().numpy()  # (1152,)

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as tmp_f:
            np.save(tmp_f, pooled)
        os.replace(tmp_path, out_path)

        processed += 1
        done_total = len(done) + processed
        if processed % PROGRESS_EVERY == 0 or done_total == total:
            print(f"processed {done_total}/{total}", flush=True)

    print(f"Done. {len(done) + processed}/{total} images embedded with the BN-HIB-pretrained SigLIP.")


if __name__ == "__main__":
    main()
