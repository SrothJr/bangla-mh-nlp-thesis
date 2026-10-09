"""
Phase 9, Step 6c (Track A2): extract features for the screened external
BN-HIB memes, using the SAME frozen encoders as the real data.

Two feature streams are produced per external meme:
  phase9_external_siglip/   SigLIP pooled image vector      (1152-d)
  phase9_external_ocr/      DAPT-BanglaBERT CLS over its
                            Bangla text                     (768-d)

Same model names, same preprocessing, same pooling and same max length as
extract_siglip_embeddings.py and extract_text_embeddings_ocr.py. If the
external memes were embedded any differently, the resulting distribution
shift would be indistinguishable from the effect being measured, and the
whole A/B would be meaningless.

WHY OCR-ONLY TEXT
-----------------
The locked pipeline's richer text feature ("e3b") pairs OCR text with
VLM-generated reasoning text. Generating that reasoning for external memes
turned out to be unreliable on this machine (the local vision model
degenerated into repeated tokens). Rather than feed external rows an empty
reasoning segment -- which would hand the classifier an obvious "this row
is external, so it is class 0" shortcut and invalidate the experiment --
Track A2 is run on an OCR-only text feature for BOTH real and external
rows. That keeps the comparison internally consistent.

The cost is stated plainly: the A/B is measured on a weaker text feature
than the locked pipeline uses, so a positive result would show that
external negatives help in principle, not that they would transfer
unchanged to the e3b pipeline. That limitation is recorded rather than
papered over.

PRIVACY
-------
BN-HIB is CC BY-NC-SA and excluded from version control, as are these
derived embedding directories (see .gitignore). No meme text is written to
any committed file.

RESUMABLE: atomic per-file writes; restart skips finished files.
"""
import os
import csv
import json

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoTokenizer, AutoModel, SiglipImageProcessor, SiglipVisionModel,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAPT_ROOT = r"C:\Users\user6\T2520814\DAPT_models"
CKPT_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5", "checkpoint-735")
TOKENIZER_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5_tapt")

CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_7_external_negative_candidates.json")
CSV_DIR = os.path.join(PROJECT_ROOT, "BN-HIB", "Train Test Val CSV")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "BN-HIB", "Images")
EMB_ROOT = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
SIGLIP_OUT = os.path.join(EMB_ROOT, "phase9_external_siglip")
OCR_OUT = os.path.join(EMB_ROOT, "phase9_external_ocr")
MANIFEST = os.path.join(PROJECT_ROOT, "outputs", "phase9_10_external_feature_manifest.json")

SIGLIP_MODEL = "google/siglip-so400m-patch14-384"
MAX_LEN = 256
PROGRESS_EVERY = 50


def stem(image_name):
    return os.path.splitext(image_name)[0]


def load_texts():
    texts = {}
    for name in ("train.csv", "val.csv", "test.csv"):
        with open(os.path.join(CSV_DIR, name), "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                texts[r["image"]] = r.get("text", "") or ""
    return texts


def save_atomic(path, vec):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, vec)
    os.replace(tmp, path)


def main():
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)["candidates"]
    texts = load_texts()
    os.makedirs(SIGLIP_OUT, exist_ok=True)
    os.makedirs(OCR_OUT, exist_ok=True)
    print(f"Candidates to embed: {len(candidates)}")

    # ------------------------------------------------------------ images ----
    todo_img = [c for c in candidates
                if not os.path.exists(os.path.join(SIGLIP_OUT, f"{stem(c['image'])}.npy"))]
    print(f"SigLIP remaining: {len(todo_img)}")
    if todo_img:
        print(f"Loading {SIGLIP_MODEL}...")
        processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL)
        vision = SiglipVisionModel.from_pretrained(SIGLIP_MODEL, dtype=torch.float16).to("cuda")
        vision.eval()
        for i, c in enumerate(todo_img, 1):
            path = os.path.join(IMAGE_DIR, c["image"])
            with Image.open(path) as im:
                im.seek(0)
                img = im.convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            with torch.no_grad():
                pooled = vision(**inputs).pooler_output[0].float().cpu().numpy()
            save_atomic(os.path.join(SIGLIP_OUT, f"{stem(c['image'])}.npy"), pooled)
            if i % PROGRESS_EVERY == 0 or i == len(todo_img):
                print(f"  siglip {i}/{len(todo_img)}", flush=True)
        del vision
        torch.cuda.empty_cache()

    # -------------------------------------------------------------- text ----
    todo_txt = [c for c in candidates
                if not os.path.exists(os.path.join(OCR_OUT, f"{stem(c['image'])}.npy"))]
    print(f"BanglaBERT remaining: {len(todo_txt)}")
    if todo_txt:
        print(f"Loading DAPT-BanglaBERT body from {CKPT_PATH} (read-only, inference only)")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        text_model = AutoModel.from_pretrained(CKPT_PATH).to("cuda")
        text_model.eval()
        with torch.no_grad():
            for i, c in enumerate(todo_txt, 1):
                t = (texts.get(c["image"]) or "").strip()
                if not t:
                    vec = np.zeros(768, dtype=np.float32)
                else:
                    enc = tokenizer(t, truncation=True, padding="max_length",
                                    max_length=MAX_LEN, return_tensors="pt").to("cuda")
                    vec = text_model(**enc).last_hidden_state[0, 0, :].float().cpu().numpy()
                save_atomic(os.path.join(OCR_OUT, f"{stem(c['image'])}.npy"), vec)
                if i % PROGRESS_EVERY == 0 or i == len(todo_txt):
                    print(f"  text {i}/{len(todo_txt)}", flush=True)
        del text_model
        torch.cuda.empty_cache()

    stems = [stem(c["image"]) for c in candidates
             if os.path.exists(os.path.join(SIGLIP_OUT, f"{stem(c['image'])}.npy"))
             and os.path.exists(os.path.join(OCR_OUT, f"{stem(c['image'])}.npy"))]
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Stems of external BN-HIB memes with both feature streams extracted. "
                     "Filenames only -- no meme text, since BN-HIB is CC BY-NC-SA."),
            "siglip_dir": os.path.relpath(SIGLIP_OUT, PROJECT_ROOT),
            "ocr_dir": os.path.relpath(OCR_OUT, PROJECT_ROOT),
            "n_complete": len(stems),
            "stems": stems,
        }, f, indent=2)
    print(f"\nComplete pairs: {len(stems)}")
    print(f"Manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
