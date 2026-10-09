"""
Phase 9, Step 5c (Track D): embed each Bangla reasoning field separately.

Takes the per-field Bangla text produced by
phase9_5_translate_reasoning_fields.py and runs each field through the SAME
frozen DAPT-BanglaBERT body used for every other text feature in this
project, producing one 768-d CLS vector per field per meme.

Three new embedding streams result:
    outputs/embeddings/phase9_bn_cause_effect/
    outputs/embeddings/phase9_bn_figurative_meaning/
    outputs/embeddings/phase9_bn_emotional_state/

These are additive. Nothing existing is read-modified or overwritten: the
existing banglabert_ocr and banglabert_e3b directories are untouched, and
the DAPT checkpoint is loaded read-only for inference exactly as
extract_text_embeddings_ocr.py and extract_text_embeddings_e3b.py do. The
depression classifier's head is never loaded and never trained.

Tokenization matches extract_text_embeddings_ocr.py (single text, max
length 256, CLS pooling), so the new streams sit in the same representation
space as the existing OCR stream and can be compared and attended over
without a scale mismatch.

An empty field (the reasoning pass failed, or translation returned nothing)
yields a zero vector, and the count is reported. Zero vectors are a
meaningful signal for the attention layer -- a stream with nothing in it
should receive little weight -- rather than an error to hide.

RESUMABLE: per-file atomic writes, restart skips finished files.

OUTPUT: three embedding directories, plus a short summary printed here.
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
FIELDS_JSONL = os.path.join(PROJECT_ROOT, "outputs", "phase9_5_reasoning_fields_bn.jsonl")
EMB_ROOT = os.path.join(PROJECT_ROOT, "outputs", "embeddings")

FIELDS = ["cause_effect", "figurative_meaning", "emotional_state"]
MAX_LEN = 256
HIDDEN = 768
PROGRESS_EVERY = 200


def main():
    records = []
    with open(FIELDS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} per-field translation records.")

    dirs = {k: os.path.join(EMB_ROOT, f"phase9_bn_{k}") for k in FIELDS}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    todo = [r for r in records
            if not all(os.path.exists(os.path.join(dirs[k], f"{r['image_index']}.npy")) for k in FIELDS)]
    print(f"Remaining to embed: {len(todo)}")
    if not todo:
        print("Nothing left to do.")
        return

    print(f"Loading DAPT-BanglaBERT body from {CKPT_PATH} (read-only, inference only)")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModel.from_pretrained(CKPT_PATH).to("cuda")
    model.eval()

    empty_counts = {k: 0 for k in FIELDS}
    done = 0
    with torch.no_grad():
        for rec in todo:
            idx = rec["image_index"]
            for k in FIELDS:
                out_path = os.path.join(dirs[k], f"{idx}.npy")
                if os.path.exists(out_path):
                    continue
                text = (rec.get(f"{k}_bn") or "").strip()
                if not text:
                    empty_counts[k] += 1
                    vec = np.zeros(HIDDEN, dtype=np.float32)
                else:
                    enc = tokenizer(text, truncation=True, padding="max_length",
                                    max_length=MAX_LEN, return_tensors="pt").to("cuda")
                    vec = model(**enc).last_hidden_state[0, 0, :].float().cpu().numpy()
                tmp = out_path + ".tmp"
                with open(tmp, "wb") as fh:
                    np.save(fh, vec)
                os.replace(tmp, out_path)
            done += 1
            if done % PROGRESS_EVERY == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)

    print("\nEmpty (zero-vector) fields:")
    for k in FIELDS:
        print(f"  {k:<22}{empty_counts[k]}")
    for k in FIELDS:
        n = len([x for x in os.listdir(dirs[k]) if x.endswith('.npy')])
        print(f"  {k:<22}{n} embeddings in {dirs[k]}")


if __name__ == "__main__":
    main()
