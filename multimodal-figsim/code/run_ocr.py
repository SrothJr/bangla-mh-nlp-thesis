"""
Day 1, Step 4a: OCR on all 973 leakage-safe FigSIM images (Section 5.1 step 1 / Section 4).

Resumable per Section 4.2: writes one JSONL line per image, flushed immediately,
and on restart skips any image_index already present in the output file.
"""
import os
import sys
import json
import csv

import easyocr
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
INDEX_CSV = os.path.join(FIGSIM_ROOT, "data", "leakage_safe", "figsim_leakage_safe_index.csv")
OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "outputs", "ocr_results.jsonl")

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


def load_done_indices():
    done = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["image_index"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    records = load_records()
    total = len(records)
    print(f"Loaded {total} leakage-safe records from index.")

    done = load_done_indices()
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [r for r in records if r["image_index"] not in done]
    if not remaining:
        print("Nothing left to do. OCR already complete.")
        return

    print("Loading EasyOCR reader (GPU)...")
    reader = easyocr.Reader(["en"], gpu=True)
    print("Reader loaded. Starting OCR loop.")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    processed_this_run = 0
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for rec in remaining:
            image_index = rec["image_index"]
            image_path = os.path.join(FIGSIM_ROOT, rec["relative_path"])

            try:
                # Use PIL to normalize any format (incl. animated GIFs, which
                # EasyOCR/PIL can otherwise choke on with a frame-shape mismatch)
                # to a single RGB frame before handing a plain array to EasyOCR.
                with Image.open(image_path) as im:
                    im.seek(0)
                    img_array = np.array(im.convert("RGB"))
                detections = reader.readtext(img_array)
                texts = [t for (_bbox, t, _conf) in detections]
                confs = [float(c) for (_bbox, _t, c) in detections]
                ocr_text = " ".join(texts)
                ocr_confidence = sum(confs) / len(confs) if confs else 0.0
                num_detections = len(detections)
                error = None
            except Exception as e:  # noqa: BLE001 - log and continue, never abort the batch
                ocr_text = ""
                ocr_confidence = 0.0
                num_detections = 0
                error = f"{type(e).__name__}: {e}"

            result = {
                "image_index": image_index,
                "relative_path": rec["relative_path"],
                "ocr_text": ocr_text,
                "ocr_confidence": ocr_confidence,
                "num_detections": num_detections,
                "error": error,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            processed_this_run += 1
            done_total = len(done) + processed_this_run
            if processed_this_run % PROGRESS_EVERY == 0 or done_total == total:
                print(f"processed {done_total}/{total}", flush=True)

    print(f"OCR complete. {len(done) + processed_this_run}/{total} images done.")


if __name__ == "__main__":
    main()
