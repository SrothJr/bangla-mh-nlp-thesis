"""
Day 1, Step 4b: NLLB-200-3.3B translation of OCR text to Bangla (Section 5.1 step 3 / Section 4).

Uses the existing NLLB-200-3.3B checkpoint at tari_translation_v3/nllb_model
(read-only reference; the DAPT_models HF-cache entry for this model is an
empty stub, so this is the real source). Not a new download.

Resumable per Section 4.2: writes one JSONL line per image, flushed immediately,
and on restart skips any image_index already present in the output file.

Captures average translation log-probability per item as a confidence signal
(Section 5.1: "Capture translation log-probability as a confidence signal").
"""
import os
import json

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NLLB_PATH = r"C:\Users\user6\T2520814\tari_translation_v3\nllb_model"
OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "ocr_results.jsonl")
OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")

SRC_LANG = "eng_Latn"
TGT_LANG = "ben_Beng"
MAX_NEW_TOKENS = 200
PROGRESS_EVERY = 50


def load_ocr_records():
    records = []
    with open(OCR_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
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
    ocr_records = load_ocr_records()
    total = len(ocr_records)
    print(f"Loaded {total} OCR records.")

    done = load_done_indices()
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [r for r in ocr_records if r["image_index"] not in done]
    if not remaining:
        print("Nothing left to do. Translation already complete.")
        return

    print(f"Loading NLLB-200-3.3B from {NLLB_PATH} (read-only reference)...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_PATH, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_PATH, dtype=torch.float16).to("cuda")
    model.eval()
    tgt_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    print("Model loaded. Starting translation loop.")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    processed_this_run = 0
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for rec in remaining:
            image_index = rec["image_index"]
            ocr_text = rec["ocr_text"]

            if not ocr_text.strip():
                # Nothing to translate for this image (no OCR text detected).
                result = {
                    "image_index": image_index,
                    "relative_path": rec["relative_path"],
                    "ocr_text_bn": "",
                    "translation_confidence": None,
                    "error": None,
                }
            else:
                try:
                    enc = tokenizer(
                        ocr_text, return_tensors="pt", truncation=True, max_length=256
                    ).to("cuda")
                    with torch.no_grad():
                        out = model.generate(
                            **enc,
                            forced_bos_token_id=tgt_token_id,
                            max_new_tokens=MAX_NEW_TOKENS,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                    translated = tokenizer.batch_decode(
                        out.sequences, skip_special_tokens=True
                    )[0]
                    transition_scores = model.compute_transition_scores(
                        out.sequences, out.scores, normalize_logits=True
                    )
                    avg_logprob = float(transition_scores[0].mean().item())
                    result = {
                        "image_index": image_index,
                        "relative_path": rec["relative_path"],
                        "ocr_text_bn": translated,
                        "translation_confidence": avg_logprob,
                        "error": None,
                    }
                except Exception as e:  # noqa: BLE001 - log and continue, never abort the batch
                    result = {
                        "image_index": image_index,
                        "relative_path": rec["relative_path"],
                        "ocr_text_bn": "",
                        "translation_confidence": None,
                        "error": f"{type(e).__name__}: {e}",
                    }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            processed_this_run += 1
            done_total = len(done) + processed_this_run
            if processed_this_run % PROGRESS_EVERY == 0 or done_total == total:
                print(f"processed {done_total}/{total}", flush=True)

    print(f"Translation complete. {len(done) + processed_this_run}/{total} images done.")


if __name__ == "__main__":
    main()
