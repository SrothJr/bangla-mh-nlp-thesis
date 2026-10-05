"""
Day 3, Step 2: translate the AI-reasoning text to Bangla via NLLB-200-3.3B
(Section 5.1 step 3 -- same translation step as OCR text, applied to
reasoning_text this time).

Same NLLB source as Day 1 (tari_translation_v3/nllb_model, read-only, not
the empty DAPT_models cache stub).

Resumable per Section 4.2: one JSONL line per image, flushed immediately,
restart skips any image_index already present in the output file.
"""
import os
import json

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NLLB_PATH = r"C:\Users\user6\T2520814\tari_translation_v3\nllb_model"
REASONING_JSONL = os.path.join(PROJECT_ROOT, "outputs", "reasoning_results.jsonl")
OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_reasoning_results.jsonl")

SRC_LANG = "eng_Latn"
TGT_LANG = "ben_Beng"
MAX_NEW_TOKENS = 300  # reasoning text is longer than OCR text
PROGRESS_EVERY = 50


def load_reasoning_records():
    records = []
    with open(REASONING_JSONL, "r", encoding="utf-8") as f:
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
    reasoning_records = load_reasoning_records()
    total = len(reasoning_records)
    print(f"Loaded {total} reasoning records.")

    done = load_done_indices()
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [r for r in reasoning_records if r["image_index"] not in done]
    if not remaining:
        print("Nothing left to do. Reasoning translation already complete.")
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
            reasoning_text = rec.get("reasoning_text", "") or ""

            if not reasoning_text.strip():
                # Reasoning generation failed/produced nothing for this image
                # (see reasoning_results.jsonl's "error" field for why).
                result = {
                    "image_index": image_index,
                    "relative_path": rec["relative_path"],
                    "reasoning_text_bn": "",
                    "translation_confidence": None,
                    "error": None,
                }
            else:
                try:
                    enc = tokenizer(
                        reasoning_text, return_tensors="pt", truncation=True, max_length=384
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
                        "reasoning_text_bn": translated,
                        "translation_confidence": avg_logprob,
                        "error": None,
                    }
                except Exception as e:  # noqa: BLE001
                    result = {
                        "image_index": image_index,
                        "relative_path": rec["relative_path"],
                        "reasoning_text_bn": "",
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

    print(f"Reasoning translation complete. {len(done) + processed_this_run}/{total} images done.")


if __name__ == "__main__":
    main()
