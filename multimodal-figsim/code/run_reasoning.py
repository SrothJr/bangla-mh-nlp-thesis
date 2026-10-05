"""
Day 3, Step 1: single AI-reasoning pass with qwen2.5vl:7b over all 973 images
(Section 5.1 step 2). Given image + OCR text, produces a structured,
evidence-grounded explanation of (a) cause-effect context, (b) figurative/
ironic/sarcastic meaning, (c) implied emotional state -- each claim must cite
specific evidence or explicitly say "uncertain" rather than invent a
confident answer.

Resumable per Section 4.2: one JSONL line per image, flushed immediately,
restart skips any image_index already present in the output file.

~13s/image measured in a smoke test -> ~3.5h for all 973. Long-running;
meant to be launched under a persistent Monitor for live progress.
"""
import os
import csv
import json
import time

import ollama

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
INDEX_CSV = os.path.join(FIGSIM_ROOT, "data", "leakage_safe", "figsim_leakage_safe_index.csv")
OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "ocr_results.jsonl")
OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "outputs", "reasoning_results.jsonl")

MODEL = "qwen2.5vl:7b"
PROGRESS_EVERY = 25
MAX_RETRIES = 2

PROMPT_TEMPLATE = """You are assisting a mental-health severity research study analyzing memes from r/SuicideMeme for academic classification purposes (not a clinical judgment, not identifying any real person).

You are given the meme image and OCR-extracted text from it: "{ocr_text}"

Analyze the meme and respond with ONLY a JSON object with exactly these three keys: cause_effect, figurative_meaning, emotional_state.
Each key's value must be an object with:
- "claim": a 1-2 sentence explanation
- "evidence": a short quote from the OCR text or a specific visual detail from the image that grounds the claim. If you cannot ground the claim in specific evidence, set this to the string "uncertain" instead of guessing.
- "uncertain": true or false

cause_effect: the likely cause-effect context behind the meme's message.
figurative_meaning: any figurative, ironic, or sarcastic meaning present and what it conveys (if the meme is literal, say so).
emotional_state: the emotional state implied by the meme.

Respond with ONLY the JSON object, no markdown, no other text."""

REQUIRED_KEYS = ("cause_effect", "figurative_meaning", "emotional_state")


def load_ocr_records():
    records = {}
    with open(OCR_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["image_index"]] = r
    return records


def load_index_records():
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


def parse_response(raw_text):
    parsed = json.loads(raw_text)
    for key in REQUIRED_KEYS:
        field = parsed[key]
        if not all(k in field for k in ("claim", "evidence", "uncertain")):
            raise ValueError(f"missing subkey in {key}: {field}")
    return parsed


def build_reasoning_text(parsed):
    parts = []
    labels = {
        "cause_effect": "Cause-effect",
        "figurative_meaning": "Figurative meaning",
        "emotional_state": "Emotional state",
    }
    for key, label in labels.items():
        claim = parsed[key]["claim"]
        parts.append(f"{label}: {claim}")
    return " ".join(parts)


def call_model(image_path, ocr_text):
    prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                format="json",
                options={"temperature": 0.2},
            )
            raw_text = resp["message"]["content"]
            parsed = parse_response(raw_text)
            return parsed, raw_text, None
        except Exception as e:  # noqa: BLE001 - retry, then log and move on
            last_err = f"{type(e).__name__}: {e}"
    return None, None, last_err


def main():
    index_records = load_index_records()
    ocr_records = load_ocr_records()
    total = len(index_records)
    print(f"Loaded {total} leakage-safe records, {len(ocr_records)} OCR records.")

    done = load_done_indices()
    print(f"Already done: {len(done)} / {total}. Resuming remaining {total - len(done)}.")

    remaining = [r for r in index_records if r["image_index"] not in done]
    if not remaining:
        print("Nothing left to do. Reasoning pass already complete.")
        return

    print("Starting reasoning loop.")
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    processed_this_run = 0
    t_start = time.time()
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for rec in remaining:
            image_index = rec["image_index"]
            image_path = os.path.join(FIGSIM_ROOT, rec["relative_path"])
            ocr_text = ocr_records.get(image_index, {}).get("ocr_text", "")

            parsed, raw_text, error = call_model(image_path, ocr_text)

            if parsed is not None:
                reasoning_text = build_reasoning_text(parsed)
                result = {
                    "image_index": image_index,
                    "relative_path": rec["relative_path"],
                    "cause_effect": parsed["cause_effect"],
                    "figurative_meaning": parsed["figurative_meaning"],
                    "emotional_state": parsed["emotional_state"],
                    "reasoning_text": reasoning_text,
                    "error": None,
                }
            else:
                result = {
                    "image_index": image_index,
                    "relative_path": rec["relative_path"],
                    "cause_effect": None,
                    "figurative_meaning": None,
                    "emotional_state": None,
                    "reasoning_text": "",
                    "error": error,
                }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            processed_this_run += 1
            done_total = len(done) + processed_this_run
            if processed_this_run % PROGRESS_EVERY == 0 or done_total == total:
                elapsed = time.time() - t_start
                rate = processed_this_run / elapsed if elapsed > 0 else 0
                remaining_n = total - done_total
                eta_min = (remaining_n / rate / 60) if rate > 0 else float("inf")
                print(
                    f"processed {done_total}/{total} "
                    f"({rate:.2f} img/s, ~{eta_min:.0f} min remaining)",
                    flush=True,
                )

    print(f"Reasoning pass complete. {len(done) + processed_this_run}/{total} images done.")


if __name__ == "__main__":
    main()
