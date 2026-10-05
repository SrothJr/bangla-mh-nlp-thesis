"""
Phase 4, item 4.2 (IMPROVEMENT_PLAN.md): prompt Qwen2.5-VL directly for
the classification task (zero-shot), rather than using it to generate
reasoning text that feeds a separately-trained classifier. A
mechanistically different approach from everything else tried in this
improvement round -- and specifically the approach FigSIM's own paper
found gave the largest gains on this exact dataset (large prompted
models, not fusion architectures).

Run on the VALIDATION split only (195 items), per this round's protocol
-- test stays locked. Resumable per the project's established pattern.
"""
import os
import csv
import json

import ollama

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
LEAKAGE_SAFE_DIR = os.path.join(FIGSIM_ROOT, "data", "leakage_safe")
INDEX_CSV = os.path.join(LEAKAGE_SAFE_DIR, "figsim_leakage_safe_index.csv")
OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "ocr_results.jsonl")
OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "outputs", "phase4_2_direct_vlm_val_results.jsonl")

MODEL = "qwen2.5vl:7b"
PROGRESS_EVERY = 25

CLASSES = ["None", "Wish to be dead", "Suicide ideation", "Suicide planning", "Suicide attempt or death"]
RAW_LABEL_TO_ID = {
    "None": 0, "Wish to be dead": 1, "Suicide ideation": 2,
    "Suicide planning": 3, "Suicide attempts": 4, "Suicide death": 4,
}

PROMPT_TEMPLATE = """You are assisting a mental-health research study analyzing memes from r/SuicideMeme for academic classification purposes (not a clinical judgment, not identifying any real person).

Given the meme image and its OCR-extracted text below, classify the meme's suicide-severity content into EXACTLY ONE of these 5 categories:

1. None - no suicide-related content
2. Wish to be dead - a passive wish to not exist or be dead, no active ideation or plan
3. Suicide ideation - active thoughts about suicide, without a specific plan or method
4. Suicide planning - a specific plan, method, or means for suicide is referenced
5. Suicide attempt or death - an actual suicide attempt or death is referenced

OCR text: "{ocr_text}"

Respond with ONLY the category number and name, e.g. "4. Suicide planning" -- nothing else."""


def load_split(name):
    with open(os.path.join(LEAKAGE_SAFE_DIR, f"{name}.txt"), "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_index():
    records = {}
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["include_in_leakage_safe_dataset"] == "True":
                idx = int(row["image_index"])
                records[idx] = {
                    "relative_path": row["relative_path"],
                    "true_label": RAW_LABEL_TO_ID[row["suicide_scale"]],
                }
    return records


def load_ocr():
    records = {}
    with open(OCR_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[r["image_index"]] = r["ocr_text"]
    return records


def parse_response(text):
    text = text.strip()
    for i, cls in enumerate(CLASSES):
        if text.startswith(f"{i + 1}.") or cls.lower() in text.lower():
            return i
    return None  # unparseable


def load_done_indices():
    done = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["image_index"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return done


def main():
    val_idx = load_split("val")
    index_records = load_index()
    ocr_records = load_ocr()
    total = len(val_idx)
    print(f"Loaded {total} validation records.")

    done = load_done_indices()
    print(f"Already done: {len(done)}/{total}. Resuming remaining {total - len(done)}.")
    remaining = [i for i in val_idx if i not in done]
    if not remaining:
        print("Nothing left to do.")
        return

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    processed = 0
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for idx in remaining:
            rec = index_records[idx]
            image_path = os.path.join(FIGSIM_ROOT, rec["relative_path"])
            ocr_text = ocr_records.get(idx, "")
            prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)

            try:
                resp = ollama.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                    options={"temperature": 0.0},
                )
                raw_text = resp["message"]["content"]
                pred = parse_response(raw_text)
                error = None if pred is not None else f"unparseable: {raw_text!r}"
            except Exception as e:  # noqa: BLE001
                raw_text = ""
                pred = None
                error = f"{type(e).__name__}: {e}"

            result = {
                "image_index": idx,
                "true_label": rec["true_label"],
                "predicted_label": pred,
                "raw_response": raw_text,
                "error": error,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            processed += 1
            done_total = len(done) + processed
            if processed % PROGRESS_EVERY == 0 or done_total == total:
                print(f"processed {done_total}/{total}", flush=True)

    print(f"Direct VLM classification complete. {len(done) + processed}/{total} done.")


if __name__ == "__main__":
    main()
