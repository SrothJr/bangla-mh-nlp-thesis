"""
Phase 9, Step 6b (Track A2): generate reasoning text for the screened
external BN-HIB memes, so they can be embedded the same way real examples
are.

WHY THIS IS NEEDED
------------------
The locked pipeline's text feature ("e3b") is BanglaBERT's CLS vector over
the pair (OCR text, reasoning text). An external meme that had no reasoning
text would therefore have a systematically different feature -- an empty
second segment -- which the model could latch onto as "this row is
external, so it is class 0". That is precisely the shortcut the Track A2
caps exist to prevent, and it would invalidate the whole experiment.

So external memes go through the SAME reasoning pass as the real ones:
same model (qwen2.5vl:7b), same prompt, same parser, same JSON contract.
The prompt and parsing helpers are imported from run_reasoning.py rather
than copied, so the two cannot drift apart.

BN-HIB memes already carry Bangla text in their CSV, which plays the role
that OCR output plays for the real data. Only Bangla-language rows survived
the Step 6a screen, so no translation is needed on that side and no
translation signature distinguishes external rows from real ones.

RESUMABLE
---------
One JSONL line per image, flushed immediately. Restart skips anything
already present, following the pattern used by every long job here.

PRIVACY
-------
BN-HIB is CC BY-NC-SA and excluded from version control. This file writes
into outputs/ and is added to .gitignore alongside the dataset itself,
because it contains derived meme content.

OUTPUT
------
outputs/phase9_8_external_reasoning.jsonl   (gitignored)
"""
import os
import csv
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_7_external_negative_candidates.json")
CSV_DIR = os.path.join(PROJECT_ROOT, "BN-HIB", "Train Test Val CSV")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "BN-HIB", "Images")
OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "outputs", "phase9_8_external_reasoning.jsonl")

# Bounded so the VLM pass stays within a sensible runtime. The later
# confidence screen trims this down to the 300 cap.
MAX_TO_PROCESS = 400
PROGRESS_EVERY = 20

from run_reasoning import PROMPT_TEMPLATE, MODEL, MAX_RETRIES, parse_response, build_reasoning_text
import ollama


def load_texts():
    texts = {}
    for name in ("train.csv", "val.csv", "test.csv"):
        with open(os.path.join(CSV_DIR, name), "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                texts[r["image"]] = r.get("text", "") or ""
    return texts


def load_done():
    done = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["image"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def call_model(image_path, ocr_text):
    prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)
    last_err = None
    for _ in range(MAX_RETRIES + 1):
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
                format="json",
                options={"temperature": 0.2},
            )
            raw = resp["message"]["content"]
            return parse_response(raw), raw, None
        except Exception as e:  # noqa: BLE001 - retry, then log and move on
            last_err = f"{type(e).__name__}: {e}"
    return None, None, last_err


def main():
    t0 = time.time()
    print("=" * 74)
    print("PHASE 9 STEP 6b (Track A2) -- reasoning pass over external BN-HIB memes")
    print("=" * 74)

    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)["candidates"]
    texts = load_texts()
    done = load_done()
    todo = [c for c in candidates if c["image"] not in done][:MAX_TO_PROCESS - len(done)]
    print(f"Candidates: {len(candidates)}  already done: {len(done)}  "
          f"processing now: {len(todo)} (cap {MAX_TO_PROCESS})")
    if not todo:
        print("Nothing left to do.")
        return

    ok, failed = 0, 0
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for i, c in enumerate(todo, 1):
            img = c["image"]
            ocr_text = texts.get(img, "")
            parsed, raw, err = call_model(os.path.join(IMAGE_DIR, img), ocr_text)
            if parsed is None:
                failed += 1
                rec = {"image": img, "ocr_text_bn": ocr_text, "reasoning_text": "",
                       "cause_effect": None, "figurative_meaning": None,
                       "emotional_state": None, "error": err}
            else:
                ok += 1
                rec = {
                    "image": img,
                    "ocr_text_bn": ocr_text,
                    "cause_effect": parsed["cause_effect"],
                    "figurative_meaning": parsed["figurative_meaning"],
                    "emotional_state": parsed["emotional_state"],
                    "reasoning_text": build_reasoning_text(parsed),
                    "error": None,
                }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            if i % PROGRESS_EVERY == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / max(el, 1e-6)
                print(f"  {i}/{len(todo)}  ok={ok} failed={failed}  "
                      f"({rate:.2f} img/s, ETA {(len(todo) - i) / max(rate, 1e-9) / 60:.1f} min)",
                      flush=True)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. ok={ok} failed={failed}")
    print(f"Output -> {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
