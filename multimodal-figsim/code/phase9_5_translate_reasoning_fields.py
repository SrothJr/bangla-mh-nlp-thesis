"""
Phase 9, Step 5a: translate each structured reasoning field to Bangla
separately, so Track D can treat them as distinct streams.

WHY
---
run_reasoning.py already produced three distinct analytic fields per meme
-- cause_effect, figurative_meaning, emotional_state -- each with a claim
and supporting evidence. build_reasoning_text() in that script then
FLATTENS all three into one English string, and translate_reasoning.py
translates that single blob into one Bangla string. The structure is
generated and then immediately discarded before any model sees it.

Track D's hypothesis is that keeping the three apart, and letting the model
weight them, helps -- because memes are figurative by nature, and being
able to weigh "what this figuratively means" separately from "what it
literally says" is directly aligned with what makes this task hard.

Testing that needs each field in Bangla on its own, which does not exist
yet. This script creates it. It does NOT modify or overwrite
translation_reasoning_results.jsonl; it writes a new file.

PRESERVATION
------------
Reads reasoning_results.jsonl read-only. Uses the same NLLB-200-3.3B model,
same source and target languages, same decoding settings as
translate_reasoning.py, so the new per-field text is directly comparable to
the existing blob translation. Nothing existing is touched.

RESUMABLE
---------
One JSONL line per image, flushed immediately. Restarting skips any
image_index already present in the output, following the pattern used by
every long-running job in this project.

OUTPUT
------
outputs/phase9_5_reasoning_fields_bn.jsonl
"""
import os
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NLLB_PATH = r"C:\Users\user6\T2520814\tari_translation_v3\nllb_model"
REASONING_JSONL = os.path.join(PROJECT_ROOT, "outputs", "reasoning_results.jsonl")
OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "outputs", "phase9_5_reasoning_fields_bn.jsonl")

SRC_LANG = "eng_Latn"
TGT_LANG = "ben_Beng"
# Each field is roughly a third of the blob, so a smaller cap than
# translate_reasoning.py's 300 is appropriate and faster.
MAX_NEW_TOKENS = 160
MAX_INPUT_LEN = 256
PROGRESS_EVERY = 25

FIELDS = ["cause_effect", "figurative_meaning", "emotional_state"]


def field_text(rec, key):
    """Claim plus evidence -- the evidence carries the grounding quote, which
    is exactly the sort of literal detail a figurative-meaning stream should
    be able to weigh against."""
    part = rec.get(key)
    if not isinstance(part, dict):
        return ""
    claim = (part.get("claim") or "").strip()
    evidence = (part.get("evidence") or "").strip()
    if claim and evidence:
        return f"{claim} {evidence}"
    return claim or evidence


def load_done():
    done = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["image_index"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    t0 = time.time()
    records = []
    with open(REASONING_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} reasoning records.")

    done = load_done()
    remaining = [r for r in records if r["image_index"] not in done]
    print(f"Already done: {len(done)}. Remaining: {len(remaining)}.")
    if not remaining:
        print("Nothing left to do.")
        return

    print(f"Loading NLLB-200-3.3B from {NLLB_PATH} (read-only reference)...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_PATH, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_PATH, dtype=torch.float16).to("cuda")
    model.eval()
    tgt_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    print("Model loaded. Translating three fields per image.")

    processed = 0
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for rec in remaining:
            texts = [field_text(rec, k) for k in FIELDS]
            out = {"image_index": rec["image_index"], "relative_path": rec.get("relative_path")}

            nonempty = [(i, t) for i, t in enumerate(texts) if t.strip()]
            translated = {i: "" for i in range(len(FIELDS))}
            if nonempty:
                try:
                    batch = [t for _, t in nonempty]
                    enc = tokenizer(batch, return_tensors="pt", truncation=True,
                                    max_length=MAX_INPUT_LEN, padding=True).to("cuda")
                    with torch.no_grad():
                        gen = model.generate(
                            **enc, forced_bos_token_id=tgt_token_id,
                            max_new_tokens=MAX_NEW_TOKENS, num_beams=1,
                        )
                    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
                    for (i, _), d in zip(nonempty, decoded):
                        translated[i] = d
                    out["error"] = None
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    out["error"] = f"{type(exc).__name__}: {exc}"
            else:
                out["error"] = None

            for i, k in enumerate(FIELDS):
                out[f"{k}_bn"] = translated[i]
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_f.flush()

            processed += 1
            if processed % PROGRESS_EVERY == 0 or processed == len(remaining):
                el = time.time() - t0
                rate = processed / max(el, 1e-6)
                eta = (len(remaining) - processed) / max(rate, 1e-9)
                print(f"  {len(done) + processed}/{len(records)}  "
                      f"({rate:.2f} img/s, ETA {eta / 60:.1f} min)", flush=True)

    print(f"Done. Wrote {processed} records in {(time.time() - t0) / 60:.1f} min.")
    print(f"Output -> {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
