"""
STEP A SCRIPT (new pipeline): Translate English Reddit posts to Bangla using
Meta's NLLB-200 model.

UPDATED: Fixed potential Windows multi-threading deadlocks and added 
step-by-step debug indicators to isolate startup freezes.
UPDATED: Added --start_index argument to allow resuming/chunking from a
specific line number in the input file.
"""

import argparse
import json
import os
import sys

# FIX 1: Explicitly disable tokenizer parallelism to prevent Windows deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Ensure local directory is in path for custom imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from mask_utils import mask_text, unmask_text
except ImportError:
    print("WARNING: Could not import mask_text/unmask_text from mask_utils.py.")
    print("Please ensure mask_utils.py is in the same directory.")
    def mask_text(t): return t, {}
    def unmask_text(t, lut): return t

SRC_LANG = "eng_Latn"
TGT_LANG = "ben_Beng"


def load_input_rows(path, text_field="text", limit=None, start_index=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_index:
                continue
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            val = obj.get(text_field, "")
            if isinstance(val, str) and val.strip():
                rows.append((i, obj))
            if limit and len(rows) >= limit:
                break
    return rows


def load_done_ids(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done.add(obj["row_id"])
                except Exception:
                    continue
    return done


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL file")
    ap.add_argument("--output", default="nllb_output.jsonl")
    ap.add_argument("--model", default="./nllb_model")
    ap.add_argument("--text_field", default="text", help="JSON key holding the post text")
    ap.add_argument("--batch_size", type=int, default=16, help="Posts per GPU batch.")
    ap.add_argument("--max_length", type=int, default=1024, help="Max output tokens per post")
    ap.add_argument("--num_beams", type=int, default=3, help="Beam width for generation")
    ap.add_argument("--checkpoint_every", type=int, default=10, help="Flush to disk every N batches")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    ap.add_argument("--start_index", type=int, default=0,
                    help="Skip rows before this index (0-based line number in the input file)")
    ap.add_argument("--reset", action="store_true",
                    help="Delete the output file and start from scratch, ignoring any prior checkpoint.")
    args = ap.parse_args()

    if args.reset and os.path.exists(args.output):
        os.remove(args.output)
        print(f"[--reset] Deleted existing output file: {args.output}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected, this will be extremely slow.")

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=SRC_LANG)
    
    # Use standard modern precision keyword arguments
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    rows = load_input_rows(args.input, text_field=args.text_field, limit=args.limit, start_index=args.start_index)
    print(f"Loaded {len(rows)} total rows from {args.input}")

    done_ids = load_done_ids(args.output)
    todo = [(i, obj) for i, obj in rows if i not in done_ids]
    print(f"{len(done_ids)} already done, {len(todo)} remaining")

    out_f = open(args.output, "a", encoding="utf-8")
    batches_since_flush = 0
    batch_idx = 0

    with tqdm(total=len(todo), desc="Translating posts", unit="post") as pbar:
        for batch in chunk(todo, args.batch_size):
            batch_idx += 1
            line_nums = [i for i, _ in batch]
            objs = [obj for _, obj in batch]
            texts = [obj[args.text_field] for obj in objs]

            # DIAGNOSTIC LOGGING FOR FIRST 3 BATCHES
            if batch_idx <= 3:
                tqdm.write(f"\n--- [Batch {batch_idx}] Diagnostic Step 1: Running mask_text...")
            
            masked_texts, luts = [], []
            for idx, t in enumerate(texts):
                try:
                    m, lut = mask_text(t)
                    masked_texts.append(m)
                    luts.append(lut)
                except Exception as mask_err:
                    tqdm.write(f"Masking failed on original row {line_nums[idx]}: {mask_err}")
                    masked_texts.append(t)
                    luts.append({})

            if batch_idx <= 3:
                tqdm.write(f"--- [Batch {batch_idx}] Diagnostic Step 2: Tokenizing inputs...")
            
            try:
                with torch.no_grad():
                    inputs = tokenizer(
                        masked_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=args.max_length,
                    ).to(device)
                    
                    if batch_idx <= 3:
                        tqdm.write(f"--- [Batch {batch_idx}] Diagnostic Step 3: Pushing to GPU / model.generate()...")
                    
                    generated = model.generate(
                        **inputs,
                        forced_bos_token_id=tgt_lang_id,
                        max_length=args.max_length,
                        num_beams=args.num_beams,
                        repetition_penalty=1.1,
                    )
                    
                    if batch_idx <= 3:
                        tqdm.write(f"--- [Batch {batch_idx}] Diagnostic Step 4: Decoding outputs...")
                        
                    outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)

                for line_num, obj, raw_out, lut in zip(line_nums, objs, outputs, luts):
                    final = unmask_text(raw_out, lut)
                    out_obj = {
                        "row_id":     line_num,
                        "subreddit":  obj.get("subreddit", ""),
                        "bangla_raw": final,
                        "error":      "",
                    }
                    out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            except Exception as e:
                for line_num, obj in zip(line_nums, objs):
                    out_obj = {
                        "row_id":     line_num,
                        "subreddit":  obj.get("subreddit", ""),
                        "bangla_raw": "",
                        "error":      str(e),
                    }
                    out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                tqdm.write(f"Batch failed (rows {line_nums[0]}-{line_nums[-1]}): {e}")

            pbar.update(len(batch))

            batches_since_flush += 1
            if batches_since_flush >= args.checkpoint_every:
                out_f.flush()
                batches_since_flush = 0

    out_f.close()
    print(f"\nDone. Output saved to {args.output}")


if __name__ == "__main__":
    main()