"""
STEP B SCRIPT (new pipeline, v2): Polish pass over NLLB's raw translations.

NLLB gives accurate, coherent Bangla but in a formal/textbook register.
This script runs a narrow, constrained LLM rewrite task on top of that —
NOT a translation task — to loosen it into casual spoken Reddit-Bangla and
reintroduce natural code-switching. This is a much easier, more constrained
job than "translate + control tone + control code-switching" all at once,
which is why the earlier all-in-one LLM pipeline struggled.

WHY THIS VERSION IS DIFFERENT FROM THE FIRST DRAFT:
The original version of this script called a local Ollama/vLLM HTTP server,
one request per row. That's slow (140K individual API calls) and vLLM isn't
available at all on native Windows (Linux/WSL2 only). This version instead
loads the model directly via `transformers` and runs TRUE BATCHED generation
(many rows per .generate() call, all processed together on the GPU) — the
same approach nllb_translate.py already uses successfully. No server, no
HTTP, no Ollama, no vLLM, no WSL2. Runs natively on Windows.

Default model is Qwen2.5-7B-Instruct rather than 32B — this rewrite task is
narrow (adjust register, don't re-translate), so a smaller model is enough,
and it means much higher throughput / batch size headroom on a single 5090.
If quality isn't good enough on real output, bump --model to
Qwen/Qwen2.5-14B-Instruct (still fits fine, just slower).

INPUT / OUTPUT FORMAT: JSONL in, JSONL out. Takes nllb_output.jsonl (from
nllb_translate.py) as input. Every output line = the full input object,
unchanged, PLUS:
    bangla_polished   <- the casual-register rewrite
    polish_error      <- empty string if OK, error message if this row failed

The "row_id" field from Stage A passes through untouched, so every row
remains traceable back to its original line number in
merged_mental_health_dapt.jsonl.

SETUP:
    pip install transformers torch accelerate

USAGE:
    python3 casual_polish.py --input nllb_output.jsonl --output polished_full.jsonl

    # quick test on a small slice first (recommended):
    python3 casual_polish.py --input nllb_test.jsonl --output polished_test.jsonl

    # if 7B quality isn't good enough, try 14B (slower, more VRAM):
    python3 casual_polish.py --input nllb_output.jsonl --output polished_full.jsonl --model Qwen/Qwen2.5-14B-Instruct --batch_size 12

    # if you hit CUDA OOM, lower --batch_size, or add --load_in_4bit for headroom:
    python3 casual_polish.py --input nllb_output.jsonl --output polished_full.jsonl --load_in_4bit
"""

import argparse
import json
import os
import time

import torch
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer

POLISH_SYSTEM_PROMPT = """You will be given a Bangla text that is an ACCURATE but FORMAL/TEXTBOOK-register translation of an English Reddit post about mental health. Your ONLY job is to rewrite it into casual spoken Bangla (কথ্য ভাষা) — the way a young Bangladeshi Redditor would actually type it.

STRICT RULES:
1. Do NOT change the meaning. Do NOT re-translate from scratch. Only adjust register/tone/word choice.
2. Any token that looks like ⟦T1⟧, ⟦T2⟧, etc. is a PROTECTED PLACEHOLDER. Copy it EXACTLY as-is, do not translate/remove/alter it.
3. Convert formal/literary verb forms (e.g. করিতেছি, হইয়াছে, করেছিলাম-style stiffness) into natural spoken forms (করছি, হচ্ছে, লাগছে).
4. It's normal and expected for a young Bangla Redditor to mix in some English words/phrases naturally (code-switching) — feel free to swap a stiff Bangla phrase for a natural English one if that's how it would actually be said (e.g. "মন খারাপ" can become "mood off" if that fits better), but the result should still be MOSTLY Bangla, not majority English.
5. Preserve sarcasm, dark humor, and emotional tone exactly as present in the input — if the input reads sarcastic, keep it sarcastic; don't soften or sanitize.
6. Preserve line breaks / paragraph structure.
7. Output ONLY the rewritten Bangla text. No notes, no explanations, no preamble.
8. IMPORTANT: Do NOT use any language other than Bangla and English (no Hindi, no other languages) in your output.
"""


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
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


def build_batch_prompts(tokenizer, bangla_texts):
    """Builds one chat-formatted prompt string per row, ready for tokenization."""
    prompts = []
    for text in bangla_texts:
        messages = [
            {"role": "system", "content": POLISH_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="nllb_output.jsonl from nllb_translate.py")
    ap.add_argument("--output", default="polished_full.jsonl")
    ap.add_argument("--model", default="./qwen_model")
    ap.add_argument("--batch_size", type=int, default=24, help="Rows per GPU batch. Lower if you hit OOM.")
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--checkpoint_every", type=int, default=10, help="Flush to disk every N batches")
    ap.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization for more VRAM headroom / larger batches")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected, this will be extremely slow.")

    print(f"Loading {args.model} ...")
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched causal-LM generation

    load_kwargs = dict(dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen2ForCausalLM.from_pretrained(args.model, **load_kwargs).to(
        device if not args.load_in_4bit else None
    )
    model.eval()

    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    done_ids = load_done_ids(args.output)
    todo = [r for r in rows if r.get("row_id") not in done_ids]
    print(f"{len(done_ids)} already done, {len(todo)} remaining")

    out_f = open(args.output, "a", encoding="utf-8")

    start_time = time.time()
    processed = 0
    batches_since_flush = 0

    for batch in chunk(todo, args.batch_size):
        bangla_texts = [r.get("bangla_raw", "") for r in batch]

        # Rows with empty bangla_raw: skip generation, write straight through
        gen_indices = [i for i, t in enumerate(bangla_texts) if t and t.strip()]
        try:
            if gen_indices:
                gen_texts = [bangla_texts[i] for i in gen_indices]
                prompts = build_batch_prompts(tokenizer, gen_texts)
                inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048,
                ).to(model.device)

                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.4,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=5,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Slice off the input prompt tokens, decode only the new output
                input_len = inputs["input_ids"].shape[1]
                decoded = tokenizer.batch_decode(
                    generated[:, input_len:], skip_special_tokens=True
                )
            else:
                decoded = []

            gen_iter = iter(decoded)
            for i, r in enumerate(batch):
                out_obj = dict(r)
                if i in gen_indices:
                    out_obj["bangla_polished"] = next(gen_iter).strip()
                    out_obj["polish_error"] = ""
                else:
                    out_obj["bangla_polished"] = ""
                    out_obj["polish_error"] = "EMPTY_INPUT"
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                processed += 1

        except Exception as e:
            for r in batch:
                out_obj = dict(r)
                out_obj["bangla_polished"] = ""
                out_obj["polish_error"] = str(e)
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                processed += 1
            print(f"Batch failed: {e}")

        batches_since_flush += 1
        if batches_since_flush >= args.checkpoint_every:
            out_f.flush()
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = len(todo) - processed
            eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
            print(f"[{processed}/{len(todo)}] rate={rate:.2f}/s ETA={eta_min:.1f}min")
            batches_since_flush = 0

    out_f.close()
    print(f"\nDone. Output saved to {args.output}")
    print("Each line still carries 'row_id' tracing back to the original input file.")
    print("Next step: run qa_check.py on this file (--bangla_field bangla_polished) for QA flags.")


if __name__ == "__main__":
    main()