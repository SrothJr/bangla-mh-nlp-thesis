"""
Bengali Mental-Health Dataset Augmentation Pipeline  (v2 - Robust)
===================================================================
Goal      : Expand dataset to a configurable ratio using a local LLM
Model     : aya-expanse:32b (recommended) or qwen2.5:32b via Ollama
Strategy  : Few-shot prompting with label-verification, deduplication,
            per-label length filters, and infinite-loop protection.

Usage
-----
  python augment_dataset.py                          # full run, all labels
  python augment_dataset.py --label 4               # single label
  python augment_dataset.py --label 4 --count 500   # exactly 500 synthetic texts
  python augment_dataset.py --label 4 --test        # dry-run (3 samples only)
  python augment_dataset.py --ratio 3               # 1:3 instead of default 1:5
  python augment_dataset.py --model aya-expanse:32b  # switch model
  python augment_dataset.py --batch-size 5          # smaller batches
  python augment_dataset.py --temperature 0.9       # higher diversity
  python augment_dataset.py --no-verify             # skip label verification pass
  python augment_dataset.py --shots 8               # more few-shot examples
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import ollama
import openpyxl
import pandas as pd

# ---------------------------------------------------------------------------
# Default config  (all overridable via CLI)
# ---------------------------------------------------------------------------
MODEL          = "qwen2.5:32b"   # default; switch to aya-expanse:32b for better Bengali
INPUT_FILE     = "train.xlsx"
OUTPUT_FILE    = "train_augmented.xlsx"
CHECKPOINT_DIR = Path("checkpoints")
BATCH_SIZE     = 10              # synthetic texts requested per LLM call
FEW_SHOT_N     = 5              # real examples shown per call (use 8 for labels 3-4)
TEMPERATURE    = 0.85
AUGMENT_RATIO  = 5              # 1 original : 5 total  =>  4x synthetic per label
SEED           = 42
MAX_EMPTY_RETRIES = 10          # give up after N consecutive empty batches
SIMILARITY_THRESHOLD = 0.85    # deduplicate if >85% similar to any original

# ---------------------------------------------------------------------------
# Length boundaries — derived from full statistical analysis of original dataset
# ---------------------------------------------------------------------------
# Analysis (character counts):
#
#  Label |  n   | Min | p5  | p10 | p25 | p50 | p75 | p90 | p95 |  Max  | Mean
#  ------+------+-----+-----+-----+-----+-----+-----+-----+-----+-------+-----
#    1   | 2097 |  16 |  31 |  36 |  50 |  93 | 202 | 409 | 588 |  2240 | 171
#    2   | 1540 |  16 |  37 |  40 |  53 |  90 | 186 | 368 | 561 |  2317 | 166
#    3   |  707 |  38 |  54 |  73 | 120 | 219 | 396 | 692 | 874 |  2185 | 316
#    4   |  553 |  36 |  60 |  82 | 139 | 282 | 537 | 906 |1192 |  2365 | 401
#
# Length bucket distribution (% of texts):
#  Label |  <50  | 50-100 | 100-200 | 200-400 | 400-600 | 600-800 | 800+
#  ------+-------+--------+---------+---------+---------+---------+-----
#    1   | 24.4% |  28.1% |   21.8% |   15.5% |    5.4% |    2.4% | 2.3%
#    2   | 21.4% |  32.5% |   24.1% |   13.4% |    4.4% |    1.7% | 2.5%
#    3   |  3.1% |  15.4% |   27.6% |   29.1% |   12.2% |    5.2% | 7.4%
#    4   |  2.5% |  12.5% |   21.5% |   26.9% |   15.2% |    8.9% |12.5%
#
# Strategy:
#   MIN_LEN = p5  (excludes bottom 5% extreme outliers, allows very short posts)
#   MAX_LEN = p90 (covers 85% of real data; excludes only the long-tail 10%)
#   This ensures synthetic texts fall within the realistic range of the originals
#   and prevents the LLM from defaulting to verbose multi-paragraph outputs.
#
#   Labels 1 & 2 are dominated by SHORT posts (52% under 100 chars each).
#   Labels 3 & 4 are more spread — but median is still well under 400 chars.
#   The LLM's natural verbose tendency pushes toward 350-600 chars, which
#   is above the median for all labels — hence the MAX_LEN cap is critical.

MIN_LEN = {1: 31,  2: 37,  3: 54,  4: 60}   # p5  of each label's distribution
MAX_LEN = {1: 409, 2: 368, 3: 692, 4: 906}   # p90 of each label's distribution

# Length descriptor bands injected into prompts (guides the LLM on target length)
# Band boundaries are label-agnostic; the prompt picks one based on a sampled original
LENGTH_BANDS = [
    (0,    80,   "very short — under 80 characters, 1 sentence or a brief raw outburst"),
    (80,   200,  "short — 80 to 200 characters, 1 to 2 sentences"),
    (200,  400,  "medium — 200 to 400 characters, 2 to 4 sentences"),
    (400,  700,  "moderately long — 400 to 700 characters, 4 to 6 sentences"),
    (700,  9999, "long — over 700 characters, a detailed personal post"),
]

# ---------------------------------------------------------------------------
# Label definitions  (injected into every prompt)
# ---------------------------------------------------------------------------
LABEL_META = {
    1: {
        "name": "Mild / Non-clinical",
        "desc": (
            "General sadness, temporary stress, or everyday frustration without "
            "clinical depression. May include venting, mild anxiety, or short-term "
            "mood dips. MUST NOT contain any mention of self-harm, hopelessness, "
            "wishing to die, or suicidal thoughts."
        ),
        "boundary": "This label does NOT include persistent sadness or loss of hope.",
    },
    2: {
        "name": "Moderate Depression",
        "desc": (
            "Persistent sadness, feelings of isolation, anxiety, low energy, loss "
            "of interest, or seeking help for mental distress. The author feels "
            "trapped or helpless. MUST NOT express a wish to die or disappear."
        ),
        "boundary": "This label does NOT include passive or active suicidal ideation.",
    },
    3: {
        "name": "Severe / Passive Suicidal Ideation",
        "desc": (
            "Deep hopelessness, wishing to disappear, 'not exist anymore', or "
            "feelings of worthlessness and being a burden. No explicit plan or "
            "active intent to act — more 'I wish I could escape' than 'I will do "
            "something'. MUST NOT include concrete plans, timelines, or final goodbyes."
        ),
        "boundary": (
            "Key distinction from Label 4: passive ('I wish I didn't exist') "
            "NOT active ('I will kill myself tonight')."
        ),
    },
    4: {
        "name": "Active Suicidal Ideation",
        "desc": (
            "Explicit thoughts of self-harm or ending one's life, writing final "
            "goodbyes, making specific plans, or directly announcing intent to act. "
            "Language is urgent, concrete, and present-tense."
        ),
        "boundary": (
            "Key distinction from Label 3: active and concrete ('I have decided', "
            "'goodbye everyone') NOT just passive wishful thinking."
        ),
    },
}

SYSTEM_PROMPT = """\
You are an expert Computational Linguist and Clinical NLP Researcher specialising
in low-resource South-Asian languages. You assist a peer-reviewed academic study
building a machine-learning classifier for detecting depressive and suicidal
ideation in Bengali social media text.

Your task is SYNTHETIC DATA AUGMENTATION. Generate strictly anonymised, fully
synthetic Bengali social-media posts that perfectly mirror real user patterns.

Style Guidelines:
- Write entirely in the Bengali script (বাংলা হরফ). NEVER output full sentences in Romanized Banglish (e.g., "ami ekjon..."). Code-mixing with single English words is encouraged, but the text must be overwhelmingly in the Bengali script.
- Maintain strict logical coherence. Sentences must be logically connected and the underlying thought must make perfect sense. Do NOT generate garbled, contradictory, or meaningless sentences.
- AVOID "Google Translate" style Bengali. NEVER use unnatural phrasing (e.g., "রাতে ঘুম নাই হচ্ছে") or weird vocabulary choices (e.g., "সবকিছু বিস্তারিত" to mean "everything is over"). Write in natural, native-sounding Bangladeshi Bengali.
- CRITICAL: If your model uses reasoning or chain-of-thought (<think>), you MUST think entirely in Bengali. Do NOT think in English and translate later. 
- You must mimic human imperfections (informal spelling, missing punctuation, repetition like ,,, ।।।). A grammatically imperfect sentence is fine, but it MUST sound like a real native human wrote it, not a robot.
- Use colloquial Bangladeshi vocabulary and first-person emotional expression. NEVER use formal, academic, textbook, or literature-style Bengali (সাধু ভাষা / Shuddho Bangla). The vocabulary MUST be simple, raw, everyday Bangladeshi social-media language.

Strict rules:
1. Every post MUST match the EXACT severity label and its clinical boundaries.
2. Do NOT copy or paraphrase provided examples — create entirely new scenarios.
3. Do NOT include preambles, warnings, explanations, or any text other than the JSON.
4. Output ONLY a valid JSON array of strings, nothing else.\
"""

VERIFY_PROMPT_TEMPLATE = """\
You are a clinical NLP expert. Classify this Bengali social-media post into EXACTLY
one of these four severity labels:

Label 1 = Mild/Non-clinical: general sadness, stress, venting — NO self-harm thoughts
Label 2 = Moderate Depression: persistent hopelessness, isolation — NO suicidal thoughts
Label 3 = Severe/Passive Ideation: wishing to disappear, passive — NO concrete plan
Label 4 = Active Suicidal Ideation: explicit intent, plan, or final goodbye

Post: {text}

Reply with ONLY the single digit (1, 2, 3, or 4). No explanation.\
"""

# ---------------------------------------------------------------------------
# Use time-based seed for few-shot sampling so each run picks different
# examples (prevents same-scenario convergence across model comparisons).
# SEED is still used for the final df shuffle to keep that reproducible.
random.seed()  # time-based


def model_slug(model_name: str) -> str:
    """
    Convert a model name to a filesystem-safe lowercase slug.
    E.g. 'qwen2.5:32b'  -> 'qwen2-5-32b'
         'deepseek-r1:32b' -> 'deepseek-r1-32b'
         'aya-expanse:32b' -> 'aya-expanse-32b'
    """
    return re.sub(r"[^\w\-]", "-", model_name).strip("-").lower()


def is_reasoning_model(model_name: str) -> bool:
    """
    Detect models that emit <think>...</think> chain-of-thought blocks.
    Covers: DeepSeek-R1 family, Qwen3 (hybrid thinking), QwQ (always thinking).
    These models need larger token budgets and <think> stripping in all calls.
    """
    name_lower = model_name.lower()
    return any(tok in name_lower for tok in (
        "deepseek-r1", "-r1", ":r1",   # DeepSeek-R1 distills
        "qwen3",                         # Qwen3 (hybrid think mode)
        "qwq",                           # QwQ (always think)
    ))



# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = []
    for r in range(2, ws.max_row + 1):
        text  = ws.cell(row=r, column=1).value
        label = ws.cell(row=r, column=2).value
        if text and label:
            rows.append({
                "posts":  str(text).strip(),
                "labels": int(label),
                "source": "original",
            })
    return pd.DataFrame(rows)


def save_checkpoint(label: int, new_texts: list, cp_dir: Path, slug: str):
    """Append accepted texts to checkpoints/label_N_<model-slug>.jsonl"""
    cp_dir.mkdir(exist_ok=True)
    cp_file = cp_dir / f"label_{label}_{slug}.jsonl"
    with open(cp_file, "a", encoding="utf-8") as f:
        for t in new_texts:
            f.write(json.dumps({"text": t, "label": label}, ensure_ascii=False) + "\n")


def load_checkpoint(label: int, cp_dir: Path, slug: str) -> list:
    """Load existing texts from checkpoints/label_N_<model-slug>.jsonl"""
    cp_file = cp_dir / f"label_{label}_{slug}.jsonl"
    if not cp_file.exists():
        return []
    texts = []
    with open(cp_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except Exception:
                pass
    return texts


def save_output(df: pd.DataFrame, path: str):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["posts", "labels", "source"])
    for _, row in df.iterrows():
        ws.append([row["posts"], int(row["labels"]), row.get("source", "synthetic")])
    wb.save(path)


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def is_too_short(text: str, label: int) -> bool:
    return len(text) < MIN_LEN.get(label, 20)


def is_too_long(text: str, label: int) -> bool:
    return len(text) > MAX_LEN.get(label, 9999)


def sample_target_length_desc(originals: list) -> str:
    """
    Randomly pick one original text, find its char-length band,
    and return a human-readable length instruction for the prompt.
    Calling this once per batch means the LLM is nudged toward the
    full spread of real lengths rather than always generating medium texts.
    """
    ref_len = len(random.choice(originals))
    for lo, hi, desc in LENGTH_BANDS:
        if lo <= ref_len < hi:
            return desc
    return LENGTH_BANDS[-1][2]  # fallback: long band


def is_too_similar(text: str, reference_pool: list, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """Return True if text is suspiciously similar to any item in reference_pool."""
    for ref in reference_pool:
        ratio = SequenceMatcher(None, text, ref).ratio()
        if ratio >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def build_user_prompt(label: int, examples: list, n: int, shots: int, originals: list) -> str:
    meta       = LABEL_META[label]
    shots_text = "\n".join(f"{i+1}. {e}" for i, e in enumerate(examples))
    len_desc   = sample_target_length_desc(originals)
    return (
        f"Target Severity Label : {label} — {meta['name']}\n"
        f"Clinical Description  : {meta['desc']}\n"
        f"Important Boundary    : {meta['boundary']}\n\n"
        f"Original examples (analyse style only — do NOT copy or paraphrase):\n"
        f"{shots_text}\n\n"
        f"LENGTH REQUIREMENT: Each post must be {len_desc}. "
        f"Write in the same raw, informal, social-media style as the examples above. "
        f"Do NOT write formal paragraphs or elaborate prose.\n\n"
        f"Generate exactly {n} NEW, highly diverse synthetic Bengali social-media posts "
        f"that strictly match Severity Label {label}.\n"
        f"Each post must represent a completely different scenario, emotional context, "
        f"and sentence structure.\n\n"
        f"Output ONLY a valid JSON array of {n} strings. No other text."
    )


def call_llm_generate(label: int, examples: list, n: int, shots: int, model: str, temp: float, originals: list) -> list:
    """
    Single generation call. Returns list of strings (may be empty on failure).
    Handles DeepSeek-R1 <think>...</think> chain-of-thought blocks automatically.
    """
    prompt = build_user_prompt(label, examples, n, shots, originals)
    # Reasoning models (DeepSeek-R1) need extra token budget for the <think> block
    token_budget = 12000 if is_reasoning_model(model) else 4096

    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": temp, "num_predict": token_budget},
            )
            raw = response["message"]["content"].strip()

            # Strip DeepSeek-R1 <think>...</think> reasoning block
            # The block always comes before the actual JSON output
            raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()

            # Handle markdown code fences
            if "```" in raw:
                parts = raw.split("```")
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            # Extract outermost JSON array
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in LLM response")

            parsed = json.loads(raw[start:end])
            if not isinstance(parsed, list):
                raise ValueError("LLM output is not a list")

            return [str(t).strip() for t in parsed if str(t).strip()]

        except Exception as exc:
            print(f"    [warn] Generation attempt {attempt+1}/3 failed: {exc}")
            if attempt < 2:
                time.sleep(3)
    return []


def call_llm_verify(text: str, expected_label: int, model: str) -> bool:
    """
    Ask the LLM to classify the generated text and confirm it matches expected_label.
    Uses temperature=0 for deterministic classification.
    Returns True if verified, False if mismatch or error.
    """
    prompt = VERIFY_PROMPT_TEMPLATE.format(text=text)
    # Reasoning models need more tokens for the think block even at temp=0
    verify_tokens = 8000 if is_reasoning_model(model) else 10
    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_predict": verify_tokens},
            )
            pred = response["message"]["content"].strip()
            # Strip <think> block from reasoning models
            pred = re.sub(r"<think>[\s\S]*?</think>", "", pred, flags=re.IGNORECASE).strip()
            # Accept just the digit, or "Label X", etc.
            digit = next((c for c in pred if c in "1234"), None)
            if digit is None:
                raise ValueError(f"No digit in verification response: {pred!r}")
            return digit == str(expected_label)
        except Exception as exc:
            print(f"    [warn] Verification attempt {attempt+1}/2 failed: {exc}")
            time.sleep(2)
    return True  # on verification error, accept the text (fail-open)


# ---------------------------------------------------------------------------
# Core augmentation loop
# ---------------------------------------------------------------------------

def augment_label(
    label: int,
    originals: list,
    n_needed: int,
    cp_dir: Path,
    model: str,
    temp: float,
    shots: int,
    do_verify: bool,
    test_mode: bool,
    batch_size: int = 10,
    slug: str = "model",
) -> list:
    """Generate n_needed synthetic texts for `label`. Returns list of new texts."""
    if test_mode:
        n_needed = min(n_needed, 3)

    existing  = load_checkpoint(label, cp_dir, slug)
    already   = len(existing)
    remaining = n_needed - already

    if remaining <= 0:
        print(f"  Label {label}: already have {already} in checkpoint — skipping.")
        return existing

    print(f"  Label {label}: need {n_needed} total, have {already} → generating {remaining} more …")

    # Reference pool for deduplication = originals + already-generated
    ref_pool  = list(originals) + list(existing)
    generated = list(existing)

    empty_streak = 0  # consecutive-empty-batch protection

    while (len(generated) - already) < remaining:
        still_need = remaining - (len(generated) - already)
        batch_n    = min(batch_size, still_need)  # uses CLI --batch-size, not global

        # Re-sample few-shot examples each call for diversity
        # Use more shots for boundary labels 3 and 4
        shot_n    = shots if label in [1, 2] else max(shots, 8)
        shot_n    = min(shot_n, len(originals))
        shot_pool = random.sample(originals, shot_n)

        raw_batch = call_llm_generate(label, shot_pool, batch_n, shot_n, model, temp, originals)

        if not raw_batch:
            empty_streak += 1
            print(f"  [warn] Empty batch (streak {empty_streak}/{MAX_EMPTY_RETRIES}), retrying …")
            if empty_streak >= MAX_EMPTY_RETRIES:
                print(f"  [error] Too many consecutive failures for label {label}. Stopping early.")
                break
            time.sleep(3)
            continue

        empty_streak = 0  # reset on success

        accepted        = []
        rejected_short  = 0
        rejected_long   = 0
        rejected_sim    = 0
        rejected_label  = 0

        for text in raw_batch:
            # Filter 1: too short (below p5 of original distribution)
            if is_too_short(text, label):
                rejected_short += 1
                continue

            # Filter 2: too long (above p90 of original distribution)
            # Prevents LLM verbose-bias from skewing the length distribution
            if is_too_long(text, label):
                rejected_long += 1
                continue

            # Filter 3: too similar to any existing text
            if is_too_similar(text, ref_pool):
                rejected_sim += 1
                continue

            # Filter 4: label verification (optional, slower)
            if do_verify:
                if not call_llm_verify(text, label, model):
                    rejected_label += 1
                    continue

            accepted.append(text)
            ref_pool.append(text)  # add to dedup pool immediately

        if accepted:
            generated.extend(accepted)
            save_checkpoint(label, accepted, cp_dir, slug)

        done = len(generated) - already
        pct  = done / remaining * 100
        rej_info = (
            f"(rejected: short={rejected_short}, long={rejected_long}, "
            f"similar={rejected_sim}, label={rejected_label})"
        )
        print(f"    -> {done}/{remaining} ({pct:.1f}%)  {rej_info}")

    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MODEL, AUGMENT_RATIO  # noqa: PLW0603


    parser = argparse.ArgumentParser(
        description="Bengali mental-health dataset augmentation via local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python augment_dataset.py                            full run, all labels, default settings
  python augment_dataset.py --label 4                 augment only label 4
  python augment_dataset.py --label 4 --count 500     generate exactly 500 synthetic texts
  python augment_dataset.py --label 4 --test          dry-run: 3 samples, no file saved
  python augment_dataset.py --ratio 3                 1:3 ratio instead of 1:5
  python augment_dataset.py --model aya-expanse:32b   use Aya Expanse model
  python augment_dataset.py --batch-size 5            smaller batches (more careful)
  python augment_dataset.py --temperature 0.95        higher creativity/diversity
  python augment_dataset.py --shots 8                 8 few-shot examples per call
  python augment_dataset.py --no-verify               skip label verification (faster)
  python augment_dataset.py --label 4 --count 200 --no-verify --batch-size 15
        """,
    )

    parser.add_argument(
        "--label", type=int, default=0, choices=[0, 1, 2, 3, 4],
        help="Label to augment (1-4). 0 = all labels. Default: 0",
    )
    parser.add_argument(
        "--count", type=int, default=0,
        help="Exact number of synthetic texts to generate (overrides --ratio). Default: 0 (use --ratio)",
    )
    parser.add_argument(
        "--ratio", type=int, default=AUGMENT_RATIO,
        help=f"Total augmentation ratio. E.g. 5 means 1 original + 4 synthetic = 5x. Default: {AUGMENT_RATIO}",
    )
    parser.add_argument(
        "--model", default=MODEL,
        help=f"Ollama model name. Default: {MODEL}",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, dest="batch_size",
        help=f"Number of texts to request per LLM call. Default: {BATCH_SIZE}",
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"LLM sampling temperature (0.0–1.0). Higher = more diverse. Default: {TEMPERATURE}",
    )
    parser.add_argument(
        "--shots", type=int, default=FEW_SHOT_N,
        help=f"Number of few-shot examples shown per LLM call. Default: {FEW_SHOT_N} (auto-bumped to 8 for labels 3-4)",
    )
    parser.add_argument(
        "--no-verify", action="store_true", dest="no_verify",
        help="Skip the label-verification pass (faster but less safe). Default: verification ON",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Dry-run: generate only 3 samples per label, do not save output file.",
    )
    parser.add_argument(
        "--input", default=INPUT_FILE,
        help=f"Input Excel file. Default: {INPUT_FILE}",
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help=f"Output Excel file. Default: {OUTPUT_FILE}",
    )

    args = parser.parse_args()

    # Apply CLI overrides to globals
    MODEL         = args.model
    AUGMENT_RATIO = args.ratio
    do_verify     = not args.no_verify

    # Model slug: filesystem-safe name embedded in all output filenames
    slug = model_slug(MODEL)

    # Auto-derive output filename from model slug unless user explicitly set --output
    if args.output == OUTPUT_FILE:  # user did not override
        out_stem = Path(OUTPUT_FILE).stem   # 'dataset_augmented'
        out_file = f"{out_stem}_{slug}.xlsx"
    else:
        out_file = args.output

    reasoning = is_reasoning_model(MODEL)

    print("=" * 65)
    print("  Bengali Dataset Augmenter  v2 (robust)")
    print("=" * 65)
    print(f"  Model          : {MODEL}{'  [reasoning — <think> stripping ON]' if reasoning else ''}")
    print(f"  Model slug     : {slug}")
    print(f"  Input          : {args.input}")
    print(f"  Output         : {out_file}")
    print(f"  Checkpoint dir : checkpoints/label_N_{slug}.jsonl")
    print(f"  Ratio          : 1 : {args.ratio}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Few-shot N     : {args.shots} (auto->8 for labels 3-4)")
    print(f"  Verification   : {'OFF' if args.no_verify else 'ON'}")
    print(f"  Mode           : {'TEST (3 samples only)' if args.test else 'FULL'}")
    print("=" * 65)

    # Load data
    df = load_dataset(args.input)
    print(f"\nLoaded {len(df)} rows from {args.input}")
    print("Original distribution:", df["labels"].value_counts().sort_index().to_dict())

    by_label: dict = defaultdict(list)
    for _, row in df.iterrows():
        by_label[int(row["labels"])].append(row["posts"])

    labels_to_process = [args.label] if args.label in [1, 2, 3, 4] else [1, 2, 3, 4]

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    synth_rows = []

    for lbl in labels_to_process:
        originals = by_label[lbl]

        # Determine how many synthetic texts to generate
        if args.count > 0:
            n_needed = args.count
        else:
            n_needed = len(originals) * (AUGMENT_RATIO - 1)

        print(f"\n[Label {lbl}] {LABEL_META[lbl]['name']}")
        print(f"  Originals : {len(originals)}")
        print(f"  Target    : {n_needed} synthetic texts")

        synth = augment_label(
            label=lbl,
            originals=originals,
            n_needed=n_needed,
            cp_dir=CHECKPOINT_DIR,
            model=MODEL,
            temp=args.temperature,
            shots=args.shots,
            do_verify=do_verify,
            test_mode=args.test,
            batch_size=args.batch_size,   # FIX: was using hardcoded global BATCH_SIZE
            slug=slug,
        )

        for t in synth:
            synth_rows.append({"posts": t, "labels": lbl, "source": "synthetic"})

    # Merge and save
    # Always pull in ALL checkpoints (not just the labels processed in this run)
    # so that running --label N never drops previously augmented labels from the output.
    all_synth_rows = []
    for lbl in [1, 2, 3, 4]:
        ckpt_texts = load_checkpoint(lbl, CHECKPOINT_DIR, slug)
        for t in ckpt_texts:
            all_synth_rows.append({"posts": t, "labels": lbl, "source": "synthetic"})

    if all_synth_rows:
        df_synth = pd.DataFrame(all_synth_rows)
        df_final = pd.concat([df, df_synth], ignore_index=True).sample(frac=1, random_state=SEED)
    else:
        df_final = df

    if not args.test:
        save_output(df_final, out_file)
        print(f"\n{'=' * 65}")
        print(f"  Saved → {out_file}")
        print(f"  Final distribution:")
        print(f"  {df_final['labels'].value_counts().sort_index().to_dict()}")
        print(f"  Total rows : {len(df_final)}")
        print(f"  Synthetic  : {len(synth_rows)}")
        print("=" * 65)
    else:
        print("\n[TEST MODE] Output file not saved.")
        if synth_rows:
            print(f"Generated {len(synth_rows)} sample(s):")
            for i, r in enumerate(synth_rows, 1):
                print(f"  {i}. [{r['labels']}] {r['posts']}")


if __name__ == "__main__":
    main()
