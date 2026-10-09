"""
Phase 9, Step 6a (Track A2): screen external BN-HIB memes as candidate
class-0 ("no expressed severity") training examples.

WHY
---
Step 1 measured class 0 as the weak point: F1 0.469, recall 0.419, on only
117 of the 777 examples. Step 3 then measured the learning curve and found
it still climbing steeply at +0.030 accuracy per +100 training examples.
Together those say the class-0 decision boundary is starved of data, and
that adding data is the lever that actually moves this task.

BN-HIB is already present locally (used by Phase 7.2) and is Bangla meme
data -- in-domain visually and linguistically. Its labels are about
trolling and hate speech, not self-harm, so its memes are overwhelmingly
TRUE NEGATIVES for suicide content. That makes them usable as extra class-0
examples, but only after screening, because "overwhelmingly" is not "all".

SCREENING, IN THREE LAYERS (this script is layers 1 and 2)
----------------------------------------------------------
Layer 1  Keyword screen. Any candidate whose text contains Bangla or
         English self-harm, suicide, or severe-distress vocabulary is
         dropped. Deliberately over-broad: a false drop costs one training
         example, a false keep injects a mislabelled example into a class
         that is already the weakest.

Layer 2  Language filter and sampling. Only Bangla-text memes are kept, so
         the OCR side needs no translation and cannot pick up a
         translation-artifact signature that distinguishes external rows
         from real ones.

Layer 3  Model-confidence screen, applied later in the pipeline once
         embeddings exist, plus a manual spot-check. Not in this file.

SHORTCUT RISK, AND THE CAPS THAT CONTROL IT
-------------------------------------------
The real danger is not mislabelling, it is the model learning "BN-HIB
visual style means class 0" instead of learning anything about suicide
content. Three controls, all fixed in advance:
  - At most 300 external examples, roughly doubling class 0's training
    count rather than swamping it.
  - External rows are used for TRAINING ONLY. They never appear in
    inner-validation or in any outer held-out fold, so every reported
    number is still measured purely on real in-domain data.
  - The whole thing is A/B'd against the identical pipeline without them.

PRIVACY
-------
BN-HIB is CC BY-NC-SA and is already excluded from version control by
.gitignore. This script writes image filenames and screening decisions
only. It never writes meme text into any output file.

OUTPUT
------
outputs/phase9_7_external_negative_candidates.json
"""
import os
import csv
import json
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BNHIB_ROOT = os.path.join(PROJECT_ROOT, "BN-HIB")
CSV_DIR = os.path.join(BNHIB_ROOT, "Train Test Val CSV")
IMAGE_DIR = os.path.join(BNHIB_ROOT, "Images")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_7_external_negative_candidates.json")

MAX_CANDIDATES = 300
# Oversample before the later model-confidence screen trims the list.
SAMPLE_POOL = 600
SAMPLE_SEED = 20240923

# Deliberately over-broad. Dropping a safe meme costs one training example;
# keeping an unsafe one corrupts the weakest class.
BANGLA_BLOCK = [
    "আত্মহত্যা", "আত্মহনন", "আত্মঘাতী", "মরে যাব", "মরে যাবো", "মরতে চাই",
    "মৃত্যু", "মরণ", "গলায় দড়ি", "ফাঁসি", "বিষ খা", "ঘুমের ওষুধ",
    "হাত কাট", "রক্ত", "বাঁচতে চাই না", "বেঁচে থেকে লাভ", "জীবন শেষ",
    "হতাশ", "বিষণ্ণ", "বিষন্ন", "ডিপ্রেশন", "মানসিক রোগ", "একা লাগে",
    "কষ্ট সহ্য", "শেষ করে দ", "নিজেকে শেষ",
]
ENGLISH_BLOCK = [
    "suicide", "suicidal", "kill myself", "killing myself", "end my life",
    "end it all", "self harm", "self-harm", "cut myself", "cutting myself",
    "overdose", "hang myself", "want to die", "wanna die", "better off dead",
    "depress", "hopeless", "worthless", "no reason to live", "die", "death",
]


def load_rows():
    rows = []
    for name in ("train.csv", "val.csv", "test.csv"):
        path = os.path.join(CSV_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_source_csv"] = name
                rows.append(r)
    return rows


def blocked_reason(text):
    low = text.lower()
    for kw in BANGLA_BLOCK:
        if kw in text:
            return f"bangla_keyword"
    for kw in ENGLISH_BLOCK:
        if kw in low:
            return f"english_keyword"
    return None


def main():
    print("=" * 74)
    print("PHASE 9 STEP 6a (Track A2) -- screening external BN-HIB negatives")
    print("=" * 74)

    rows = load_rows()
    print(f"BN-HIB rows loaded: {len(rows)}")

    stats = {
        "total": len(rows),
        "dropped_not_bangla": 0,
        "dropped_no_text": 0,
        "dropped_missing_image": 0,
        "dropped_bangla_keyword": 0,
        "dropped_english_keyword": 0,
        "survived_screen": 0,
    }

    survivors = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            stats["dropped_no_text"] += 1
            continue
        if r.get("language") != "b":
            stats["dropped_not_bangla"] += 1
            continue
        img = r.get("image", "").strip()
        if not img or not os.path.exists(os.path.join(IMAGE_DIR, img)):
            stats["dropped_missing_image"] += 1
            continue
        reason = blocked_reason(text)
        if reason:
            stats[f"dropped_{reason}"] += 1
            continue
        survivors.append({
            "image": img,
            "source_csv": r["_source_csv"],
            "bnhib_class": r.get("class"),
            "text_length": len(text),
        })

    stats["survived_screen"] = len(survivors)
    print("\nScreening results:")
    for k, v in stats.items():
        print(f"  {k:<28}{v}")

    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(survivors)
    pool = survivors[:SAMPLE_POOL]
    print(f"\nSampled pool for downstream processing: {len(pool)} "
          f"(will be trimmed to at most {MAX_CANDIDATES} after the "
          f"model-confidence screen)")

    by_class = {}
    for c in pool:
        by_class[c["bnhib_class"]] = by_class.get(c["bnhib_class"], 0) + 1
    print("Pool composition by original BN-HIB class:")
    for k, v in sorted(by_class.items()):
        print(f"  {k:<28}{v}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Screened BN-HIB candidates for use as extra class-0 "
                     "('no expressed severity') TRAINING examples only. Filenames and "
                     "screening decisions only -- no meme text is written here, since "
                     "BN-HIB is CC BY-NC-SA and excluded from version control."),
            "screening_layers": {
                "layer_1_keyword": "Bangla + English self-harm/suicide/distress vocabulary, over-broad by design",
                "layer_2_language_and_sampling": "Bangla-text memes only, so no translation artifacts distinguish external rows",
                "layer_3_pending": "model-confidence screen + manual spot-check, applied downstream",
            },
            "caps": {
                "max_candidates": MAX_CANDIDATES,
                "sample_pool": SAMPLE_POOL,
                "training_only": True,
                "never_in_validation_or_heldout": True,
            },
            "statistics": stats,
            "sample_seed": SAMPLE_SEED,
            "pool_composition_by_bnhib_class": by_class,
            "candidates": pool,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
