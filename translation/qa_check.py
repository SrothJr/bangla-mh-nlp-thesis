"""
STEP C SCRIPT: QA pass. Runs multilingual sentiment analysis on both English
original and Bangla translation, flags rows where sentiment doesn't match
for manual review. Also flags obvious mechanical failures (empty output,
un-replaced placeholder tokens, output that's suspiciously similar length
to a totally different language, etc.)

INPUT / OUTPUT FORMAT: JSONL in, JSONL out. Reads polished_full.jsonl (or
nllb_output.jsonl if you want to QA before the polish step). Output file
contains only the FLAGGED rows, each with its original "row_id" intact so
you can trace it straight back to the source line in
merged_mental_health_dapt.jsonl, plus an "issues" field listing what tripped.

Usage:
    pip install transformers torch
    python3 qa_check.py --input polished_full.jsonl --bangla_field bangla_polished --output flagged_for_review.jsonl
"""

import argparse
import json
import re

from transformers import pipeline

# Multilingual sentiment model that supports Bengali reasonably well
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME, truncation=True)


def mechanical_checks(english, bangla):
    """Cheap checks that don't need a model — catch obvious failures fast."""
    issues = []
    if not bangla or not bangla.strip():
        issues.append("EMPTY_OUTPUT")
        return issues
    if "⟦T" in bangla or "T1⟧" in bangla:
        issues.append("UNRESOLVED_PLACEHOLDER")
    # crude check: bangla output should contain some Bengali unicode range chars
    bengali_chars = re.findall(r"[\u0980-\u09FF]", bangla)
    if len(bengali_chars) < 5 and len(bangla) > 20:
        issues.append("LOW_BENGALI_CONTENT")
    if len(bangla) < len(english) * 0.15:
        issues.append("SUSPICIOUSLY_SHORT")
    return issues


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="polished_full.jsonl from casual_polish.py (or nllb_output.jsonl)")
    ap.add_argument("--output", default="flagged_for_review.jsonl")
    ap.add_argument("--text_field", default="text", help="JSON key holding the original English text")
    ap.add_argument("--bangla_field", default="bangla_polished",
                     help="Key holding the Bangla text to check (bangla_polished or bangla_raw)")
    ap.add_argument("--sample_rate", type=float, default=1.0,
                     help="Fraction of rows to run sentiment check on (1.0 = all, slower)")
    args = ap.parse_args()

    print("Loading sentiment model (first run downloads it)...")
    sentiment = get_sentiment_pipeline()

    rows = load_rows(args.input)
    print(f"Loaded {len(rows)} rows, checking field '{args.bangla_field}'")

    import random
    random.seed(42)
    flagged = []

    for i, row in enumerate(rows):
        english = row.get(args.text_field, "")
        bangla = row.get(args.bangla_field, "")

        issues = mechanical_checks(english, bangla)

        if args.sample_rate >= 1.0 or random.random() < args.sample_rate:
            if bangla.strip():
                try:
                    eng_sent = sentiment(english[:512])[0]
                    ban_sent = sentiment(bangla[:512])[0]
                    if eng_sent["label"] != ban_sent["label"]:
                        issues.append(f"SENTIMENT_MISMATCH(en={eng_sent['label']},bn={ban_sent['label']})")
                except Exception as e:
                    issues.append(f"SENTIMENT_CHECK_FAILED:{e}")

        if issues:
            out_obj = dict(row)  # preserves row_id + every original field
            out_obj["issues"] = issues
            flagged.append(out_obj)

        if i % 500 == 0:
            print(f"Checked {i}/{len(rows)}, flagged so far: {len(flagged)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for obj in flagged:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(flagged)}/{len(rows)} rows flagged for review -> {args.output}")
    print("Each flagged line carries 'row_id' tracing back to the original input file.")
    print("Priority order: fix EMPTY_OUTPUT and UNRESOLVED_PLACEHOLDER first (mechanical bugs),")
    print("then review SENTIMENT_MISMATCH rows (tone/translation quality issues).")


if __name__ == "__main__":
    main()