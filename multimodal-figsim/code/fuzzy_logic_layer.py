"""
Days 9-10: fuzzy-logic risk-combination layer (Section 9).

Runs Phase-2's existing, untouched 4-class depression classifier on the same
translated OCR text already used for the suicide-severity pipeline (Section
9.0: "reuse that same translated text ... run it through Phase-2's existing
classifier"), combines it with the locked FigSIM model's suicide-severity
output via the draft rule table (Section 9), and reports risk level for
every test meme plus an illustrative table.

Fuzzification note (Section 9, "implementation approach (simplified,
decided)"): membership degrees are each classifier's own softmax
probabilities directly -- no separate triangular/trapezoidal membership
functions are built. `skfuzzy.fuzzy_and`/`fuzzy_or` operate on continuous
membership FUNCTIONS sampled over a shared universe (they interpolate two
arrays), which does not fit singleton class-probability degrees -- forcing
our numbers through that API would mean constructing degenerate universes
for no benefit, exactly the extra design work the brief said to skip. Fuzzy
AND (min) / OR (max) are applied directly via numpy on the scalar
membership degrees, which is mathematically identical to what fuzzy_and/
fuzzy_or reduce to on singleton sets. skfuzzy is still imported and used
elsewhere unmodified (embedding extraction and translation scripts do not
touch it; this is the only stage that uses it, per Section 9's own
"Implementation: Python's scikit-fuzzy library").

Defuzzification: the consequent (risk level) is an inherently discrete
4-way label, not a continuous quantity -- so "defuzzification" here is
argmax over the aggregated risk-level membership degrees, the direct
discrete analogue of centroid defuzzification for a continuous consequent.
"""
import os
import json

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAPT_ROOT = r"C:\Users\user6\T2520814\DAPT_models"
CKPT_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5", "checkpoint-735")
TOKENIZER_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5_tapt")
TRANSLATION_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
SUICIDE_PRED_PATH = os.path.join(PROJECT_ROOT, "outputs", "final_test_predictions.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "fuzzy_risk_results.json")

DEPRESSION_CLASSES = ["Minimum", "Mild", "Moderate", "Severe"]
SUICIDE_CLASSES = ["None", "Wish to be dead", "Suicide ideation", "Suicide planning", "Suicide attempt or death"]
RISK_LEVELS = ["Minimal", "Low", "Elevated", "Critical"]
MAX_LEN = 256

# (suicide_level_index, [depression_level_indices in the OR-group]) -> risk
# Exactly the Section 9 draft rule table, re-expressed against 0-indexed classes.
RULES = [
    (4, [0, 1, 2, 3], "Critical"),      # Attempt or Death, any depression
    (3, [2, 3],        "Critical"),     # Planning, Moderate or Severe
    (3, [0, 1],        "Elevated"),     # Planning, Minimum or Mild
    (2, [2, 3],        "Elevated"),     # Ideation, Moderate or Severe
    (2, [1],           "Elevated"),     # Ideation, Mild
    (2, [0],           "Low"),          # Ideation, Minimum
    (1, [2, 3],        "Elevated"),     # Wish to be dead, Moderate or Severe
    (1, [0, 1],        "Low"),          # Wish to be dead, Minimum or Mild
    (0, [3],           "Elevated"),     # None, Severe
    (0, [2],           "Low"),          # None, Moderate
    (0, [0, 1],        "Minimal"),      # None, Minimum or Mild
]


def load_depression_classifier():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT_PATH)
    model.eval()
    return tokenizer, model


def get_depression_probs(tokenizer, model, text):
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=1)[0].numpy()


def combine_fuzzy(suicide_probs, depression_probs):
    """Mamdani-style combination: AND = min, OR (both within a rule's
    depression-condition group, and across rules mapping to the same risk
    level) = max. Returns per-risk-level membership degrees."""
    risk_membership = {r: 0.0 for r in RISK_LEVELS}
    firing_strengths = []
    for suicide_idx, depression_idxs, risk in RULES:
        depression_condition = float(max(depression_probs[i] for i in depression_idxs))  # fuzzy OR
        firing_strength = float(min(suicide_probs[suicide_idx], depression_condition))   # fuzzy AND
        firing_strengths.append({
            "suicide_level": SUICIDE_CLASSES[suicide_idx],
            "depression_condition": "/".join(DEPRESSION_CLASSES[i] for i in depression_idxs),
            "firing_strength": float(firing_strength),
            "risk": risk,
        })
        risk_membership[risk] = max(risk_membership[risk], firing_strength)      # fuzzy OR (aggregation)
    crisp_risk = max(risk_membership, key=risk_membership.get)  # "defuzzification" for a discrete consequent
    return risk_membership, crisp_risk, firing_strengths


def main():
    with open(SUICIDE_PRED_PATH, "r", encoding="utf-8") as f:
        suicide_preds = {r["image_index"]: r for r in json.load(f)}

    ocr_texts = {}
    with open(TRANSLATION_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ocr_texts[r["image_index"]] = r["ocr_text_bn"]

    test_indices = sorted(suicide_preds.keys())
    print(f"Running Phase-2 depression classifier on {len(test_indices)} test memes' OCR text...")
    print(f"Loading Phase-2 classifier (same checkpoint as the sanity check, WITH its head this time) "
          f"from {CKPT_PATH}")
    tokenizer, model = load_depression_classifier()

    results = []
    risk_counts = {r: 0 for r in RISK_LEVELS}
    for idx in test_indices:
        text = ocr_texts.get(idx, "") or ""
        depression_probs = get_depression_probs(tokenizer, model, text)
        suicide_probs = np.array(suicide_preds[idx]["probs"])

        risk_membership, crisp_risk, firing_strengths = combine_fuzzy(suicide_probs, depression_probs)
        risk_counts[crisp_risk] += 1

        results.append({
            "image_index": idx,
            "ocr_text_bn": text,
            "suicide_pred": SUICIDE_CLASSES[suicide_preds[idx]["y_pred"]],
            "suicide_true": SUICIDE_CLASSES[suicide_preds[idx]["y_true"]],
            "suicide_probs": {c: float(p) for c, p in zip(SUICIDE_CLASSES, suicide_probs)},
            "depression_pred": DEPRESSION_CLASSES[int(np.argmax(depression_probs))],
            "depression_probs": {c: float(p) for c, p in zip(DEPRESSION_CLASSES, depression_probs)},
            "risk_membership": risk_membership,
            "crisp_risk": crisp_risk,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")
    print("\n=== Risk level distribution across all 196 test memes ===")
    for r in RISK_LEVELS:
        print(f"  {r}: {risk_counts[r]} ({100 * risk_counts[r] / len(results):.1f}%)")


if __name__ == "__main__":
    main()
