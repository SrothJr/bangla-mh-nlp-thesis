"""
Phase 8, Step 2 (PHASE8_MODALITY_AWARE_SYSTEM_PLAN.md): a parity-tested
wrapper around the Phase 2 depression classifier. The wrapper does not
reimplement anything -- it imports and calls the exact same functions
already used throughout this project (fuzzy_logic_layer.py's
load_depression_classifier() and get_depression_probs()), so identical
behavior is guaranteed by construction, not just by similarity.

Parity is verified against outputs/fuzzy_risk_results.json -- per-
example depression probabilities already saved for all 196 test memes
from the original Days 9-10 fuzzy-logic run. This wrapper must
reproduce those exact numbers (within floating-point tolerance) when
run on the identical text today, or the parity test fails and this
wrapper must not be considered done.
"""
import os
import json

import numpy as np

from fuzzy_logic_layer import load_depression_classifier, get_depression_probs, DEPRESSION_CLASSES

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_FUZZY_RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "fuzzy_risk_results.json")
PARITY_REPORT_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase8_2_depression_parity_report.json")

_tokenizer = None
_model = None


def _get_model():
    """Lazy singleton -- the classifier loads once, reused across calls."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer, _model = load_depression_classifier()
    return _tokenizer, _model


def predict_depression(text_bn):
    """The stable, documented interface for text-only depression prediction.

    predict_depression(text_bn) -> {
        "label": "Minimum|Mild|Moderate|Severe",
        "probabilities": [p_minimum, p_mild, p_moderate, p_severe],
    }

    Calls the exact original Phase 2 checkpoint and preprocessing --
    nothing new is trained or initialized here.
    """
    tokenizer, model = _get_model()
    probs = get_depression_probs(tokenizer, model, text_bn)
    return {
        "label": DEPRESSION_CLASSES[int(np.argmax(probs))],
        "probabilities": [float(p) for p in probs],
    }


def run_parity_test(tolerance=1e-5):
    """Re-runs the wrapper on the same 196 test memes' OCR text already
    used for the original Days 9-10 fuzzy-logic run, and checks the
    wrapper's output against those saved probabilities exactly."""
    print(f"Loading known-good depression probabilities from {ORIGINAL_FUZZY_RESULTS_PATH}...")
    with open(ORIGINAL_FUZZY_RESULTS_PATH, "r", encoding="utf-8") as f:
        original_records = json.load(f)
    print(f"Loaded {len(original_records)} original records.")

    max_abs_diff = 0.0
    label_mismatches = 0
    per_record_results = []

    for rec in original_records:
        text_bn = rec["ocr_text_bn"]
        original_probs = np.array([rec["depression_probs"][c] for c in DEPRESSION_CLASSES])
        original_label = rec["depression_pred"]

        wrapper_out = predict_depression(text_bn)
        wrapper_probs = np.array(wrapper_out["probabilities"])
        wrapper_label = wrapper_out["label"]

        abs_diff = np.abs(wrapper_probs - original_probs).max()
        max_abs_diff = max(max_abs_diff, abs_diff)
        label_match = wrapper_label == original_label
        if not label_match:
            label_mismatches += 1

        per_record_results.append({
            "image_index": rec["image_index"],
            "label_match": bool(label_match),
            "max_abs_prob_diff": float(abs_diff),
        })

    n = len(original_records)
    label_agreement_pct = 100 * (n - label_mismatches) / n
    passed = (label_mismatches == 0) and (max_abs_diff <= tolerance)

    print(f"\n=== Parity test result ===")
    print(f"Records checked:        {n}")
    print(f"Label agreement:        {label_agreement_pct:.2f}% ({n - label_mismatches}/{n})")
    print(f"Max absolute prob diff: {max_abs_diff:.2e}")
    print(f"Tolerance:              {tolerance:.2e}")
    print(f"PASSED: {passed}")

    if not passed:
        print("\n!!! PARITY TEST FAILED -- do not consider this wrapper done. !!!")
        mismatched = [r for r in per_record_results if not r["label_match"]]
        for r in mismatched[:5]:
            print(f"  mismatch at image_index={r['image_index']}: diff={r['max_abs_prob_diff']:.4f}")

    with open(PARITY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_records": n,
            "label_agreement_pct": label_agreement_pct,
            "label_mismatches": label_mismatches,
            "max_abs_prob_diff": float(max_abs_diff),
            "tolerance": tolerance,
            "passed": bool(passed),
            "per_record_results": per_record_results,
        }, f, indent=2)
    print(f"\nSaved full parity report to {PARITY_REPORT_PATH}")
    return passed


if __name__ == "__main__":
    run_parity_test()
