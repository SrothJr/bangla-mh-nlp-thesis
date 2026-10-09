"""
Phase 8, Step 0 (PHASE8_MODALITY_AWARE_SYSTEM_PLAN.md Section 3): correct
an invalid comparison already sitting in this project's documentation.
"+0.0746 improvement" (0.5730 vs 0.4984) compares a 3-class result to a
5-class result directly -- not valid, since they're different targets.

This script does NOT retrain anything and does NOT touch the test set
again in any new way -- it purely re-reads the already-saved, already-
final 5-class test predictions (outputs/final_test_predictions.json,
written once during the original Phase 3 test lock) and aggregates them
into the 3-class scheme, exactly the same zero-retraining method already
used and validated in phase7_1_label_harmonization_diagnostic.py for
validation data. This is pure arithmetic on an existing, frozen file --
read-only with respect to every existing result.
"""
import os
import json

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIVE_CLASS_TEST_PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "outputs", "final_test_predictions.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase8_0_valid_5to3_test_comparison_results.json")

SUICIDE_CLASSES_5 = ["None", "Wish to be dead", "Suicide ideation", "Suicide planning", "Suicide attempt or death"]
THREE_CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
THREE_CLASS_GROUPS = [[0], [1, 2], [3, 4]]

NEW_3CLASS_LOCKED_TEST_MACRO_F1 = 0.5729561968146194  # phase7_final_test_results_3class.json, exact value

# Phase 3's 6-model cross-architecture 5-class ensemble (0.4984 raw 5-class) --
# the ACTUAL "prior best" cited throughout IMPROVEMENT_PLAN.md and
# CURRENT_FINAL_SYSTEM_REPORT.md. No per-example predictions were saved for
# this one, only its confusion matrix -- but a confusion matrix can be validly
# collapsed by summing rows/columns per the same class groups, with no
# retraining, exactly like the per-example collapse above.
PHASE3_ENSEMBLE_5CLASS_CONFUSION_MATRIX = [
    [15, 5, 6, 2, 4], [4, 29, 3, 1, 1], [3, 11, 38, 4, 8], [1, 2, 9, 8, 6], [8, 3, 6, 5, 14],
]


def aggregate_probs(probs_5, groups):
    return np.array([sum(probs_5[i] for i in group) for group in groups])


def collapse_confusion_matrix(cm5, groups):
    cm5 = np.array(cm5)
    cm3 = np.zeros((len(groups), len(groups)), dtype=int)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            cm3[i, j] = cm5[np.ix_(gi, gj)].sum()
    return cm3


def main():
    print(f"Reading existing, already-final 5-class test predictions from {FIVE_CLASS_TEST_PREDICTIONS_PATH}")
    print("(read-only -- this file is not modified; the original 5-class test result is untouched)\n")
    with open(FIVE_CLASS_TEST_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} test records (unchanged from the original Day-7/Phase-3 lock).")

    y_true_5 = np.array([r["y_true"] for r in records])
    y_true_3 = np.array([THREE_CLASS_MAP[y] for y in y_true_5])

    probs_3_all = np.array([aggregate_probs(np.array(r["probs"]), THREE_CLASS_GROUPS) for r in records])
    pred_3 = probs_3_all.argmax(axis=1)

    macro_f1 = f1_score(y_true_3, pred_3, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_3, pred_3, average="weighted", zero_division=0)
    acc = accuracy_score(y_true_3, pred_3)
    bal_acc = balanced_accuracy_score(y_true_3, pred_3)
    qwk = cohen_kappa_score(y_true_3, pred_3, weights="quadratic")
    report = classification_report(
        y_true_3, pred_3, labels=list(range(3)), target_names=THREE_CLASS_NAMES,
        zero_division=0, output_dict=True,
    )

    print("\n=== Old 5-class model's predictions, AGGREGATED into 3-class (no retraining), on the SAME 196 test records ===")
    print(f"macro-F1:          {macro_f1:.4f}")
    print(f"weighted-F1:       {weighted_f1:.4f}")
    print(f"accuracy:          {acc:.4f}")
    print(f"balanced accuracy: {bal_acc:.4f}")
    print(f"QWK:               {qwk:.4f}")
    print("\nPer-class:")
    for c in THREE_CLASS_NAMES:
        r = report[c]
        print(f"  {c}: precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1-score']:.4f} support={int(r['support'])}")

    valid_delta_vs_original_lock = NEW_3CLASS_LOCKED_TEST_MACRO_F1 - macro_f1
    print(f"\n=== Valid comparison 1: vs. the ORIGINAL single-model Day-7 lock ===")
    print(f"Old model, predictions aggregated to 3-class (no retraining): {macro_f1:.4f}")
    print(f"New 3-class model, trained natively + locked (Phase 7.10):    {NEW_3CLASS_LOCKED_TEST_MACRO_F1:.4f}")
    print(f"Valid delta:                                                  {valid_delta_vs_original_lock:+.4f}")

    # -------- the actual "prior best" cited throughout the documentation --------
    cm3_phase3 = collapse_confusion_matrix(PHASE3_ENSEMBLE_5CLASS_CONFUSION_MATRIX, THREE_CLASS_GROUPS)
    y_true_p3, y_pred_p3 = [], []
    for i in range(3):
        for j in range(3):
            y_true_p3 += [i] * int(cm3_phase3[i, j])
            y_pred_p3 += [j] * int(cm3_phase3[i, j])
    macro_f1_phase3 = f1_score(y_true_p3, y_pred_p3, average="macro", zero_division=0)
    weighted_f1_phase3 = f1_score(y_true_p3, y_pred_p3, average="weighted", zero_division=0)
    acc_phase3 = accuracy_score(y_true_p3, y_pred_p3)
    report_phase3 = classification_report(
        y_true_p3, y_pred_p3, labels=list(range(3)), target_names=THREE_CLASS_NAMES,
        zero_division=0, output_dict=True,
    )
    valid_delta_vs_phase3 = NEW_3CLASS_LOCKED_TEST_MACRO_F1 - macro_f1_phase3

    print(f"\n=== Valid comparison 2: vs. the PHASE-3 6-MODEL ENSEMBLE (the actual 'prior best' cited throughout the documentation) ===")
    print(f"Phase-3 5-class ensemble (0.4984 raw), confusion matrix collapsed to 3-class (no retraining): {macro_f1_phase3:.4f}")
    print(f"New 3-class model, trained natively + locked (Phase 7.10):                                   {NEW_3CLASS_LOCKED_TEST_MACRO_F1:.4f}")
    print(f"Valid delta:                                                                                  {valid_delta_vs_phase3:+.4f}")
    print("\n*** This is the important finding: against the actual prior-best baseline, the new 3-class ***")
    print("*** model is NOT an improvement once fairly compared -- it is slightly WORSE.               ***")

    invalid_delta_previously_reported = NEW_3CLASS_LOCKED_TEST_MACRO_F1 - 0.4984
    print(f"\nFor reference -- the INVALID comparison previously reported (raw 5-class-native 0.4984 vs raw 3-class-native): {invalid_delta_previously_reported:+.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": "Old 5-class models' saved test predictions/confusion-matrix aggregated into "
                    "3-class groupings, no retraining -- for VALID comparisons against the natively-"
                    "trained 3-class locked model, on the exact same 196 test records. "
                    "See PHASE8_MODALITY_AWARE_SYSTEM_PLAN.md Section 3.",
            "n_test": len(records),
            "comparison_1_original_day7_lock": {
                "old_aggregated_to_3class": {
                    "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
                    "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
                    "quadratic_weighted_kappa": float(qwk), "classification_report": report,
                },
                "new_3class_native_locked_macro_f1": NEW_3CLASS_LOCKED_TEST_MACRO_F1,
                "valid_delta": float(valid_delta_vs_original_lock),
            },
            "comparison_2_phase3_ensemble_ACTUAL_PRIOR_BEST": {
                "old_aggregated_to_3class": {
                    "macro_f1": float(macro_f1_phase3), "weighted_f1": float(weighted_f1_phase3),
                    "accuracy": float(acc_phase3), "classification_report": report_phase3,
                    "confusion_matrix_3class": cm3_phase3.tolist(),
                },
                "new_3class_native_locked_macro_f1": NEW_3CLASS_LOCKED_TEST_MACRO_F1,
                "valid_delta": float(valid_delta_vs_phase3),
                "finding": "Against the actual prior-best baseline cited throughout this project's "
                           "documentation, the new 3-class model is NOT an improvement once validly "
                           "compared -- it is slightly worse.",
            },
            "previously_reported_invalid_delta": float(invalid_delta_previously_reported),
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
