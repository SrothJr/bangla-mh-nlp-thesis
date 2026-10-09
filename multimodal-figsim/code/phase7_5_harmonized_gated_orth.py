"""
Phase 7, item 7.5 (IMPROVEMENT_PLAN.md): the current best fusion
architecture (E6 gated+orth, tuned dropout=0.4, EXISTING Stage A
alignment -- unchanged) applied to the 3-class harmonized FigSIM target,
using the ORIGINAL frozen SigLIP embeddings (the BN-HIB-pretrained
variant was already tested and rejected in Step 4).

Closes a gap: Step 4 only tested simple concat (deliberately, for a
clean single-variable comparison of the pretraining question) and never
checked whether the architecture already proven better on the 5-class
task (gated+orth beat concat throughout Phases 1-6) also does better on
the 3-class target. Near-zero risk -- reuses proven, unchanged
components (Stage A alignment, E6GatedOrthTunable, ordinal smoothing,
multi-seed majority vote) with only the label target changed.
"""
import os
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report

from train_e6 import load_aligned_data, SEEDS, DEVICE
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
from phase4_1_hyperparam_sweep import train_one as train_gated_tuned, DEFAULT

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase7_5_harmonized_gated_orth_results.json")

TUNED_CFG = {**DEFAULT, "dropout": 0.4}
NUM_CLASSES_3 = 3
THREE_CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}

# Reference points, from prior phases (see IMPROVEMENT_PLAN.md)
REF_SIMPLE_CONCAT_3CLASS = 0.6425      # Phase 7 Step 4, variant A
REF_GATED_TUNED_5CLASS = 0.5418        # Phase 4.1, tuned gated alone


def to_3class(data):
    for split in ("train", "val"):
        data[split]["y"] = np.array([THREE_CLASS_MAP[y] for y in data[split]["y"]], dtype=np.int64)
    return data


def class_weights_3(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES_3).astype(np.float32)
    weights = len(y_train) / (NUM_CLASSES_3 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def majority_vote(preds_list, num_classes):
    preds = np.stack(preds_list)
    return np.array([
        np.bincount(preds[:, i], minlength=num_classes).argmax()
        for i in range(preds.shape[1])
    ])


def main():
    print("Loading aligned data (existing Stage A alignment, unchanged) and remapping to 3-class...")
    data = load_aligned_data()
    data = to_3class(data)
    print(f"Train n={len(data['train']['y'])}, Val n={len(data['val']['y'])}")
    print(f"Train class distribution: {np.bincount(data['train']['y'], minlength=NUM_CLASSES_3)}")

    class_weights = class_weights_3(data["train"]["y"])
    smoothing_matrix = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=DEVICE)

    # phase4_1_hyperparam_sweep.train_one is hardcoded to train_e6.NUM_CLASSES (5) internally
    # via E6GatedOrthTunable(NUM_CLASSES, ...) -- patch it locally for this 3-class run.
    import phase4_1_hyperparam_sweep as sweep_module
    sweep_module.NUM_CLASSES = NUM_CLASSES_3

    seed_f1s = []
    seed_preds = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_gated_tuned(seed, data, class_weights, smoothing_matrix, TUNED_CFG)
        pred = probs.argmax(axis=1)
        seed_f1s.append(val_f1)
        seed_preds.append(pred)
        print(f"seed {seed}: val macro-F1 = {val_f1:.4f}")

    majority_pred = majority_vote(seed_preds, NUM_CLASSES_3)
    majority_f1 = f1_score(y_val, majority_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, majority_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_val, majority_pred)
    bal_acc = balanced_accuracy_score(y_val, majority_pred)
    report = classification_report(
        y_val, majority_pred, labels=list(range(NUM_CLASSES_3)), target_names=THREE_CLASS_NAMES,
        zero_division=0, output_dict=True,
    )

    print(f"\n=== Gated+orth (tuned), 3-class harmonized: majority-vote macro-F1 = {majority_f1:.4f} ===")
    print(f"weighted-F1={weighted_f1:.4f}  accuracy={acc:.4f}  balanced_accuracy={bal_acc:.4f}")
    print(f"\nComparison -- simple concat, 3-class (Phase 7 Step 4): {REF_SIMPLE_CONCAT_3CLASS}")
    print(f"Delta vs simple concat: {majority_f1 - REF_SIMPLE_CONCAT_3CLASS:+.4f}")
    print(f"Comparison -- gated+orth (tuned), 5-class (Phase 4.1): {REF_GATED_TUNED_5CLASS}")
    print("\nPer-class F1:")
    for c in THREE_CLASS_NAMES:
        print(f"  {c}: {report[c]['f1-score']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "seed_f1s": seed_f1s,
            "majority_vote_macro_f1": float(majority_f1),
            "weighted_f1": float(weighted_f1),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "classification_report": report,
            "comparison_simple_concat_3class": REF_SIMPLE_CONCAT_3CLASS,
            "delta_vs_simple_concat": float(majority_f1 - REF_SIMPLE_CONCAT_3CLASS),
            "comparison_gated_tuned_5class": REF_GATED_TUNED_5CLASS,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
