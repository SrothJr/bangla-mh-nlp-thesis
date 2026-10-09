"""
Phase 7, Step 1 (PHASE7_REVISED_PRETRAINING_PLAN.md): free label-scheme
diagnostic. Aggregates the current best ensemble's OWN validation
predictions into a 3-class and a 4-class scheme (per
FIGSIM_HARMONIZATION_AND_PRETRAINING.md's harmonization table) to see
whether either grouping looks meaningfully better/more reliable, WITHOUT
committing to the large external-pretraining engineering effort.

No raw per-example validation probabilities had ever been saved to disk
by any prior script (only aggregate metrics were persisted, and the only
saved per-example predictions are on the LOCKED TEST SET, which must not
be used for label-scheme development per this project's protocol). So
this script re-runs the current best recipe once -- 3 tuned-gated seeds +
3 cross-attention seeds, exactly matching Phase 4.4's ensemble -- and
this time saves the per-example vote fractions, which is what actually
gets aggregated below. This is a small, cheap re-run of an already-
proven pipeline (frozen cached embeddings, same seeds), NOT the large
external-meme-pretraining phase.

Aggregation uses vote fractions (fraction of the 6 models voting for
each class), the same "probability" signal already used for Phase 4.4's
calibration -- consistent with this project's finding that majority
voting outperforms soft-probability averaging at every comparison so far.
"""
import os
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from train_e6 import load_aligned_data, class_weights_from, NUM_CLASSES, CLASSES, SEEDS, DEVICE as GATED_DEVICE
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
from phase4_1_hyperparam_sweep import train_one as train_gated_tuned, DEFAULT
from phase3_2_cross_attention import load_data as load_data_xattn, train_one as train_xattn, DEVICE as XATTN_DEVICE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase7_1_label_harmonization_diagnostic_results.json")

TUNED_CFG = {**DEFAULT, "dropout": 0.4}

# Harmonization mapping, from FIGSIM_HARMONIZATION_AND_PRETRAINING.md
# 5-class indices: 0 None, 1 Wish, 2 Ideation, 3 Planning, 4 Attempt/Death
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
THREE_CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
THREE_CLASS_GROUPS = [[0], [1, 2], [3, 4]]  # which 5-class indices sum into each 3-class bucket

FOUR_CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3}
FOUR_CLASS_NAMES = ["No expressed severity", "Passive death wish", "Active ideation or planning", "Suicide behavior or outcome"]
FOUR_CLASS_GROUPS = [[0], [1], [2, 3], [4]]


def majority_vote_fractions(prob_list, num_classes):
    preds = np.stack([p.argmax(axis=1) for p in prob_list])
    n = preds.shape[1]
    fractions = np.zeros((n, num_classes))
    for c in range(num_classes):
        fractions[:, c] = (preds == c).sum(axis=0) / preds.shape[0]
    return fractions


def aggregate_fractions(fractions_5class, groups):
    """Sum 5-class vote fractions into fewer buckets, per `groups`."""
    n = fractions_5class.shape[0]
    out = np.zeros((n, len(groups)))
    for new_idx, old_indices in enumerate(groups):
        out[:, new_idx] = fractions_5class[:, old_indices].sum(axis=1)
    return out


def evaluate_scheme(y_true_mapped, pred, names, scheme_label):
    macro_f1 = f1_score(y_true_mapped, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_mapped, pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true_mapped, pred)
    bal_acc = balanced_accuracy_score(y_true_mapped, pred)
    cm = confusion_matrix(y_true_mapped, pred, labels=list(range(len(names))))
    report = classification_report(
        y_true_mapped, pred, labels=list(range(len(names))), target_names=names,
        zero_division=0, output_dict=True,
    )
    print(f"\n=== {scheme_label} ===")
    print(f"macro-F1={macro_f1:.4f}  weighted-F1={weighted_f1:.4f}  "
          f"accuracy={acc:.4f}  balanced_accuracy={bal_acc:.4f}")
    for c in names:
        print(f"  {c}: F1={report[c]['f1-score']:.4f}  support={int(report[c]['support'])}")
    return {
        "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm.tolist(), "classification_report": report,
    }


def main():
    print("Re-running the current best ensemble recipe (3 tuned-gated + 3 cross-attention "
          "seeds) once, to capture per-example validation vote fractions -- this has never "
          "been saved to disk before. Cheap, frozen-embedding, same recipe as Phase 4.4.\n")

    gated_data = load_aligned_data()
    gated_class_weights = class_weights_from(gated_data["train"]["y"])
    smoothing_matrix_gated = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=GATED_DEVICE)

    gated_probs = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_gated_tuned(seed, gated_data, gated_class_weights, smoothing_matrix_gated, TUNED_CFG)
        gated_probs.append(probs)
        print(f"gated (tuned) seed {seed}: val macro-F1 = {val_f1:.4f}")

    xattn_data = load_data_xattn()
    xattn_class_weights = class_weights_from(xattn_data["train"]["y"]).to(XATTN_DEVICE)
    smoothing_matrix_xattn = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=XATTN_DEVICE)

    xattn_probs = []
    for seed in SEEDS:
        val_f1, probs, _ = train_xattn(seed, xattn_data, xattn_class_weights, smoothing_matrix_xattn)
        xattn_probs.append(probs)
        print(f"cross-attn seed {seed}: val macro-F1 = {val_f1:.4f}")

    all_probs = gated_probs + xattn_probs
    fractions_5class = majority_vote_fractions(all_probs, NUM_CLASSES)

    # --- sanity check: 5-class result should closely match the known Phase 4.1 ensemble number
    pred_5class = fractions_5class.argmax(axis=1)
    five_class_result = evaluate_scheme(y_val, pred_5class, CLASSES, "5-class (sanity check vs. known 0.5638)")

    # --- 3-class harmonized diagnostic
    y_val_3class = np.array([THREE_CLASS_MAP[y] for y in y_val])
    fractions_3class = aggregate_fractions(fractions_5class, THREE_CLASS_GROUPS)
    pred_3class = fractions_3class.argmax(axis=1)
    three_class_result = evaluate_scheme(y_val_3class, pred_3class, THREE_CLASS_NAMES, "3-class harmonized (diagnostic)")

    # --- 4-class harmonized diagnostic (merges the ideation/planning boundary)
    y_val_4class = np.array([FOUR_CLASS_MAP[y] for y in y_val])
    fractions_4class = aggregate_fractions(fractions_5class, FOUR_CLASS_GROUPS)
    pred_4class = fractions_4class.argmax(axis=1)
    four_class_result = evaluate_scheme(y_val_4class, pred_4class, FOUR_CLASS_NAMES, "4-class harmonized (diagnostic, merges ideation+planning)")

    print(f"\n=== Summary ===")
    print(f"5-class macro-F1: {five_class_result['macro_f1']:.4f}")
    print(f"3-class macro-F1: {three_class_result['macro_f1']:.4f}")
    print(f"4-class macro-F1: {four_class_result['macro_f1']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_val": len(y_val),
            "five_class": five_class_result,
            "three_class": three_class_result,
            "four_class": four_class_result,
            "note": "Diagnostic only -- aggregated from the existing 5-class ensemble's own "
                    "predictions, no retraining on any harmonized label scheme. Per "
                    "PHASE7_REVISED_PRETRAINING_PLAN.md Step 1.",
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
