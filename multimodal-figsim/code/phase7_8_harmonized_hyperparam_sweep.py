"""
Phase 7, item 7.8: re-run Phase 4.1's hyperparameter sweep for the
E6 gated+orth classifier head, but on the 3-class harmonized target
instead of 5-class. Phase 7.5 applied dropout=0.4 (the value found best
FOR 5-CLASS) as-is to 3-class without re-checking whether it's actually
the right setting for this coarser target -- this fills that gap before
concluding gated+orth can't compete with simple concat (0.6425) on
3-class.

Same one-at-a-time coordinate search as Phase 4.1, same architecture
(E6GatedOrthTunable, existing unchanged Stage A alignment), only the
label target and NUM_CLASSES differ.
"""
import os
import json

import numpy as np
import torch

from train_e6 import load_aligned_data, SEEDS, DEVICE, ALIGN_SHARED_DIM
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
import phase4_1_hyperparam_sweep as sweep_module
from phase4_1_hyperparam_sweep import SWEEP_CONFIGS, run_config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase7_8_harmonized_hyperparam_sweep_results.json")

NUM_CLASSES_3 = 3
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
REF_SIMPLE_CONCAT_3CLASS = 0.6425
REF_GATED_DEFAULT_3CLASS = 0.6144  # Phase 7.5, dropout=0.4 (5-class-tuned value, applied as-is)


def to_3class(data):
    for split in ("train", "val"):
        data[split]["y"] = np.array([THREE_CLASS_MAP[y] for y in data[split]["y"]], dtype=np.int64)
    return data


def class_weights_3(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES_3).astype(np.float32)
    weights = len(y_train) / (NUM_CLASSES_3 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def main():
    # Patch the sweep module's NUM_CLASSES so E6GatedOrthTunable(NUM_CLASSES, ...)
    # builds a 3-way head instead of 5-way (same technique as Phase 7.5).
    sweep_module.NUM_CLASSES = NUM_CLASSES_3

    print("Loading aligned data (existing Stage A alignment, unchanged) and remapping to 3-class...")
    data = load_aligned_data()
    data = to_3class(data)
    print(f"Train n={len(data['train']['y'])}, Val n={len(data['val']['y'])}")

    class_weights = class_weights_3(data["train"]["y"])
    smoothing_matrix = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=DEVICE)

    results = {}
    for name, cfg in SWEEP_CONFIGS:
        f1 = run_config(name, cfg, data, class_weights, smoothing_matrix)
        results[name] = {"config": cfg, "majority_vote_macro_f1": float(f1)}

    best_name = max(results, key=lambda k: results[k]["majority_vote_macro_f1"])
    best_f1 = results[best_name]["majority_vote_macro_f1"]
    print(f"\nBest config: {best_name} ({best_f1:.4f})")
    print(f"Default config (hidden_dim=256, dropout=0.2): {results['default']['majority_vote_macro_f1']:.4f}")
    print(f"\nComparison -- simple concat, 3-class (Phase 7 Step 4, current best): {REF_SIMPLE_CONCAT_3CLASS}")
    print(f"Comparison -- gated+orth, 3-class, 5-class-tuned dropout=0.4 (Phase 7.5): {REF_GATED_DEFAULT_3CLASS}")
    print(f"Delta of best-tuned-for-3-class vs simple concat: {best_f1 - REF_SIMPLE_CONCAT_3CLASS:+.4f}")
    print(f"Delta of best-tuned-for-3-class vs Phase 7.5's untuned-for-3-class gated: {best_f1 - REF_GATED_DEFAULT_3CLASS:+.4f}")

    decision = "gated+orth CAN compete with simple concat once properly tuned for 3-class" \
        if best_f1 >= REF_SIMPLE_CONCAT_3CLASS else \
        "gated+orth still does not beat simple concat, even after tuning for 3-class specifically"
    print(f"\nDecision: {decision}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "sweep_results": results,
            "best_config_name": best_name,
            "best_config_macro_f1": best_f1,
            "comparison_simple_concat_3class": REF_SIMPLE_CONCAT_3CLASS,
            "comparison_gated_5class_tuned_applied_asis": REF_GATED_DEFAULT_3CLASS,
            "delta_vs_simple_concat": best_f1 - REF_SIMPLE_CONCAT_3CLASS,
            "delta_vs_phase7_5": best_f1 - REF_GATED_DEFAULT_3CLASS,
            "decision": decision,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
