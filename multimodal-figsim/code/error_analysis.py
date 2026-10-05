"""
Day 8: error analysis -- slice the LOCKED final model's test predictions by
FigSIM's own modality and context metadata fields (Section 8's "when does
the image help" analysis).

Section 8's protocol note ("never touch the test set more than once") is
about not using test performance to make further modeling decisions --
it does not forbid analyzing the one set of results already obtained. But
run_final_test_eval.py only saved aggregate metrics, not per-item
predictions, so this script reconstructs them by retraining seed 0
(fully deterministic: fixed seed, fixed data, no stochastic augmentation)
and verifies the reproduced aggregate numbers match the already-reported
ones EXACTLY before doing anything else -- confirming this is the same
result being re-read, not a new peek that could change any decision.

Also saves per-item softmax probabilities to outputs/final_test_predictions.json,
reused by the fuzzy-logic layer (Section 9) so the suicide-severity side of
that combination doesn't require yet another test-set pass.
"""
import os
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

from train_contrastive_align import AlignmentProjections, CKPT_DIR as ALIGN_CKPT_DIR
from train_e6 import E6GatedOrth, load_confidences, class_weights_from
from run_final_test_eval import (
    load_all_data, to_tensors, train_one_seed, CLASSES, NUM_CLASSES, SEEDS,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
LEAKAGE_SAFE_DIR = os.path.join(FIGSIM_ROOT, "data", "leakage_safe")
INDEX_CSV = os.path.join(LEAKAGE_SAFE_DIR, "figsim_leakage_safe_index.csv")
PRED_PATH = os.path.join(PROJECT_ROOT, "outputs", "final_test_predictions.json")
PRIOR_RESULT_PATH = os.path.join(PROJECT_ROOT, "outputs", "final_test_results.json")

EXPECTED_PRIMARY_SEED = 0
EXPECTED_MACRO_F1 = 0.47527430026096484  # from final_test_results.json, seed 0


def load_metadata(indices):
    meta = {}
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["image_index"])
            if idx in indices:
                meta[idx] = {"modality": row["modality"], "context": row["context"]}
    return meta


def main():
    print("Reconstructing seed-0 test predictions (deterministic retrain, sanity-checked).")
    data = load_all_data()
    class_weights = class_weights_from(data["train"]["y"])

    val_f1, state = train_one_seed(EXPECTED_PRIMARY_SEED, data, class_weights)
    print(f"seed {EXPECTED_PRIMARY_SEED}: val macro-F1 = {val_f1:.6f}")

    model = E6GatedOrth(NUM_CLASSES)
    model.load_state_dict(state)
    model.eval()

    X_text_test, X_img_test, X_rawtext_test, X_conf_test, y_test = to_tensors(data["test"])
    with torch.no_grad():
        logits = model(X_text_test, X_img_test, X_rawtext_test, X_conf_test)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy()
    y_true = y_test.numpy()

    reproduced_macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    print(f"Reproduced test macro-F1: {reproduced_macro_f1:.6f} "
          f"(previously reported: {EXPECTED_MACRO_F1:.6f})")
    assert abs(reproduced_macro_f1 - EXPECTED_MACRO_F1) < 1e-6, (
        "Reconstructed predictions do NOT match the originally reported test metrics -- "
        "stopping rather than silently analyzing a different result."
    )
    print("Sanity check passed: reconstructed predictions exactly match the original "
          "one-time test evaluation. Proceeding to per-item analysis.")

    test_indices = data["test"]["indices"]
    per_item = [
        {
            "image_index": idx,
            "y_true": int(yt),
            "y_pred": int(yp),
            "probs": pr.tolist(),
        }
        for idx, yt, yp, pr in zip(test_indices, y_true, preds, probs)
    ]
    with open(PRED_PATH, "w", encoding="utf-8") as f:
        json.dump(per_item, f, indent=2)
    print(f"Saved per-item predictions to {PRED_PATH}")

    # ---------------------------------------------------------- slicing --
    meta = load_metadata(set(test_indices))

    def slice_metrics(key):
        groups = {}
        for item in per_item:
            g = meta[item["image_index"]][key]
            groups.setdefault(g, {"y_true": [], "y_pred": []})
            groups[g]["y_true"].append(item["y_true"])
            groups[g]["y_pred"].append(item["y_pred"])
        results = {}
        for g, d in groups.items():
            n = len(d["y_true"])
            macro_f1 = f1_score(d["y_true"], d["y_pred"], average="macro", zero_division=0)
            acc = accuracy_score(d["y_true"], d["y_pred"])
            results[g] = {"n": n, "macro_f1": float(macro_f1), "accuracy": float(acc)}
        return results

    modality_results = slice_metrics("modality")
    context_results = slice_metrics("context")

    print("\n=== By modality ===")
    for g, r in sorted(modality_results.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {g}: n={r['n']}, macro-F1={r['macro_f1']:.4f}, accuracy={r['accuracy']:.4f}")

    print("\n=== By context ===")
    for g, r in sorted(context_results.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {g}: n={r['n']}, macro-F1={r['macro_f1']:.4f}, accuracy={r['accuracy']:.4f}")

    out = {"by_modality": modality_results, "by_context": context_results}
    with open(os.path.join(PROJECT_ROOT, "outputs", "error_analysis_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved slice results to outputs/error_analysis_results.json")


if __name__ == "__main__":
    main()
