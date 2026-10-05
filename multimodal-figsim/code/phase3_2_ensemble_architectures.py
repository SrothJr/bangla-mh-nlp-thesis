"""
Phase 3, item 3.2 follow-up: the cross-attention and E6-gated+orth
architectures showed complementary per-class strengths (cross-attention:
much stronger on Suicide planning/ideation; E6 gated+orth: much stronger
on Wish to be dead). This tests whether ensembling predictions ACROSS the
two different architectures (not just across seeds of the same one, as in
Phase 1.1) captures the best of both -- the same logic as Phase 4's
planned "ensemble the trained classifier with a differently-mechanismed
model" idea (4.3), applied here since two different trained architectures
are already available.
"""
import os
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from train_e6 import E6GatedOrth, load_aligned_data, class_weights_from, NUM_CLASSES, CLASSES, SEEDS, DEVICE as GATED_DEVICE
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU, train_one as train_gated_ordinal
from phase3_2_cross_attention import load_data as load_data_xattn, train_one as train_xattn, DEVICE as XATTN_DEVICE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase3_2_ensemble_results.json")


def majority_vote(prob_list, num_classes):
    preds = np.stack([p.argmax(axis=1) for p in prob_list])
    return np.array([
        np.bincount(preds[:, i], minlength=num_classes).argmax()
        for i in range(preds.shape[1])
    ])


def main():
    print("=== Training E6 gated+orth (ordinal smoothing), 3 seeds ===")
    gated_data = load_aligned_data()
    gated_class_weights = class_weights_from(gated_data["train"]["y"])
    smoothing_matrix_gated = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=GATED_DEVICE)

    gated_probs = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_gated_ordinal(seed, gated_data, gated_class_weights, smoothing_matrix_gated)
        gated_probs.append(probs)
        print(f"  gated seed {seed}: val macro-F1 = {val_f1:.4f}")

    print("\n=== Training cross-attention fusion, 3 seeds ===")
    xattn_data = load_data_xattn()
    xattn_class_weights = class_weights_from(xattn_data["train"]["y"]).to(XATTN_DEVICE)
    smoothing_matrix_xattn = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=XATTN_DEVICE)

    xattn_probs = []
    for seed in SEEDS:
        val_f1, probs, _ = train_xattn(seed, xattn_data, xattn_class_weights, smoothing_matrix_xattn)
        xattn_probs.append(probs)
        print(f"  cross-attn seed {seed}: val macro-F1 = {val_f1:.4f}")

    gated_majority = majority_vote(gated_probs, NUM_CLASSES)
    xattn_majority = majority_vote(xattn_probs, NUM_CLASSES)
    gated_f1 = f1_score(y_val, gated_majority, average="macro", zero_division=0)
    xattn_f1 = f1_score(y_val, xattn_majority, average="macro", zero_division=0)

    # Cross-architecture ensemble: all 6 seed-predictions (3 gated + 3 cross-attn) vote together
    all_probs = gated_probs + xattn_probs
    combined_majority = majority_vote(all_probs, NUM_CLASSES)
    combined_f1 = f1_score(y_val, combined_majority, average="macro", zero_division=0)

    # Also try soft-averaging all 6 probability distributions
    avg_probs = np.mean(all_probs, axis=0)
    avg_pred = avg_probs.argmax(axis=1)
    avg_f1 = f1_score(y_val, avg_pred, average="macro", zero_division=0)

    print(f"\n=== Results ===")
    print(f"E6 gated+orth alone (majority of 3):     {gated_f1:.4f}")
    print(f"Cross-attention alone (majority of 3):   {xattn_f1:.4f}")
    print(f"Cross-architecture majority vote (6):    {combined_f1:.4f}")
    print(f"Cross-architecture soft average (6):     {avg_f1:.4f}")

    report = classification_report(
        y_val, combined_majority, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
        zero_division=0, output_dict=True,
    )
    print(f"\nPer-class F1 (cross-architecture majority vote):")
    for c in CLASSES:
        print(f"  {c}: {report[c]['f1-score']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "gated_alone_macro_f1": float(gated_f1),
            "xattn_alone_macro_f1": float(xattn_f1),
            "cross_arch_majority_vote_macro_f1": float(combined_f1),
            "cross_arch_soft_average_macro_f1": float(avg_f1),
            "classification_report": report,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
