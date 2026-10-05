"""
Phase 4.1 follow-up: does the tuned gated component (dropout=0.4) improve
the full cross-architecture ensemble (gated + cross-attention, majority
vote of 6), not just the gated architecture standalone?
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from train_e6 import load_aligned_data, class_weights_from, NUM_CLASSES, CLASSES, SEEDS, DEVICE as GATED_DEVICE
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
from phase4_1_hyperparam_sweep import train_one as train_gated_tuned, DEFAULT
from phase3_2_cross_attention import load_data as load_data_xattn, train_one as train_xattn, DEVICE as XATTN_DEVICE

TUNED_CFG = {**DEFAULT, "dropout": 0.4}


def majority_vote(prob_list, num_classes):
    preds = np.stack([p.argmax(axis=1) for p in prob_list])
    return np.array([
        np.bincount(preds[:, i], minlength=num_classes).argmax()
        for i in range(preds.shape[1])
    ])


def main():
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
    combined_majority = majority_vote(all_probs, NUM_CLASSES)
    combined_f1 = f1_score(y_val, combined_majority, average="macro", zero_division=0)

    gated_majority = majority_vote(gated_probs, NUM_CLASSES)
    gated_f1 = f1_score(y_val, gated_majority, average="macro", zero_division=0)

    print(f"\nGated (tuned, dropout=0.4) alone: {gated_f1:.4f}")
    print(f"Cross-architecture ensemble (tuned gated + cross-attn): {combined_f1:.4f}")
    print(f"Comparison -- Phase 3 locked ensemble (untuned gated): 0.5592")
    print(f"Delta: {combined_f1 - 0.5592:+.4f}")

    report = classification_report(
        y_val, combined_majority, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
        zero_division=0, output_dict=True,
    )
    print("\nPer-class F1:")
    for c in CLASSES:
        print(f"  {c}: {report[c]['f1-score']:.4f}")


if __name__ == "__main__":
    main()
