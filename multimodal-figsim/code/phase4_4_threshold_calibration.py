"""
Phase 4, item 4.4 (IMPROVEMENT_PLAN.md): post-hoc per-class decision
calibration on top of the current best ensemble (Phase 4.1's tuned gated
+ cross-attention, 6 models, majority vote). No retraining -- this only
re-weights the existing 6-model vote before taking argmax, targeting two
patterns diagnosed repeatedly across this whole project:
  - "Wish to be dead": consistently over-predicted (high recall, weak
    precision every time it's been measured) -- a fallback guess.
  - "Suicide planning": consistently under-predicted into "Suicide
    ideation" -- the single most repeated failure pattern in this project.

Base signal: vote_fraction[c] = (number of the 6 models voting for class
c) / 6 -- preserves the majority-vote combination method (already shown
to beat soft probability averaging) rather than replacing it; calibration
multiplies this vote fraction by a per-class weight before the final
argmax, so ties can also be broken in a principled direction.

Search: coordinate ascent over per-class multipliers (cheap -- each
evaluation is instant, no training involved), several passes over all 5
classes, since a full grid (5 classes, ~5-7 values e.g.) would be
thousands of combinations for no real benefit over cycling once
convergence is reached.
"""
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from train_e6 import load_aligned_data, class_weights_from, NUM_CLASSES, CLASSES, SEEDS, DEVICE as GATED_DEVICE
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
from phase4_1_hyperparam_sweep import train_one as train_gated_tuned, DEFAULT
from phase3_2_cross_attention import load_data as load_data_xattn, train_one as train_xattn, DEVICE as XATTN_DEVICE

TUNED_CFG = {**DEFAULT, "dropout": 0.4}
CANDIDATE_MULTIPLIERS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]


def get_vote_fractions(all_probs, num_classes):
    preds = np.stack([p.argmax(axis=1) for p in all_probs])  # (6, n)
    n_models = preds.shape[0]
    n = preds.shape[1]
    fractions = np.zeros((n, num_classes))
    for c in range(num_classes):
        fractions[:, c] = (preds == c).sum(axis=0) / n_models
    return fractions


def coordinate_ascent(vote_fractions, y_true, num_classes, n_passes=3):
    weights = np.ones(num_classes)
    best_f1 = f1_score(y_true, (vote_fractions * weights).argmax(axis=1), average="macro", zero_division=0)
    print(f"Starting macro-F1 (all weights=1.0, i.e. plain majority vote): {best_f1:.4f}")

    for p in range(n_passes):
        improved = False
        for c in range(num_classes):
            best_w_for_c = weights[c]
            for w in CANDIDATE_MULTIPLIERS:
                trial = weights.copy()
                trial[c] = w
                pred = (vote_fractions * trial).argmax(axis=1)
                f1 = f1_score(y_true, pred, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_w_for_c = w
                    improved = True
            weights[c] = best_w_for_c
        print(f"pass {p}: weights={np.round(weights, 2).tolist()} macro-F1={best_f1:.4f}")
        if not improved:
            break
    return weights, best_f1


def main():
    print("Retraining the 6 models (tuned gated + cross-attention) to get validation vote fractions...")
    gated_data = load_aligned_data()
    gated_class_weights = class_weights_from(gated_data["train"]["y"])
    smoothing_matrix_gated = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=GATED_DEVICE)

    gated_probs = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_gated_tuned(seed, gated_data, gated_class_weights, smoothing_matrix_gated, TUNED_CFG)
        gated_probs.append(probs)

    xattn_data = load_data_xattn()
    xattn_class_weights = class_weights_from(xattn_data["train"]["y"]).to(XATTN_DEVICE)
    smoothing_matrix_xattn = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=XATTN_DEVICE)

    xattn_probs = []
    for seed in SEEDS:
        val_f1, probs, _ = train_xattn(seed, xattn_data, xattn_class_weights, smoothing_matrix_xattn)
        xattn_probs.append(probs)

    all_probs = gated_probs + xattn_probs
    vote_fractions = get_vote_fractions(all_probs, NUM_CLASSES)

    print("\nRunning coordinate-ascent calibration search on validation...")
    weights, calibrated_f1 = coordinate_ascent(vote_fractions, y_val, NUM_CLASSES)

    uncalibrated_pred = vote_fractions.argmax(axis=1)
    uncalibrated_f1 = f1_score(y_val, uncalibrated_pred, average="macro", zero_division=0)
    calibrated_pred = (vote_fractions * weights).argmax(axis=1)

    print(f"\n=== Uncalibrated majority vote: {uncalibrated_f1:.4f} ===")
    print(f"=== Calibrated (weights={dict(zip(CLASSES, np.round(weights,2).tolist()))}): {calibrated_f1:.4f} ===")
    print(f"Delta: {calibrated_f1 - uncalibrated_f1:+.4f}")

    report_before = classification_report(
        y_val, uncalibrated_pred, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
        zero_division=0, output_dict=True,
    )
    report_after = classification_report(
        y_val, calibrated_pred, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
        zero_division=0, output_dict=True,
    )
    print("\nPer-class F1 before -> after:")
    for c in CLASSES:
        print(f"  {c}: {report_before[c]['f1-score']:.4f} -> {report_after[c]['f1-score']:.4f}")

    with open("../outputs/phase4_4_calibration_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "weights": dict(zip(CLASSES, weights.tolist())),
            "uncalibrated_macro_f1": float(uncalibrated_f1),
            "calibrated_macro_f1": float(calibrated_f1),
            "delta": float(calibrated_f1 - uncalibrated_f1),
            "report_before": report_before,
            "report_after": report_after,
        }, f, indent=2)
    print("\nSaved results to outputs/phase4_4_calibration_results.json")


if __name__ == "__main__":
    main()
