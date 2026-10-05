"""
Phase 5, item 5.1 (IMPROVEMENT_PLAN.md): a rigor upgrade, not a new
modeling idea. Every decision in Phases 1-4 was made against ONE fixed
195-item validation split -- seed-to-seed noise observed throughout the
project has been roughly +/-0.01-0.03 macro-F1, which is large enough
that some "this is slightly worse, reject it" calls could plausibly have
gone the other way on a different validation sample. This script re-
evaluates the current locked Phase 4.4 pipeline (tuned gated+orth +
cross-attention ensemble, majority vote, per-class calibration weights)
under 5-fold stratified cross-validation over train+val combined (777
examples), so we get a variance estimate around the single-split result
of 0.5695 instead of a single point value. The test set (196 examples,
touched exactly twice so far in this project) is NOT touched here.

Simplification, noted explicitly: Stage A's contrastive alignment
projections are frozen and shared across all 5 folds (not retrained per
fold). This is consistent with the project's existing design -- Stage A
is trained once, self-supervised, on its own schedule, and reused
everywhere else as a frozen input feature (see train_e6.py's docstring)
-- and retraining it 5x would be a materially larger undertaking outside
this item's rigor-check scope. Per-fold, only the downstream 6 classifier
heads (3 gated-tuned seeds + 3 cross-attention seeds) are retrained from
scratch on that fold's training portion; confidence-feature normalization
statistics are also recomputed per fold from that fold's train subset
only, to avoid any cross-fold leakage.
"""
import os
import json

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from train_e6 import (
    load_labels, load_split, load_embeddings, load_confidences, class_weights_from,
    NUM_CLASSES, CLASSES, SIGLIP_DIR, BANGLABERT_E3B_DIR, DEVICE as GATED_DEVICE,
)
from train_contrastive_align import AlignmentProjections, CKPT_DIR as ALIGN_CKPT_DIR
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
from phase4_1_hyperparam_sweep import train_one as train_gated_tuned, DEFAULT
from phase3_2_cross_attention import load_patches, load_tokens, train_one as train_xattn, DEVICE as XATTN_DEVICE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase5_1_kfold_cv_results.json")

TUNED_CFG = {**DEFAULT, "dropout": 0.4}
SEEDS = [0, 1, 2]
N_FOLDS = 5
CV_RANDOM_STATE = 42
# Phase 4.4 locked per-class calibration weights, in CLASSES order.
CALIBRATION_WEIGHTS = np.array([0.7, 1.1, 1.0, 1.0, 1.0])
SINGLE_SPLIT_LOCKED_RESULT = 0.5695


def load_combined_indices():
    return load_split("train") + load_split("val")


def build_gated_pool(combined_idx):
    align_model = AlignmentProjections()
    ckpt = torch.load(os.path.join(ALIGN_CKPT_DIR, "current.pt"), map_location="cpu")
    align_model.load_state_dict(ckpt["state_dict"])
    align_model.eval()
    raw_text = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, combined_idx))
    raw_image = torch.tensor(load_embeddings(SIGLIP_DIR, combined_idx))
    with torch.no_grad():
        aligned_text, aligned_image = align_model(raw_text, raw_image)
    labels = load_labels()
    return {
        "text": aligned_text.numpy(),
        "image": aligned_image.numpy(),
        "raw_text": raw_text.numpy(),
        "conf_raw": load_confidences(combined_idx),
        "y": np.array([labels[i] for i in combined_idx], dtype=np.int64),
    }


def build_xattn_pool(combined_idx):
    labels = load_labels()
    tokens, masks = load_tokens(combined_idx)
    return {
        "tokens": tokens,
        "mask": masks,
        "patches": load_patches(combined_idx),
        "raw_text": load_embeddings(BANGLABERT_E3B_DIR, combined_idx),
        "y": np.array([labels[i] for i in combined_idx], dtype=np.int64),
    }


def slice_gated(pool, idx):
    return {k: pool[k][idx] for k in ("text", "image", "raw_text", "conf_raw", "y")}


def slice_xattn(pool, idx):
    return {k: pool[k][idx] for k in ("tokens", "mask", "patches", "raw_text", "y")}


def make_gated_fold(pool, train_pos, val_pos):
    fold = {"train": slice_gated(pool, train_pos), "val": slice_gated(pool, val_pos)}
    mean = fold["train"]["conf_raw"].mean(axis=0, keepdims=True)
    std = fold["train"]["conf_raw"].std(axis=0, keepdims=True) + 1e-6
    for split in ("train", "val"):
        fold[split]["conf"] = (fold[split]["conf_raw"] - mean) / std
    return fold


def majority_vote_fractions(prob_list, num_classes):
    preds = np.stack([p.argmax(axis=1) for p in prob_list])
    n = preds.shape[1]
    fractions = np.zeros((n, num_classes))
    for c in range(num_classes):
        fractions[:, c] = (preds == c).sum(axis=0) / preds.shape[0]
    return fractions


def main():
    combined_idx = load_combined_indices()
    print(f"Combined pool (train+val): {len(combined_idx)} examples. Test set NOT touched.")

    print("Building gated-pipeline feature pool (Stage A alignment applied once, frozen)...")
    gated_pool = build_gated_pool(combined_idx)
    print("Building cross-attention-pipeline feature pool (token/patch level, frozen encoders)...")
    xattn_pool = build_xattn_pool(combined_idx)

    y_all = gated_pool["y"]
    assert np.array_equal(y_all, xattn_pool["y"]), "label mismatch between the two feature pools"

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_results = []

    for fold_i, (train_pos, val_pos) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
        print(f"\n=== Fold {fold_i + 1}/{N_FOLDS}: train n={len(train_pos)}, held-out n={len(val_pos)} ===")

        gated_fold = make_gated_fold(gated_pool, train_pos, val_pos)
        gated_class_weights = class_weights_from(gated_fold["train"]["y"])
        smoothing_matrix_gated = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=GATED_DEVICE)

        gated_probs = []
        y_val_fold = None
        for seed in SEEDS:
            val_f1, probs, y_val_fold = train_gated_tuned(
                seed, gated_fold, gated_class_weights, smoothing_matrix_gated, TUNED_CFG
            )
            gated_probs.append(probs)
            print(f"  gated seed {seed}: fold macro-F1 = {val_f1:.4f}")

        xattn_fold = {"train": slice_xattn(xattn_pool, train_pos), "val": slice_xattn(xattn_pool, val_pos)}
        xattn_class_weights = class_weights_from(xattn_fold["train"]["y"]).to(XATTN_DEVICE)
        smoothing_matrix_xattn = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=XATTN_DEVICE)

        xattn_probs = []
        for seed in SEEDS:
            val_f1, probs, _ = train_xattn(seed, xattn_fold, xattn_class_weights, smoothing_matrix_xattn)
            xattn_probs.append(probs)
            print(f"  cross-attn seed {seed}: fold macro-F1 = {val_f1:.4f}")

        all_probs = gated_probs + xattn_probs
        vote_fractions = majority_vote_fractions(all_probs, NUM_CLASSES)

        uncalibrated_pred = vote_fractions.argmax(axis=1)
        uncalibrated_f1 = f1_score(y_val_fold, uncalibrated_pred, average="macro", zero_division=0)

        calibrated_pred = (vote_fractions * CALIBRATION_WEIGHTS).argmax(axis=1)
        calibrated_f1 = f1_score(y_val_fold, calibrated_pred, average="macro", zero_division=0)

        print(f"Fold {fold_i + 1} result: uncalibrated={uncalibrated_f1:.4f}  calibrated={calibrated_f1:.4f}")
        fold_results.append({
            "fold": fold_i,
            "n_train": len(train_pos),
            "n_val": len(val_pos),
            "uncalibrated_macro_f1": float(uncalibrated_f1),
            "calibrated_macro_f1": float(calibrated_f1),
        })

    uncal = [r["uncalibrated_macro_f1"] for r in fold_results]
    cal = [r["calibrated_macro_f1"] for r in fold_results]

    print("\n=== 5-fold CV summary (train+val combined, test untouched) ===")
    print(f"Uncalibrated ensemble: mean={np.mean(uncal):.4f} std={np.std(uncal):.4f}  "
          f"per-fold={[round(v, 4) for v in uncal]}")
    print(f"Calibrated ensemble:   mean={np.mean(cal):.4f} std={np.std(cal):.4f}  "
          f"per-fold={[round(v, 4) for v in cal]}")
    print(f"Single fixed-split locked result (Phase 4.4): {SINGLE_SPLIT_LOCKED_RESULT}")
    print(f"Single-split value vs. 5-fold mean, delta: {SINGLE_SPLIT_LOCKED_RESULT - np.mean(cal):+.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_folds": N_FOLDS,
            "random_state": CV_RANDOM_STATE,
            "fold_results": fold_results,
            "uncalibrated_mean": float(np.mean(uncal)),
            "uncalibrated_std": float(np.std(uncal)),
            "calibrated_mean": float(np.mean(cal)),
            "calibrated_std": float(np.std(cal)),
            "single_split_locked_result": SINGLE_SPLIT_LOCKED_RESULT,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
