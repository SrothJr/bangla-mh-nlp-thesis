"""
Phase 2, item 2.3: E6 gated+orth classifier on top of the SUPERVISED-
contrastive-aligned representations (phase2_3_align_supervised.py),
using Phase 1's kept configuration (ordinal label smoothing + majority
vote), same comparison basis as 2.1 and 2.2.
"""
import os
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from phase2_3_align_supervised import AlignmentProjectionsSupervised, CKPT_DIR as ALIGN_CKPT_DIR_SUP
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
from phase2_2_mixup_augmentation import to_tensors  # reuse tensor conversion helper
from train_e6 import (
    load_confidences, class_weights_from, NUM_CLASSES, CLASSES, SEEDS,
    DEVICE, load_labels, load_split, load_embeddings, SIGLIP_DIR, BANGLABERT_E3B_DIR,
)
from phase1_3_ordinal_smoothing import train_one as train_one_ordinal  # exact same training loop

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase2_3_supervised_align_results.json")


def load_data_supervised_aligned():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    align_model = AlignmentProjectionsSupervised()
    ckpt = torch.load(os.path.join(ALIGN_CKPT_DIR_SUP, "best.pt"), map_location=DEVICE)
    align_model.load_state_dict(ckpt["state_dict"])
    align_model.eval()
    print(f"Loaded supervised-contrastive Stage A checkpoint from epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}")

    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        raw_text = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, idx))
        raw_image = torch.tensor(load_embeddings(SIGLIP_DIR, idx))
        with torch.no_grad():
            aligned_text, aligned_image = align_model(raw_text, raw_image)
        data[split_name] = {
            "text": aligned_text.numpy(),
            "image": aligned_image.numpy(),
            "raw_text": raw_text.numpy(),
            "conf": load_confidences(idx),
            "y": np.array([labels[i] for i in idx], dtype=np.int64),
        }

    conf_mean = data["train"]["conf"].mean(axis=0, keepdims=True)
    conf_std = data["train"]["conf"].std(axis=0, keepdims=True) + 1e-6
    for split_name in ("train", "val"):
        data[split_name]["conf"] = (data[split_name]["conf"] - conf_mean) / conf_std

    return data


def main():
    print("Loading supervised-contrastive-aligned embeddings...")
    data = load_data_supervised_aligned()
    class_weights = class_weights_from(data["train"]["y"])
    smoothing_matrix = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=DEVICE)

    seed_f1s = []
    seed_probs = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_one_ordinal(seed, data, class_weights, smoothing_matrix)
        seed_f1s.append(val_f1)
        seed_probs.append(probs)
        print(f"seed {seed}: val macro-F1 = {val_f1:.4f}")

    seed_preds = np.stack([p.argmax(axis=1) for p in seed_probs])
    majority_pred = np.array([
        np.bincount(seed_preds[:, i], minlength=NUM_CLASSES).argmax()
        for i in range(seed_preds.shape[1])
    ])
    majority_macro_f1 = f1_score(y_val, majority_pred, average="macro", zero_division=0)
    report = classification_report(
        y_val, majority_pred, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
        zero_division=0, output_dict=True,
    )

    print(f"\n=== Supervised-contrastive alignment, majority-vote macro-F1: {majority_macro_f1:.4f} ===")
    print(f"Comparison -- Phase 1 best (self-supervised alignment): 0.5330")
    print(f"Delta: {majority_macro_f1 - 0.5330:+.4f}")
    print(f"\nPer-class F1:")
    for c in CLASSES:
        print(f"  {c}: {report[c]['f1-score']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "seed_f1s": seed_f1s,
            "majority_vote_macro_f1": float(majority_macro_f1),
            "comparison_phase1_best": 0.5330,
            "delta": float(majority_macro_f1 - 0.5330),
            "classification_report": report,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
