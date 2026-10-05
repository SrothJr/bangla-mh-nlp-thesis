"""
Phase 1, item 1.1 (IMPROVEMENT_PLAN.md): ensemble the 3 seeds' predicted
probabilities instead of reporting only the single best-by-validation seed.

Reuses E6GatedOrth and the aligned-embedding data loading from train_e6.py
unmodified -- this script only adds probability capture and averaging on
top, it does not change how any individual seed is trained. Validation
split only, per the improvement round's protocol note (test stays locked
until a new final configuration is chosen).
"""
import os
import json
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from train_e6 import (
    E6GatedOrth, load_aligned_data, class_weights_from, NUM_CLASSES, CLASSES, SEEDS,
    MAX_EPOCHS, PATIENCE, LR, DEVICE,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase1_ensemble_results.json")


def to_tensors(split_data):
    return (
        torch.tensor(split_data["text"], device=DEVICE),
        torch.tensor(split_data["image"], device=DEVICE),
        torch.tensor(split_data["raw_text"], device=DEVICE),
        torch.tensor(split_data["conf"], device=DEVICE),
        torch.tensor(split_data["y"], device=DEVICE),
    )


def train_one_capture_probs(seed, data, class_weights):
    """Same training loop as train_e6.train_one, but also captures the
    predicted probability matrix at the best validation epoch, needed for
    soft-probability ensembling (argmax predictions alone aren't enough)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = E6GatedOrth(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr, y_tr = to_tensors(data["train"])
    X_text_val, X_img_val, X_rawtext_val, X_conf_val, y_val = to_tensors(data["val"])

    best_val_f1 = -1.0
    best_val_probs = None
    epochs_without_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr)
        loss = nn.functional.cross_entropy(out, y_tr, weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_text_val, X_img_val, X_rawtext_val, X_conf_val)
            probs_val = torch.softmax(out_val, dim=1)
            pred_val = probs_val.argmax(dim=1)
            val_f1 = f1_score(y_val.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_probs = probs_val.cpu().numpy().copy()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                break

    return best_val_f1, best_val_probs, y_val.cpu().numpy()


def main():
    print("Loading aligned embeddings (same as the locked E6 gated+orth baseline)...")
    data = load_aligned_data()
    class_weights = class_weights_from(data["train"]["y"])

    seed_f1s = []
    seed_probs = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_one_capture_probs(seed, data, class_weights)
        seed_f1s.append(val_f1)
        seed_probs.append(probs)
        print(f"seed {seed}: val macro-F1 = {val_f1:.4f} (individual)")

    best_single_f1 = max(seed_f1s)
    print(f"\nBest single seed (this is the Day 5-6/7 baseline number): {best_single_f1:.4f}")

    # ---- Ensemble: average predicted probabilities across all 3 seeds ----
    avg_probs = np.mean(seed_probs, axis=0)
    ensemble_pred = avg_probs.argmax(axis=1)
    ensemble_macro_f1 = f1_score(y_val, ensemble_pred, average="macro", zero_division=0)
    ensemble_weighted_f1 = f1_score(y_val, ensemble_pred, average="weighted", zero_division=0)
    ensemble_acc = accuracy_score(y_val, ensemble_pred)

    print(f"\n=== Ensemble (average of 3 seeds' softmax probabilities) ===")
    print(f"macro-F1:    {ensemble_macro_f1:.4f}")
    print(f"weighted-F1: {ensemble_weighted_f1:.4f}")
    print(f"accuracy:    {ensemble_acc:.4f}")

    delta = ensemble_macro_f1 - best_single_f1
    print(f"\nDelta vs. best single seed: {delta:+.4f}")

    # Also try majority vote as a second ensembling strategy, for comparison
    seed_preds = np.stack([p.argmax(axis=1) for p in seed_probs])  # (3, n_val)
    majority_pred = np.array([
        np.bincount(seed_preds[:, i], minlength=NUM_CLASSES).argmax()
        for i in range(seed_preds.shape[1])
    ])
    majority_macro_f1 = f1_score(y_val, majority_pred, average="macro", zero_division=0)
    print(f"Majority vote macro-F1: {majority_macro_f1:.4f} (for comparison)")

    report = classification_report(
        y_val, ensemble_pred, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_val, ensemble_pred, labels=list(range(NUM_CLASSES)))

    results = {
        "seed_f1s": seed_f1s,
        "best_single_seed_f1": float(best_single_f1),
        "ensemble_soft_avg": {
            "macro_f1": float(ensemble_macro_f1),
            "weighted_f1": float(ensemble_weighted_f1),
            "accuracy": float(ensemble_acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        },
        "ensemble_majority_vote": {"macro_f1": float(majority_macro_f1)},
        "delta_vs_best_single_seed": float(delta),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
