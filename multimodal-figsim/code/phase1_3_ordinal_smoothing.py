"""
Phase 1, item 1.3 (IMPROVEMENT_PLAN.md): ordinal label smoothing in place
of hard-target weighted cross-entropy.

Targets the specific, repeated finding that "Suicide planning" errors
concentrate on the adjacent class "Suicide ideation" in every experiment
since E1 -- an ordinal-adjacency problem. Instead of a one-hot target,
each sample's target is a distribution over all 5 classes that decays
exponentially with ordinal distance from the true label, so a prediction
of the adjacent class costs less loss than a prediction of a distant one
-- the same motivation as CORAL, without CORAL's diagnosed flaw (collapsing
to a single shared scalar rank score). This uses the full 5-way logit
output (no capacity bottleneck), only the *target* is softened, not the
model's output capacity.

Base architecture unchanged from the Phase-1.1-kept configuration:
E6GatedOrth, 2 confidence signals (OCR + translation; reasoning-certainty
was tried in 1.2 and not kept). Majority-vote ensembling across 3 seeds,
per 1.1.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from train_e6 import (
    E6GatedOrth, load_aligned_data, class_weights_from, NUM_CLASSES, CLASSES, SEEDS,
    MAX_EPOCHS, PATIENCE, LR, DEVICE,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase1_3_ordinal_smoothing_results.json")

TAU = 1.0  # decay rate for the ordinal-smoothing kernel; smaller = closer to one-hot


def build_smoothing_matrix(num_classes, tau):
    """(num_classes, num_classes) matrix; row i = target distribution when
    the true label is class i. weight(d) = exp(-d/tau), normalized to sum
    to 1 across the row."""
    classes = np.arange(num_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    for i in classes:
        dist = np.abs(classes - i)
        weights = np.exp(-dist / tau)
        matrix[i] = weights / weights.sum()
    return matrix


def to_tensors(split_data):
    return (
        torch.tensor(split_data["text"], device=DEVICE),
        torch.tensor(split_data["image"], device=DEVICE),
        torch.tensor(split_data["raw_text"], device=DEVICE),
        torch.tensor(split_data["conf"], device=DEVICE),
        torch.tensor(split_data["y"], device=DEVICE),
    )


def soft_target_loss(logits, y_true, smoothing_matrix, class_weights):
    """Cross-entropy against a soft (ordinal-smoothed) target distribution,
    with per-sample weighting by the TRUE class's weight (same
    inverse-frequency weights used for the hard-CE baseline, applied at the
    sample level since target is now a distribution, not an index)."""
    soft_targets = smoothing_matrix[y_true]  # (N, num_classes)
    log_probs = torch.log_softmax(logits, dim=1)
    per_sample_loss = -(soft_targets * log_probs).sum(dim=1)  # (N,)
    sample_weights = class_weights[y_true]
    return (per_sample_loss * sample_weights).mean()


def train_one(seed, data, class_weights, smoothing_matrix):
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
        loss = soft_target_loss(out, y_tr, smoothing_matrix, class_weights)
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
    smoothing_matrix = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=DEVICE)

    print(f"\nOrdinal-smoothing kernel (tau={TAU}), row = true class, cols = target weight per class:")
    for i, cls in enumerate(CLASSES):
        row = smoothing_matrix[i].cpu().numpy()
        print(f"  {cls:28s}: " + " ".join(f"{w:.3f}" for w in row))

    seed_f1s = []
    seed_probs = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_one(seed, data, class_weights, smoothing_matrix)
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

    print(f"\nMajority-vote macro-F1 (ordinal label smoothing): {majority_macro_f1:.4f}")
    print(f"Comparison -- Phase 1.1 majority-vote macro-F1 (hard-target weighted CE): 0.5311")
    print(f"Delta: {majority_macro_f1 - 0.5311:+.4f}")
    print(f"\n'Suicide planning' F1 (this run): {report['Suicide planning']['f1-score']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "tau": TAU,
            "seed_f1s": seed_f1s,
            "majority_vote_macro_f1": float(majority_macro_f1),
            "comparison_phase1_1_majority_vote": 0.5311,
            "delta": float(majority_macro_f1 - 0.5311),
            "classification_report": report,
        }, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
