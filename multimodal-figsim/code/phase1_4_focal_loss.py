"""
Phase 1, item 1.4 (IMPROVEMENT_PLAN.md): focal loss in place of plain
weighted cross-entropy, to concentrate gradient on hard/misclassified
examples rather than only reweighting by class frequency.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), alpha_t = the same
inverse-frequency class weight already used for weighted CE, gamma = 2.0
(the standard default from Lin et al. 2017, RetinaNet).

Same base architecture as the Phase-1.1-kept config (E6GatedOrth, 2
confidence signals), majority-vote across 3 seeds, tested standalone
against the hard-CE baseline before any combination with 1.3.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

from train_e6 import (
    E6GatedOrth, load_aligned_data, class_weights_from, NUM_CLASSES, CLASSES, SEEDS,
    MAX_EPOCHS, PATIENCE, LR, DEVICE,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase1_4_focal_loss_results.json")

GAMMA = 1.0


def to_tensors(split_data):
    return (
        torch.tensor(split_data["text"], device=DEVICE),
        torch.tensor(split_data["image"], device=DEVICE),
        torch.tensor(split_data["raw_text"], device=DEVICE),
        torch.tensor(split_data["conf"], device=DEVICE),
        torch.tensor(split_data["y"], device=DEVICE),
    )


def focal_loss(logits, y_true, class_weights, gamma):
    log_probs = torch.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    log_p_t = log_probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
    p_t = probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
    alpha_t = class_weights[y_true]
    loss = -alpha_t * (1 - p_t).pow(gamma) * log_p_t
    return loss.mean()


def train_one(seed, data, class_weights):
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
        loss = focal_loss(out, y_tr, class_weights, GAMMA)
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
        val_f1, probs, y_val = train_one(seed, data, class_weights)
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

    print(f"\nMajority-vote macro-F1 (focal loss, gamma={GAMMA}): {majority_macro_f1:.4f}")
    print(f"Comparison -- Phase 1.1 majority-vote macro-F1 (hard weighted CE): 0.5311")
    print(f"Comparison -- Phase 1.3 majority-vote macro-F1 (ordinal smoothing): 0.5330")
    print(f"Delta vs 1.1: {majority_macro_f1 - 0.5311:+.4f}")
    print(f"\nPer-class F1:")
    for c in CLASSES:
        print(f"  {c}: {report[c]['f1-score']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "gamma": GAMMA,
            "seed_f1s": seed_f1s,
            "majority_vote_macro_f1": float(majority_macro_f1),
            "comparison_phase1_1_majority_vote": 0.5311,
            "comparison_phase1_3_majority_vote": 0.5330,
            "delta_vs_phase1_1": float(majority_macro_f1 - 0.5311),
            "classification_report": report,
        }, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
