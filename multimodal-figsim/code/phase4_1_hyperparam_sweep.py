"""
Phase 4, item 4.1 (IMPROVEMENT_PLAN.md): systematic hyperparameter sweep
of the E6 gated+orth classifier head -- hidden_dim, dropout, learning
rate, weight_decay have all used one reasonable-guess value (256, 0.2,
1e-3, 1e-4) for every experiment in this entire project, never swept.

One-at-a-time coordinate search from the current default (not a full
grid, which would be 50+ configs) -- cheap, and sufficient to check
whether the default was already reasonable or leaving something on the
table. Each config trains 3 seeds, majority-vote macro-F1 on validation,
same recipe (ordinal label smoothing) as the current best.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from train_e6 import (
    load_aligned_data, class_weights_from, NUM_CLASSES, SEEDS,
    MAX_EPOCHS, PATIENCE, DEVICE, ALIGN_SHARED_DIM,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase4_1_hyperparam_sweep_results.json")

DEFAULT = {"hidden_dim": 256, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4}

SWEEP_CONFIGS = [
    ("default", DEFAULT),
    ("hidden_dim=128", {**DEFAULT, "hidden_dim": 128}),
    ("hidden_dim=512", {**DEFAULT, "hidden_dim": 512}),
    ("dropout=0.1", {**DEFAULT, "dropout": 0.1}),
    ("dropout=0.4", {**DEFAULT, "dropout": 0.4}),
    ("lr=5e-4", {**DEFAULT, "lr": 5e-4}),
    ("lr=2e-3", {**DEFAULT, "lr": 2e-3}),
    ("weight_decay=1e-3", {**DEFAULT, "weight_decay": 1e-3}),
]


class E6GatedOrthTunable(nn.Module):
    def __init__(self, num_outputs, hidden_dim, dropout):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(ALIGN_SHARED_DIM * 2 + 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gated_norm = nn.LayerNorm(ALIGN_SHARED_DIM)
        self.orth_norm = nn.LayerNorm(ALIGN_SHARED_DIM)
        self.net = nn.Sequential(
            nn.Linear(ALIGN_SHARED_DIM * 2 + 768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, text_vec, image_vec, raw_text_vec, conf):
        t, i = text_vec, image_vec
        gate_in = torch.cat([t, i, conf], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in))
        gated = alpha * t + (1 - alpha) * i
        t_norm_sq = (t * t).sum(dim=-1, keepdim=True) + 1e-6
        proj_coeff = (i * t).sum(dim=-1, keepdim=True) / t_norm_sq
        i_orth = i - proj_coeff * t
        z = torch.cat([self.gated_norm(gated), self.orth_norm(i_orth), raw_text_vec], dim=-1)
        return self.net(z)


def to_tensors(split_data):
    return (
        torch.tensor(split_data["text"], device=DEVICE),
        torch.tensor(split_data["image"], device=DEVICE),
        torch.tensor(split_data["raw_text"], device=DEVICE),
        torch.tensor(split_data["conf"], device=DEVICE),
        torch.tensor(split_data["y"], device=DEVICE),
    )


def train_one(seed, data, class_weights, smoothing_matrix, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = E6GatedOrthTunable(NUM_CLASSES, cfg["hidden_dim"], cfg["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

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


def run_config(name, cfg, data, class_weights, smoothing_matrix):
    seed_f1s = []
    seed_probs = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_one(seed, data, class_weights, smoothing_matrix, cfg)
        seed_f1s.append(val_f1)
        seed_probs.append(probs)

    seed_preds = np.stack([p.argmax(axis=1) for p in seed_probs])
    majority_pred = np.array([
        np.bincount(seed_preds[:, i], minlength=NUM_CLASSES).argmax()
        for i in range(seed_preds.shape[1])
    ])
    majority_macro_f1 = f1_score(y_val, majority_pred, average="macro", zero_division=0)
    print(f"{name}: seeds={[round(f,4) for f in seed_f1s]}  majority-vote macro-F1={majority_macro_f1:.4f}")
    return majority_macro_f1


def main():
    data = load_aligned_data()
    class_weights = class_weights_from(data["train"]["y"])
    smoothing_matrix = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=DEVICE)

    results = {}
    for name, cfg in SWEEP_CONFIGS:
        f1 = run_config(name, cfg, data, class_weights, smoothing_matrix)
        results[name] = {"config": cfg, "majority_vote_macro_f1": float(f1)}

    best_name = max(results, key=lambda k: results[k]["majority_vote_macro_f1"])
    print(f"\nBest config: {best_name} ({results[best_name]['majority_vote_macro_f1']:.4f})")
    print(f"Default config: {results['default']['majority_vote_macro_f1']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
