"""
Phase 3, item 3.1 -- step A: establish a fair frozen-encoder baseline
using simple concatenation (E4-style, no Stage A alignment needed), with
Phase 1's kept recipe (ordinal label smoothing + majority-vote ensembling),
on CACHED embeddings. This is the number partial encoder unfreezing needs
to beat -- unfreezing will require raw-input, mini-batched, end-to-end
training (a materially different training loop), so an apples-to-apples
frozen baseline under that same simple-concat architecture is needed
first, rather than comparing against the more complex aligned-gated E6
number.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

from train_e6 import (
    load_confidences, class_weights_from, NUM_CLASSES, CLASSES, SEEDS,
    MAX_EPOCHS, PATIENCE, LR, DEVICE, load_labels, load_split, load_embeddings,
    SIGLIP_DIR, BANGLABERT_E3B_DIR, HIDDEN_DIM,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase3_1_baseline_concat_frozen_results.json")

TEXT_DIM = 768
IMAGE_DIM = 1152


class ConcatClassifier(nn.Module):
    """Same shape as E4Concat in train_e4_e5.py -- raw (unaligned)
    concatenation of text and image vectors."""

    def __init__(self, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_DIM + IMAGE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, text_vec, image_vec):
        return self.net(torch.cat([text_vec, image_vec], dim=-1))


def load_data():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")
    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        data[split_name] = {
            "text": load_embeddings(BANGLABERT_E3B_DIR, idx),
            "image": load_embeddings(SIGLIP_DIR, idx),
            "y": np.array([labels[i] for i in idx], dtype=np.int64),
        }
    return data


def train_one(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ConcatClassifier(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    X_text_tr = torch.tensor(data["train"]["text"], device=DEVICE)
    X_img_tr = torch.tensor(data["train"]["image"], device=DEVICE)
    y_tr = torch.tensor(data["train"]["y"], device=DEVICE)
    X_text_val = torch.tensor(data["val"]["text"], device=DEVICE)
    X_img_val = torch.tensor(data["val"]["image"], device=DEVICE)
    y_val = torch.tensor(data["val"]["y"], device=DEVICE)

    best_val_f1 = -1.0
    best_val_probs = None
    epochs_without_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_text_tr, X_img_tr)
        loss = soft_target_loss(out, y_tr, smoothing_matrix, class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_text_val, X_img_val)
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
    data = load_data()
    class_weights = class_weights_from(data["train"]["y"])
    smoothing_matrix = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=DEVICE)

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

    print(f"\n=== Frozen-encoder concat baseline, majority-vote macro-F1: {majority_macro_f1:.4f} ===")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "seed_f1s": seed_f1s,
            "majority_vote_macro_f1": float(majority_macro_f1),
            "classification_report": report,
        }, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
