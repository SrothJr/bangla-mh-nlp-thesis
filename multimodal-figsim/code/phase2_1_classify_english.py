"""
Phase 2, item 2.1: E6 gated+orth classifier on top of the English-aligned
representations, using Phase 1's kept configuration (ordinal label
smoothing + majority-vote ensembling across 3 seeds) so the comparison to
the Bangla pipeline isn't confounded by also changing the loss/ensembling
recipe -- only the text branch's language changes.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

from phase2_1_align_english import AlignmentProjectionsEnglish, CKPT_DIR as ALIGN_CKPT_DIR_EN
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU
from train_e6 import (
    E6GatedOrth, load_confidences, class_weights_from, NUM_CLASSES, CLASSES, SEEDS,
    MAX_EPOCHS, PATIENCE, LR, DEVICE, load_labels, load_split, load_embeddings,
    SIGLIP_DIR,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENGLISH_TEXT_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "mentalroberta_english")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase2_1_english_results.json")


def load_data_english():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    align_model = AlignmentProjectionsEnglish()
    ckpt = torch.load(os.path.join(ALIGN_CKPT_DIR_EN, "best.pt"), map_location=DEVICE)
    align_model.load_state_dict(ckpt["state_dict"])
    align_model.eval()
    print(f"Loaded English Stage A checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        raw_text = torch.tensor(load_embeddings(ENGLISH_TEXT_DIR, idx))
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


def to_tensors(split_data):
    return (
        torch.tensor(split_data["text"], device=DEVICE),
        torch.tensor(split_data["image"], device=DEVICE),
        torch.tensor(split_data["raw_text"], device=DEVICE),
        torch.tensor(split_data["conf"], device=DEVICE),
        torch.tensor(split_data["y"], device=DEVICE),
    )


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
    print("Loading English-aligned embeddings...")
    data = load_data_english()
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

    print(f"\n=== English text branch, majority-vote macro-F1: {majority_macro_f1:.4f} ===")
    print(f"Comparison -- Phase 1 best (Bangla pipeline, same recipe): 0.5330")
    print(f"Delta: {majority_macro_f1 - 0.5330:+.4f}")
    print(f"\nPer-class F1:")
    for c in CLASSES:
        print(f"  {c}: {report[c]['f1-score']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "seed_f1s": seed_f1s,
            "majority_vote_macro_f1": float(majority_macro_f1),
            "comparison_phase1_best_bangla": 0.5330,
            "delta": float(majority_macro_f1 - 0.5330),
            "classification_report": report,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
