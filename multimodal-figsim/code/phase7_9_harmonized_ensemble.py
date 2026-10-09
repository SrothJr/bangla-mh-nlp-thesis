"""
Phase 7, item 7.9: cross-architecture ensembling on the 3-class
harmonized target -- the one proven-biggest lever from the original
project (Phase 3.2: gated+orth 0.5330 alone + cross-attention 0.5205
alone -> 0.5592 ensembled, on 5-class) never tried for 3-class.

Trains THREE architectures on 3-class for the first time together:
  - simple concat (current best standalone, 0.6425)
  - gated+orth, tuned specifically for 3-class (hidden_dim=512,
    Phase 7.8's best config, 0.6330 standalone)
  - cross-attention (NEW -- never tested on 3-class at all before this)

Then reports every majority-vote combination (pairs and the full triple)
to see whether ensembling recovers or exceeds simple concat's standalone
lead, the same way it did for 5-class.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

from train_e6 import (
    load_aligned_data, load_labels, load_split, load_embeddings,
    BANGLABERT_E3B_DIR, SIGLIP_DIR, HIDDEN_DIM, SEEDS, DEVICE as GATED_DEVICE,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU
import phase4_1_hyperparam_sweep as sweep_module
from phase4_1_hyperparam_sweep import train_one as train_gated_tuned
import phase3_2_cross_attention as xattn_module
from phase3_2_cross_attention import load_data as load_data_xattn, train_one as train_xattn, DEVICE as XATTN_DEVICE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase7_9_harmonized_ensemble_results.json")

NUM_CLASSES_3 = 3
THREE_CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
BEST_3CLASS_CFG = {"hidden_dim": 512, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4}  # Phase 7.8 winner
REF_SIMPLE_CONCAT = 0.6425
REF_GATED_TUNED_3CLASS = 0.6330


def to_3class(data):
    for split in ("train", "val"):
        data[split]["y"] = np.array([THREE_CLASS_MAP[y] for y in data[split]["y"]], dtype=np.int64)
    return data


def class_weights_3(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES_3).astype(np.float32)
    weights = len(y_train) / (NUM_CLASSES_3 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


class ConcatClassifier3Class(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768 + 1152, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2), nn.Linear(HIDDEN_DIM, NUM_CLASSES_3),
        )

    def forward(self, text_vec, image_vec):
        return self.net(torch.cat([text_vec, image_vec], dim=-1))


def train_concat_3class(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConcatClassifier3Class()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X_text_tr = torch.tensor(data["train"]["text"])
    X_img_tr = torch.tensor(data["train"]["image"])
    y_tr = torch.tensor(data["train"]["y"])
    X_text_val = torch.tensor(data["val"]["text"])
    X_img_val = torch.tensor(data["val"]["image"])
    y_val = torch.tensor(data["val"]["y"])

    best_val_f1 = -1.0
    best_probs = None
    epochs_without_improve = 0
    for epoch in range(300):
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
            val_f1 = f1_score(y_val.numpy(), pred_val.numpy(), average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_probs = probs_val.detach().numpy().copy()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= 30:
                break
    return best_val_f1, best_probs, y_val.numpy()


def majority_vote(prob_list, num_classes):
    preds = np.stack([p.argmax(axis=1) for p in prob_list])
    return np.array([
        np.bincount(preds[:, i], minlength=num_classes).argmax()
        for i in range(preds.shape[1])
    ])


def main():
    sweep_module.NUM_CLASSES = NUM_CLASSES_3
    xattn_module.NUM_CLASSES = NUM_CLASSES_3

    print("=== Training simple concat, 3-class (reference) ===")
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")
    concat_data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        y5 = np.array([labels[i] for i in idx], dtype=np.int64)
        y3 = np.array([THREE_CLASS_MAP[y] for y in y5], dtype=np.int64)
        concat_data[split_name] = {
            "text": load_embeddings(BANGLABERT_E3B_DIR, idx),
            "image": load_embeddings(SIGLIP_DIR, idx),
            "y": y3,
        }
    concat_class_weights = class_weights_3(concat_data["train"]["y"])
    concat_smoothing = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device="cpu")
    concat_probs = []
    y_val_concat = None
    for seed in SEEDS:
        f1, probs, y_val_concat = train_concat_3class(seed, concat_data, concat_class_weights, concat_smoothing)
        concat_probs.append(probs)
        print(f"  concat seed {seed}: val macro-F1 = {f1:.4f}")

    print("\n=== Training gated+orth, 3-class (Phase 7.8 best config, hidden_dim=512) ===")
    gated_data = load_aligned_data()
    gated_data = to_3class(gated_data)
    gated_class_weights = class_weights_3(gated_data["train"]["y"])
    gated_smoothing = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=GATED_DEVICE)
    gated_probs = []
    y_val_gated = None
    for seed in SEEDS:
        f1, probs, y_val_gated = train_gated_tuned(seed, gated_data, gated_class_weights, gated_smoothing, BEST_3CLASS_CFG)
        gated_probs.append(probs)
        print(f"  gated seed {seed}: val macro-F1 = {f1:.4f}")

    print("\n=== Training cross-attention, 3-class (NEW -- never tested before) ===")
    xattn_data = load_data_xattn()
    for split in ("train", "val"):
        xattn_data[split]["y"] = np.array([THREE_CLASS_MAP[y] for y in xattn_data[split]["y"]], dtype=np.int64)
    xattn_class_weights = class_weights_3(xattn_data["train"]["y"]).to(XATTN_DEVICE)
    xattn_smoothing = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=XATTN_DEVICE)
    xattn_probs = []
    y_val_xattn = None
    for seed in SEEDS:
        f1, probs, y_val_xattn = train_xattn(seed, xattn_data, xattn_class_weights, xattn_smoothing)
        xattn_probs.append(probs)
        print(f"  cross-attn seed {seed}: val macro-F1 = {f1:.4f}")

    assert np.array_equal(y_val_concat, y_val_gated) and np.array_equal(y_val_gated, y_val_xattn)
    y_val = y_val_concat

    combos = {
        "concat_alone": concat_probs,
        "gated_alone": gated_probs,
        "xattn_alone": xattn_probs,
        "concat+gated": concat_probs + gated_probs,
        "concat+xattn": concat_probs + xattn_probs,
        "gated+xattn": gated_probs + xattn_probs,
        "concat+gated+xattn": concat_probs + gated_probs + xattn_probs,
    }

    print("\n=== Majority-vote combinations ===")
    results = {}
    for name, probs_list in combos.items():
        pred = majority_vote(probs_list, NUM_CLASSES_3)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        report = classification_report(
            y_val, pred, labels=list(range(NUM_CLASSES_3)), target_names=THREE_CLASS_NAMES,
            zero_division=0, output_dict=True,
        )
        results[name] = {"macro_f1": float(f1), "n_models": len(probs_list), "classification_report": report}
        print(f"  {name} ({len(probs_list)} models): macro-F1 = {f1:.4f}")

    best_combo = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\nBest combination: {best_combo} ({results[best_combo]['macro_f1']:.4f})")
    print(f"Comparison -- simple concat alone (previous best): {REF_SIMPLE_CONCAT}")
    print(f"Delta: {results[best_combo]['macro_f1'] - REF_SIMPLE_CONCAT:+.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "best_combination": best_combo,
            "best_macro_f1": results[best_combo]["macro_f1"],
            "comparison_simple_concat_alone": REF_SIMPLE_CONCAT,
            "delta_vs_simple_concat": results[best_combo]["macro_f1"] - REF_SIMPLE_CONCAT,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
