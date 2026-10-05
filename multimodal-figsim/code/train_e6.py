"""
Day 5-6: E6 -- E4/E5 built on top of Stage A's contrastively-aligned
representations instead of raw text/image vectors (Section 8: E6 = E5 +
contrastive alignment; per direction from the user, tested primarily on E4
since E4 was the stronger, more stable Day-4 base, and secondarily on E5 to
test whether pre-alignment fixes the gate-collapse diagnosed after Day 4).

Loads the Stage A projection weights trained by train_contrastive_align.py
(the "current" checkpoint -- closest to peak validation retrieval before
early stopping intervened on val_loss plateauing) and freezes them: Stage A
is trained self-supervised on its own, once, then reused here exactly as
frozen input features -- consistent with Section 5.6's "Stage A alignment
projections: Trained (self-supervised)" being a separate stage from Stage
D's supervised classifier training.

Weighted CE only (CORAL dropped per Day-4 decision). 3 seeds.
"""
import os
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from train_contrastive_align import (
    AlignmentProjections, SHARED_DIM as ALIGN_SHARED_DIM, CKPT_DIR as ALIGN_CKPT_DIR,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
LEAKAGE_SAFE_DIR = os.path.join(FIGSIM_ROOT, "data", "leakage_safe")
INDEX_CSV = os.path.join(LEAKAGE_SAFE_DIR, "figsim_leakage_safe_index.csv")
SIGLIP_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip")
BANGLABERT_E3B_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_e3b")
OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "ocr_results.jsonl")
TRANSLATION_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "e6_results.json")
PRIOR_E4E5_PATH = os.path.join(PROJECT_ROOT, "outputs", "e4_e5_results.json")

CLASSES = [
    "None", "Wish to be dead", "Suicide ideation",
    "Suicide planning", "Suicide attempt or death",
]
NUM_CLASSES = 5
RAW_LABEL_TO_ID = {
    "None": 0, "Wish to be dead": 1, "Suicide ideation": 2,
    "Suicide planning": 3, "Suicide attempts": 4, "Suicide death": 4,
}
HIDDEN_DIM = 256
SEEDS = [0, 1, 2]
MAX_EPOCHS = 300
PATIENCE = 30
LR = 1e-3
DEVICE = "cpu"


def load_labels():
    labels = {}
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["include_in_leakage_safe_dataset"] == "True":
                labels[int(row["image_index"])] = RAW_LABEL_TO_ID[row["suicide_scale"]]
    return labels


def load_split(name):
    with open(os.path.join(LEAKAGE_SAFE_DIR, f"{name}.txt"), "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_embeddings(emb_dir, indices):
    return np.stack([np.load(os.path.join(emb_dir, f"{i}.npy")) for i in indices]).astype(np.float32)


def load_confidences(indices):
    ocr_conf = {}
    with open(OCR_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ocr_conf[r["image_index"]] = r["ocr_confidence"]
    trans_conf = {}
    with open(TRANSLATION_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            v = r["translation_confidence"]
            trans_conf[r["image_index"]] = v if v is not None else 0.0
    oc = np.array([ocr_conf[i] for i in indices], dtype=np.float32)
    tc = np.array([trans_conf[i] for i in indices], dtype=np.float32)
    return np.stack([oc, tc], axis=1)


def class_weights_from(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    weights = len(y_train) / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def load_aligned_data():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    align_model = AlignmentProjections()
    ckpt_path = os.path.join(ALIGN_CKPT_DIR, "current.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    align_model.load_state_dict(ckpt["state_dict"])
    align_model.eval()
    print(f"Loaded Stage A alignment checkpoint from epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}")

    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        raw_text = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, idx))
        raw_image = torch.tensor(load_embeddings(SIGLIP_DIR, idx))
        with torch.no_grad():
            aligned_text, aligned_image = align_model(raw_text, raw_image)
        data[split_name] = {
            "text": aligned_text.numpy(),       # (N, SHARED_DIM), L2-normalized
            "image": aligned_image.numpy(),     # (N, SHARED_DIM), L2-normalized
            "raw_text": raw_text.numpy(),        # kept for E5's "concat with raw text" step
            "conf": load_confidences(idx),
            "y": np.array([labels[i] for i in idx], dtype=np.int64),
        }

    conf_mean = data["train"]["conf"].mean(axis=0, keepdims=True)
    conf_std = data["train"]["conf"].std(axis=0, keepdims=True) + 1e-6
    for split_name in ("train", "val"):
        data[split_name]["conf"] = (data[split_name]["conf"] - conf_mean) / conf_std

    return data


# --------------------------------------------------------------- models ---

class E6Concat(nn.Module):
    """E4-style simple concatenation, but on ALIGNED text/image vectors."""

    def __init__(self, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ALIGN_SHARED_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, text_vec, image_vec, raw_text_vec, conf):
        z = torch.cat([text_vec, image_vec], dim=-1)
        return self.net(z)


class E6GatedOrth(nn.Module):
    """E5-style gated fusion + orthogonal feature, but the shared-dimension
    projection is now Stage A's pre-trained alignment (frozen), not a fresh
    supervised projection learned from scratch -- this is the actual test of
    whether pre-alignment fixes the E5 gate-collapse diagnosed after Day 4."""

    def __init__(self, num_outputs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(ALIGN_SHARED_DIM * 2 + 2, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )
        self.gated_norm = nn.LayerNorm(ALIGN_SHARED_DIM)
        self.orth_norm = nn.LayerNorm(ALIGN_SHARED_DIM)
        self.net = nn.Sequential(
            nn.Linear(ALIGN_SHARED_DIM * 2 + 768, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )
        self.last_alpha = None

    def forward(self, text_vec, image_vec, raw_text_vec, conf):
        t, i = text_vec, image_vec  # already aligned + frozen

        gate_in = torch.cat([t, i, conf], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in))
        self.last_alpha = alpha.detach()
        gated = alpha * t + (1 - alpha) * i

        t_norm_sq = (t * t).sum(dim=-1, keepdim=True) + 1e-6
        proj_coeff = (i * t).sum(dim=-1, keepdim=True) / t_norm_sq
        i_parallel = proj_coeff * t
        i_orth = i - i_parallel

        z = torch.cat([self.gated_norm(gated), self.orth_norm(i_orth), raw_text_vec], dim=-1)
        return self.net(z)


def train_one(model_cls, seed, data, class_weights):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_cls(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    X_text_tr = torch.tensor(data["train"]["text"], device=DEVICE)
    X_img_tr = torch.tensor(data["train"]["image"], device=DEVICE)
    X_rawtext_tr = torch.tensor(data["train"]["raw_text"], device=DEVICE)
    X_conf_tr = torch.tensor(data["train"]["conf"], device=DEVICE)
    y_tr = torch.tensor(data["train"]["y"], device=DEVICE)

    X_text_val = torch.tensor(data["val"]["text"], device=DEVICE)
    X_img_val = torch.tensor(data["val"]["image"], device=DEVICE)
    X_rawtext_val = torch.tensor(data["val"]["raw_text"], device=DEVICE)
    X_conf_val = torch.tensor(data["val"]["conf"], device=DEVICE)
    y_val = torch.tensor(data["val"]["y"], device=DEVICE)

    best_val_f1 = -1.0
    best_val_pred = None
    best_alpha = None
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
            pred_val = out_val.argmax(dim=1)
            val_f1 = f1_score(y_val.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_pred = pred_val.cpu().numpy().copy()
            best_alpha = model.last_alpha.cpu().numpy().copy() if getattr(model, "last_alpha", None) is not None else None
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                break

    return best_val_f1, best_val_pred, y_val.cpu().numpy(), best_alpha


def main():
    print("Loading aligned embeddings (Stage A frozen projections applied)...")
    data = load_aligned_data()
    class_weights = class_weights_from(data["train"]["y"])
    print(f"Train n={len(data['train']['y'])}, Val n={len(data['val']['y'])}")

    results = {}
    for name, model_cls in (("E6_concat", E6Concat), ("E6_gated_orth", E6GatedOrth)):
        print(f"\n--- {name}: {len(SEEDS)} seeds ---")
        seed_f1s = []
        seed_preds = []
        seed_alphas = []
        for seed in SEEDS:
            val_f1, val_pred, y_val, alpha = train_one(model_cls, seed, data, class_weights)
            seed_f1s.append(val_f1)
            seed_preds.append(val_pred)
            seed_alphas.append(alpha)
            print(f"  seed {seed}: macro-F1 = {val_f1:.4f}", flush=True)

        mean_f1 = float(np.mean(seed_f1s))
        std_f1 = float(np.std(seed_f1s))
        best_idx = int(np.argmax(seed_f1s))
        best_pred = seed_preds[best_idx]

        weighted_f1 = f1_score(y_val, best_pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_val, best_pred)
        cm = confusion_matrix(y_val, best_pred, labels=list(range(NUM_CLASSES)))
        report = classification_report(
            y_val, best_pred, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
            zero_division=0, output_dict=True,
        )

        print(f"  {name}: macro-F1 = {mean_f1:.4f} +/- {std_f1:.4f} "
              f"(seeds: {[round(f, 4) for f in seed_f1s]})")

        entry = {
            "seeds": SEEDS, "macro_f1_per_seed": seed_f1s,
            "macro_f1_mean": mean_f1, "macro_f1_std": std_f1,
            "best_seed_weighted_f1": float(weighted_f1), "best_seed_accuracy": float(acc),
            "best_seed_confusion_matrix": cm.tolist(),
            "best_seed_classification_report": report, "n_val": len(y_val),
        }
        if name == "E6_gated_orth" and seed_alphas[best_idx] is not None:
            alpha_arr = seed_alphas[best_idx].flatten()
            entry["gate_alpha_diagnostic"] = {
                "mean": float(alpha_arr.mean()), "std": float(alpha_arr.std()),
                "min": float(alpha_arr.min()), "max": float(alpha_arr.max()),
                "pct_below_0.1": float((alpha_arr < 0.1).mean() * 100),
                "pct_above_0.9": float((alpha_arr > 0.9).mean() * 100),
            }
            print(f"  gate alpha (best seed): mean={alpha_arr.mean():.4f} std={alpha_arr.std():.4f} "
                  f"min={alpha_arr.min():.4f} max={alpha_arr.max():.4f}")
        results[name] = entry

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")

    print("\n=== SUMMARY (macro-F1 mean +/- std, validation split) ===")
    if os.path.exists(PRIOR_E4E5_PATH):
        with open(PRIOR_E4E5_PATH, "r", encoding="utf-8") as f:
            prior = json.load(f)
        print(f"E4 (raw concat, weighted CE):      {prior['E4_ce']['macro_f1_mean']:.4f} +/- {prior['E4_ce']['macro_f1_std']:.4f}")
        print(f"E5 (raw gated+orth, weighted CE):  {prior['E5_ce']['macro_f1_mean']:.4f} +/- {prior['E5_ce']['macro_f1_std']:.4f}")
    for name, r in results.items():
        print(f"{name}: {r['macro_f1_mean']:.4f} +/- {r['macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()
