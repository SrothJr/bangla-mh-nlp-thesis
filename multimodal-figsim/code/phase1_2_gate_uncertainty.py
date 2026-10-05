"""
Phase 1, item 1.2 (IMPROVEMENT_PLAN.md): feed the reasoning model's own
`uncertain` flags (cause_effect/figurative_meaning/emotional_state, from
reasoning_results.jsonl) into the gate as a third confidence signal,
alongside the existing OCR confidence and OCR-translation confidence.

Caveat found before running: the signal is sparse -- 918/973 memes (94%)
have zero uncertain flags, only 55 have any. Testing anyway since it's
free and directly responds to the Day-4 spot-check finding that this flag
is a real (if imperfectly triggered) signal.

Validation split only, per the improvement round's protocol note. Uses
majority-vote ensembling across 3 seeds (Phase 1.1's kept change) as the
comparison basis, not best-single-seed.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from train_contrastive_align import AlignmentProjections, CKPT_DIR as ALIGN_CKPT_DIR
from train_e6 import (
    load_confidences, class_weights_from, NUM_CLASSES, CLASSES, SEEDS,
    MAX_EPOCHS, PATIENCE, LR, DEVICE, ALIGN_SHARED_DIM,
    load_embeddings, load_labels, load_split,
    SIGLIP_DIR, BANGLABERT_E3B_DIR, HIDDEN_DIM,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REASONING_JSONL = os.path.join(PROJECT_ROOT, "outputs", "reasoning_results.jsonl")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase1_2_gate_uncertainty_results.json")


def load_reasoning_certainty(indices):
    """1.0 = fully certain (0 uncertain flags), 0.0 = maximally uncertain
    (all 3 flags true). Missing/errored reasoning defaults to 1.0 (treat
    as certain -- no evidence of uncertainty is not evidence of it)."""
    certainty = {}
    with open(REASONING_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["error"] or r["cause_effect"] is None:
                certainty[r["image_index"]] = 1.0
                continue
            uncertain_count = sum(
                1 for k in ("cause_effect", "figurative_meaning", "emotional_state")
                if r[k]["uncertain"]
            )
            certainty[r["image_index"]] = 1.0 - (uncertain_count / 3.0)
    return np.array([certainty.get(i, 1.0) for i in indices], dtype=np.float32).reshape(-1, 1)


def load_data_with_uncertainty():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    align_model = AlignmentProjections()
    ckpt = torch.load(os.path.join(ALIGN_CKPT_DIR, "current.pt"), map_location=DEVICE)
    align_model.load_state_dict(ckpt["state_dict"])
    align_model.eval()

    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        raw_text = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, idx))
        raw_image = torch.tensor(load_embeddings(SIGLIP_DIR, idx))
        with torch.no_grad():
            aligned_text, aligned_image = align_model(raw_text, raw_image)
        conf2 = load_confidences(idx)                    # (N, 2): ocr, translation
        certainty = load_reasoning_certainty(idx)          # (N, 1): reasoning certainty
        conf3 = np.concatenate([conf2, certainty], axis=1)  # (N, 3)
        data[split_name] = {
            "text": aligned_text.numpy(),
            "image": aligned_image.numpy(),
            "raw_text": raw_text.numpy(),
            "conf": conf3,
            "y": np.array([labels[i] for i in idx], dtype=np.int64),
        }

    conf_mean = data["train"]["conf"].mean(axis=0, keepdims=True)
    conf_std = data["train"]["conf"].std(axis=0, keepdims=True) + 1e-6
    for split_name in ("train", "val"):
        data[split_name]["conf"] = (data[split_name]["conf"] - conf_mean) / conf_std

    return data


class E6GatedOrthV2(nn.Module):
    """Same as E6GatedOrth (train_e6.py) except the gate takes 3 confidence
    signals (OCR, translation, reasoning-certainty) instead of 2."""

    def __init__(self, num_outputs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(ALIGN_SHARED_DIM * 2 + 3, HIDDEN_DIM // 2),
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


def train_one(seed, data, class_weights):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = E6GatedOrthV2(NUM_CLASSES).to(DEVICE)
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
    print("Loading aligned embeddings + 3-signal confidence (OCR, translation, reasoning certainty)...")
    data = load_data_with_uncertainty()
    class_weights = class_weights_from(data["train"]["y"])
    print(f"Train n={len(data['train']['y'])}, Val n={len(data['val']['y'])}")

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

    print(f"\nMajority-vote macro-F1 (3 signals, incl. reasoning certainty): {majority_macro_f1:.4f}")
    print(f"Comparison -- Phase 1.1 majority-vote macro-F1 (2 signals, no reasoning certainty): 0.5311")
    print(f"Delta: {majority_macro_f1 - 0.5311:+.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "seed_f1s": seed_f1s,
            "majority_vote_macro_f1": float(majority_macro_f1),
            "comparison_phase1_1_majority_vote": 0.5311,
            "delta": float(majority_macro_f1 - 0.5311),
        }, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
