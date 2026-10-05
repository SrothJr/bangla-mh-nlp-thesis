"""
Phase 3, item 3.2: cross-attention fusion, replacing the scalar gate
entirely. Text tokens (256, 768 from frozen BanglaBERT) act as queries
attending over image patches (729, 1152 from frozen SigLIP) as keys/
values -- expressing "which image regions are relevant to which words,"
which a single pooled scalar-gated vector (E5/E6's design) cannot.

Both encoders stay frozen (Phase 3.1 already showed unfreezing regresses
on this dataset size) -- only small linear projections (into a shared
dim) and a standard multi-head cross-attention layer are trained, plus
the classifier head. Attended text tokens are masked-mean-pooled (using
the real attention mask, ignoring padding) to a single fused vector,
concatenated with the raw pooled text vector (matching E5/E6's "concat
with raw text" pattern), then classified.

Full-batch training on GPU (unlike Phase 3.1's mini-batching, the
frozen-feature tensors here fit comfortably in memory -- no encoder
forward/backward pass needed, only the small attention+projection+head
modules are trained). Phase 1's kept recipe (ordinal label smoothing +
majority-vote ensembling across 3 seeds).
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

from train_e6 import (
    class_weights_from, NUM_CLASSES, CLASSES, SEEDS,
    MAX_EPOCHS, PATIENCE, LR, load_labels, load_split, load_embeddings,
    BANGLABERT_E3B_DIR, HIDDEN_DIM,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATCH_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip_patches")
TOKEN_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_tokens")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase3_2_cross_attention_results.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_DIM = 1152
TOKEN_DIM = 768
RAW_TEXT_DIM = 768
SHARED_DIM = 256
NUM_HEADS = 4


def load_patches(indices):
    return np.stack([np.load(os.path.join(PATCH_DIR, f"{i}.npy")).astype(np.float32) for i in indices])


def load_tokens(indices):
    tokens = np.stack([np.load(os.path.join(TOKEN_DIR, f"{i}.npy")).astype(np.float32) for i in indices])
    masks = np.stack([np.load(os.path.join(TOKEN_DIR, f"{i}_mask.npy")).astype(np.float32) for i in indices])
    return tokens, masks


class CrossAttentionFusion(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.text_proj = nn.Linear(TOKEN_DIM, SHARED_DIM)
        self.image_proj = nn.Linear(PATCH_DIM, SHARED_DIM)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=SHARED_DIM, num_heads=NUM_HEADS, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(SHARED_DIM)
        self.head = nn.Sequential(
            nn.Linear(SHARED_DIM + RAW_TEXT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, text_tokens, text_mask, image_patches, raw_text_vec):
        q = self.text_proj(text_tokens)       # (B, 256, D)
        kv = self.image_proj(image_patches)   # (B, 729, D)
        # text tokens attend over image patches; padded text positions are
        # masked out of the pooling step below (they still "query," but
        # their output is discarded, which is simpler and numerically safer
        # than masking queries inside attention)
        attended, _ = self.cross_attn(q, kv, kv)  # (B, 256, D)
        attended = self.norm(attended + q)         # residual + norm

        mask = text_mask.unsqueeze(-1)  # (B, 256, 1)
        pooled = (attended * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # masked mean pool

        z = torch.cat([pooled, raw_text_vec], dim=-1)
        return self.head(z)


def load_data():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")
    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        tokens, masks = load_tokens(idx)
        data[split_name] = {
            "tokens": tokens,
            "mask": masks,
            "patches": load_patches(idx),
            "raw_text": load_embeddings(BANGLABERT_E3B_DIR, idx),
            "y": np.array([labels[i] for i in idx], dtype=np.int64),
        }
    return data


def to_tensors(split_data):
    return (
        torch.tensor(split_data["tokens"], device=DEVICE),
        torch.tensor(split_data["mask"], device=DEVICE),
        torch.tensor(split_data["patches"], device=DEVICE),
        torch.tensor(split_data["raw_text"], device=DEVICE),
        torch.tensor(split_data["y"], device=DEVICE),
    )


def train_one(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CrossAttentionFusion(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    tok_tr, mask_tr, patch_tr, rawtext_tr, y_tr = to_tensors(data["train"])
    tok_val, mask_val, patch_val, rawtext_val, y_val = to_tensors(data["val"])

    best_val_f1 = -1.0
    best_val_probs = None
    epochs_without_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(tok_tr, mask_tr, patch_tr, rawtext_tr)
        loss = soft_target_loss(out, y_tr, smoothing_matrix, class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(tok_val, mask_val, patch_val, rawtext_val)
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

        if epoch % 20 == 0:
            print(f"  seed {seed} epoch {epoch}: val macro-F1 = {val_f1:.4f}", flush=True)

    return best_val_f1, best_val_probs, y_val.cpu().numpy()


def main():
    print("Loading token-level and patch-level features (this uses more memory than pooled vectors)...")
    data = load_data()
    print(f"Train n={len(data['train']['y'])}, Val n={len(data['val']['y'])}")
    class_weights = class_weights_from(data["train"]["y"]).to(DEVICE)
    smoothing_matrix = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=DEVICE)

    seed_f1s = []
    seed_probs = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_one(seed, data, class_weights, smoothing_matrix)
        seed_f1s.append(val_f1)
        seed_probs.append(probs)
        print(f"seed {seed}: BEST val macro-F1 = {val_f1:.4f}")

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

    print(f"\n=== Cross-attention fusion, majority-vote macro-F1: {majority_macro_f1:.4f} ===")
    print(f"Comparison -- Phase 1 best (scalar gate, aligned): 0.5330")
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
