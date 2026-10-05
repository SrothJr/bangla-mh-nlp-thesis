"""
Day 5-6: contrastive alignment, Stage A (Section 5.3). Small trainable
projection layers map the text vector (E3b: OCR + reasoning, 768-dim) and
image vector (SigLIP, 1152-dim) into a shared dimension, trained with
symmetric InfoNCE (CLIP-style) on the TRAINING split only -- self-supervised,
no labels used. A meme's own (text, image) pair is the positive; every other
meme in the batch is a negative. Full-batch training (582 items fits easily
in one batch; the small dataset makes richer in-batch negatives more useful
than mini-batching here).

STOP-LOSS RULE (Section 5.3): if this is not showing sensible, decreasing
loss / non-trivial retrieval accuracy after a reasonable debugging effort,
drop it entirely and lock E4 (weighted CE) as the final model. Tracked here
via train/val InfoNCE loss AND top-1 text->image retrieval accuracy on the
validation split (a more interpretable convergence signal than loss alone).

Checkpoint retention (Section 4.3): best / current / previous only.
"""
import os
import csv
import json

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
LEAKAGE_SAFE_DIR = os.path.join(FIGSIM_ROOT, "data", "leakage_safe")
SIGLIP_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip")
BANGLABERT_E3B_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_e3b")
CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "stage_a_alignment")

TEXT_DIM = 768
IMAGE_DIM = 1152
SHARED_DIM = 256
TEMPERATURE = 0.07
LR = 3e-4
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 1000
PATIENCE = 50
DEVICE = "cpu"


def load_split(name):
    with open(os.path.join(LEAKAGE_SAFE_DIR, f"{name}.txt"), "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_embeddings(emb_dir, indices):
    return np.stack([np.load(os.path.join(emb_dir, f"{i}.npy")) for i in indices]).astype(np.float32)


class AlignmentProjections(nn.Module):
    def __init__(self):
        super().__init__()
        # Plain single linear layer per modality (matches the brief's "small
        # trainable projection layers" literally) -- deliberately leaner than
        # a first attempt with a hidden ReLU layer, which overfit badly on
        # 582 training pairs (val loss diverged while train loss kept
        # falling). Dropout added as a second regularizer.
        self.text_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(TEXT_DIM, SHARED_DIM))
        self.image_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(IMAGE_DIM, SHARED_DIM))

    def forward(self, text_vec, image_vec):
        t = nn.functional.normalize(self.text_proj(text_vec), dim=-1)
        i = nn.functional.normalize(self.image_proj(image_vec), dim=-1)
        return t, i


def info_nce_loss(t, i, temperature):
    logits = t @ i.T / temperature  # (N, N)
    labels = torch.arange(t.shape[0], device=t.device)
    loss_t2i = nn.functional.cross_entropy(logits, labels)
    loss_i2t = nn.functional.cross_entropy(logits.T, labels)
    return (loss_t2i + loss_i2t) / 2


def top1_retrieval_accuracy(t, i):
    logits = t @ i.T
    pred = logits.argmax(dim=1)
    labels = torch.arange(t.shape[0], device=t.device)
    return (pred == labels).float().mean().item()


def save_checkpoint(model, epoch, val_loss, tag):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{tag}.pt")
    tmp_path = path + ".tmp"
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, tmp_path)
    os.replace(tmp_path, path)


def rotate_checkpoints(model, epoch, val_loss):
    # best/current/previous per Section 4.3 -- current becomes previous,
    # new save becomes current, best only overwritten on genuine improvement.
    current_path = os.path.join(CKPT_DIR, "current.pt")
    previous_path = os.path.join(CKPT_DIR, "previous.pt")
    if os.path.exists(current_path):
        os.replace(current_path, previous_path)
    save_checkpoint(model, epoch, val_loss, "current")


def main():
    print("Loading cached embeddings...")
    train_idx = load_split("train")
    val_idx = load_split("val")

    X_text_tr = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, train_idx), device=DEVICE)
    X_img_tr = torch.tensor(load_embeddings(SIGLIP_DIR, train_idx), device=DEVICE)
    X_text_val = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, val_idx), device=DEVICE)
    X_img_val = torch.tensor(load_embeddings(SIGLIP_DIR, val_idx), device=DEVICE)
    print(f"Train n={len(train_idx)}, Val n={len(val_idx)}")

    model = AlignmentProjections().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    epochs_without_improve = 0
    history = []

    print("Starting contrastive alignment training (full-batch InfoNCE)...")
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        t_tr, i_tr = model(X_text_tr, X_img_tr)
        loss = info_nce_loss(t_tr, i_tr, TEMPERATURE)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            t_val, i_val = model(X_text_val, X_img_val)
            val_loss = info_nce_loss(t_val, i_val, TEMPERATURE).item()
            val_retrieval = top1_retrieval_accuracy(t_val, i_val)
            train_retrieval = top1_retrieval_accuracy(t_tr, i_tr)

        rotate_checkpoints(model, epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, epoch, val_loss, "best")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        history.append({
            "epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss,
            "train_retrieval_acc": train_retrieval, "val_retrieval_acc": val_retrieval,
        })

        if epoch % 20 == 0 or epoch == MAX_EPOCHS - 1:
            chance = 1.0 / len(val_idx)
            print(f"epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f} "
                  f"train_retrieval={train_retrieval:.3f} val_retrieval={val_retrieval:.3f} "
                  f"(chance={chance:.4f})", flush=True)

        if epochs_without_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val_loss improvement for {PATIENCE} epochs).")
            break

    with open(os.path.join(PROJECT_ROOT, "outputs", "stage_a_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Final val retrieval accuracy: {history[-1]['val_retrieval_acc']:.3f} "
          f"(chance level: {1.0/len(val_idx):.4f})")


if __name__ == "__main__":
    main()
