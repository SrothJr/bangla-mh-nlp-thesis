"""
Phase 2, item 2.4: retrain Stage A contrastive alignment with CLIP image
embeddings (1024-dim) in place of SigLIP (1152-dim) -- Bangla text branch
unchanged (confirmed best after 2.1). Same self-supervised InfoNCE
architecture/hyperparameters as the original, already-fixed alignment run
(train_contrastive_align.py), only the image encoder source changes.
"""
import os
import json

import torch
import torch.nn as nn

from train_contrastive_align import (
    info_nce_loss, top1_retrieval_accuracy, load_split, load_embeddings,
    TEMPERATURE, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, DEVICE, SHARED_DIM, TEXT_DIM,
    BANGLABERT_E3B_DIR,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLIP_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "clip_vit_large")
CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "stage_a_alignment_clip")

CLIP_IMAGE_DIM = 1024


class AlignmentProjectionsClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(TEXT_DIM, SHARED_DIM))
        self.image_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(CLIP_IMAGE_DIM, SHARED_DIM))

    def forward(self, text_vec, image_vec):
        t = nn.functional.normalize(self.text_proj(text_vec), dim=-1)
        i = nn.functional.normalize(self.image_proj(image_vec), dim=-1)
        return t, i


def save_checkpoint(model, epoch, val_loss, tag):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{tag}.pt")
    tmp_path = path + ".tmp"
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, tmp_path)
    os.replace(tmp_path, path)


def rotate_checkpoints(model, epoch, val_loss):
    current_path = os.path.join(CKPT_DIR, "current.pt")
    previous_path = os.path.join(CKPT_DIR, "previous.pt")
    if os.path.exists(current_path):
        os.replace(current_path, previous_path)
    save_checkpoint(model, epoch, val_loss, "current")


def main():
    train_idx = load_split("train")
    val_idx = load_split("val")

    X_text_tr = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, train_idx), device=DEVICE)
    X_img_tr = torch.tensor(load_embeddings(CLIP_DIR, train_idx), device=DEVICE)
    X_text_val = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, val_idx), device=DEVICE)
    X_img_val = torch.tensor(load_embeddings(CLIP_DIR, val_idx), device=DEVICE)
    print(f"Train n={len(train_idx)}, Val n={len(val_idx)}")

    model = AlignmentProjectionsClip().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    epochs_without_improve = 0
    history = []

    print("Starting contrastive alignment training (CLIP image branch)...")
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

        history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss,
                         "train_retrieval_acc": train_retrieval, "val_retrieval_acc": val_retrieval})

        if epoch % 20 == 0 or epoch == MAX_EPOCHS - 1:
            chance = 1.0 / len(val_idx)
            print(f"epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f} "
                  f"train_retrieval={train_retrieval:.3f} val_retrieval={val_retrieval:.3f} "
                  f"(chance={chance:.4f})", flush=True)

        if epochs_without_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    with open(os.path.join(PROJECT_ROOT, "outputs", "phase2_4_stage_a_clip_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Final val retrieval accuracy: {history[-1]['val_retrieval_acc']:.3f} "
          f"(chance level: {1.0/len(val_idx):.4f})")


if __name__ == "__main__":
    main()
