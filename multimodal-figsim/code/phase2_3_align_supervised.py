"""
Phase 2, item 2.3 (IMPROVEMENT_PLAN.md): strengthen Stage A contrastive
alignment with a supervised contrastive loss instead of pure self-
supervised InfoNCE.

Original Stage A (train_contrastive_align.py): for text vector t_i, the
only positive is its OWN meme's image vector i_i; every other meme in the
batch is a negative, regardless of class. Purely self-supervised -- no
label information used.

This version (SupCon-style, adapted cross-modal): for text vector t_i, ALL
image vectors from memes sharing t_i's suicide-severity class are treated
as positives (including but not limited to i_i itself), and only
cross-class image vectors are negatives. This gives the alignment a
task-relevant signal -- pulling together representations that share a
severity label, not just a meme's own two modalities -- which is the
motivation named in IMPROVEMENT_PLAN.md item 2.3.

Same architecture (plain linear projections, same regularization) as the
already-fixed train_contrastive_align.py, so any difference is
attributable to the loss function, not a confound from also changing the
projection capacity.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn

from train_contrastive_align import (
    top1_retrieval_accuracy, load_split, load_embeddings,
    LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, DEVICE, SHARED_DIM, TEXT_DIM, IMAGE_DIM,
    BANGLABERT_E3B_DIR, SIGLIP_DIR, TEMPERATURE,
)
from train_e6 import load_labels

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "stage_a_alignment_supervised")


class AlignmentProjectionsSupervised(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(TEXT_DIM, SHARED_DIM))
        self.image_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(IMAGE_DIM, SHARED_DIM))

    def forward(self, text_vec, image_vec):
        t = nn.functional.normalize(self.text_proj(text_vec), dim=-1)
        i = nn.functional.normalize(self.image_proj(image_vec), dim=-1)
        return t, i


def supervised_contrastive_loss(t, i, labels, temperature):
    """Cross-modal SupCon: for each text anchor, positives = all image
    vectors sharing the anchor's class label; negatives = the rest.
    Symmetric in both directions (text->image and image->text)."""
    sim_t2i = t @ i.T / temperature  # (N, N)
    same_class = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # (N, N), includes diagonal

    def directional_loss(sim, pos_mask):
        # log-sum-exp over all candidates (denominator), log-sum-exp over positives only (numerator)
        log_denom = torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob = sim - log_denom  # (N, N) log P(j | i) for every candidate j
        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count
        return -mean_log_prob_pos.mean()

    loss_t2i = directional_loss(sim_t2i, same_class)
    loss_i2t = directional_loss(sim_t2i.T, same_class.T)
    return (loss_t2i + loss_i2t) / 2


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
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    X_text_tr = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, train_idx), device=DEVICE)
    X_img_tr = torch.tensor(load_embeddings(SIGLIP_DIR, train_idx), device=DEVICE)
    X_text_val = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, val_idx), device=DEVICE)
    X_img_val = torch.tensor(load_embeddings(SIGLIP_DIR, val_idx), device=DEVICE)
    y_tr = torch.tensor([labels[i] for i in train_idx], device=DEVICE)
    y_val = torch.tensor([labels[i] for i in val_idx], device=DEVICE)
    print(f"Train n={len(train_idx)}, Val n={len(val_idx)}")

    model = AlignmentProjectionsSupervised().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    epochs_without_improve = 0
    history = []

    print("Starting SUPERVISED contrastive alignment training...")
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        t_tr, i_tr = model(X_text_tr, X_img_tr)
        loss = supervised_contrastive_loss(t_tr, i_tr, y_tr, TEMPERATURE)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            t_val, i_val = model(X_text_val, X_img_val)
            val_loss = supervised_contrastive_loss(t_val, i_val, y_val, TEMPERATURE).item()
            # retrieval accuracy still measured against the meme's OWN pair (not class-level),
            # for comparability with the self-supervised run's reported numbers
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
            print(f"epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f} "
                  f"train_retrieval(own-pair)={train_retrieval:.3f} val_retrieval(own-pair)={val_retrieval:.3f}",
                  flush=True)

        if epochs_without_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    with open(os.path.join(PROJECT_ROOT, "outputs", "phase2_3_stage_a_supervised_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Final own-pair retrieval accuracy: {history[-1]['val_retrieval_acc']:.3f} "
          f"(for comparison to the original self-supervised run's 0.067 at a similar stopping point)")


if __name__ == "__main__":
    main()
