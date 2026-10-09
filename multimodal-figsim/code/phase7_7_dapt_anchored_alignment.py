"""
Phase 7, Option 2 (PHASE7_SHARED_ENCODER_OPTION2_PLAN.md): contrastive
alignment anchored to DAPT-BanglaBERT's own native 768-dim output space,
instead of learning a brand-new arbitrary shared space (the existing
Stage A design, train_contrastive_align.py).

Only the image side gets a trainable projection (1152 -> 768). The text
side is DAPT's raw output, used as-is, L2-normalized but NOT projected
or otherwise modified -- DAPT-BanglaBERT itself is never loaded or
touched by this script at all (it operates on DAPT's already-cached
embeddings, same as the existing Stage A script). Same InfoNCE
contrastive objective, same train-split-only self-supervised protocol,
same checkpoint retention pattern (best/current/previous) -- saved to a
NEW directory, never overwriting the existing Stage A checkpoint.

Step 3 of the plan: report top-1 retrieval accuracy and compare directly
against the existing Stage A's known value (0.051 validation, vs.
chance ~0.005) BEFORE any downstream classifier training -- a cheap,
early decision gate.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn

from train_contrastive_align import (
    load_split, load_embeddings, info_nce_loss, top1_retrieval_accuracy,
    SIGLIP_DIR, BANGLABERT_E3B_DIR, TEXT_DIM, IMAGE_DIM, TEMPERATURE, LR, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, DEVICE,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "stage_a_dapt_anchored")
HISTORY_PATH = os.path.join(PROJECT_ROOT, "outputs", "stage_a_dapt_anchored_training_history.json")

# Reference point: existing Stage A's known validation retrieval accuracy
# (outputs/stage_a_training_history.json, epoch 63, the "current.pt"
# checkpoint actually used throughout this project).
REF_EXISTING_STAGE_A_VAL_RETRIEVAL = 0.0513


class DAPTAnchoredProjection(nn.Module):
    """Only the image side is projected -- into DAPT's own native space."""

    def __init__(self):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(IMAGE_DIM, TEXT_DIM))

    def forward(self, text_vec, image_vec):
        t = nn.functional.normalize(text_vec, dim=-1)  # DAPT's own space, unprojected
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
    print("Loading cached embeddings (DAPT-BanglaBERT and SigLIP were never re-run -- "
          "using their existing cached outputs, same as the original Stage A script)...")
    train_idx = load_split("train")
    val_idx = load_split("val")

    X_text_tr = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, train_idx), device=DEVICE)
    X_img_tr = torch.tensor(load_embeddings(SIGLIP_DIR, train_idx), device=DEVICE)
    X_text_val = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, val_idx), device=DEVICE)
    X_img_val = torch.tensor(load_embeddings(SIGLIP_DIR, val_idx), device=DEVICE)
    print(f"Train n={len(train_idx)}, Val n={len(val_idx)}")

    model = DAPTAnchoredProjection().to(DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,} (image-side projection only; "
          f"existing Stage A trains both sides, ~{768*256 + 1152*256:,} total)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    epochs_without_improve = 0
    history = []

    print("Starting DAPT-anchored contrastive alignment training (full-batch InfoNCE)...")
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

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # Report the retrieval accuracy of the "current" (final-epoch) checkpoint, matching
    # how train_e6.load_aligned_data() uses "current.pt" for the existing Stage A --
    # not necessarily the val_loss-best epoch, since retrieval accuracy and val_loss
    # don't always move together (documented behavior of the existing Stage A too).
    final_retrieval = history[-1]["val_retrieval_acc"]
    print(f"\n=== Final (current.pt) val retrieval accuracy: {final_retrieval:.4f} ===")
    print(f"Comparison -- existing Stage A's val retrieval accuracy: {REF_EXISTING_STAGE_A_VAL_RETRIEVAL:.4f}")
    print(f"Delta: {final_retrieval - REF_EXISTING_STAGE_A_VAL_RETRIEVAL:+.4f}")
    chance = 1.0 / len(val_idx)
    print(f"Chance level: {chance:.4f}")

    decision = "PROCEED to downstream classifier training" if final_retrieval >= REF_EXISTING_STAGE_A_VAL_RETRIEVAL \
        else "STOP HERE -- alignment quality did not match existing Stage A, not worth downstream cost"
    print(f"\nDecision gate (Step 3 of the plan): {decision}")

    with open(os.path.join(PROJECT_ROOT, "outputs", "phase7_7_dapt_anchored_alignment_results.json"), "w") as f:
        json.dump({
            "final_val_retrieval_acc": final_retrieval,
            "reference_existing_stage_a_val_retrieval_acc": REF_EXISTING_STAGE_A_VAL_RETRIEVAL,
            "delta": final_retrieval - REF_EXISTING_STAGE_A_VAL_RETRIEVAL,
            "chance_level": chance,
            "n_trainable_params": n_trainable,
            "decision": decision,
        }, f, indent=2)
    print(f"Saved results to outputs/phase7_7_dapt_anchored_alignment_results.json")


if __name__ == "__main__":
    main()
