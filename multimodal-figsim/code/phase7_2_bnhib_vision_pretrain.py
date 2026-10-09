"""
Phase 7, Step 3 -- Track A (PHASE7_REVISED_PRETRAINING_PLAN.md): DAPT-safe
external meme pretraining, BN-HIB only (CMBAN is not yet fully downloaded
-- see the Step 2 audit; only a 133-row sample with no English text is
present, so it is skipped here rather than faked).

This script does NOT load, forward-pass, or modify DAPT-BanglaBERT in any
way -- no text encoder is used at all. It only adapts SigLIP (the image
encoder) via small LoRA adapters, trained on BN-HIB's own Hate /
Inflammatory / Benign labels as a vision-side auxiliary classification
task. The goal is a SigLIP checkpoint that is more sensitive to meme-
style images in general, which can optionally be tried later as the
image branch's starting point for the real FigSIM classifier -- but
that's a separate, later, validation-gated decision (Step 4/5 of the
plan), not decided here.

Label mapping, per BN-HIB's own README (IMPORTANT): the `choice` column
is the authoritative label -- Targeted Trolling -> Hate, Harmless
Trolling -> Benign, Provocative_Trolls -> Inflammatory. The `class`
column is explicitly NOT a label (language-mix metadata only) and is not
used for supervision here.

Uses BN-HIB's own official train/val split for training and model
selection. BN-HIB's test.csv is never touched by this script, matching
this project's established test-set discipline (extended here to the
new external dataset, not just FigSIM's own test set).

Run this directly in a terminal to watch live per-epoch progress:
    python phase7_2_bnhib_vision_pretrain.py
"""
import os
import csv
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import f1_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BNHIB_ROOT = os.path.join(PROJECT_ROOT, "BN-HIB")
CSV_DIR = os.path.join(BNHIB_ROOT, "Train Test Val CSV")
IMAGES_DIR = os.path.join(BNHIB_ROOT, "Images")
SIGLIP_MODEL = "google/siglip-so400m-patch14-384"

CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "phase7_bnhib_siglip_lora")
RESUME_CKPT_PATH = os.path.join(CKPT_DIR, "resume_state.pt")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase7_2_bnhib_vision_pretrain_results.json")

# BN-HIB's own authoritative label mapping (README (IMPORTANT).md)
CHOICE_TO_LABEL = {
    "Targeted Trolling": "Hate",
    "Harmless Trolling": "Benign",
    "Provocative_Trolls": "Inflammatory",
}
LABEL_NAMES = ["Benign", "Inflammatory", "Hate"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_LR = 1e-4
HEAD_LR = 1e-3
HIDDEN_DIM = 256
BATCH_SIZE = 32
MAX_EPOCHS = 15
PATIENCE = 4
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_split(name):
    """Returns a list of dicts (no pandas -- its compiled DLL is blocked by this
    machine's Application Control policy; csv.DictReader is pure Python
    stdlib and unaffected, matching how train_e6.py's load_labels() already
    reads CSVs elsewhere in this project)."""
    records = []
    with open(os.path.join(CSV_DIR, f"{name}.csv"), "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            choice = row["choice"]
            assert choice in CHOICE_TO_LABEL, f"unmapped choice value {choice!r} in {name}.csv"
            label_name = CHOICE_TO_LABEL[choice]
            records.append({
                "image": row["image"],
                "label_name": label_name,
                "label_id": LABEL_TO_ID[label_name],
            })
    return records


class BnhibImageDataset(Dataset):
    def __init__(self, records, image_processor):
        self.records = records
        self.image_processor = image_processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        image_path = os.path.join(IMAGES_DIR, row["image"])
        with Image.open(image_path) as im:
            im.seek(0)
            img = im.convert("RGB")
        pixel_values = self.image_processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel_values, "label": row["label_id"]}


class LoraVisionClassifier(nn.Module):
    def __init__(self, image_encoder, num_outputs):
        super().__init__()
        self.image_encoder = image_encoder
        self.head = nn.Sequential(
            nn.Linear(1152, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, pixel_values):
        out = self.image_encoder(pixel_values=pixel_values)
        return self.head(out.pooler_output)


def build_lora_image_encoder():
    base = SiglipVisionModel.from_pretrained(SIGLIP_MODEL)
    config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                         target_modules=["q_proj", "v_proj"], bias="none")
    return get_peft_model(base, config)


def count_trainable(model, name):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_trainable:,} / {n_total:,} trainable ({100 * n_trainable / n_total:.3f}%)", flush=True)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["pixel_values"].to(DEVICE))
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["label"].numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return f1, preds, labels


def main():
    print(f"Device: {DEVICE}", flush=True)
    print("Loading BN-HIB official train/val splits (test.csv is NOT touched)...", flush=True)
    train_df = load_split("train")
    val_df = load_split("val")
    print(f"Train n={len(train_df)}, Val n={len(val_df)}", flush=True)
    train_label_counts = Counter(r["label_name"] for r in train_df)
    print("Train label distribution:", dict(train_label_counts), flush=True)

    image_processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL)
    train_ds = BnhibImageDataset(train_df, image_processor)
    val_ds = BnhibImageDataset(val_df, image_processor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=(DEVICE == "cuda"), persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                             num_workers=2, pin_memory=(DEVICE == "cuda"), persistent_workers=True)

    id_counts = Counter(r["label_id"] for r in train_df)
    class_counts = np.array([id_counts[i] for i in range(len(LABEL_NAMES))], dtype=np.float32)
    class_weights = torch.tensor(len(train_df) / (len(LABEL_NAMES) * class_counts), dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {dict(zip(LABEL_NAMES, class_weights.cpu().numpy().round(3)))}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Building LoRA-adapted SigLIP...", flush=True)
    image_encoder = build_lora_image_encoder()
    count_trainable(image_encoder, "SigLIP image encoder (LoRA)")
    model = LoraVisionClassifier(image_encoder, len(LABEL_NAMES)).to(DEVICE)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n]
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LORA_LR, "weight_decay": 0.01},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": 1e-4},
    ])

    start_epoch = 0
    best_val_f1 = -1.0
    best_state = None
    epochs_without_improve = 0

    os.makedirs(CKPT_DIR, exist_ok=True)
    if os.path.exists(RESUME_CKPT_PATH):
        print(f"\nFound a resume checkpoint at {RESUME_CKPT_PATH} -- resuming instead of "
              f"starting over.", flush=True)
        ckpt = torch.load(RESUME_CKPT_PATH, map_location=DEVICE)
        # Only trainable params were saved (LoRA adapters + head) -- the frozen base SigLIP
        # weights are already correct as freshly loaded above and were never saved to disk,
        # to avoid writing ~1.7GB of unchanging frozen weights on every epoch.
        set_peft_model_state_dict(model.image_encoder, ckpt["lora_state"])
        model.head.load_state_dict(ckpt["head_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
        best_state = ckpt["best_state"]
        epochs_without_improve = ckpt["epochs_without_improve"]
        torch.manual_seed(SEED + start_epoch)  # avoid replaying the exact same shuffle order
        print(f"Resuming from epoch {start_epoch} (best val macro-F1 so far: {best_val_f1:.4f})\n", flush=True)

    print("\nStarting training (Ctrl+C-safe: progress after each finished epoch is saved to disk;\n"
          "if the machine loses power or is closed, just re-run this same command and it will\n"
          "pick back up from the last completed epoch)...\n", flush=True)
    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch["pixel_values"].to(DEVICE))
            y_batch = batch["label"].to(DEVICE)
            loss = nn.functional.cross_entropy(out, y_batch, weight=class_weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        val_f1, val_preds, val_labels = evaluate(model, val_loader)
        avg_loss = running_loss / n_batches
        print(f"epoch {epoch}: train_loss={avg_loss:.4f}  val macro-F1={val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            import copy
            best_state = {
                "lora_state": copy.deepcopy(get_peft_model_state_dict(model.image_encoder)),
                "head_state": copy.deepcopy(model.head.state_dict()),
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        # Save progress after every epoch, atomically (temp file + rename), so a power loss
        # or closed terminal never loses more than the epoch currently in flight. Only the
        # trainable LoRA + head weights are written (a few MB) -- NOT the frozen ~429M-
        # parameter SigLIP base, which would otherwise add ~1.7GB per save and is wasteful
        # given this machine's limited free disk space; the base is simply reloaded fresh
        # from the (already-cached) pretrained checkpoint each time this script starts.
        tmp_path = RESUME_CKPT_PATH + ".tmp"
        torch.save({
            "epoch": epoch,
            "lora_state": get_peft_model_state_dict(model.image_encoder),
            "head_state": model.head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "best_state": best_state,
            "epochs_without_improve": epochs_without_improve,
        }, tmp_path)
        os.replace(tmp_path, RESUME_CKPT_PATH)

        if epochs_without_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val macro-F1={best_val_f1:.4f})", flush=True)
            break

    print(f"\n=== Best validation macro-F1: {best_val_f1:.4f} ===", flush=True)

    set_peft_model_state_dict(model.image_encoder, best_state["lora_state"])
    model.head.load_state_dict(best_state["head_state"])
    final_f1, final_preds, final_labels = evaluate(model, val_loader)
    report = classification_report(
        final_labels, final_preds, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES,
        zero_division=0, output_dict=True,
    )
    print("\nPer-class F1 (best checkpoint, val set):", flush=True)
    for c in LABEL_NAMES:
        print(f"  {c}: {report[c]['f1-score']:.4f}", flush=True)

    lora_save_path = os.path.join(CKPT_DIR, "best")
    model.image_encoder.save_pretrained(lora_save_path)
    print(f"\nSaved LoRA adapter weights to {lora_save_path}", flush=True)

    if os.path.exists(RESUME_CKPT_PATH):
        os.remove(RESUME_CKPT_PATH)
        print("Training finished normally -- removed the now-unneeded resume checkpoint.", flush=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "BN-HIB only (CMBAN not fully downloaded, see Step 2 audit)",
            "n_train": len(train_df), "n_val": len(val_df),
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_lr": LORA_LR, "head_lr": HEAD_LR,
            "best_val_macro_f1": float(best_val_f1),
            "classification_report": report,
            "checkpoint_path": lora_save_path,
            "note": "DAPT-BanglaBERT was never loaded or modified in this script -- vision-only "
                    "auxiliary pretraining, per PHASE7_REVISED_PRETRAINING_PLAN.md Track A.",
        }, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
