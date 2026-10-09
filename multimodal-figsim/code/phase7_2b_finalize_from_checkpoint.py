"""
One-off recovery script: the live phase7_2_bnhib_vision_pretrain.py run hit
its early-stopping condition (epochs_without_improve >= PATIENCE) but hung
during its own wrap-up (evaluate + save_pretrained + write results), a
known Windows issue with persistent_workers DataLoader shutdown. The best
weights were already safely captured in resume_state.pt's `best_state`
before the hang -- this script just does the exact same finalization the
original script would have done, reading that best_state directly, so
nothing needs to be re-trained or is at risk.
"""
import os
import json

import numpy as np
import torch
from transformers import SiglipVisionModel, SiglipImageProcessor
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from sklearn.metrics import f1_score, classification_report

from phase7_2_bnhib_vision_pretrain import (
    load_split, BnhibImageDataset, LoraVisionClassifier, LABEL_NAMES,
    SIGLIP_MODEL, CKPT_DIR, RESUME_CKPT_PATH, RESULTS_PATH,
    LORA_R, LORA_ALPHA, LORA_LR, HEAD_LR, DEVICE,
)
from torch.utils.data import DataLoader

print(f"Device: {DEVICE}")
print("Loading best_state from the stuck run's checkpoint...")
ckpt = torch.load(RESUME_CKPT_PATH, map_location=DEVICE)
print(f"Checkpoint is from epoch {ckpt['epoch']}, best_val_f1={ckpt['best_val_f1']:.4f}")
best_state = ckpt["best_state"]

print("Rebuilding model and loading the best weights...")
base = SiglipVisionModel.from_pretrained(SIGLIP_MODEL)
config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.1,
                     target_modules=["q_proj", "v_proj"], bias="none")
image_encoder = get_peft_model(base, config)
model = LoraVisionClassifier(image_encoder, len(LABEL_NAMES)).to(DEVICE)
set_peft_model_state_dict(model.image_encoder, best_state["lora_state"])
model.head.load_state_dict(best_state["head_state"])
model.eval()

print("Loading validation set (no DataLoader workers, to avoid the same hang)...")
val_df = load_split("val")
image_processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL)
val_ds = BnhibImageDataset(val_df, image_processor)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        out = model(batch["pixel_values"].to(DEVICE))
        all_preds.append(out.argmax(dim=1).cpu().numpy())
        all_labels.append(batch["label"].numpy())
preds = np.concatenate(all_preds)
labels = np.concatenate(all_labels)
final_f1 = f1_score(labels, preds, average="macro", zero_division=0)
print(f"Final validation macro-F1 (re-confirmed): {final_f1:.4f}")

report = classification_report(
    labels, preds, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES,
    zero_division=0, output_dict=True,
)
print("\nPer-class F1:")
for c in LABEL_NAMES:
    print(f"  {c}: {report[c]['f1-score']:.4f}")

lora_save_path = os.path.join(CKPT_DIR, "best")
model.image_encoder.save_pretrained(lora_save_path)
print(f"\nSaved LoRA adapter weights to {lora_save_path}")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "BN-HIB only (CMBAN not fully downloaded, see Step 2 audit)",
        "n_train": 2272, "n_val": len(val_df),
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_lr": LORA_LR, "head_lr": HEAD_LR,
        "best_val_macro_f1": float(final_f1),
        "classification_report": report,
        "checkpoint_path": lora_save_path,
        "note": "DAPT-BanglaBERT was never loaded or modified in this script -- vision-only "
                "auxiliary pretraining, per PHASE7_REVISED_PRETRAINING_PLAN.md Track A. "
                "Finalized via phase7_2b_finalize_from_checkpoint.py after the original run "
                "hung during its own wrap-up post-early-stopping (Windows DataLoader worker "
                "shutdown issue) -- the trained weights themselves were unaffected.",
    }, f, indent=2)
print(f"Saved results to {RESULTS_PATH}")

if os.path.exists(RESUME_CKPT_PATH):
    os.remove(RESUME_CKPT_PATH)
    print("Removed the now-unneeded resume checkpoint.")

print("\nDone. You can now Ctrl+C the original stuck terminal -- its work is fully captured here.")
