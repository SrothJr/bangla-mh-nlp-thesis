"""
Phase 6, item 6.1 (IMPROVEMENT_PLAN.md): parameter-efficient (LoRA)
fine-tuning of BOTH frozen encoders, end-to-end with the classifier head
-- the one remaining "heavier learning" lever available without API
access (a bigger SigLIP doesn't exist; a bigger text encoder was already
tried and lost in Phase 2.1).

Different mechanism from Phase 3.1's partial-unfreezing experiment
(which failed, -0.026 macro-F1, by overfitting fast on 582 examples
after making full transformer layers trainable). LoRA freezes 100% of
the original pretrained weights and injects small low-rank adapter
matrices into the attention query/value projections -- well under 1% of
each encoder's total parameters are trainable -- a much more heavily
regularized way to let the encoders adapt slightly to this domain.

Architecture is deliberately the same "simple concat" design Phase 3.1
used (text CLS token concat SigLIP pooled vector -> small MLP head), so
any difference from the 0.5139 frozen-baseline reference is attributable
to the fine-tuning mechanism, not a fusion-architecture confound.
"""
import os
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import AutoTokenizer, AutoModel, SiglipVisionModel, SiglipImageProcessor
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, classification_report

from train_e6 import class_weights_from, NUM_CLASSES, CLASSES, SEEDS, load_labels, load_split, HIDDEN_DIM
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
INDEX_CSV = os.path.join(FIGSIM_ROOT, "data", "leakage_safe", "figsim_leakage_safe_index.csv")
DAPT_ROOT = r"C:\Users\user6\T2520814\DAPT_models"
CKPT_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5", "checkpoint-735")
TOKENIZER_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5_tapt")
SIGLIP_MODEL = "google/siglip-so400m-patch14-384"
TRANSLATION_OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
TRANSLATION_REASONING_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_reasoning_results.jsonl")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase6_1_lora_finetune_results.json")

TEXT_DIM = 768
IMAGE_DIM = 1152
MAX_LEN = 256
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_LR = 1e-4
HEAD_LR = 1e-3
BATCH_SIZE = 16
MAX_EPOCHS = 15
PATIENCE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPARISON_FROZEN_BASELINE = 0.5139


def load_jsonl(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[r["image_index"]] = r
    return records


def load_image_paths():
    paths = {}
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["include_in_leakage_safe_dataset"] == "True":
                paths[int(row["image_index"])] = os.path.join(FIGSIM_ROOT, row["relative_path"])
    return paths


class MemeMultimodalDataset(Dataset):
    def __init__(self, indices, ocr_records, reasoning_records, image_paths, labels, tokenizer, image_processor):
        self.indices = indices
        self.ocr_records = ocr_records
        self.reasoning_records = reasoning_records
        self.image_paths = image_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        ocr_bn = self.ocr_records[idx].get("ocr_text_bn", "") or ""
        reasoning_bn = self.reasoning_records[idx].get("reasoning_text_bn", "") or ""
        enc = self.tokenizer(
            text=ocr_bn, text_pair=reasoning_bn,
            truncation=True, padding="max_length", max_length=MAX_LEN,
            return_tensors="pt",
        )
        with Image.open(self.image_paths[idx]) as im:
            im.seek(0)
            img = im.convert("RGB")
        pixel_values = self.image_processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "token_type_ids": enc["token_type_ids"][0],
            "pixel_values": pixel_values,
            "label": self.labels[i],
        }


class LoraFusionModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, num_outputs):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.head = nn.Sequential(
            nn.Linear(TEXT_DIM + IMAGE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)
        text_vec = text_out.last_hidden_state[:, 0, :]  # CLS token
        image_out = self.image_encoder(pixel_values=pixel_values)
        image_vec = image_out.pooler_output
        z = torch.cat([text_vec, image_vec], dim=-1)
        return self.head(z)


def build_lora_text_encoder():
    base = AutoModel.from_pretrained(CKPT_PATH)
    config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                         target_modules=["query", "value"], bias="none")
    model = get_peft_model(base, config)
    return model


def build_lora_image_encoder():
    base = SiglipVisionModel.from_pretrained(SIGLIP_MODEL)
    config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                         target_modules=["q_proj", "v_proj"], bias="none")
    model = get_peft_model(base, config)
    return model


def count_trainable(model, name):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_trainable:,} / {n_total:,} trainable ({100 * n_trainable / n_total:.3f}%)")


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE),
                batch["token_type_ids"].to(DEVICE), batch["pixel_values"].to(DEVICE),
            )
            all_probs.append(torch.softmax(out, dim=1).cpu().numpy())
            all_labels.append(batch["label"].numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return f1, probs, labels


def train_one(seed, train_ds, val_ds, class_weights, smoothing_matrix):
    torch.manual_seed(seed)
    np.random.seed(seed)

    text_encoder = build_lora_text_encoder()
    image_encoder = build_lora_image_encoder()
    if seed == SEEDS[0]:
        count_trainable(text_encoder, "text encoder (LoRA)")
        count_trainable(image_encoder, "image encoder (LoRA)")

    model = LoraFusionModel(text_encoder, image_encoder, NUM_CLASSES).to(DEVICE)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n]
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LORA_LR, "weight_decay": 0.01},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": 1e-4},
    ])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    best_val_f1 = -1.0
    best_val_probs = None
    epochs_without_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE),
                batch["token_type_ids"].to(DEVICE), batch["pixel_values"].to(DEVICE),
            )
            y_batch = batch["label"].to(DEVICE)
            loss = soft_target_loss(out, y_batch, smoothing_matrix, class_weights)
            loss.backward()
            optimizer.step()

        val_f1, val_probs, val_labels = evaluate(model, val_loader)
        print(f"  seed {seed} epoch {epoch}: val macro-F1 = {val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_probs = val_probs
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                print(f"  seed {seed}: early stopping at epoch {epoch}")
                break

    del model, text_encoder, image_encoder
    torch.cuda.empty_cache()
    return best_val_f1, best_val_probs, val_labels


def main():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    ocr_records = load_jsonl(TRANSLATION_OCR_JSONL)
    reasoning_records = load_jsonl(TRANSLATION_REASONING_JSONL)
    image_paths = load_image_paths()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    image_processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL)

    train_y = np.array([labels[i] for i in train_idx], dtype=np.int64)
    val_y = np.array([labels[i] for i in val_idx], dtype=np.int64)

    train_ds = MemeMultimodalDataset(train_idx, ocr_records, reasoning_records, image_paths, train_y,
                                      tokenizer, image_processor)
    val_ds = MemeMultimodalDataset(val_idx, ocr_records, reasoning_records, image_paths, val_y,
                                    tokenizer, image_processor)

    class_weights = class_weights_from(train_y).to(DEVICE)
    smoothing_matrix = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=DEVICE)

    seed_f1s = []
    seed_probs = []
    y_val_final = None
    for seed in SEEDS:
        val_f1, probs, y_val_final = train_one(seed, train_ds, val_ds, class_weights, smoothing_matrix)
        seed_f1s.append(val_f1)
        seed_probs.append(probs)
        print(f"seed {seed}: BEST val macro-F1 = {val_f1:.4f}\n")

    seed_preds = np.stack([p.argmax(axis=1) for p in seed_probs])
    majority_pred = np.array([
        np.bincount(seed_preds[:, i], minlength=NUM_CLASSES).argmax()
        for i in range(seed_preds.shape[1])
    ])
    majority_macro_f1 = f1_score(y_val_final, majority_pred, average="macro", zero_division=0)
    report = classification_report(
        y_val_final, majority_pred, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
        zero_division=0, output_dict=True,
    )

    print(f"\n=== LoRA fine-tuning (both encoders), majority-vote macro-F1: {majority_macro_f1:.4f} ===")
    print(f"Comparison -- Phase 3.1 frozen-encoder concat baseline: {COMPARISON_FROZEN_BASELINE}")
    print(f"Delta: {majority_macro_f1 - COMPARISON_FROZEN_BASELINE:+.4f}")
    print(f"Comparison -- Phase 3.1 partial unfreezing (failed): 0.4875")
    print("\nPer-class F1:")
    for c in CLASSES:
        print(f"  {c}: {report[c]['f1-score']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "lora_lr": LORA_LR, "head_lr": HEAD_LR,
            "seed_f1s": seed_f1s,
            "majority_vote_macro_f1": float(majority_macro_f1),
            "comparison_frozen_baseline": COMPARISON_FROZEN_BASELINE,
            "comparison_partial_unfreeze": 0.4875,
            "delta_vs_frozen_baseline": float(majority_macro_f1 - COMPARISON_FROZEN_BASELINE),
            "classification_report": report,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
