"""
Phase 3, item 3.1: partially unfreeze BanglaBERT's last 2 transformer
layers (of 12) and fine-tune end-to-end, jointly with the classifier head,
on the actual downstream task -- rather than treating the encoder as a
fixed feature extractor as every prior experiment in this project has.

SigLIP stays frozen (cached embeddings reused as-is) so this is a clean
single-variable test of "does adapting the text encoder to this domain
help," not a simultaneous test of unfreezing both encoders at once.

Architecture: same simple-concat classifier as the Phase 3.1 frozen
baseline (phase3_1_baseline_concat_frozen.py) -- text CLS token concat
SigLIP vector -> small MLP head -- so any difference is attributable to
the encoder being partially trainable, not a fusion-mechanism confound.

Differential learning rates: encoder's unfrozen layers at 2e-5 (standard
fine-tuning LR, much lower than the head's), classifier head at 1e-3
(same as every other experiment). Mini-batched (582 examples don't fit
the full-batch pattern used for frozen-embedding training once gradients
must flow through a 110M-parameter encoder). Early stopping on validation
macro-F1 with a much shorter patience than the frozen-embedding
experiments, since transformer fine-tuning converges in single-digit-to-
low-double-digit epochs, not hundreds.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report

from train_e6 import (
    class_weights_from, NUM_CLASSES, CLASSES, SEEDS,
    load_labels, load_split, load_embeddings, SIGLIP_DIR, HIDDEN_DIM,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAPT_ROOT = r"C:\Users\user6\T2520814\DAPT_models"
CKPT_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5", "checkpoint-735")
TOKENIZER_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5_tapt")
TRANSLATION_OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
TRANSLATION_REASONING_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_reasoning_results.jsonl")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase3_1_unfreeze_results.json")

TEXT_DIM = 768
IMAGE_DIM = 1152
MAX_LEN = 256
NUM_UNFROZEN_LAYERS = 2
ENCODER_LR = 5e-6
HEAD_LR = 1e-3
BATCH_SIZE = 16
MAX_EPOCHS = 15
PATIENCE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[r["image_index"]] = r
    return records


class MemeTextDataset(Dataset):
    def __init__(self, indices, ocr_records, reasoning_records, image_vecs, labels, tokenizer):
        self.indices = indices
        self.ocr_records = ocr_records
        self.reasoning_records = reasoning_records
        self.image_vecs = image_vecs
        self.labels = labels
        self.tokenizer = tokenizer

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
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "token_type_ids": enc["token_type_ids"][0],
            "image_vec": torch.tensor(self.image_vecs[i], dtype=torch.float32),
            "label": self.labels[i],
        }


class PartiallyUnfrozenModel(nn.Module):
    def __init__(self, encoder, num_outputs):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(TEXT_DIM + IMAGE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, image_vec):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        text_vec = out.last_hidden_state[:, 0, :]  # CLS token
        z = torch.cat([text_vec, image_vec], dim=-1)
        return self.head(z)


def build_encoder(num_unfrozen_layers):
    encoder = AutoModel.from_pretrained(CKPT_PATH)
    for param in encoder.parameters():
        param.requires_grad = False
    total_layers = len(encoder.encoder.layer)
    for layer in encoder.encoder.layer[total_layers - num_unfrozen_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder: {n_trainable:,} / {n_total:,} parameters trainable "
          f"(last {num_unfrozen_layers} of {total_layers} layers)")
    return encoder


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE),
                batch["token_type_ids"].to(DEVICE), batch["image_vec"].to(DEVICE),
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

    encoder = build_encoder(NUM_UNFROZEN_LAYERS)
    model = PartiallyUnfrozenModel(encoder, NUM_CLASSES).to(DEVICE)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": ENCODER_LR, "weight_decay": 0.01},
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
                batch["token_type_ids"].to(DEVICE), batch["image_vec"].to(DEVICE),
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

    del model, encoder
    torch.cuda.empty_cache()
    return best_val_f1, best_val_probs, val_labels


def main():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    ocr_records = load_jsonl(TRANSLATION_OCR_JSONL)
    reasoning_records = load_jsonl(TRANSLATION_REASONING_JSONL)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    train_img = load_embeddings(SIGLIP_DIR, train_idx)
    val_img = load_embeddings(SIGLIP_DIR, val_idx)
    train_y = np.array([labels[i] for i in train_idx], dtype=np.int64)
    val_y = np.array([labels[i] for i in val_idx], dtype=np.int64)

    train_ds = MemeTextDataset(train_idx, ocr_records, reasoning_records, train_img, train_y, tokenizer)
    val_ds = MemeTextDataset(val_idx, ocr_records, reasoning_records, val_img, val_y, tokenizer)

    class_weights = class_weights_from(train_y)
    smoothing_matrix = build_smoothing_matrix(NUM_CLASSES, TAU)
    class_weights_t = torch.tensor(class_weights, device=DEVICE) if not torch.is_tensor(class_weights) else class_weights.to(DEVICE)
    smoothing_matrix_t = torch.tensor(smoothing_matrix, device=DEVICE)

    seed_f1s = []
    seed_probs = []
    y_val_final = None
    for seed in SEEDS:
        val_f1, probs, y_val_final = train_one(seed, train_ds, val_ds, class_weights_t, smoothing_matrix_t)
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

    print(f"\n=== Partial unfreezing (last {NUM_UNFROZEN_LAYERS} BanglaBERT layers), "
          f"majority-vote macro-F1: {majority_macro_f1:.4f} ===")
    print(f"Comparison -- Phase 3.1 frozen-encoder concat baseline: 0.5139")
    print(f"Delta: {majority_macro_f1 - 0.5139:+.4f}")
    print(f"\nPer-class F1:")
    for c in CLASSES:
        print(f"  {c}: {report[c]['f1-score']:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "num_unfrozen_layers": NUM_UNFROZEN_LAYERS,
            "encoder_lr": ENCODER_LR, "head_lr": HEAD_LR,
            "seed_f1s": seed_f1s,
            "majority_vote_macro_f1": float(majority_macro_f1),
            "comparison_frozen_baseline": 0.5139,
            "delta": float(majority_macro_f1 - 0.5139),
            "classification_report": report,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
