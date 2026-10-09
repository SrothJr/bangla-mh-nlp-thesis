"""
Phase 7, Step 4 (PHASE7_REVISED_PRETRAINING_PLAN.md): the real test.
Trains an actual 3-class FigSIM classifier (harmonized target, per
FIGSIM_HARMONIZATION_AND_PRETRAINING.md's mapping) -- not just the
Step 1 diagnostic's post-hoc aggregation of a 5-class model -- and
compares two variants on the exact same architecture, seeds, and
protocol, differing in only one thing:

  A. original frozen SigLIP embeddings (outputs/embeddings/siglip/)
  B. BN-HIB-pretrained LoRA-adapted SigLIP embeddings
     (outputs/embeddings/siglip_bnhib_pretrained/, Phase 7 Step 3)

DAPT-BanglaBERT is identical and untouched in both variants -- this
isolates exactly one variable: does the external BN-HIB pretraining
help, on top of the harmonized 3-class target?

Architecture: simple concat (matching Phase 3.1's and Phase 6's
comparison-baseline convention -- raw text + raw image concat -> small
MLP head), NOT the full Stage-A-aligned gated+orth pipeline. This is
deliberate: Stage A's alignment projection was specifically fit to the
ORIGINAL SigLIP embedding space, so routing variant B's shifted
embedding space through that same fixed alignment would confound the
comparison. A clean, simple, un-aligned architecture isolates the one
variable being tested, same reasoning as every prior single-variable
ablation in this project.

Validation-only; FigSIM's test set is not touched by this script.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report

from train_e6 import load_labels, load_split, load_embeddings, BANGLABERT_E3B_DIR, SIGLIP_DIR, HIDDEN_DIM
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIGLIP_PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip_bnhib_pretrained")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase7_4_harmonized_3class_comparison_results.json")

TEXT_DIM = 768
IMAGE_DIM = 1152
NUM_CLASSES_3 = 3
THREE_CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}  # from original 5-class FigSIM labels

SEEDS = [0, 1, 2]
MAX_EPOCHS = 300
PATIENCE = 30
LR = 1e-3
DEVICE = "cpu"


def load_data(image_emb_dir):
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")
    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        y5 = np.array([labels[i] for i in idx], dtype=np.int64)
        y3 = np.array([THREE_CLASS_MAP[y] for y in y5], dtype=np.int64)
        data[split_name] = {
            "text": load_embeddings(BANGLABERT_E3B_DIR, idx),
            "image": load_embeddings(image_emb_dir, idx),
            "y": y3,
        }
    return data


def class_weights_3(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES_3).astype(np.float32)
    weights = len(y_train) / (NUM_CLASSES_3 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


class ConcatClassifier3Class(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_DIM + IMAGE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES_3),
        )

    def forward(self, text_vec, image_vec):
        return self.net(torch.cat([text_vec, image_vec], dim=-1))


def train_one(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ConcatClassifier3Class().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    X_text_tr = torch.tensor(data["train"]["text"], device=DEVICE)
    X_img_tr = torch.tensor(data["train"]["image"], device=DEVICE)
    y_tr = torch.tensor(data["train"]["y"], device=DEVICE)
    X_text_val = torch.tensor(data["val"]["text"], device=DEVICE)
    X_img_val = torch.tensor(data["val"]["image"], device=DEVICE)
    y_val = torch.tensor(data["val"]["y"], device=DEVICE)

    best_val_f1 = -1.0
    best_val_pred = None
    epochs_without_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_text_tr, X_img_tr)
        loss = soft_target_loss(out, y_tr, smoothing_matrix, class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_text_val, X_img_val)
            pred_val = out_val.argmax(dim=1)
            val_f1 = f1_score(y_val.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_pred = pred_val.cpu().numpy().copy()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                break

    return best_val_f1, best_val_pred, y_val.cpu().numpy()


def majority_vote(preds_list, num_classes):
    preds = np.stack(preds_list)
    return np.array([
        np.bincount(preds[:, i], minlength=num_classes).argmax()
        for i in range(preds.shape[1])
    ])


def run_variant(name, image_emb_dir):
    print(f"\n=== Variant: {name} ===")
    data = load_data(image_emb_dir)
    print(f"Train n={len(data['train']['y'])}, Val n={len(data['val']['y'])}")
    class_weights = class_weights_3(data["train"]["y"])
    smoothing_matrix = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=DEVICE)

    seed_f1s = []
    seed_preds = []
    y_val = None
    for seed in SEEDS:
        val_f1, pred, y_val = train_one(seed, data, class_weights, smoothing_matrix)
        seed_f1s.append(val_f1)
        seed_preds.append(pred)
        print(f"  seed {seed}: val macro-F1 = {val_f1:.4f}")

    majority_pred = majority_vote(seed_preds, NUM_CLASSES_3)
    majority_f1 = f1_score(y_val, majority_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, majority_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_val, majority_pred)
    bal_acc = balanced_accuracy_score(y_val, majority_pred)
    report = classification_report(
        y_val, majority_pred, labels=list(range(NUM_CLASSES_3)), target_names=THREE_CLASS_NAMES,
        zero_division=0, output_dict=True,
    )
    print(f"  majority-vote macro-F1: {majority_f1:.4f}  weighted-F1: {weighted_f1:.4f}  "
          f"accuracy: {acc:.4f}  balanced_accuracy: {bal_acc:.4f}")
    for c in THREE_CLASS_NAMES:
        print(f"    {c}: F1={report[c]['f1-score']:.4f}")

    return {
        "seed_f1s": seed_f1s,
        "majority_vote_macro_f1": float(majority_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "classification_report": report,
    }


def main():
    result_original = run_variant("A: original frozen SigLIP", SIGLIP_DIR)
    result_pretrained = run_variant("B: BN-HIB-pretrained SigLIP", SIGLIP_PRETRAINED_DIR)

    delta = result_pretrained["majority_vote_macro_f1"] - result_original["majority_vote_macro_f1"]
    print(f"\n=== Comparison ===")
    print(f"A (original SigLIP):      {result_original['majority_vote_macro_f1']:.4f}")
    print(f"B (BN-HIB-pretrained):    {result_pretrained['majority_vote_macro_f1']:.4f}")
    print(f"Delta (B - A):            {delta:+.4f}")
    print(f"\nFor reference -- Step 1 diagnostic (aggregated, not trained): 0.6394")
    print(f"For reference -- 5-class baseline (Phase 4.1): 0.5638")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "variant_A_original_siglip": result_original,
            "variant_B_bnhib_pretrained_siglip": result_pretrained,
            "delta_B_minus_A": float(delta),
            "step1_diagnostic_reference": 0.6394,
            "five_class_baseline_reference": 0.5638,
            "decision": "kept" if delta > 0 else "not kept -- BN-HIB pretraining did not help FigSIM's own task",
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
