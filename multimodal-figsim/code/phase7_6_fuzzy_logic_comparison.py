"""
PHASE7_FUZZY_LOGIC_5CLASS_VS_3CLASS_PLAN.md -- full workflow.

Runs the fuzzy-logic risk-combination layer twice on FigSIM's
VALIDATION split (not test -- the 3-class model has never been locked
or touched against test, so both runs use validation for a fair,
equal-footing comparison):

  1. Current best 5-class model (tuned gated + cross-attention
     ensemble, matching Phase 4.4) + the existing 11-row 5-class rule
     table.
  2. Current best 3-class model (simple concat, original SigLIP,
     matching Phase 7 Step 4 variant A) + the new 6-row 3-class rule
     table derived in the plan doc via a documented cautious-merge
     policy.

Phase 2's depression classifier is identical and untouched in both --
run once on the validation split's translated OCR text, reused for
both combinations.
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

from train_e6 import (
    load_aligned_data, class_weights_from, NUM_CLASSES, SEEDS, DEVICE as GATED_DEVICE,
    load_labels, load_split, load_embeddings, BANGLABERT_E3B_DIR, SIGLIP_DIR, HIDDEN_DIM,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
from phase4_1_hyperparam_sweep import train_one as train_gated_tuned, DEFAULT
from phase3_2_cross_attention import load_data as load_data_xattn, train_one as train_xattn, DEVICE as XATTN_DEVICE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAPT_ROOT = r"C:\Users\user6\T2520814\DAPT_models"
CKPT_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5", "checkpoint-735")
TOKENIZER_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5_tapt")
TRANSLATION_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase7_6_fuzzy_logic_comparison_results.json")

TUNED_CFG = {**DEFAULT, "dropout": 0.4}
DEPRESSION_CLASSES = ["Minimum", "Mild", "Moderate", "Severe"]
SUICIDE_CLASSES_5 = ["None", "Wish to be dead", "Suicide ideation", "Suicide planning", "Suicide attempt or death"]
SUICIDE_CLASSES_3 = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
RISK_LEVELS = ["Minimal", "Low", "Elevated", "Critical"]
MAX_LEN = 256

RULES_5CLASS = [
    (4, [0, 1, 2, 3], "Critical"),
    (3, [2, 3], "Critical"),
    (3, [0, 1], "Elevated"),
    (2, [2, 3], "Elevated"),
    (2, [1], "Elevated"),
    (2, [0], "Low"),
    (1, [2, 3], "Elevated"),
    (1, [0, 1], "Low"),
    (0, [3], "Elevated"),
    (0, [2], "Low"),
    (0, [0, 1], "Minimal"),
]

RULES_3CLASS = [
    (2, [0, 1, 2, 3], "Critical"),
    (1, [1, 2, 3], "Elevated"),
    (1, [0], "Low"),
    (0, [3], "Elevated"),
    (0, [2], "Low"),
    (0, [0, 1], "Minimal"),
]


def combine_fuzzy(suicide_probs, depression_probs, rules, suicide_classes):
    risk_membership = {r: 0.0 for r in RISK_LEVELS}
    firing_strengths = []
    for suicide_idx, depression_idxs, risk in rules:
        depression_condition = float(max(depression_probs[i] for i in depression_idxs))
        firing_strength = float(min(suicide_probs[suicide_idx], depression_condition))
        firing_strengths.append({
            "suicide_level": suicide_classes[suicide_idx],
            "depression_condition": "/".join(DEPRESSION_CLASSES[i] for i in depression_idxs),
            "firing_strength": firing_strength,
            "risk": risk,
        })
        risk_membership[risk] = max(risk_membership[risk], firing_strength)
    crisp_risk = max(risk_membership, key=risk_membership.get)
    return risk_membership, crisp_risk, firing_strengths


def majority_vote_fractions(prob_list, num_classes):
    preds = np.stack([p.argmax(axis=1) for p in prob_list])
    n = preds.shape[1]
    fractions = np.zeros((n, num_classes))
    for c in range(num_classes):
        fractions[:, c] = (preds == c).sum(axis=0) / preds.shape[0]
    return fractions


def get_5class_val_vote_fractions():
    """Reruns the current best 5-class ensemble (tuned gated + cross-attention,
    matching Phase 4.4) and returns per-example vote fractions on validation."""
    print("=== Regenerating 5-class ensemble validation predictions ===")
    gated_data = load_aligned_data()
    gated_class_weights = class_weights_from(gated_data["train"]["y"])
    smoothing_matrix_gated = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=GATED_DEVICE)

    gated_probs = []
    y_val = None
    for seed in SEEDS:
        val_f1, probs, y_val = train_gated_tuned(seed, gated_data, gated_class_weights, smoothing_matrix_gated, TUNED_CFG)
        gated_probs.append(probs)
        print(f"  gated seed {seed}: val macro-F1 = {val_f1:.4f}")

    xattn_data = load_data_xattn()
    xattn_class_weights = class_weights_from(xattn_data["train"]["y"]).to(XATTN_DEVICE)
    smoothing_matrix_xattn = torch.tensor(build_smoothing_matrix(NUM_CLASSES, TAU), device=XATTN_DEVICE)

    xattn_probs = []
    for seed in SEEDS:
        val_f1, probs, _ = train_xattn(seed, xattn_data, xattn_class_weights, smoothing_matrix_xattn)
        xattn_probs.append(probs)
        print(f"  cross-attn seed {seed}: val macro-F1 = {val_f1:.4f}")

    all_probs = gated_probs + xattn_probs
    vote_fractions = majority_vote_fractions(all_probs, NUM_CLASSES)
    pred = vote_fractions.argmax(axis=1)
    f1 = f1_score(y_val, pred, average="macro", zero_division=0)
    print(f"5-class ensemble majority-vote macro-F1 (sanity check vs known 0.5638-0.5695): {f1:.4f}")
    return vote_fractions, y_val


def get_3class_val_vote_fractions():
    """Reruns the current best 3-class model (simple concat, original SigLIP,
    matching Phase 7 Step 4 variant A) and returns per-example vote fractions."""
    print("\n=== Regenerating 3-class model validation predictions ===")
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        y5 = np.array([labels[i] for i in idx], dtype=np.int64)
        y3 = np.array([THREE_CLASS_MAP[y] for y in y5], dtype=np.int64)
        data[split_name] = {
            "text": load_embeddings(BANGLABERT_E3B_DIR, idx),
            "image": load_embeddings(SIGLIP_DIR, idx),
            "y": y3,
        }

    class ConcatClassifier3Class(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(768 + 1152, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2), nn.Linear(HIDDEN_DIM, 3),
            )

        def forward(self, text_vec, image_vec):
            return self.net(torch.cat([text_vec, image_vec], dim=-1))

    from phase1_3_ordinal_smoothing import soft_target_loss

    def class_weights_3(y_train):
        counts = np.bincount(y_train, minlength=3).astype(np.float32)
        weights = len(y_train) / (3 * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float32)

    class_weights = class_weights_3(data["train"]["y"])
    smoothing_matrix = torch.tensor(build_smoothing_matrix(3, TAU), device="cpu")

    seed_probs = []
    y_val = None
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ConcatClassifier3Class()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        X_text_tr = torch.tensor(data["train"]["text"])
        X_img_tr = torch.tensor(data["train"]["image"])
        y_tr = torch.tensor(data["train"]["y"])
        X_text_val = torch.tensor(data["val"]["text"])
        X_img_val = torch.tensor(data["val"]["image"])
        y_val = torch.tensor(data["val"]["y"])

        best_val_f1 = -1.0
        best_probs = None
        epochs_without_improve = 0
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            out = model(X_text_tr, X_img_tr)
            loss = soft_target_loss(out, y_tr, smoothing_matrix, class_weights)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out_val = model(X_text_val, X_img_val)
                probs_val = torch.softmax(out_val, dim=1)
                pred_val = probs_val.argmax(dim=1)
                val_f1 = f1_score(y_val.numpy(), pred_val.numpy(), average="macro", zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_probs = probs_val.detach().numpy().copy()
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= 30:
                    break
        seed_probs.append(best_probs)
        print(f"  seed {seed}: val macro-F1 = {best_val_f1:.4f}")

    vote_fractions = majority_vote_fractions(seed_probs, 3)
    pred = vote_fractions.argmax(axis=1)
    f1 = f1_score(y_val.numpy(), pred, average="macro", zero_division=0)
    print(f"3-class majority-vote macro-F1 (sanity check vs known 0.6425): {f1:.4f}")
    return vote_fractions, y_val.numpy()


def get_depression_probs_for_val():
    print("\n=== Running Phase-2 depression classifier on validation OCR text ===")
    val_idx = load_split("val")
    ocr_texts = {}
    with open(TRANSLATION_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ocr_texts[r["image_index"]] = r["ocr_text_bn"]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT_PATH)
    model.eval()

    all_probs = []
    for idx in val_idx:
        text = ocr_texts.get(idx, "") or ""
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0].numpy()
        all_probs.append(probs)
    print(f"Depression classifier run on {len(val_idx)} validation memes.")
    return val_idx, np.stack(all_probs)


def main():
    probs_5class, y_val_5 = get_5class_val_vote_fractions()
    probs_3class, y_val_3 = get_3class_val_vote_fractions()
    val_idx, depression_probs = get_depression_probs_for_val()

    results_5class = []
    results_3class = []
    risk_counts_5 = {r: 0 for r in RISK_LEVELS}
    risk_counts_3 = {r: 0 for r in RISK_LEVELS}
    agreement_matrix = {f"{a}->{b}": 0 for a in RISK_LEVELS for b in RISK_LEVELS}

    for i, idx in enumerate(val_idx):
        dep_probs = depression_probs[i]

        risk_membership_5, crisp_risk_5, _ = combine_fuzzy(probs_5class[i], dep_probs, RULES_5CLASS, SUICIDE_CLASSES_5)
        risk_counts_5[crisp_risk_5] += 1
        results_5class.append({
            "image_index": idx,
            "suicide_pred": SUICIDE_CLASSES_5[int(probs_5class[i].argmax())],
            "suicide_true_5class": SUICIDE_CLASSES_5[int(y_val_5[i])],
            "depression_pred": DEPRESSION_CLASSES[int(dep_probs.argmax())],
            "crisp_risk": crisp_risk_5,
        })

        risk_membership_3, crisp_risk_3, _ = combine_fuzzy(probs_3class[i], dep_probs, RULES_3CLASS, SUICIDE_CLASSES_3)
        risk_counts_3[crisp_risk_3] += 1
        results_3class.append({
            "image_index": idx,
            "suicide_pred": SUICIDE_CLASSES_3[int(probs_3class[i].argmax())],
            "suicide_true_3class": SUICIDE_CLASSES_3[int(y_val_3[i])],
            "depression_pred": DEPRESSION_CLASSES[int(dep_probs.argmax())],
            "crisp_risk": crisp_risk_3,
        })

        agreement_matrix[f"{crisp_risk_5}->{crisp_risk_3}"] += 1

    n = len(val_idx)
    agree_count = sum(agreement_matrix[f"{r}->{r}"] for r in RISK_LEVELS)

    print(f"\n=== 5-class-fuzzy risk distribution (n={n}) ===")
    for r in RISK_LEVELS:
        print(f"  {r}: {risk_counts_5[r]} ({100 * risk_counts_5[r] / n:.1f}%)")
    print(f"\n=== 3-class-fuzzy risk distribution (n={n}) ===")
    for r in RISK_LEVELS:
        print(f"  {r}: {risk_counts_3[r]} ({100 * risk_counts_3[r] / n:.1f}%)")
    print(f"\n=== Agreement: {agree_count}/{n} ({100 * agree_count / n:.1f}%) same final risk level ===")
    print("Disagreement breakdown (5-class -> 3-class):")
    for a in RISK_LEVELS:
        for b in RISK_LEVELS:
            if a != b and agreement_matrix[f"{a}->{b}"] > 0:
                print(f"  {a} -> {b}: {agreement_matrix[f'{a}->{b}']}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_val": n,
            "risk_distribution_5class": risk_counts_5,
            "risk_distribution_3class": risk_counts_3,
            "agreement_count": agree_count,
            "agreement_pct": 100 * agree_count / n,
            "agreement_matrix": agreement_matrix,
            "results_5class": results_5class,
            "results_3class": results_3class,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
