"""
Day 2, Step 3: E0 (majority-class), E1 (image-only, SigLIP), E3 (text-only,
DAPT-BanglaBERT OCR-only) from Section 8's ablation ladder.

Protocol (Section 8): fixed 582/195/196 split. The test set stays LOCKED
until the final configuration is chosen on Day 7 -- so every number here is
reported on the VALIDATION split (195 items), not test.

Label scale (Section 3, 5-class merged):
  0 None
  1 Wish to be dead
  2 Suicide ideation
  3 Suicide planning
  4 Suicide attempt or death   (merged: "Suicide attempts" + "Suicide death")
"""
import os
import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
LEAKAGE_SAFE_DIR = os.path.join(FIGSIM_ROOT, "data", "leakage_safe")
INDEX_CSV = os.path.join(LEAKAGE_SAFE_DIR, "figsim_leakage_safe_index.csv")
SIGLIP_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip")
BANGLABERT_OCR_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_ocr")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "e0_e1_e3_results.json")

CLASSES = [
    "None", "Wish to be dead", "Suicide ideation",
    "Suicide planning", "Suicide attempt or death",
]
RAW_LABEL_TO_ID = {
    "None": 0,
    "Wish to be dead": 1,
    "Suicide ideation": 2,
    "Suicide planning": 3,
    "Suicide attempts": 4,
    "Suicide death": 4,
}


def load_labels():
    labels = {}
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["include_in_leakage_safe_dataset"] == "True":
                idx = int(row["image_index"])
                labels[idx] = RAW_LABEL_TO_ID[row["suicide_scale"]]
    return labels


def load_split(name):
    path = os.path.join(LEAKAGE_SAFE_DIR, f"{name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_embeddings(emb_dir, indices):
    return np.stack([np.load(os.path.join(emb_dir, f"{i}.npy")) for i in indices])


def report(name, y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} (validation split, n={len(y_true)}) ===")
    print(f"macro-F1:    {macro_f1:.4f}")
    print(f"weighted-F1: {weighted_f1:.4f}")
    print(f"accuracy:    {acc:.4f}")
    print(classification_report(
        y_true, y_pred, labels=list(range(5)), target_names=CLASSES, zero_division=0
    ))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)
    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "n_val": len(y_true),
    }


def main():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")
    test_idx = load_split("test")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} "
          f"(test is LOCKED -- not touched here)")

    y_train = np.array([labels[i] for i in train_idx])
    y_val = np.array([labels[i] for i in val_idx])

    train_dist = {CLASSES[c]: int((y_train == c).sum()) for c in range(5)}
    print(f"Train label distribution: {train_dist}")

    results = {}

    # ---------------- E0: majority-class baseline ----------------
    majority_class = int(np.bincount(y_train, minlength=5).argmax())
    y_pred_e0 = np.full_like(y_val, majority_class)
    results["E0"] = report(
        f"E0 majority-class baseline (predicts '{CLASSES[majority_class]}' always)",
        y_val, y_pred_e0,
    )

    # ---------------- E1: image-only (SigLIP) ----------------
    X_train_img = load_embeddings(SIGLIP_DIR, train_idx)
    X_val_img = load_embeddings(SIGLIP_DIR, val_idx)
    clf_img = LogisticRegression(max_iter=2000)
    clf_img.fit(X_train_img, y_train)
    y_pred_e1 = clf_img.predict(X_val_img)
    results["E1"] = report("E1 image-only (SigLIP embeddings + logistic regression)", y_val, y_pred_e1)

    # ---------------- E3: text-only, DAPT-BanglaBERT, OCR-only ----------------
    X_train_txt = load_embeddings(BANGLABERT_OCR_DIR, train_idx)
    X_val_txt = load_embeddings(BANGLABERT_OCR_DIR, val_idx)
    clf_txt = LogisticRegression(max_iter=2000)
    clf_txt.fit(X_train_txt, y_train)
    y_pred_e3 = clf_txt.predict(X_val_txt)
    results["E3"] = report(
        "E3 text-only (DAPT-BanglaBERT, OCR-only, fold5/checkpoint-735 body)", y_val, y_pred_e3
    )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")

    print("\n=== SUMMARY (macro-F1, validation split) ===")
    for k in ("E0", "E1", "E3"):
        print(f"{k}: {results[k]['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
