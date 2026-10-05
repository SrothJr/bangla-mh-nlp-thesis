"""
Day 3, Step 4: E3b -- DAPT-BanglaBERT, OCR text + AI reasoning text
(Section 8). Same LogisticRegression classifier config as E1/E3 in
build_baselines_e0_e1_e3.py, so the E3 vs E3b comparison is clean (only the
input embeddings differ, not the classifier).

Protocol (Section 8): reported on the VALIDATION split -- test stays locked
until Day 7.
"""
import os
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
E3B_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_e3b")
PRIOR_RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "e0_e1_e3_results.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "e3b_results.json")

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
    import csv
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


def main():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    y_train = np.array([labels[i] for i in train_idx])
    y_val = np.array([labels[i] for i in val_idx])

    X_train = load_embeddings(E3B_DIR, train_idx)
    X_val = load_embeddings(E3B_DIR, val_idx)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_val, y_pred)

    print(f"\n=== E3b text (DAPT-BanglaBERT, OCR + AI reasoning) (validation split, n={len(y_val)}) ===")
    print(f"macro-F1:    {macro_f1:.4f}")
    print(f"weighted-F1: {weighted_f1:.4f}")
    print(f"accuracy:    {acc:.4f}")
    print(classification_report(
        y_val, y_pred, labels=list(range(5)), target_names=CLASSES, zero_division=0
    ))
    cm = confusion_matrix(y_val, y_pred, labels=list(range(5)))
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    result = {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "n_val": len(y_val),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")

    if os.path.exists(PRIOR_RESULTS_PATH):
        with open(PRIOR_RESULTS_PATH, "r", encoding="utf-8") as f:
            prior = json.load(f)
        e3_f1 = prior["E3"]["macro_f1"]
        delta = macro_f1 - e3_f1
        print(f"\n=== E3 vs E3b (macro-F1, validation split) ===")
        print(f"E3  (OCR only):          {e3_f1:.4f}")
        print(f"E3b (OCR + reasoning):   {macro_f1:.4f}")
        print(f"Delta:                   {delta:+.4f}")


if __name__ == "__main__":
    main()
