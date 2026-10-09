"""
Phase 8: verify the Phase 2 depression classifier's real-world accuracy
against its own original labeled dataset (dataset.xlsx, 4,897 records,
columns "posts"/"labels", raw labels 1-4).

Uses ONLY the existing parity-tested wrapper (phase8_2_depression_parity_wrapper.py)
-- the exact frozen checkpoint, unchanged. Nothing is trained, fine-tuned,
or modified; this is pure inference + comparison against ground truth,
same spirit as the parity test but checking correctness against real
labels instead of self-consistency against previously saved outputs.

Label mapping verified against the checkpoint's own config.json
(id2label: 0=Minimum, 1=Mild, 2=Moderate, 3=Severe) -- raw dataset label
(1-4) minus 1 = model class index.

IMPORTANT CAVEAT, stated before any results: this checkpoint
("BanglaBERT_fold5") was trained on a ~4/5 fold of this exact dataset
(confirmed via step-count analysis in Phase 8 Step 1). Evaluating on
the FULL 4,897 records therefore mixes the model's own training data
with whatever held-out portion exists -- this is a sanity/behavior
check, not an unbiased held-out accuracy estimate. Read the resulting
number as "does the model behave as expected on its own domain," not
as a true generalization measurement.
"""
import os
import json

import numpy as np
import openpyxl
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from phase8_2_depression_parity_wrapper import predict_depression, DEPRESSION_CLASSES

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset.xlsx")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase8_3_depression_dataset_verification_results.json")
PROGRESS_EVERY = 250


def load_dataset():
    wb = openpyxl.load_workbook(DATASET_PATH, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    assert header == ("posts", "labels"), f"unexpected header: {header}"
    posts = [r[0] for r in rows[1:]]
    raw_labels = [r[1] for r in rows[1:]]
    return posts, raw_labels


def main():
    print(f"Loading {DATASET_PATH} (read-only, no changes made to this file)...")
    posts, raw_labels = load_dataset()
    print(f"Loaded {len(posts)} labeled records.")

    y_true = [int(r) - 1 for r in raw_labels]  # 1-4 -> 0-3, matching model's id2label
    assert min(y_true) == 0 and max(y_true) == 3, "unexpected label range after mapping"

    print("Running the parity-tested wrapper (frozen checkpoint, unchanged) on every post...")
    y_pred = []
    for i, text in enumerate(posts):
        result = predict_depression(text)
        y_pred.append(DEPRESSION_CLASSES.index(result["label"]))
        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == len(posts):
            print(f"  processed {i + 1}/{len(posts)}", flush=True)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    report = classification_report(
        y_true, y_pred, labels=list(range(4)), target_names=DEPRESSION_CLASSES,
        zero_division=0, output_dict=True,
    )

    print(f"\n=== Verification result on dataset.xlsx (n={len(posts)}) ===")
    print("CAVEAT: this checkpoint was trained on ~4/5 of this exact dataset -- ")
    print("this is a behavior sanity check, NOT an unbiased held-out accuracy estimate.\n")
    print(f"macro-F1:    {macro_f1:.4f}")
    print(f"weighted-F1: {weighted_f1:.4f}")
    print(f"accuracy:    {acc:.4f}")
    print("\nPer-class:")
    for c in DEPRESSION_CLASSES:
        r = report[c]
        print(f"  {c}: precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1-score']:.4f} support={int(r['support'])}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    for row in cm.tolist():
        print(" ", row)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "caveat": "This checkpoint was trained on ~4/5 of this exact dataset "
                      "(confirmed via Phase 8 Step 1's step-count analysis). This result "
                      "is a behavior sanity check against the model's own training-domain "
                      "data, NOT an unbiased held-out generalization estimate.",
            "n_records": len(posts),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
