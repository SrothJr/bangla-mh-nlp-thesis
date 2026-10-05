"""
Day 7: final locked configuration -- E6 gated+orth (Stage A aligned
representations, weighted CE) -- trained with 3 seeds exactly as in Day 5-6,
then the test split (196 items, untouched since the leakage-safe split was
built) is unlocked EXACTLY ONCE to report final metrics.

Protocol discipline (Section 8): model selection (which epoch to keep, which
seed is "primary" for the detailed report) is decided ENTIRELY from
validation performance, before test is touched at all. Test is scored once,
here, in a single non-interactive pass -- no threshold tuning or re-runs
based on what test shows.

Reports macro-F1, weighted-F1, per-class precision/recall/F1, confusion
matrix, and quadratic weighted kappa (Section 8), for each seed and as a
3-seed mean +/- std, plus a detailed breakdown for the seed with the best
VALIDATION macro-F1 (chosen without looking at test).
"""
import os
import csv
import json
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix, cohen_kappa_score,
)

from train_contrastive_align import AlignmentProjections, CKPT_DIR as ALIGN_CKPT_DIR
from train_e6 import E6GatedOrth, load_confidences, class_weights_from, ALIGN_SHARED_DIM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
LEAKAGE_SAFE_DIR = os.path.join(FIGSIM_ROOT, "data", "leakage_safe")
INDEX_CSV = os.path.join(LEAKAGE_SAFE_DIR, "figsim_leakage_safe_index.csv")
SIGLIP_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip")
BANGLABERT_E3B_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_e3b")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "final_test_results.json")

CLASSES = [
    "None", "Wish to be dead", "Suicide ideation",
    "Suicide planning", "Suicide attempt or death",
]
NUM_CLASSES = 5
RAW_LABEL_TO_ID = {
    "None": 0, "Wish to be dead": 1, "Suicide ideation": 2,
    "Suicide planning": 3, "Suicide attempts": 4, "Suicide death": 4,
}
SEEDS = [0, 1, 2]
MAX_EPOCHS = 300
PATIENCE = 30
LR = 1e-3
DEVICE = "cpu"


def load_labels():
    labels = {}
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["include_in_leakage_safe_dataset"] == "True":
                labels[int(row["image_index"])] = RAW_LABEL_TO_ID[row["suicide_scale"]]
    return labels


def load_split(name):
    with open(os.path.join(LEAKAGE_SAFE_DIR, f"{name}.txt"), "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_embeddings(emb_dir, indices):
    return np.stack([np.load(os.path.join(emb_dir, f"{i}.npy")) for i in indices]).astype(np.float32)


def load_all_data():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")
    test_idx = load_split("test")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    align_model = AlignmentProjections()
    ckpt_path = os.path.join(ALIGN_CKPT_DIR, "current.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    align_model.load_state_dict(ckpt["state_dict"])
    align_model.eval()
    print(f"Loaded Stage A alignment checkpoint from epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}")

    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        raw_text = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, idx))
        raw_image = torch.tensor(load_embeddings(SIGLIP_DIR, idx))
        with torch.no_grad():
            aligned_text, aligned_image = align_model(raw_text, raw_image)
        data[split_name] = {
            "text": aligned_text.numpy(),
            "image": aligned_image.numpy(),
            "raw_text": raw_text.numpy(),
            "conf": load_confidences(idx),
            "y": np.array([labels[i] for i in idx], dtype=np.int64),
            "indices": idx,
        }

    conf_mean = data["train"]["conf"].mean(axis=0, keepdims=True)
    conf_std = data["train"]["conf"].std(axis=0, keepdims=True) + 1e-6
    for split_name in ("train", "val", "test"):
        data[split_name]["conf"] = (data[split_name]["conf"] - conf_mean) / conf_std

    return data


def to_tensors(split_data):
    return (
        torch.tensor(split_data["text"], device=DEVICE),
        torch.tensor(split_data["image"], device=DEVICE),
        torch.tensor(split_data["raw_text"], device=DEVICE),
        torch.tensor(split_data["conf"], device=DEVICE),
        torch.tensor(split_data["y"], device=DEVICE),
    )


def train_one_seed(seed, data, class_weights):
    """Train on train split, model-select on val split ONLY. Returns the
    best-val-epoch model (deep-copied state), never touches test."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = E6GatedOrth(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr, y_tr = to_tensors(data["train"])
    X_text_val, X_img_val, X_rawtext_val, X_conf_val, y_val = to_tensors(data["val"])

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr)
        loss = nn.functional.cross_entropy(out, y_tr, weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_text_val, X_img_val, X_rawtext_val, X_conf_val)
            pred_val = out_val.argmax(dim=1)
            val_f1 = f1_score(y_val.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                break

    return best_val_f1, best_state


def evaluate_on_test(state_dict, data):
    model = E6GatedOrth(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    X_text_test, X_img_test, X_rawtext_test, X_conf_test, y_test = to_tensors(data["test"])
    with torch.no_grad():
        out_test = model(X_text_test, X_img_test, X_rawtext_test, X_conf_test)
        pred_test = out_test.argmax(dim=1)
        alpha = model.last_alpha.cpu().numpy().flatten()

    y_true = y_test.cpu().numpy()
    y_pred = pred_test.cpu().numpy()

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASSES,
        zero_division=0, output_dict=True,
    )

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(acc),
        "quadratic_weighted_kappa": float(qwk),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_test": len(y_true),
        "gate_alpha": {
            "mean": float(alpha.mean()), "std": float(alpha.std()),
            "min": float(alpha.min()), "max": float(alpha.max()),
        },
    }


def main():
    print("=" * 70)
    print("STEP 1: train 3 seeds, model-select on VALIDATION ONLY.")
    print("Test set (196 items) is not touched during this step.")
    print("=" * 70)
    data = load_all_data()
    class_weights = class_weights_from(data["train"]["y"])

    seed_val_f1s = {}
    seed_states = {}
    for seed in SEEDS:
        val_f1, state = train_one_seed(seed, data, class_weights)
        seed_val_f1s[seed] = val_f1
        seed_states[seed] = state
        print(f"seed {seed}: best VAL macro-F1 = {val_f1:.4f}", flush=True)

    primary_seed = max(seed_val_f1s, key=seed_val_f1s.get)
    print(f"\nPrimary seed selected by VALIDATION performance only: {primary_seed} "
          f"(val macro-F1={seed_val_f1s[primary_seed]:.4f})")

    print("\n" + "=" * 70)
    print("STEP 2: unlocking the test set ONCE. Evaluating all 3 seeds.")
    print("=" * 70)

    per_seed_test = {}
    for seed in SEEDS:
        test_result = evaluate_on_test(seed_states[seed], data)
        per_seed_test[seed] = test_result
        print(f"seed {seed}: TEST macro-F1={test_result['macro_f1']:.4f} "
              f"weighted-F1={test_result['weighted_f1']:.4f} "
              f"accuracy={test_result['accuracy']:.4f} "
              f"QWK={test_result['quadratic_weighted_kappa']:.4f}")

    macro_f1s = [per_seed_test[s]["macro_f1"] for s in SEEDS]
    weighted_f1s = [per_seed_test[s]["weighted_f1"] for s in SEEDS]
    accs = [per_seed_test[s]["accuracy"] for s in SEEDS]
    qwks = [per_seed_test[s]["quadratic_weighted_kappa"] for s in SEEDS]

    summary = {
        "macro_f1_mean": float(np.mean(macro_f1s)), "macro_f1_std": float(np.std(macro_f1s)),
        "weighted_f1_mean": float(np.mean(weighted_f1s)), "weighted_f1_std": float(np.std(weighted_f1s)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "qwk_mean": float(np.mean(qwks)), "qwk_std": float(np.std(qwks)),
    }

    print("\n=== FINAL TEST METRICS (3-seed mean +/- std) ===")
    print(f"macro-F1:    {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    print(f"weighted-F1: {summary['weighted_f1_mean']:.4f} +/- {summary['weighted_f1_std']:.4f}")
    print(f"accuracy:    {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"QWK:         {summary['qwk_mean']:.4f} +/- {summary['qwk_std']:.4f}")

    primary_result = per_seed_test[primary_seed]
    print(f"\n=== DETAILED BREAKDOWN, primary seed {primary_seed} "
          f"(selected by VAL, not test) ===")
    print(f"macro-F1: {primary_result['macro_f1']:.4f}, "
          f"weighted-F1: {primary_result['weighted_f1']:.4f}, "
          f"accuracy: {primary_result['accuracy']:.4f}, "
          f"QWK: {primary_result['quadratic_weighted_kappa']:.4f}")
    print("\nPer-class:")
    for cls in CLASSES:
        r = primary_result["classification_report"][cls]
        print(f"  {cls}: precision={r['precision']:.4f} recall={r['recall']:.4f} "
              f"f1={r['f1-score']:.4f} support={int(r['support'])}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    for row in primary_result["confusion_matrix"]:
        print(" ", row)
    print(f"\nGate alpha on test (primary seed): {primary_result['gate_alpha']}")

    full_results = {
        "primary_seed": primary_seed,
        "seed_val_f1_used_for_selection": seed_val_f1s,
        "per_seed_test_results": per_seed_test,
        "three_seed_summary": summary,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
