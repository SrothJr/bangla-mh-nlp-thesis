"""
utils_aug.py — Shared utilities for augmented-dataset training scripts.

Extended from DAPT_models/scripts/utils.py with:
  - Per-class F1/P/R/support in both Trainer-compatible and standalone forms
  - Structured metrics JSON saving
  - Consistent plot style with original

Do NOT import from DAPT_models/scripts/utils.py — this file is self-contained
so both train_aug_base.py and train_aug_dapt.py can be run from any CWD.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


CLASSES = ["Minimum", "Mild", "Moderate", "Severe"]


# ---------------------------------------------------------------------------
# Trainer-compatible compute_metrics
# Used as the callback inside TrainingArguments / Trainer.
# Returns weighted aggregates only — Trainer uses these for early stopping
# and best-model selection (metric_for_best_model="f1").
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """
    Trainer-compatible callback.
    Returns: accuracy, f1 (weighted), precision (weighted), recall (weighted).
    """
    labels = eval_pred.label_ids
    predictions = (
        eval_pred.predictions[0]
        if isinstance(eval_pred.predictions, tuple)
        else eval_pred.predictions
    )
    preds = predictions.argmax(-1) if len(predictions.shape) > 1 else predictions

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy":  float(acc),
        "f1":        float(f1),
        "precision": float(precision),
        "recall":    float(recall),
    }


# ---------------------------------------------------------------------------
# Standalone full metrics — called once on the final test set
# ---------------------------------------------------------------------------
def compute_full_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute weighted aggregates AND per-class metrics from raw label lists.
    Called after training for the final test-set evaluation.

    Returns a structured dict suitable for saving to metrics.json.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    # Per-class breakdown — critical for thesis minority-class analysis
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for cls in CLASSES:
        per_class[cls] = {
            "f1":        round(float(report[cls]["f1-score"]),  4),
            "precision": round(float(report[cls]["precision"]), 4),
            "recall":    round(float(report[cls]["recall"]),    4),
            "support":   int(report[cls]["support"]),
        }

    return {
        "accuracy":            round(float(acc),       4),
        "f1_weighted":         round(float(f1),        4),
        "precision_weighted":  round(float(precision), 4),
        "recall_weighted":     round(float(recall),    4),
        "per_class":           per_class,
    }


# ---------------------------------------------------------------------------
# Save structured metrics JSON
# ---------------------------------------------------------------------------
def save_metrics(
    path: str,
    model_name: str,
    experiment: str,
    val_metrics: dict,
    test_metrics: dict,
    training_params: dict,
) -> None:
    """
    Write a complete, structured metrics file.

    Parameters
    ----------
    path            : Full path to the output .json file.
    model_name      : e.g. "BanglaBERT"
    experiment      : "aug_base" or "aug_dapt"
    val_metrics     : Dict returned by compute_metrics (from Trainer, on val set).
    test_metrics    : Dict returned by compute_full_metrics (on locked test set).
    training_params : Dict of key hyperparameters for reproducibility.
    """
    payload = {
        "model":            model_name,
        "experiment":       experiment,
        "val_metrics":      val_metrics,
        "test_metrics":     test_metrics,
        "training_params":  training_params,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  [Metrics saved] {path}")


# ---------------------------------------------------------------------------
# Confusion matrix plot — identical style to original utils.py
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    title: str,
    output_path: str,
) -> None:
    """
    Save a 300-dpi confusion matrix heatmap.
    Uses CLASSES = ["Minimum", "Mild", "Moderate", "Severe"].
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [Figure saved] {output_path}")


# ---------------------------------------------------------------------------
# Performance comparison bar chart
# ---------------------------------------------------------------------------
def plot_performance(results_dict: dict, output_path: str, title: str) -> None:
    """
    Bar chart comparing accuracy and weighted F1 across models.

    Parameters
    ----------
    results_dict : { model_name: {"accuracy": float, "f1_weighted": float} }
    output_path  : Save path for the figure.
    title        : Chart title string.
    """
    models = list(results_dict.keys())
    accs   = [results_dict[m]["accuracy"]    for m in models]
    f1s    = [results_dict[m]["f1_weighted"] for m in models]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, accs, width, label="Accuracy",  color="#2b6b8e")
    rects2 = ax.bar(x + width / 2, f1s,  width, label="F1 (Weighted)", color="#45a081")

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="lower right")
    ax.set_ylim([0.0, 1.0])

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [Figure saved] {output_path}")
