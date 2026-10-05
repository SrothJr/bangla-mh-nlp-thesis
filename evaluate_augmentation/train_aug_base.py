"""
train_aug_base.py — Fine-tune BASE models on the augmented dataset.

Experiment:  aug_base
Models:      BanglaBERT, sahajBERT, mBERT  (HuggingFace hub weights)
Train data:  train_augmented_qwen3-32b.xlsx  (17,131 rows: 3,426 original + 13,705 synthetic)
Val data:    locked/val.xlsx                 (733 rows, original only — early stopping)
Test data:   locked/test.xlsx                (738 rows, original only — ONE final evaluation)

Evaluation protocol:
  Fixed 70-15-15 split. 5-fold CV is NOT used because the augmentation
  was generated from the fixed 70% train partition; folding would create
  data leakage between synthetic posts and their source originals.
  Both val and test contain ONLY original data, so evaluation is always
  against the true data distribution — results are directly comparable
  to the baseline 5-fold CV numbers.

Hardware: RTX 5090, 32 GB VRAM.
Batch sizes are model-specific (proven in DAPT worklog):
  BanglaBERT — batch=16, accum=1  (12 layers, 768 hidden, 32K vocab)
  sahajBERT  — batch=8,  accum=2  (24-layer ALBERT, activations not shared → OOM at 16)
  mBERT      — batch=8,  accum=2  (119K vocab → huge output tensor at 16)
  All effective batch size = 16, matching fine-tuning convention (Devlin et al., 2019).

Consistency with prior experiments:
  max_length=256     — matches DAPT pretraining and baseline fine-tuning exactly
  learning_rate=2e-5 — same as train_baseline.py and train_dapt_eval.py
  seed=42            — same as all prior experiments
  metric_for_best_model="f1" — same as train_baseline.py (line 116) and
                                train_dapt_eval.py (line 315)
  fp16=True          — same as all prior experiments
"""

import os
import gc
import time
import json
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from utils_aug import (
    compute_metrics,
    compute_full_metrics,
    save_metrics,
    plot_confusion_matrix,
    plot_performance,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths — all relative to this script's location (eval_augmentation/)
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(_HERE, "train_augmented_qwen3-32b.xlsx")
VAL_PATH    = os.path.join(_HERE, "locked", "val.xlsx")
TEST_PATH   = os.path.join(_HERE, "locked", "test.xlsx")
OUTPUT_DIR  = os.path.join(_HERE, "results", "aug_base")
FIG_DIR     = os.path.join(_HERE, "results", "figures")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "BanglaBERT": "csebuetnlp/banglabert",
    "sahajBERT":  "neuropark/sahajBERT",
    "mBERT":      "bert-base-multilingual-cased",
}

# Per-model batch configuration (hardware-proven, see DAPT worklog)
# Effective batch = per_device_train_batch_size * gradient_accumulation_steps = 16
BATCH_CFG = {
    "BanglaBERT": {"batch": 16, "accum": 1},
    "sahajBERT":  {"batch": 8,  "accum": 2},
    "mBERT":      {"batch": 8,  "accum": 2},
}

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------
CLASSES  = ["Minimum", "Mild", "Moderate", "Severe"]
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Hyperparameters — documented with justification
# ---------------------------------------------------------------------------
MAX_LENGTH            = 256    # Matches DAPT pretraining context window. Non-negotiable.
NUM_EPOCHS            = 10     # ~5x more data than baseline. EarlyStopping handles ceiling.
LEARNING_RATE         = 2e-5   # Standard BERT fine-tuning LR (Devlin et al., 2019).
WEIGHT_DECAY          = 0.01   # AdamW regularisation. Prevents overfitting on synthetic data.
WARMUP_STEPS          = 500    # ~5% of total steps (17131 rows / eff.batch 16 * 10 epochs = 10,706 steps).
                                # Matches scale used in DAPT pretraining (warmup_steps=500).
EARLY_STOPPING_PAT    = 3      # Stop if val F1 doesn't improve for 3 consecutive epochs.
EVAL_BATCH_SIZE       = 32     # No gradient computation during eval — can double safely.
SEED                  = 42     # Matches all prior experiments. Non-negotiable.
SAVE_TOTAL_LIMIT      = 2      # Keep best + 1 backup. Avoids checkpoint bloat.


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class BanglaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# ---------------------------------------------------------------------------
# Label loading helper
# ---------------------------------------------------------------------------
def load_labels(raw_labels: list) -> list:
    """
    Convert 1-indexed integer labels (1, 2, 3, 4) to 0-indexed (0, 1, 2, 3).
    Handles both int and string-typed label columns defensively.
    """
    labels = []
    for l in raw_labels:
        if isinstance(l, str):
            labels.append(LABEL2ID.get(l.capitalize(), 0))
        else:
            labels.append(int(l) - 1)
    return labels


# ---------------------------------------------------------------------------
# VRAM flush — critical on Windows (WDDM releases VRAM lazily)
# ---------------------------------------------------------------------------
def flush_gpu(obj1=None, obj2=None):
    """Delete model/trainer objects, empty CUDA cache, GC, sleep 10s."""
    if obj1 is not None:
        del obj1
    if obj2 is not None:
        del obj2
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- Load datasets ----
    print("Loading datasets...")
    df_train = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df_val   = pd.read_excel(VAL_PATH,   engine="openpyxl")
    df_test  = pd.read_excel(TEST_PATH,  engine="openpyxl")

    # Training set includes both 'original' and 'synthetic' rows.
    # The 'source' column is not used during training — the model sees all rows equally.
    train_texts  = df_train["posts"].astype(str).tolist()
    train_labels = load_labels(df_train["labels"].tolist())

    val_texts    = df_val["posts"].astype(str).tolist()
    val_labels   = load_labels(df_val["labels"].tolist())

    test_texts   = df_test["posts"].astype(str).tolist()
    test_labels  = load_labels(df_test["labels"].tolist())

    print(f"  Train: {len(train_texts)} rows | Val: {len(val_texts)} rows | Test: {len(test_texts)} rows")
    print(f"  Train source breakdown: {df_train['source'].value_counts().to_dict()}")

    # ---- Per-model training loop ----
    chart_data = {}

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*65}")
        print(f"  Model: {model_name}  ({model_id})")
        print(f"{'='*65}")

        model_out_dir = os.path.join(OUTPUT_DIR, model_name)
        metrics_file  = os.path.join(model_out_dir, "metrics.json")

        # Skip if already done (allows safe reruns)
        if os.path.exists(metrics_file):
            print(f"  [SKIP] {model_name} already evaluated. Loading saved metrics.")
            with open(metrics_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            chart_data[model_name] = {
                "accuracy":    saved["test_metrics"]["accuracy"],
                "f1_weighted": saved["test_metrics"]["f1_weighted"],
            }
            continue

        os.makedirs(model_out_dir, exist_ok=True)

        cfg = BATCH_CFG[model_name]
        print(f"  Batch config: per_device={cfg['batch']}, accum={cfg['accum']}, "
              f"effective={cfg['batch'] * cfg['accum']}")

        # ---- Tokenise ----
        print(f"  Loading tokenizer from HuggingFace: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        test_encodings = tokenizer(
            test_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

        train_dataset = BanglaDataset(train_encodings, train_labels)
        val_dataset   = BanglaDataset(val_encodings,   val_labels)
        test_dataset  = BanglaDataset(test_encodings,  test_labels)

        # ---- Model ----
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=4,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        # ---- Training arguments ----
        training_args = TrainingArguments(
            output_dir=os.path.join(model_out_dir, "checkpoints"),

            # Epochs and batch
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=cfg["batch"],
            gradient_accumulation_steps=cfg["accum"],
            per_device_eval_batch_size=EVAL_BATCH_SIZE,

            # Optimiser
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=WARMUP_STEPS,

            # Evaluation and checkpointing
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",       # Weighted F1 — consistent with baseline
            greater_is_better=True,
            save_total_limit=SAVE_TOTAL_LIMIT,

            # Logging
            logging_steps=50,
            report_to="none",

            # Reproducibility and hardware
            seed=SEED,
            fp16=torch.cuda.is_available(),   # FP16 — consistent with all prior runs
            dataloader_num_workers=4,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PAT)],
        )

        # ---- Train ----
        print(f"  Starting training...")
        trainer.train()

        # ---- Val metrics (from Trainer state — already computed during training) ----
        # Get the best val metrics from the trainer's log history
        best_val = {}
        for log in reversed(trainer.state.log_history):
            if "eval_f1" in log:
                best_val = {
                    "accuracy":  round(log.get("eval_accuracy", 0.0), 4),
                    "f1":        round(log.get("eval_f1",       0.0), 4),
                    "precision": round(log.get("eval_precision",0.0), 4),
                    "recall":    round(log.get("eval_recall",   0.0), 4),
                    "epoch":     log.get("epoch", -1),
                }
                break
        print(f"  Best val → Accuracy: {best_val.get('accuracy', '?'):.4f} | "
              f"F1: {best_val.get('f1', '?'):.4f} (epoch {best_val.get('epoch', '?')})")

        # ---- Test evaluation — called exactly ONCE ----
        # trainer has already restored best checkpoint via load_best_model_at_end=True
        print(f"  Running final test evaluation (locked/test.xlsx)...")
        test_preds  = trainer.predict(test_dataset)
        predictions = (
            test_preds.predictions[0]
            if isinstance(test_preds.predictions, tuple)
            else test_preds.predictions
        )
        y_pred = list(predictions.argmax(-1))
        y_true = test_labels

        test_metrics = compute_full_metrics(y_true, y_pred)

        print(f"  TEST → Accuracy: {test_metrics['accuracy']:.4f} | "
              f"F1: {test_metrics['f1_weighted']:.4f} | "
              f"Precision: {test_metrics['precision_weighted']:.4f} | "
              f"Recall: {test_metrics['recall_weighted']:.4f}")
        print("  Per-class F1:")
        for cls in CLASSES:
            pc = test_metrics["per_class"][cls]
            print(f"    {cls:10s} → F1: {pc['f1']:.4f}  P: {pc['precision']:.4f}  "
                  f"R: {pc['recall']:.4f}  n={pc['support']}")

        # ---- Save metrics ----
        training_params = {
            "max_length":     MAX_LENGTH,
            "num_epochs":     NUM_EPOCHS,
            "learning_rate":  LEARNING_RATE,
            "weight_decay":   WEIGHT_DECAY,
            "warmup_steps":   WARMUP_STEPS,
            "effective_batch": cfg["batch"] * cfg["accum"],
            "seed":           SEED,
            "fp16":           torch.cuda.is_available(),
            "train_rows":     len(train_texts),
            "val_rows":       len(val_texts),
            "test_rows":      len(test_texts),
            "early_stopping_patience": EARLY_STOPPING_PAT,
        }
        save_metrics(
            path=metrics_file,
            model_name=model_name,
            experiment="aug_base",
            val_metrics=best_val,
            test_metrics=test_metrics,
            training_params=training_params,
        )

        # ---- Confusion matrix ----
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            title=f"Base+Aug {model_name} — Test Confusion Matrix",
            output_path=os.path.join(FIG_DIR, f"cm_aug_base_{model_name.lower()}.png"),
        )

        # ---- Save best model weights ----
        best_model_dir = os.path.join(model_out_dir, "best_model")
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        print(f"  Best model saved → {best_model_dir}")

        chart_data[model_name] = {
            "accuracy":    test_metrics["accuracy"],
            "f1_weighted": test_metrics["f1_weighted"],
        }

        # ---- Free VRAM before next model ----
        flush_gpu(model, trainer)

    # ---- Comparison chart ----
    if chart_data:
        plot_performance(
            results_dict=chart_data,
            output_path=os.path.join(FIG_DIR, "aug_base_performance_comparison.png"),
            title="Base Models + Augmented Data — Test Performance",
        )

    print(f"\n{'='*65}")
    print(f"  All aug_base models finished.")
    print(f"  Metrics: {OUTPUT_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
