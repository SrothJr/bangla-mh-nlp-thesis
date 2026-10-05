"""
train_aug_dapt.py — Fine-tune DAPT-pretrained models on the augmented dataset.

Experiment:  aug_dapt
Models:      BanglaBERT, sahajBERT, mBERT  (local DAPT weights)
DAPT weights: dapt_pretrained/{model}/model.safetensors
Train data:  train_augmented_qwen3-32b.xlsx  (17,131 rows: 3,426 original + 13,705 synthetic)
Val data:    locked/val.xlsx                 (733 rows, original only — early stopping)
Test data:   locked/test.xlsx                (738 rows, original only — ONE final evaluation)

Pipeline per model:
  1. TAPT (Task-Adaptive Pretraining) via MLM on the augmented train texts.
     - Bridges vocabulary from DAPT corpus (Reddit/mental health) to this dataset's
       specific Bengali vocabulary, now including synthetic posts.
     - Run ONLY on train texts — val and test are never seen by TAPT.
     - Early stopped on MLM loss (90/10 internal split of train texts).
  2. Fine-tuning classification head from TAPT weights.
     - Early stopped on val F1 (locked/val.xlsx).
  3. Final test evaluation on locked/test.xlsx — called exactly once.

Consistency with prior experiments:
  max_length=256     — matches DAPT pretraining context window exactly
  learning_rate=2e-5 — same as train_dapt_eval.py fine-tuning (line 310)
  TAPT_LR=3e-5       — same as train_dapt_eval.py TAPT (line 60)
  seed=42            — same as all prior experiments
  metric_for_best_model="f1" — same as train_dapt_eval.py fine-tuning (line 315)
  fp16=True          — same as all prior experiments
  WDDM flush pattern — same as train_dapt_eval.py (gc + 10s sleep between stages)
"""

import os
import gc
import time
import json
import warnings

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
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
DAPT_DIR    = os.path.join(_HERE, "dapt_pretrained")
OUTPUT_DIR  = os.path.join(_HERE, "results", "aug_dapt")
FIG_DIR     = os.path.join(_HERE, "results", "figures")

# ---------------------------------------------------------------------------
# Model registry — loaded from LOCAL DAPT weights (not HuggingFace hub)
# ---------------------------------------------------------------------------
MODELS = {
    "BanglaBERT": os.path.join(DAPT_DIR, "BanglaBERT"),
    "sahajBERT":  os.path.join(DAPT_DIR, "sahajBERT"),
    "mBERT":      os.path.join(DAPT_DIR, "mBERT"),
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
# Fine-tuning hyperparameters
# ---------------------------------------------------------------------------
FINETUNE_MAX_LEN      = 256    # Must match DAPT pretraining. Non-negotiable.
NUM_EPOCHS            = 10     # ~5x more data. EarlyStopping handles ceiling.
LEARNING_RATE         = 2e-5   # Standard BERT fine-tuning LR.
WEIGHT_DECAY          = 0.01   # AdamW regularisation.
WARMUP_STEPS    = 500   # ~5% of total fine-tuning steps. Matches DAPT pretraining scale.
EARLY_STOPPING_PAT    = 3      # Stop if val F1 doesn't improve for 3 epochs.
EVAL_BATCH_SIZE       = 32     # No gradients at eval time.
SEED                  = 42
SAVE_TOTAL_LIMIT      = 2

# ---------------------------------------------------------------------------
# TAPT hyperparameters
# ---------------------------------------------------------------------------
TAPT_MAX_LEN    = 256   # Must match. Non-negotiable.
TAPT_EPOCHS     = 10    # Augmented train set is large — fewer epochs needed. EarlyStopping handles rest.
TAPT_LR         = 3e-5  # Slightly higher LR for continued pretraining (matches original pipeline).
TAPT_BATCH      = 16    # Conservative for 256-token MLM on 32 GB VRAM.
TAPT_PATIENCE   = 3     # Stop TAPT if MLM loss plateaus for 3 epochs.
MLM_PROBABILITY = 0.15  # Standard BERT masking rate (Devlin et al., 2019).


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
def flush_gpu(*objs):
    """Delete objects, empty CUDA cache, GC, sleep 10s."""
    for obj in objs:
        if obj is not None:
            del obj
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(10)


# ---------------------------------------------------------------------------
# TAPT — Task-Adaptive Pretraining
# ---------------------------------------------------------------------------
def run_tapt(
    dapt_model_path: str,
    tokenizer,
    train_texts: list,
    tapt_output_dir: str,
    model_name: str,
) -> str:
    """
    Run Task-Adaptive Pretraining (TAPT) via MLM on the augmented train texts.

    DATA ISOLATION GUARANTEE:
    -------------------------
    - Input:  train_texts — ALL 17,131 texts from train_augmented_qwen3-32b.xlsx
              (both 'original' and 'synthetic' rows). Labels are NOT used — this
              is unsupervised MLM.
    - The 90/10 internal split is entirely within train_texts.
      locked/val.xlsx and locked/test.xlsx are NEVER passed here.
    - Output: tapt_output_dir — best MLM checkpoint, used to initialise the
              classification head in the next stage.

    Why TAPT on the augmented set?
    --------------------------------
    TAPT bridges vocabulary from the DAPT corpus (Reddit/mental health English→Bengali)
    to this specific dataset's vocabulary. The augmented set's 13,705 synthetic posts
    contain the same mental health Bengali vocabulary the classifier needs to learn.
    Including them in TAPT maximises vocabulary overlap between pretraining and
    fine-tuning, consistent with Gururangan et al. (ACL 2020).

    Parameters
    ----------
    dapt_model_path : str    Path to completed DAPT weights (ForMaskedLM).
    tokenizer               Tokenizer loaded from the DAPT model directory.
    train_texts   : list    Raw Bengali text strings from train_augmented.
    tapt_output_dir : str   Directory to save the best TAPT checkpoint.
    model_name    : str     Used for logging only.

    Returns
    -------
    str  Path to the saved TAPT model directory.
    """
    os.makedirs(tapt_output_dir, exist_ok=True)

    print(f"    [TAPT] Building HuggingFace dataset from {len(train_texts)} train texts...")
    hf_dataset = Dataset.from_dict({"text": train_texts})

    # Internal 90/10 split of train_texts for TAPT early stopping.
    # This is purely within the training partition — no held-out data involved.
    split      = hf_dataset.train_test_split(test_size=0.1, seed=SEED)
    tapt_train = split["train"]    # ~15,417 rows
    tapt_val   = split["test"]     # ~1,714 rows

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=TAPT_MAX_LEN,
        )

    tapt_train = tapt_train.map(tokenize_fn, batched=True, remove_columns=["text"])
    tapt_val   = tapt_val.map(tokenize_fn,   batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    print(f"    [TAPT] Loading DAPT weights for MLM: {dapt_model_path}")
    tapt_model = AutoModelForMaskedLM.from_pretrained(dapt_model_path)

    tapt_args = TrainingArguments(
        output_dir=tapt_output_dir,

        num_train_epochs=TAPT_EPOCHS,
        per_device_train_batch_size=TAPT_BATCH,
        per_device_eval_batch_size=TAPT_BATCH * 2,

        learning_rate=TAPT_LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=100,          # ~10% of first epoch at TAPT scale (matches original pipeline)

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",   # MLM has no F1 — minimise loss
        greater_is_better=False,
        save_total_limit=2,

        logging_steps=50,
        report_to="none",

        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    tapt_trainer = Trainer(
        model=tapt_model,
        args=tapt_args,
        train_dataset=tapt_train,
        eval_dataset=tapt_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=TAPT_PATIENCE)],
    )

    print(f"    [TAPT] Starting MLM training for {model_name}...")
    tapt_trainer.train()

    # Save best TAPT model and tokenizer
    tapt_trainer.save_model(tapt_output_dir)
    tokenizer.save_pretrained(tapt_output_dir)
    print(f"    [TAPT] Complete → {tapt_output_dir}")

    # Free VRAM before classification fine-tuning
    flush_gpu(tapt_model, tapt_trainer)

    return tapt_output_dir


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

    train_texts  = df_train["posts"].astype(str).tolist()
    train_labels = load_labels(df_train["labels"].tolist())

    val_texts    = df_val["posts"].astype(str).tolist()
    val_labels   = load_labels(df_val["labels"].tolist())

    test_texts   = df_test["posts"].astype(str).tolist()
    test_labels  = load_labels(df_test["labels"].tolist())

    print(f"  Train: {len(train_texts)} rows | Val: {len(val_texts)} rows | Test: {len(test_texts)} rows")
    print(f"  Train source breakdown: {df_train['source'].value_counts().to_dict()}")

    # ---- Per-model loop ----
    chart_data = {}

    for model_name, dapt_model_path in MODELS.items():
        print(f"\n{'='*65}")
        print(f"  Model: {model_name}")
        print(f"  DAPT weights: {dapt_model_path}")
        print(f"{'='*65}")

        # Verify DAPT weights exist
        if not os.path.isdir(dapt_model_path):
            print(f"  [ERROR] DAPT weights not found at {dapt_model_path}. Skipping.")
            continue

        model_out_dir  = os.path.join(OUTPUT_DIR, model_name)
        metrics_file   = os.path.join(model_out_dir, "metrics.json")
        tapt_dir       = os.path.join(model_out_dir, "tapt")

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

        # ---- Load tokenizer from DAPT weights (not HuggingFace hub) ----
        print(f"  Loading tokenizer from DAPT weights...")
        tokenizer = AutoTokenizer.from_pretrained(dapt_model_path)

        # ==================================================================
        # STAGE 1: TAPT — Task-Adaptive Pretraining
        # ONLY train_texts are used. val and test are withheld entirely.
        # ==================================================================
        tapt_weights_path = run_tapt(
            dapt_model_path=dapt_model_path,
            tokenizer=tokenizer,
            train_texts=train_texts,
            tapt_output_dir=tapt_dir,
            model_name=model_name,
        )

        # ==================================================================
        # STAGE 2: Fine-tuning from TAPT weights
        # ==================================================================
        print(f"  [Fine-tune] Tokenising datasets...")
        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding="max_length",
            max_length=FINETUNE_MAX_LEN,
        )
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding="max_length",
            max_length=FINETUNE_MAX_LEN,
        )
        test_encodings = tokenizer(
            test_texts,
            truncation=True,
            padding="max_length",
            max_length=FINETUNE_MAX_LEN,
        )

        train_dataset = BanglaDataset(train_encodings, train_labels)
        val_dataset   = BanglaDataset(val_encodings,   val_labels)
        test_dataset  = BanglaDataset(test_encodings,  test_labels)

        # Load classification model from TAPT weights.
        # ignore_mismatched_sizes=True is required:
        #   The saved TAPT model has an MLM head (lm_head / cls).
        #   We replace it with a fresh 4-class classification head.
        #   This is intentional and identical to train_dapt_eval.py line 302.
        model = AutoModelForSequenceClassification.from_pretrained(
            tapt_weights_path,
            num_labels=4,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

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
            metric_for_best_model="f1",       # Weighted F1 — consistent with original pipeline
            greater_is_better=True,
            save_total_limit=SAVE_TOTAL_LIMIT,

            # Logging
            logging_steps=50,
            report_to="none",

            # Reproducibility and hardware
            seed=SEED,
            fp16=torch.cuda.is_available(),
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

        print(f"  [Fine-tune] Starting classification training...")
        trainer.train()

        # ---- Val metrics (from Trainer log history) ----
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
        print(f"  [Fine-tune] Running final test evaluation (locked/test.xlsx)...")
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
            "max_length":         FINETUNE_MAX_LEN,
            "num_epochs":         NUM_EPOCHS,
            "learning_rate":      LEARNING_RATE,
            "weight_decay":       WEIGHT_DECAY,
            "warmup_steps":       WARMUP_STEPS,
            "effective_batch":    cfg["batch"] * cfg["accum"],
            "seed":               SEED,
            "fp16":               torch.cuda.is_available(),
            "tapt_lr":            TAPT_LR,
            "tapt_epochs":        TAPT_EPOCHS,
            "tapt_batch":         TAPT_BATCH,
            "mlm_probability":    MLM_PROBABILITY,
            "tapt_patience":      TAPT_PATIENCE,
            "train_rows":         len(train_texts),
            "val_rows":           len(val_texts),
            "test_rows":          len(test_texts),
            "early_stopping_patience": EARLY_STOPPING_PAT,
            "dapt_weights":       dapt_model_path,
        }
        save_metrics(
            path=metrics_file,
            model_name=model_name,
            experiment="aug_dapt",
            val_metrics=best_val,
            test_metrics=test_metrics,
            training_params=training_params,
        )

        # ---- Confusion matrix ----
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            title=f"DAPT+Aug {model_name} — Test Confusion Matrix",
            output_path=os.path.join(FIG_DIR, f"cm_aug_dapt_{model_name.lower()}.png"),
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
            output_path=os.path.join(FIG_DIR, "aug_dapt_performance_comparison.png"),
            title="DAPT Models + Augmented Data — Test Performance",
        )

    print(f"\n{'='*65}")
    print(f"  All aug_dapt models finished.")
    print(f"  Metrics: {OUTPUT_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
