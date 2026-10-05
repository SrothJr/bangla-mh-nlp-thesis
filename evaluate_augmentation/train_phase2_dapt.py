"""
train_phase2_dapt.py  —  Phase 2: TAPT (Qwen corpus) + BT Fine-Tuning Ablations

Pipeline per model  (BanglaBERT / sahajBERT / mBERT):
  1. TAPT   : Start from DAPT pretrained weights.
               Run MLM on 24,699 Qwen-generated posts (tapt_corpus_clean.txt).
               Output: results/phase2_dapt/{model}/tapt_shared/

  2. Exp A  : Fine-tune on Real (3,426) + BT Label-3 & Label-4
  3. Exp B  : Fine-tune on Real (3,426) + BT All Labels
  4. Exp C  : Fine-tune on Real (3,426) + BT Label-3 only

Crash-safety:
  - TAPT   resumes from the latest checkpoint-* if one exists.
  - Each Exp resumes from latest checkpoint-* if metrics.json is absent.
  - metrics.json present  →  experiment is fully done, skip.
  - save_total_limit=1    →  only 1 intermediate checkpoint kept at any time.
  - Intermediate checkpoint-* folders are deleted after best_model/ is saved.

Storage:
  - save_total_limit = 1 throughout (never bloats disk).
  - Automatic sweep of checkpoint-* after every stage completes.

NOTE: Does NOT import 'datasets' (HuggingFace) — blocked on this machine by
      Application Control policy.  Uses pure-PyTorch Dataset classes instead.
"""

import os
import gc
import glob
import json
import shutil
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from utils_aug import (
    compute_metrics,
    compute_full_metrics,
    save_metrics,
    plot_confusion_matrix,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
TAPT_TXT   = os.path.join(_HERE, "tapt_qwen_data",           "tapt_corpus_clean.txt")
BT_PATH    = os.path.join(_HERE, "nllb_backtranslated_data",  "train_backtranslated_cleaned.xlsx")
BASE_PATH  = os.path.join(_HERE, "train_augmented_qwen3-32b.xlsx")
VAL_PATH   = os.path.join(_HERE, "locked", "val.xlsx")
TEST_PATH  = os.path.join(_HERE, "locked", "test.xlsx")
DAPT_DIR   = os.path.join(_HERE, "dapt_pretrained")
OUTPUT_DIR = os.path.join(_HERE, "results", "phase2_dapt")
FIG_DIR    = os.path.join(_HERE, "results", "figures")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "BanglaBERT": os.path.join(DAPT_DIR, "BanglaBERT"),
    "sahajBERT":  os.path.join(DAPT_DIR, "sahajBERT"),
    "mBERT":      os.path.join(DAPT_DIR, "mBERT"),
}

BATCH_CFG = {
    "BanglaBERT": {"batch": 16, "accum": 1},
    "sahajBERT":  {"batch": 8,  "accum": 2},
    "mBERT":      {"batch": 8,  "accum": 2},
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES  = ["Minimum", "Mild", "Moderate", "Severe"]
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

MAX_LEN            = 256
NUM_EPOCHS         = 10
LEARNING_RATE      = 2e-5
WEIGHT_DECAY       = 0.01
WARMUP_STEPS       = 500
EARLY_STOPPING_PAT = 3
EVAL_BATCH_SIZE    = 32
SEED               = 42

TAPT_MAX_LEN    = 256
TAPT_EPOCHS     = 10
TAPT_LR         = 3e-5
TAPT_BATCH      = 16
TAPT_PATIENCE   = 3
MLM_PROBABILITY = 0.15


# ---------------------------------------------------------------------------
# Pure-PyTorch Dataset classes  (no HuggingFace 'datasets' required)
# ---------------------------------------------------------------------------
class TAPTDataset(torch.utils.data.Dataset):
    """Wraps a tokenized text list for MLM pre-training."""
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


class BanglaDataset(torch.utils.data.Dataset):
    """Classification dataset with 0-indexed labels."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_labels(raw) -> list:
    return [int(l) - 1 for l in raw]


def flush_gpu(*objs):
    for obj in objs:
        if obj is not None:
            del obj
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)


def clean_checkpoints(directory: str):
    """Delete checkpoint-* folders to reclaim storage after stage completes."""
    for cp in glob.glob(os.path.join(directory, "checkpoint-*")):
        try:
            shutil.rmtree(cp)
            print(f"    [Cleanup] Removed {os.path.basename(cp)}")
        except Exception as e:
            print(f"    [Cleanup] Could not remove {cp}: {e}")


def tapt_done(tapt_dir: str) -> bool:
    return os.path.exists(os.path.join(tapt_dir, "model.safetensors"))


def exp_done(exp_dir: str) -> bool:
    return os.path.exists(os.path.join(exp_dir, "metrics.json"))


def all_exps_done(model_dir: str) -> bool:
    return all(exp_done(os.path.join(model_dir, f"exp_{x}")) for x in ["a", "b", "c"])


# ---------------------------------------------------------------------------
# TAPT routine  (pure-PyTorch, no HF datasets)
# ---------------------------------------------------------------------------
def run_tapt(dapt_path: str, tokenizer, tapt_dir: str, model_name: str):
    os.makedirs(tapt_dir, exist_ok=True)

    # Load the corpus text file (one post per line)
    with open(TAPT_TXT, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if len(l.strip()) > 10]

    print(f"    [TAPT] {len(lines)} posts loaded from corpus.")

    # Tokenize in one shot — fast on CPU, fits in RAM for 25k short texts
    print("    [TAPT] Tokenizing...")
    enc = tokenizer(
        lines,
        truncation=True,
        padding="max_length",
        max_length=TAPT_MAX_LEN,
        return_token_type_ids=False,
    )

    # 90/10 train/val split
    n       = len(lines)
    n_val   = max(1, int(n * 0.10))
    n_train = n - n_val

    train_enc = {k: v[:n_train] for k, v in enc.items()}
    val_enc   = {k: v[n_train:] for k, v in enc.items()}

    ds_train  = TAPTDataset(train_enc)
    ds_val    = TAPTDataset(val_enc)
    collator  = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
    )

    print(f"    [TAPT] Loading DAPT weights: {dapt_path}")
    model = AutoModelForMaskedLM.from_pretrained(dapt_path)

    args = TrainingArguments(
        output_dir             = tapt_dir,
        num_train_epochs       = TAPT_EPOCHS,
        per_device_train_batch_size = TAPT_BATCH,
        per_device_eval_batch_size  = TAPT_BATCH * 2,
        learning_rate          = TAPT_LR,
        weight_decay           = WEIGHT_DECAY,
        warmup_steps           = 100,
        eval_strategy          = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,
        save_total_limit       = 1,
        logging_steps          = 100,
        report_to              = "none",
        seed                   = SEED,
        fp16                   = torch.cuda.is_available(),
        dataloader_num_workers = 0,      # avoids Windows multiprocessing issues
    )

    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = ds_train,
        eval_dataset  = ds_val,
        data_collator = collator,
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=TAPT_PATIENCE)],
    )

    last_ckpt = get_last_checkpoint(tapt_dir)
    if last_ckpt:
        print(f"    [TAPT] Resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model(tapt_dir)
    tokenizer.save_pretrained(tapt_dir)
    print(f"    [TAPT] Complete -> {tapt_dir}")

    flush_gpu(model, trainer)
    clean_checkpoints(tapt_dir)


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    exp_id: str,
    tapt_path: str,
    tokenizer,
    train_texts: list,
    train_labels: list,
    val_texts: list,
    val_labels: list,
    test_texts: list,
    test_labels: list,
    batch_cfg: dict,
    model_name: str,
    exp_dir: str,
):
    os.makedirs(exp_dir, exist_ok=True)
    metrics_file  = os.path.join(exp_dir, "metrics.json")
    best_model_dir = os.path.join(exp_dir, "best_model")
    ckpt_dir      = os.path.join(exp_dir, "checkpoints")

    if exp_done(exp_dir):
        print(f"    [Exp {exp_id.upper()}] Already done — skipping.")
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)["test_metrics"]

    print(f"    [Exp {exp_id.upper()}] Tokenising {len(train_texts)} rows...")
    train_enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=MAX_LEN)
    val_enc   = tokenizer(val_texts,   truncation=True, padding="max_length", max_length=MAX_LEN)
    test_enc  = tokenizer(test_texts,  truncation=True, padding="max_length", max_length=MAX_LEN)

    ds_train = BanglaDataset(train_enc, train_labels)
    ds_val   = BanglaDataset(val_enc,   val_labels)
    ds_test  = BanglaDataset(test_enc,  test_labels)

    print(f"    [Exp {exp_id.upper()}] Loading classification model from TAPT weights...")
    model = AutoModelForSequenceClassification.from_pretrained(
        tapt_path,
        num_labels             = 4,
        id2label               = ID2LABEL,
        label2id               = LABEL2ID,
        ignore_mismatched_sizes = True,
    )

    args = TrainingArguments(
        output_dir             = ckpt_dir,
        num_train_epochs       = NUM_EPOCHS,
        per_device_train_batch_size = batch_cfg["batch"],
        gradient_accumulation_steps = batch_cfg["accum"],
        per_device_eval_batch_size  = EVAL_BATCH_SIZE,
        learning_rate          = LEARNING_RATE,
        weight_decay           = WEIGHT_DECAY,
        warmup_steps           = WARMUP_STEPS,
        eval_strategy          = "epoch",
        save_strategy          = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model  = "f1",
        greater_is_better      = True,
        save_total_limit       = 1,
        logging_steps          = 50,
        report_to              = "none",
        seed                   = SEED,
        fp16                   = torch.cuda.is_available(),
        dataloader_num_workers = 0,
    )

    trainer = Trainer(
        model          = model,
        args           = args,
        train_dataset  = ds_train,
        eval_dataset   = ds_val,
        compute_metrics = compute_metrics,
        callbacks      = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PAT)],
    )

    last_ckpt = get_last_checkpoint(ckpt_dir)
    if last_ckpt:
        print(f"    [Exp {exp_id.upper()}] Resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # Best val metrics from log history
    best_val = {}
    for log in reversed(trainer.state.log_history):
        if "eval_f1" in log:
            best_val = {
                "accuracy":  round(log.get("eval_accuracy",  0.0), 4),
                "f1":        round(log.get("eval_f1",        0.0), 4),
                "precision": round(log.get("eval_precision", 0.0), 4),
                "recall":    round(log.get("eval_recall",    0.0), 4),
                "epoch":     log.get("epoch", -1),
            }
            break
    print(f"    [Exp {exp_id.upper()}] Best val -> "
          f"Acc: {best_val.get('accuracy', '?')}  F1: {best_val.get('f1', '?')} "
          f"(epoch {best_val.get('epoch', '?')})")

    # Final test evaluation
    print(f"    [Exp {exp_id.upper()}] Final test evaluation...")
    preds_out = trainer.predict(ds_test)
    raw_preds = preds_out.predictions
    if isinstance(raw_preds, tuple):
        raw_preds = raw_preds[0]
    y_pred = list(raw_preds.argmax(-1))

    test_metrics = compute_full_metrics(test_labels, y_pred)
    print(f"    [Exp {exp_id.upper()}] TEST -> "
          f"Acc: {test_metrics['accuracy']:.4f}  "
          f"F1: {test_metrics['f1_weighted']:.4f}  "
          f"P: {test_metrics['precision_weighted']:.4f}  "
          f"R: {test_metrics['recall_weighted']:.4f}")
    for cls in CLASSES:
        pc = test_metrics["per_class"][cls]
        print(f"      {cls:10s}  F1={pc['f1']:.4f}  P={pc['precision']:.4f}  "
              f"R={pc['recall']:.4f}  n={pc['support']}")

    # Save model and metrics
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    save_metrics(
        path            = metrics_file,
        model_name      = model_name,
        experiment      = f"phase2_dapt_{exp_id}",
        val_metrics     = best_val,
        test_metrics    = test_metrics,
        training_params = {
            "experiment_id":   exp_id,
            "max_length":      MAX_LEN,
            "num_epochs":      NUM_EPOCHS,
            "learning_rate":   LEARNING_RATE,
            "weight_decay":    WEIGHT_DECAY,
            "warmup_steps":    WARMUP_STEPS,
            "effective_batch": batch_cfg["batch"] * batch_cfg["accum"],
            "seed":            SEED,
            "fp16":            torch.cuda.is_available(),
            "tapt_lr":         TAPT_LR,
            "tapt_epochs":     TAPT_EPOCHS,
            "tapt_batch":      TAPT_BATCH,
            "mlm_probability": MLM_PROBABILITY,
            "tapt_corpus":     "tapt_corpus_clean.txt (24,699 Qwen posts)",
            "finetune_rows":   len(train_texts),
            "val_rows":        len(val_texts),
            "test_rows":       len(test_texts),
        },
    )

    plot_confusion_matrix(
        y_true      = test_labels,
        y_pred      = y_pred,
        title       = f"Phase2 DAPT - {model_name} Exp {exp_id.upper()}",
        output_path = os.path.join(FIG_DIR, f"cm_phase2_dapt_{model_name.lower()}_exp{exp_id}.png"),
    )

    flush_gpu(model, trainer)
    clean_checkpoints(ckpt_dir)

    return test_metrics


# ---------------------------------------------------------------------------
# Per-model summary chart
# ---------------------------------------------------------------------------
def plot_model_comparison(model_name: str, exp_results: dict):
    labels = [
        "Exp A\n(Real + BT L3/L4)",
        "Exp B\n(Real + BT All)",
        "Exp C\n(Real + BT L3)",
    ]
    f1s  = [exp_results.get(x, {}).get("f1_weighted", 0) for x in ["a", "b", "c"]]
    accs = [exp_results.get(x, {}).get("accuracy",    0) for x in ["a", "b", "c"]]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, accs, w, label="Accuracy",      color="#4C72B0")
    ax.bar(x + w/2, f1s,  w, label="F1 (Weighted)", color="#DD8452")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.002, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, f + 0.002, f"{f:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.70, 1.00)
    ax.set_title(f"Phase 2 DAPT — {model_name} — Experiment Comparison", fontsize=13)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"phase2_dapt_{model_name.lower()}_comparison.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    [Chart] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,    exist_ok=True)

    # ---- Load datasets ----
    print("Loading datasets...")
    df_base = pd.read_excel(BASE_PATH, engine="openpyxl")
    df_real = df_base[df_base["source"] == "original"].copy().reset_index(drop=True)
    print(f"  Real rows:            {len(df_real)}")

    df_bt   = pd.read_excel(BT_PATH, engine="openpyxl")
    # Rename so everything shares column names
    df_bt   = df_bt.rename(columns={"backtranslated_post": "posts", "label": "labels"})
    df_bt["source"] = "synthetic"
    print(f"  Back-translated rows: {len(df_bt)}")

    df_val  = pd.read_excel(VAL_PATH,  engine="openpyxl")
    df_test = pd.read_excel(TEST_PATH, engine="openpyxl")

    val_texts   = df_val["posts"].astype(str).tolist()
    val_labels  = load_labels(df_val["labels"].tolist())
    test_texts  = df_test["posts"].astype(str).tolist()
    test_labels = load_labels(df_test["labels"].tolist())
    print(f"  Val rows: {len(val_texts)} | Test rows: {len(test_texts)}")

    # ---- Build per-experiment training sets (done once, reused for all models) ----
    # Exp A: Real + BT Label-3 and Label-4
    df_bt_34 = df_bt[df_bt["labels"].isin([3, 4])].copy()
    df_exp_a = pd.concat([df_real, df_bt_34], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Exp B: Real + BT All Labels
    df_exp_b = pd.concat([df_real, df_bt], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Exp C: Real + BT Label-3 only
    df_bt_3  = df_bt[df_bt["labels"] == 3].copy()
    df_exp_c = pd.concat([df_real, df_bt_3], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  Exp A rows: {len(df_exp_a)}  (Real + BT L3/L4)")
    print(f"  Exp B rows: {len(df_exp_b)}  (Real + BT All)")
    print(f"  Exp C rows: {len(df_exp_c)}  (Real + BT L3 only)")

    all_results = {}

    for model_name, dapt_path in MODELS.items():
        print(f"\n{'='*65}")
        print(f"  Model : {model_name}")
        print(f"  DAPT  : {dapt_path}")
        print(f"{'='*65}")

        if not os.path.isdir(dapt_path):
            print(f"  [ERROR] DAPT weights not found: {dapt_path}. Skipping.")
            continue

        model_dir  = os.path.join(OUTPUT_DIR, model_name)
        tapt_dir   = os.path.join(model_dir,  "tapt_shared")
        bcfg       = BATCH_CFG[model_name]
        os.makedirs(model_dir, exist_ok=True)

        if all_exps_done(model_dir):
            print(f"  [SKIP] All 3 experiments done for {model_name}.")
            exp_results = {}
            for x in ["a", "b", "c"]:
                mf = os.path.join(model_dir, f"exp_{x}", "metrics.json")
                with open(mf, "r", encoding="utf-8") as f:
                    exp_results[x] = json.load(f)["test_metrics"]
            all_results[model_name] = exp_results
            plot_model_comparison(model_name, exp_results)
            continue

        # Load tokenizer once per model
        tokenizer = AutoTokenizer.from_pretrained(dapt_path)

        # ---- 1. TAPT ----
        if tapt_done(tapt_dir):
            print(f"  [SKIP] TAPT complete — reusing {tapt_dir}")
        else:
            print(f"  Running TAPT...")
            run_tapt(dapt_path, tokenizer, tapt_dir, model_name)

        # ---- 2. Experiments ----
        EXP_DATA = {
            "a": (df_exp_a, "Real + BT L3/L4"),
            "b": (df_exp_b, "Real + BT All"),
            "c": (df_exp_c, "Real + BT L3 only"),
        }

        exp_results = {}
        for exp_id, (df_exp, desc) in EXP_DATA.items():
            print(f"\n  --- Experiment {exp_id.upper()} [{desc}] ({model_name}) ---")
            exp_dir = os.path.join(model_dir, f"exp_{exp_id}")

            train_texts  = df_exp["posts"].astype(str).tolist()
            train_labels = load_labels(df_exp["labels"].tolist())

            test_metrics = run_experiment(
                exp_id      = exp_id,
                tapt_path   = tapt_dir,
                tokenizer   = tokenizer,
                train_texts = train_texts,
                train_labels= train_labels,
                val_texts   = val_texts,
                val_labels  = val_labels,
                test_texts  = test_texts,
                test_labels = test_labels,
                batch_cfg   = bcfg,
                model_name  = model_name,
                exp_dir     = exp_dir,
            )
            exp_results[exp_id] = test_metrics

        all_results[model_name] = exp_results
        plot_model_comparison(model_name, exp_results)
        flush_gpu()

    # ---- Final Summary ----
    print(f"\n{'='*65}")
    print("  PHASE 2 SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<12} {'Exp':<8} {'Accuracy':>10} {'F1-Wt':>10}")
    print(f"  {'-'*44}")
    for mn, er in all_results.items():
        for x, tm in er.items():
            print(f"  {mn:<12} Exp {x.upper():<4} "
                  f"{tm.get('accuracy',0):>10.4f} {tm.get('f1_weighted',0):>10.4f}")
    print(f"\n  Results -> {OUTPUT_DIR}")
    print(f"  Figures -> {FIG_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
