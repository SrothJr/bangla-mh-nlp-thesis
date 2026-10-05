"""
train_aug_fixed_dapt.py — Three targeted fixes for DAPT models on augmented data.

PROBLEM IDENTIFIED (see AUG_TRAINING_WORKLOG.md):
  DAPT+Aug degraded BanglaBERT by -3.74% and mBERT by -2.92% vs DAPT+TAPT baseline.
  Only sahajBERT partially recovered (-0.95%) because its weaker Bangla baseline
  gave it more room to benefit from domain exposure.

THREE FIXES, sharing one TAPT checkpoint per model:

  Experiment A — "tapt_real_only"
    TAPT  : All 17,131 augmented texts (MLM, no labels)
    Tune  : Only the 3,426 original real rows
    Pipeline: DAPT → TAPT(17k aug) → Fine-tune(3.4k real)
    This is the methodologically cleanest fix: 3-stage pretraining with
    each stage using data it's suited for.

  Experiment B — "tapt_minority_aug"
    TAPT  : All 17,131 augmented texts (same as A)
    Tune  : 3,426 original + selective synthetic for Labels 3+4 only
            Target: Moderate(494→1000), Severe(387→1000)
            Total: ~4,545 rows, 24.6% synthetic, minority classes augmented
    Pipeline: DAPT → TAPT(17k aug) → Fine-tune(4.5k mixed)

  Experiment C — "tapt_weighted_full"
    TAPT  : All 17,131 augmented texts (same as A)
    Tune  : All 17,131 rows, real=1.0 weight, synthetic=0.25 weight
            Reduces effective synthetic gradient influence by 4x
    Pipeline: DAPT → TAPT(17k aug) → WeightedFinetune(17k)

EFFICIENCY:
  - TAPT reuses prior aug_dapt checkpoint if available (same data, same params)
    Checks: results/aug_dapt/{model}/tapt/ → reuse if model.safetensors exists
    This potentially saves ~5h of TAPT compute across all 3 models.
  - TAPT runs once per model; A/B/C share the checkpoint.

SKIP LOGIC:
  - TAPT: reuse aug_dapt TAPT checkpoint if available, else check tapt_shared/,
          else run fresh
  - Each experiment: skipped if metrics.json already exists

Consistency: max_length=256, lr=2e-5, seed=42, fp16=True,
             metric_for_best_model="f1", warmup_steps=500
"""

import os
import gc
import time
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

import shutil

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
TRAIN_PATH = os.path.join(_HERE, "train_augmented_qwen3-32b.xlsx")
VAL_PATH   = os.path.join(_HERE, "locked", "val.xlsx")
TEST_PATH  = os.path.join(_HERE, "locked", "test.xlsx")
DAPT_DIR   = os.path.join(_HERE, "dapt_pretrained")
OUTPUT_DIR = os.path.join(_HERE, "results", "fix_dapt")
FIG_DIR    = os.path.join(_HERE, "results", "figures")

# Prior aug_dapt TAPT paths — reuse if they exist
PRIOR_TAPT = {
    "BanglaBERT": os.path.join(_HERE, "results", "aug_dapt", "BanglaBERT", "tapt"),
    "sahajBERT":  os.path.join(_HERE, "results", "aug_dapt", "sahajBERT",  "tapt"),
    "mBERT":      os.path.join(_HERE, "results", "aug_dapt", "mBERT",      "tapt"),
}

# ---------------------------------------------------------------------------
# Model registry — LOCAL DAPT weights
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
# Label definitions
# ---------------------------------------------------------------------------
CLASSES  = ["Minimum", "Mild", "Moderate", "Severe"]
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Hyperparameters (identical to all prior experiments)
# ---------------------------------------------------------------------------
MAX_LEN              = 256
NUM_EPOCHS           = 10
LEARNING_RATE        = 2e-5
WEIGHT_DECAY         = 0.01
WARMUP_STEPS         = 500
EARLY_STOPPING_PAT   = 3
EVAL_BATCH_SIZE      = 32
SEED                 = 42
SAVE_TOTAL_LIMIT     = 2

TAPT_MAX_LEN         = 256
TAPT_EPOCHS          = 10
TAPT_LR              = 3e-5
TAPT_BATCH           = 16
TAPT_PATIENCE        = 3
MLM_PROBABILITY      = 0.15

MINORITY_LABELS_RAW  = [3, 4]
MINORITY_TARGET      = 1000
SYNTHETIC_WEIGHT     = 0.25


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
class BanglaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class WeightedBanglaDataset(torch.utils.data.Dataset):
    """Carries per-sample loss weights for Experiment C."""
    def __init__(self, encodings, labels, weights):
        self.encodings = encodings
        self.labels    = labels
        self.weights   = weights

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"]        = torch.tensor(self.labels[idx], dtype=torch.long)
        item["sample_weight"] = torch.tensor(self.weights[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Applies per-sample loss weights during training. Val uses standard CE."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        weights = inputs.pop("sample_weight", None)
        outputs = model(**inputs)
        logits  = outputs.logits

        if weights is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss     = loss_fct(logits, labels)
            loss     = (loss * weights).mean()
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss     = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_labels(raw: list) -> list:
    return [int(l) - 1 for l in raw]


def flush_gpu(*objs):
    for obj in objs:
        if obj is not None:
            del obj
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(10)


def cleanup_checkpoints(exp_dir: str):
    """Delete checkpoints/ after best_model/ is saved. Frees 1-3 GB per experiment."""
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"    [Cleanup] Deleted checkpoints/ → freed disk space")


def tapt_done(tapt_dir: str) -> bool:
    return os.path.exists(os.path.join(tapt_dir, "model.safetensors"))


def all_experiments_done(model_dir: str) -> bool:
    return all(
        os.path.exists(os.path.join(model_dir, exp, "metrics.json"))
        for exp in ["exp_a", "exp_b", "exp_c"]
    )


# ---------------------------------------------------------------------------
# Data preparation per experiment
# ---------------------------------------------------------------------------
def prepare_exp_a(df_train):
    df    = df_train[df_train["source"] == "original"].copy()
    texts = df["posts"].astype(str).tolist()
    lbls  = load_labels(df["labels"].tolist())
    print(f"    [Exp A] Fine-tune: {len(texts)} rows (original only, 0% synthetic)")
    return texts, lbls, None


def prepare_exp_b(df_train):
    df_orig  = df_train[df_train["source"] == "original"].copy()
    df_synth = df_train[df_train["source"] == "synthetic"].copy()
    parts    = [df_orig]

    for label_raw in MINORITY_LABELS_RAW:
        orig_n = len(df_orig[df_orig["labels"] == label_raw])
        need   = max(0, MINORITY_TARGET - orig_n)
        pool   = df_synth[df_synth["labels"] == label_raw]
        n_add  = min(need, len(pool))
        if n_add > 0:
            parts.append(pool.sample(n=n_add, random_state=SEED))
            print(f"    [Exp B] Label {label_raw}: {orig_n} orig + {n_add} synth = {orig_n+n_add}")

    df_mix  = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    pct     = 100.0 * (len(df_mix) - len(df_orig)) / len(df_mix)
    print(f"    [Exp B] Total: {len(df_mix)} rows ({pct:.1f}% synthetic)")
    return df_mix["posts"].astype(str).tolist(), load_labels(df_mix["labels"].tolist()), None


def prepare_exp_c(df_train):
    texts   = df_train["posts"].astype(str).tolist()
    lbls    = load_labels(df_train["labels"].tolist())
    weights = [1.0 if s == "original" else SYNTHETIC_WEIGHT
               for s in df_train["source"].tolist()]
    n_real  = sum(1 for s in df_train["source"] if s == "original")
    print(f"    [Exp C] Fine-tune: {len(texts)} rows "
          f"(real={n_real}×1.0, synthetic={len(texts)-n_real}×{SYNTHETIC_WEIGHT})")
    return texts, lbls, weights


# ---------------------------------------------------------------------------
# TAPT — starts from DAPT weights (ForMaskedLM)
# ---------------------------------------------------------------------------
def run_tapt(dapt_model_path: str, tokenizer, tapt_texts: list,
             tapt_dir: str, model_name: str) -> str:
    """
    Run TAPT starting from DAPT-pretrained weights (not HuggingFace hub).
    Runs MLM on all 17,131 augmented texts. Labels are NOT used.
    Checkpoint is shared across all 3 fine-tuning experiments.
    """
    os.makedirs(tapt_dir, exist_ok=True)

    hf_dataset = Dataset.from_dict({"text": tapt_texts})
    split      = hf_dataset.train_test_split(test_size=0.1, seed=SEED)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=TAPT_MAX_LEN)

    tapt_train = split["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    tapt_val   = split["test"].map(tokenize_fn,  batched=True, remove_columns=["text"])
    collator   = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                  mlm=True, mlm_probability=MLM_PROBABILITY)

    print(f"    [TAPT] Loading DAPT weights: {dapt_model_path}")
    tapt_model = AutoModelForMaskedLM.from_pretrained(dapt_model_path)

    tapt_args = TrainingArguments(
        output_dir=tapt_dir,
        num_train_epochs=TAPT_EPOCHS,
        per_device_train_batch_size=TAPT_BATCH,
        per_device_eval_batch_size=TAPT_BATCH * 2,
        learning_rate=TAPT_LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=100,
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
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=TAPT_PATIENCE)],
    )

    print(f"    [TAPT] Training on {len(tapt_texts)} texts ({model_name})...")
    tapt_trainer.train()
    tapt_trainer.save_model(tapt_dir)
    tokenizer.save_pretrained(tapt_dir)
    print(f"    [TAPT] Complete → {tapt_dir}")

    flush_gpu(tapt_model, tapt_trainer)
    return tapt_dir


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------
def run_experiment(exp_id: str, tapt_path: str, tokenizer,
                   df_train: pd.DataFrame,
                   val_texts: list, val_labels: list,
                   test_texts: list, test_labels: list,
                   batch_cfg: dict, model_name: str,
                   exp_dir: str) -> dict:

    os.makedirs(exp_dir, exist_ok=True)
    metrics_file = os.path.join(exp_dir, "metrics.json")

    if os.path.exists(metrics_file):
        print(f"    [Exp {exp_id.upper()}] Already done. Loading saved metrics.")
        with open(metrics_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return saved["test_metrics"]

    if exp_id == "a":
        train_texts, train_labels, train_weights = prepare_exp_a(df_train)
        experiment_tag = "fix_dapt_a"
    elif exp_id == "b":
        train_texts, train_labels, train_weights = prepare_exp_b(df_train)
        experiment_tag = "fix_dapt_b"
    elif exp_id == "c":
        train_texts, train_labels, train_weights = prepare_exp_c(df_train)
        experiment_tag = "fix_dapt_c"
    else:
        raise ValueError(f"Unknown experiment id: {exp_id}")

    # --- Tokenise ---
    print(f"    [Exp {exp_id.upper()}] Tokenising...")
    train_enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=MAX_LEN)
    val_enc   = tokenizer(val_texts,   truncation=True, padding="max_length", max_length=MAX_LEN)
    test_enc  = tokenizer(test_texts,  truncation=True, padding="max_length", max_length=MAX_LEN)

    if train_weights is not None:
        train_dataset = WeightedBanglaDataset(train_enc, train_labels, train_weights)
    else:
        train_dataset = BanglaDataset(train_enc, train_labels)

    val_dataset  = BanglaDataset(val_enc,  val_labels)
    test_dataset = BanglaDataset(test_enc, test_labels)

    # Load classification head from TAPT checkpoint
    # ignore_mismatched_sizes=True: replaces MLM head with fresh 4-class classification head
    print(f"    [Exp {exp_id.upper()}] Loading classification head from TAPT weights...")
    model = AutoModelForSequenceClassification.from_pretrained(
        tapt_path,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(exp_dir, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_cfg["batch"],
        gradient_accumulation_steps=batch_cfg["accum"],
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    TrainerClass = WeightedTrainer if exp_id == "c" else Trainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PAT)],
    )

    print(f"    [Exp {exp_id.upper()}] Training {model_name}...")
    trainer.train()

    # --- Best val metrics ---
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
    print(f"    [Exp {exp_id.upper()}] Best val → Acc: {best_val.get('accuracy','?'):.4f}  "
          f"F1: {best_val.get('f1','?'):.4f} (epoch {best_val.get('epoch','?')})")

    # --- Final test evaluation (ONE call) ---
    print(f"    [Exp {exp_id.upper()}] Final test evaluation...")
    test_preds = trainer.predict(test_dataset)
    preds = test_preds.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    y_pred = list(preds.argmax(-1))

    test_metrics = compute_full_metrics(test_labels, y_pred)
    print(f"    [Exp {exp_id.upper()}] TEST → Acc: {test_metrics['accuracy']:.4f}  "
          f"F1: {test_metrics['f1_weighted']:.4f}  "
          f"P: {test_metrics['precision_weighted']:.4f}  "
          f"R: {test_metrics['recall_weighted']:.4f}")
    print(f"    Per-class F1:")
    for cls in CLASSES:
        pc = test_metrics["per_class"][cls]
        print(f"      {cls:10s}  F1={pc['f1']:.4f}  P={pc['precision']:.4f}  "
              f"R={pc['recall']:.4f}  n={pc['support']}")

    # --- Save ---
    save_metrics(
        path=metrics_file,
        model_name=model_name,
        experiment=experiment_tag,
        val_metrics=best_val,
        test_metrics=test_metrics,
        training_params={
            "experiment_id":     exp_id,
            "max_length":        MAX_LEN,
            "num_epochs":        NUM_EPOCHS,
            "learning_rate":     LEARNING_RATE,
            "weight_decay":      WEIGHT_DECAY,
            "warmup_steps":      WARMUP_STEPS,
            "effective_batch":   batch_cfg["batch"] * batch_cfg["accum"],
            "seed":              SEED,
            "fp16":              torch.cuda.is_available(),
            "tapt_lr":           TAPT_LR,
            "tapt_epochs":       TAPT_EPOCHS,
            "tapt_batch":        TAPT_BATCH,
            "mlm_probability":   MLM_PROBABILITY,
            "tapt_texts":        17131,
            "finetune_rows":     len(train_texts),
            "val_rows":          len(val_texts),
            "test_rows":         len(test_texts),
            "synthetic_weight":  SYNTHETIC_WEIGHT if exp_id == "c" else "N/A",
            "minority_target":   MINORITY_TARGET   if exp_id == "b" else "N/A",
        },
    )

    plot_confusion_matrix(
        y_true=test_labels, y_pred=y_pred,
        title=f"DAPT+Aug Fix-{exp_id.upper()} {model_name}",
        output_path=os.path.join(FIG_DIR, f"cm_fix_dapt_{model_name.lower()}_exp{exp_id}.png"),
    )

    best_model_dir = os.path.join(exp_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Delete training checkpoints immediately — best_model/ preserves the weights.
    cleanup_checkpoints(exp_dir)

    flush_gpu(model, trainer)
    return test_metrics


# ---------------------------------------------------------------------------
# Per-model comparison chart
# ---------------------------------------------------------------------------
def plot_model_comparison(model_name: str, exp_results: dict, output_path: str):
    labels = ["Exp A\n(TAPT+Real)", "Exp B\n(TAPT+MinorityAug)", "Exp C\n(TAPT+Weighted)"]
    f1s    = [exp_results.get("a", {}).get("f1_weighted", 0),
              exp_results.get("b", {}).get("f1_weighted", 0),
              exp_results.get("c", {}).get("f1_weighted", 0)]
    accs   = [exp_results.get("a", {}).get("accuracy",   0),
              exp_results.get("b", {}).get("accuracy",   0),
              exp_results.get("c", {}).get("accuracy",   0)]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, accs, w, label="Accuracy",      color="#4C72B0")
    ax.bar(x + w/2, f1s,  w, label="F1 (Weighted)", color="#DD8452")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.002, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, f + 0.002, f"{f:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.70, 1.00)
    ax.set_title(f"fix_dapt — {model_name} — Experiment Comparison", fontsize=13)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Chart → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading datasets...")
    df_train = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df_val   = pd.read_excel(VAL_PATH,   engine="openpyxl")
    df_test  = pd.read_excel(TEST_PATH,  engine="openpyxl")

    tapt_texts  = df_train["posts"].astype(str).tolist()
    val_texts   = df_val["posts"].astype(str).tolist()
    val_labels  = load_labels(df_val["labels"].tolist())
    test_texts  = df_test["posts"].astype(str).tolist()
    test_labels = load_labels(df_test["labels"].tolist())

    print(f"  TAPT corpus: {len(tapt_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
    print(f"  Source breakdown: {df_train['source'].value_counts().to_dict()}")

    all_results = {}

    for model_name, dapt_model_path in MODELS.items():
        print(f"\n{'='*65}")
        print(f"  Model: {model_name}")
        print(f"  DAPT weights: {dapt_model_path}")
        print(f"{'='*65}")

        if not os.path.isdir(dapt_model_path):
            print(f"  [ERROR] DAPT weights not found at {dapt_model_path}. Skipping.")
            continue

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        if all_experiments_done(model_dir):
            print(f"  [SKIP] All 3 experiments done for {model_name}.")
            exp_results = {}
            for exp_id in ["a", "b", "c"]:
                mf = os.path.join(model_dir, f"exp_{exp_id}", "metrics.json")
                with open(mf, "r") as f:
                    exp_results[exp_id] = json.load(f)["test_metrics"]
            all_results[model_name] = exp_results
            continue

        cfg = BATCH_CFG[model_name]
        print(f"  Batch: per_device={cfg['batch']}, accum={cfg['accum']}, "
              f"effective={cfg['batch']*cfg['accum']}")

        # --- Load tokenizer from DAPT weights ---
        print(f"  Loading tokenizer from DAPT weights...")
        tokenizer = AutoTokenizer.from_pretrained(dapt_model_path)

        # --- TAPT: reuse prior aug_dapt checkpoint if available ---
        prior_tapt = PRIOR_TAPT.get(model_name, "")
        tapt_dir   = os.path.join(model_dir, "tapt_shared")

        if tapt_done(tapt_dir):
            print(f"  [SKIP] TAPT already done (tapt_shared/) → reusing")
        elif tapt_done(prior_tapt):
            print(f"  [REUSE] Prior aug_dapt TAPT checkpoint found → {prior_tapt}")
            print(f"          Symlinking/copying to tapt_shared/ is skipped; "
                  f"tapt_path will point directly to prior checkpoint.")
            tapt_dir = prior_tapt  # use prior checkpoint directly
        else:
            print(f"  Running TAPT on {len(tapt_texts)} texts...")
            run_tapt(dapt_model_path, tokenizer, tapt_texts, tapt_dir, model_name)

        # --- Three experiments sharing the same TAPT checkpoint ---
        exp_results = {}
        for exp_id in ["a", "b", "c"]:
            print(f"\n  --- Experiment {exp_id.upper()} ({model_name}) ---")
            exp_dir = os.path.join(model_dir, f"exp_{exp_id}")
            test_metrics = run_experiment(
                exp_id=exp_id,
                tapt_path=tapt_dir,
                tokenizer=tokenizer,
                df_train=df_train,
                val_texts=val_texts,
                val_labels=val_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                batch_cfg=cfg,
                model_name=model_name,
                exp_dir=exp_dir,
            )
            exp_results[exp_id] = test_metrics

        all_results[model_name] = exp_results

        plot_model_comparison(
            model_name=model_name,
            exp_results=exp_results,
            output_path=os.path.join(FIG_DIR, f"fix_dapt_{model_name.lower()}_comparison.png"),
        )

        flush_gpu()

    # --- Summary ---
    print(f"\n{'='*65}")
    print(f"  fix_dapt SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<12} {'Exp':>6} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for model_name, exp_results in all_results.items():
        for exp_id, tm in exp_results.items():
            print(f"  {model_name:<12} {'Exp '+exp_id.upper():>6} "
                  f"{tm.get('accuracy',0):.4f}     {tm.get('f1_weighted',0):.4f}")
    print(f"\n  Results → {OUTPUT_DIR}")
    print(f"  Figures → {FIG_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
