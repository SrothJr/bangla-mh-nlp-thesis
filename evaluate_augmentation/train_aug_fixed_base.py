"""
train_aug_fixed_base.py — Three targeted fixes for BASE models on augmented data.

PROBLEM IDENTIFIED (see AUG_TRAINING_WORKLOG.md):
  Raw augmentation at 5x ratio (80% synthetic training data) degraded ALL three
  base models by 1.7–3.4% F1. Root cause: LLM-synthetic distribution shift. The
  model learns the style of qwen3-32b's Bengali output, not real social media Bengali.
  Training loss bottomed at epoch 2 then doubled by epoch 6 across all models —
  textbook overfitting to a synthetic distribution evaluated on a real-only test set.

THREE FIXES, all sharing a single TAPT checkpoint per model:

  Experiment A — "tapt_real_only"
    TAPT  : All 17,131 augmented texts (MLM, no labels) → vocabulary exposure
    Tune  : Only the 3,426 original real rows → clean label boundaries
    Theory: Synthetic text is valuable as an unlabelled vocabulary corpus.
            It should NOT be used as labelled training data when its quality
            is imperfect. This separates the vocabulary benefit from label noise.

  Experiment B — "tapt_minority_aug"
    TAPT  : All 17,131 augmented texts (same as A)
    Tune  : 3,426 original + selective synthetic for Labels 3+4 only
            Target: bring Moderate (494→1000) and Severe (387→1000) to parity
            Total: ~4,545 rows, only 24.6% synthetic, concentrated in minority
    Theory: Synthetic text is most defensible for minority-class oversampling
            where the class boundaries are clearer (L4 Severe is clearest).
            L3/L4 need the help; L1/L2 already have enough real data.

  Experiment C — "tapt_weighted_full"
    TAPT  : All 17,131 augmented texts (same as A)
    Tune  : All 17,131 rows, but real=1.0 weight, synthetic=0.25 weight
            The gradient influence of each synthetic sample is reduced 4x.
    Theory: The model benefits from synthetic diversity but shouldn't overfit
            to synthetic patterns. Weighting reduces the effective synthetic
            ratio from 80% to ~50% without changing dataset composition.

EFFICIENCY: TAPT runs ONCE per model → checkpoint reused for A, B, and C.
            This saves ~2h of TAPT compute vs running it 3 times.

SKIP LOGIC:
  - TAPT: skipped if tapt_shared/ contains a valid model.safetensors
  - Each experiment: skipped if its metrics.json already exists
  - Full model: skipped if all 3 experiments have metrics.json

Consistency with prior experiments:
  max_length=256, lr=2e-5, seed=42, fp16=True, metric_for_best_model="f1",
  batch sizes per model identical to aug_base/aug_dapt.
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
OUTPUT_DIR = os.path.join(_HERE, "results", "fix_base")
FIG_DIR    = os.path.join(_HERE, "results", "figures")

# ---------------------------------------------------------------------------
# Model registry — HuggingFace hub IDs (base models)
# ---------------------------------------------------------------------------
MODELS = {
    "BanglaBERT": "csebuetnlp/banglabert",
    "sahajBERT":  "neuropark/sahajBERT",
    "mBERT":      "bert-base-multilingual-cased",
}

# Per-model batch sizes (hardware-proven — see DAPT worklog)
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
# Fine-tuning hyperparameters (identical to all prior experiments)
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

# TAPT hyperparameters
TAPT_MAX_LEN         = 256
TAPT_EPOCHS          = 10
TAPT_LR              = 3e-5
TAPT_BATCH           = 16
TAPT_PATIENCE        = 3
MLM_PROBABILITY      = 0.15

# Experiment B — minority augmentation targets
# Labels are 1-indexed in file (3=Moderate, 4=Severe)
MINORITY_LABELS_RAW  = [3, 4]         # 1-indexed labels to augment
MINORITY_TARGET      = 1000           # bring each minority class to this count

# Experiment C — sample weighting
SYNTHETIC_WEIGHT     = 0.25           # down-weight synthetic 4x vs real


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
    """BanglaDataset variant that carries per-sample loss weights for Exp C."""
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
# Weighted Trainer — pops sample_weight before forwarding to model
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies per-sample loss weights.
    Training dataset must carry a 'sample_weight' field.
    Validation dataset does NOT need it — the pop() defaults to None.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        weights = inputs.pop("sample_weight", None)  # None during evaluation
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
def load_labels(raw_labels: list) -> list:
    """Convert 1-indexed file labels to 0-indexed."""
    return [int(l) - 1 for l in raw_labels]


def flush_gpu(*objs):
    """Windows WDDM VRAM lazy-release fix — always call between stages."""
    for obj in objs:
        if obj is not None:
            del obj
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(10)


def cleanup_checkpoints(exp_dir: str):
    """
    Delete the checkpoints/ subfolder inside exp_dir after training completes.
    best_model/ is already saved separately — checkpoints are only needed
    during training for resumability. Deleting them frees 1-2 GB per experiment.
    """
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        print(f"    [Cleanup] Deleted checkpoints/ → freed disk space")


def all_experiments_done(model_dir: str) -> bool:
    return all(
        os.path.exists(os.path.join(model_dir, exp, "metrics.json"))
        for exp in ["exp_a", "exp_b", "exp_c"]
    )


def tapt_done(tapt_dir: str) -> bool:
    return os.path.exists(os.path.join(tapt_dir, "model.safetensors"))


# ---------------------------------------------------------------------------
# Data preparation for each experiment
# ---------------------------------------------------------------------------
def prepare_exp_a(df_train: pd.DataFrame):
    """Exp A: Original real rows only. 3,426 rows, 0% synthetic."""
    df = df_train[df_train["source"] == "original"].copy()
    texts  = df["posts"].astype(str).tolist()
    labels = load_labels(df["labels"].tolist())
    print(f"    [Exp A] Fine-tune rows: {len(texts)} (original only, 0% synthetic)")
    return texts, labels, None  # no weights


def prepare_exp_b(df_train: pd.DataFrame):
    """
    Exp B: Original + synthetic minority (Labels 3+4) to ~1,000 per class.
    Keeps label 1 and 2 purely real. Addresses class imbalance without
    polluting the majority classes with synthetic noise.
    """
    df_orig  = df_train[df_train["source"] == "original"].copy()
    df_synth = df_train[df_train["source"] == "synthetic"].copy()
    parts    = [df_orig]

    for label_raw in MINORITY_LABELS_RAW:
        orig_count  = len(df_orig[df_orig["labels"] == label_raw])
        need        = max(0, MINORITY_TARGET - orig_count)
        synth_pool  = df_synth[df_synth["labels"] == label_raw]
        n_add       = min(need, len(synth_pool))
        if n_add > 0:
            parts.append(synth_pool.sample(n=n_add, random_state=SEED))
            print(f"    [Exp B] Label {label_raw}: {orig_count} orig + {n_add} synth = {orig_count + n_add}")

    df_mix = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    pct_synth = 100.0 * (len(df_mix) - len(df_orig)) / len(df_mix)
    print(f"    [Exp B] Total fine-tune rows: {len(df_mix)} ({pct_synth:.1f}% synthetic)")
    texts  = df_mix["posts"].astype(str).tolist()
    labels = load_labels(df_mix["labels"].tolist())
    return texts, labels, None  # no weights for B — composition handles ratio


def prepare_exp_c(df_train: pd.DataFrame):
    """
    Exp C: All 17,131 rows with per-sample loss weighting.
    Real samples: weight=1.0. Synthetic: weight=0.25 (4x down-weight).
    """
    texts   = df_train["posts"].astype(str).tolist()
    labels  = load_labels(df_train["labels"].tolist())
    weights = [1.0 if s == "original" else SYNTHETIC_WEIGHT
               for s in df_train["source"].tolist()]
    n_real  = sum(1 for s in df_train["source"] if s == "original")
    n_synth = len(df_train) - n_real
    print(f"    [Exp C] Fine-tune rows: {len(texts)} "
          f"(real={n_real}×1.0, synthetic={n_synth}×{SYNTHETIC_WEIGHT})")
    return texts, labels, weights


# ---------------------------------------------------------------------------
# TAPT — Task-Adaptive Pretraining via MLM
# ---------------------------------------------------------------------------
def run_tapt(hf_model_id: str, tokenizer, tapt_texts: list,
             tapt_dir: str, model_name: str) -> str:
    """
    Run TAPT on all 17,131 augmented texts (both real and synthetic) using MLM.
    Labels are NOT used — this is unsupervised vocabulary/domain adaptation.
    Runs once per model; checkpoint is shared by all 3 experiments.
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

    print(f"    [TAPT] Loading base weights from HuggingFace: {hf_model_id}")
    tapt_model = AutoModelForMaskedLM.from_pretrained(hf_model_id)

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
    """
    Run one fine-tuning experiment (A, B, or C) from a shared TAPT checkpoint.
    Returns test metrics dict.
    """
    os.makedirs(exp_dir, exist_ok=True)
    metrics_file = os.path.join(exp_dir, "metrics.json")

    if os.path.exists(metrics_file):
        print(f"    [Exp {exp_id.upper()}] Already done. Loading saved metrics.")
        with open(metrics_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return saved["test_metrics"]

    # --- Prepare experiment-specific training data ---
    if exp_id == "a":
        train_texts, train_labels, train_weights = prepare_exp_a(df_train)
        experiment_tag = "fix_base_a"
    elif exp_id == "b":
        train_texts, train_labels, train_weights = prepare_exp_b(df_train)
        experiment_tag = "fix_base_b"
    elif exp_id == "c":
        train_texts, train_labels, train_weights = prepare_exp_c(df_train)
        experiment_tag = "fix_base_c"
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

    # --- Load classification head from TAPT checkpoint ---
    # ignore_mismatched_sizes=True replaces MLM head with fresh 4-class head
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

    # Use WeightedTrainer for Exp C, standard Trainer otherwise
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
    preds      = test_preds.predictions
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

    # --- Save metrics ---
    training_params = {
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
    }
    save_metrics(
        path=metrics_file,
        model_name=model_name,
        experiment=experiment_tag,
        val_metrics=best_val,
        test_metrics=test_metrics,
        training_params=training_params,
    )

    # --- Confusion matrix ---
    plot_confusion_matrix(
        y_true=test_labels, y_pred=y_pred,
        title=f"Base+Aug Fix-{exp_id.upper()} {model_name}",
        output_path=os.path.join(FIG_DIR, f"cm_fix_base_{model_name.lower()}_exp{exp_id}.png"),
    )

    # --- Save best model weights ---
    best_model_dir = os.path.join(exp_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Delete training checkpoints immediately — best_model/ preserves the weights.
    # Frees 1-3 GB per experiment depending on model size.
    cleanup_checkpoints(exp_dir)

    flush_gpu(model, trainer)
    return test_metrics


# ---------------------------------------------------------------------------
# Per-model comparison chart
# ---------------------------------------------------------------------------
def plot_model_comparison(model_name: str, exp_results: dict, output_path: str):
    """Bar chart comparing F1 across the 3 experiments for one model."""
    labels = [f"Exp A\n(TAPT+Real)", f"Exp B\n(TAPT+MinorityAug)", f"Exp C\n(TAPT+Weighted)"]
    f1s    = [exp_results.get("a", {}).get("f1_weighted", 0),
              exp_results.get("b", {}).get("f1_weighted", 0),
              exp_results.get("c", {}).get("f1_weighted", 0)]
    accs   = [exp_results.get("a", {}).get("accuracy",   0),
              exp_results.get("b", {}).get("accuracy",   0),
              exp_results.get("c", {}).get("accuracy",   0)]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, accs, w, label="Accuracy", color="#4C72B0")
    ax.bar(x + w/2, f1s,  w, label="F1 (Weighted)", color="#DD8452")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.002, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, f + 0.002, f"{f:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.70, 1.00)
    ax.set_title(f"fix_base — {model_name} — Experiment Comparison", fontsize=13)
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

    tapt_texts  = df_train["posts"].astype(str).tolist()   # all 17,131 for TAPT
    val_texts   = df_val["posts"].astype(str).tolist()
    val_labels  = load_labels(df_val["labels"].tolist())
    test_texts  = df_test["posts"].astype(str).tolist()
    test_labels = load_labels(df_test["labels"].tolist())

    print(f"  TAPT corpus: {len(tapt_texts)} rows | Val: {len(val_texts)} | Test: {len(test_texts)}")
    print(f"  Source breakdown: {df_train['source'].value_counts().to_dict()}")

    all_results = {}   # {model: {exp_id: test_metrics}}

    for model_name, hf_model_id in MODELS.items():
        print(f"\n{'='*65}")
        print(f"  Model: {model_name}  ({hf_model_id})")
        print(f"{'='*65}")

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        tapt_dir  = os.path.join(model_dir, "tapt_shared")
        os.makedirs(model_dir, exist_ok=True)

        if all_experiments_done(model_dir):
            print(f"  [SKIP] All experiments done for {model_name}. Loading saved metrics.")
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

        # --- Load tokenizer ---
        print(f"  Loading tokenizer from HuggingFace: {hf_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

        # --- Stage 1: TAPT (runs once, shared by all 3 experiments) ---
        if tapt_done(tapt_dir):
            print(f"  [SKIP] TAPT already done → {tapt_dir}")
        else:
            print(f"  Running TAPT on {len(tapt_texts)} texts...")
            run_tapt(hf_model_id, tokenizer, tapt_texts, tapt_dir, model_name)

        # --- Stages 2A/2B/2C: Three experiments from shared TAPT checkpoint ---
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

        # --- Per-model comparison chart ---
        plot_model_comparison(
            model_name=model_name,
            exp_results=exp_results,
            output_path=os.path.join(FIG_DIR, f"fix_base_{model_name.lower()}_comparison.png"),
        )

        # --- Flush after all 3 experiments ---
        flush_gpu()

    # --- Final summary ---
    print(f"\n{'='*65}")
    print(f"  fix_base SUMMARY")
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
