"""
train_phase3_dapt.py  —  Phase 3: Mega-TAPT + Back-Translation Ablations

KEY FIXES OVER PHASE 2:
  1. Mega-TAPT corpus (~45k posts) combines all three sources:
       - 24,699 unlabeled Qwen posts (diverse vocabulary)
       - 17,131 Qwen augmented posts (clinical label-conditioned vocabulary)
       - 3,230 NLLB back-translated posts (paraphrased gold-standard language)
     → Labels are DROPPED for all sources. This is unsupervised MLM only.

  2. NUM_EPOCHS raised to 20, EARLY_STOPPING_PAT raised to 5.
     Phase 2 and fix_dapt showed models were still improving at epoch 10.
     Giving 20 epochs ensures full convergence.

  3. Four experiments per model (A/B/C/D) sharing one TAPT checkpoint:
       Exp D: Real only (3,426)         — mirrors the fix_dapt SOTA winner
       Exp A: Real + BT Label 3 & 4    — surgical minority boost
       Exp B: Real + BT All Labels     — uniform BT augmentation
       Exp C: Real + BT Label 3 only   — hyper-targeted Label-3 fix

  4. Storage: save_total_limit=1, checkpoints in isolated subdir,
     auto-cleanup after each stage, no intermediate weights left on disk.

  5. Crash-resume: get_last_checkpoint used for both TAPT and each exp.
     Re-running the script skips any completed stage automatically.

  6. Pure PyTorch Dataset — no HuggingFace 'datasets' import needed.

  7. Mega-TAPT corpus is built and written to disk once, then reused.
"""

import gc
import glob
import json
import os
import shutil
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from utils_aug import (
    compute_full_metrics,
    compute_metrics,
    plot_confusion_matrix,
    save_metrics,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
TAPT_25K    = os.path.join(_HERE, "tapt_qwen_data",           "tapt_corpus_clean.txt")
AUG_17K     = os.path.join(_HERE, "train_augmented_qwen3-32b.xlsx")
BT_PATH     = os.path.join(_HERE, "nllb_backtranslated_data", "train_backtranslated_cleaned.xlsx")
BASE_TRAIN  = os.path.join(_HERE, "train_augmented_qwen3-32b.xlsx")   # real rows filtered by source
VAL_PATH    = os.path.join(_HERE, "locked", "val.xlsx")
TEST_PATH   = os.path.join(_HERE, "locked", "test.xlsx")
DAPT_DIR    = os.path.join(_HERE, "dapt_pretrained")
OUTPUT_DIR  = os.path.join(_HERE, "results", "phase3_dapt")
FIG_DIR     = os.path.join(_HERE, "results", "figures")
MEGA_CORPUS = os.path.join(_HERE, "tapt_qwen_data", "mega_tapt_corpus.txt")

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
# Constants — all hyperparameters identical to prior runs EXCEPT epochs/patience
# ---------------------------------------------------------------------------
CLASSES  = ["Minimum", "Mild", "Moderate", "Severe"]
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

MAX_LEN            = 256
NUM_EPOCHS         = 20       # raised from 10 — models were still improving at ep 10
LEARNING_RATE      = 2e-5
WEIGHT_DECAY       = 0.01
WARMUP_STEPS       = 500
EARLY_STOPPING_PAT = 5        # raised from 3 — more patience to find true best
EVAL_BATCH_SIZE    = 32
SEED               = 42

TAPT_MAX_LEN    = 256
TAPT_EPOCHS     = 10
TAPT_LR         = 3e-5
TAPT_BATCH      = 16
TAPT_PATIENCE   = 3
MLM_PROBABILITY = 0.15


# ---------------------------------------------------------------------------
# Pure-PyTorch Dataset classes — no HuggingFace 'datasets' needed
# ---------------------------------------------------------------------------
class TAPTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


class BanglaDataset(torch.utils.data.Dataset):
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
    """Delete checkpoint-* subdirs after stage completes to free storage."""
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
    return all(
        exp_done(os.path.join(model_dir, f"exp_{x}"))
        for x in ["a", "b", "c", "d"]
    )


# ---------------------------------------------------------------------------
# Build Mega-TAPT corpus (runs once, cached to disk)
# ---------------------------------------------------------------------------
def build_mega_corpus():
    if os.path.exists(MEGA_CORPUS):
        with open(MEGA_CORPUS, "r", encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"  [Mega-TAPT] Corpus already built: {n} posts at {MEGA_CORPUS}")
        return

    print("  [Mega-TAPT] Building mega corpus from 3 sources...")
    lines = []

    # Source 1: 25k Qwen unlabeled posts
    with open(TAPT_25K, "r", encoding="utf-8") as f:
        src1 = [l.strip() for l in f if len(l.strip()) > 10]
    print(f"    Source 1 (Qwen unlabeled): {len(src1)} posts")
    lines.extend(src1)

    # Source 2: 17k Qwen augmented dataset — text only, labels dropped
    df_aug = pd.read_excel(AUG_17K, engine="openpyxl")
    src2 = [str(t).strip() for t in df_aug["posts"].tolist() if len(str(t).strip()) > 10]
    print(f"    Source 2 (Qwen augmented): {len(src2)} posts")
    lines.extend(src2)

    # Source 3: BT back-translated posts — text only, labels dropped
    df_bt = pd.read_excel(BT_PATH, engine="openpyxl")
    src3 = [str(t).strip() for t in df_bt["backtranslated_post"].tolist() if len(str(t).strip()) > 10]
    print(f"    Source 3 (BT translated): {len(src3)} posts")
    lines.extend(src3)

    # Deduplicate and shuffle
    lines = list(dict.fromkeys(lines))  # preserve order while deduplicating
    import random
    random.seed(SEED)
    random.shuffle(lines)

    os.makedirs(os.path.dirname(MEGA_CORPUS), exist_ok=True)
    with open(MEGA_CORPUS, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"  [Mega-TAPT] Built: {len(lines)} unique posts -> {MEGA_CORPUS}")


# ---------------------------------------------------------------------------
# TAPT Routine
# ---------------------------------------------------------------------------
def run_tapt(dapt_path: str, tokenizer, tapt_dir: str, model_name: str):
    os.makedirs(tapt_dir, exist_ok=True)

    with open(MEGA_CORPUS, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if len(l.strip()) > 10]

    print(f"    [TAPT] {len(lines)} posts loaded from mega corpus.")
    print("    [TAPT] Tokenizing...")

    enc = tokenizer(
        lines,
        truncation=True,
        padding="max_length",
        max_length=TAPT_MAX_LEN,
        return_token_type_ids=False,
    )

    n       = len(lines)
    n_val   = max(1, int(n * 0.10))
    n_train = n - n_val

    ds_train = TAPTDataset({k: v[:n_train] for k, v in enc.items()})
    ds_val   = TAPTDataset({k: v[n_train:] for k, v in enc.items()})
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROBABILITY
    )

    print(f"    [TAPT] Loading DAPT weights: {dapt_path}")
    model = AutoModelForMaskedLM.from_pretrained(dapt_path)

    args = TrainingArguments(
        output_dir                  = tapt_dir,
        num_train_epochs            = TAPT_EPOCHS,
        per_device_train_batch_size = TAPT_BATCH,
        per_device_eval_batch_size  = TAPT_BATCH * 2,
        learning_rate               = TAPT_LR,
        weight_decay                = WEIGHT_DECAY,
        warmup_steps                = 100,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 1,
        logging_steps               = 100,
        report_to                   = "none",
        seed                        = SEED,
        fp16                        = torch.cuda.is_available(),
        dataloader_num_workers      = 0,
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
        print(f"    [TAPT] Resuming from {os.path.basename(last_ckpt)}")
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
    exp_id:       str,
    tapt_path:    str,
    tokenizer,
    train_texts:  list,
    train_labels: list,
    val_texts:    list,
    val_labels:   list,
    test_texts:   list,
    test_labels:  list,
    batch_cfg:    dict,
    model_name:   str,
    exp_dir:      str,
):
    metrics_file   = os.path.join(exp_dir, "metrics.json")
    best_model_dir = os.path.join(exp_dir, "best_model")
    ckpt_dir       = os.path.join(exp_dir, "checkpoints")
    os.makedirs(exp_dir, exist_ok=True)

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
        num_labels              = 4,
        id2label                = ID2LABEL,
        label2id                = LABEL2ID,
        ignore_mismatched_sizes = True,
    )

    args = TrainingArguments(
        output_dir                  = ckpt_dir,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = batch_cfg["batch"],
        gradient_accumulation_steps = batch_cfg["accum"],
        per_device_eval_batch_size  = EVAL_BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        warmup_steps                = WARMUP_STEPS,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        save_total_limit            = 1,
        logging_steps               = 50,
        report_to                   = "none",
        seed                        = SEED,
        fp16                        = torch.cuda.is_available(),
        dataloader_num_workers      = 0,
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = ds_train,
        eval_dataset    = ds_val,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PAT)],
    )

    last_ckpt = get_last_checkpoint(ckpt_dir)
    if last_ckpt:
        print(f"    [Exp {exp_id.upper()}] Resuming from {os.path.basename(last_ckpt)}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # Extract best val metrics from log history
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
          f"Acc: {best_val.get('accuracy','?')}  "
          f"F1: {best_val.get('f1','?')}  "
          f"(epoch {best_val.get('epoch','?')}/{NUM_EPOCHS})")

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

    # Save weights and metrics before cleanup (order matters for crash safety)
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    save_metrics(
        path            = metrics_file,
        model_name      = model_name,
        experiment      = f"phase3_dapt_{exp_id}",
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
            "early_stop_pat":  EARLY_STOPPING_PAT,
            "seed":            SEED,
            "fp16":            torch.cuda.is_available(),
            "tapt_corpus":     f"mega_tapt_corpus.txt (~45k posts)",
            "tapt_lr":         TAPT_LR,
            "tapt_epochs":     TAPT_EPOCHS,
            "mlm_probability": MLM_PROBABILITY,
            "finetune_rows":   len(train_texts),
            "val_rows":        len(val_texts),
            "test_rows":       len(test_texts),
        },
    )

    plot_confusion_matrix(
        y_true      = test_labels,
        y_pred      = y_pred,
        title       = f"Phase 3 DAPT - {model_name} Exp {exp_id.upper()}",
        output_path = os.path.join(FIG_DIR, f"cm_phase3_{model_name.lower()}_exp{exp_id}.png"),
    )

    flush_gpu(model, trainer)
    clean_checkpoints(ckpt_dir)  # free storage AFTER best_model/ and metrics.json are safely written

    return test_metrics


# ---------------------------------------------------------------------------
# Per-model summary chart
# ---------------------------------------------------------------------------
def plot_model_comparison(model_name: str, exp_results: dict):
    exp_labels = {
        "d": "Exp D\n(Real only)",
        "a": "Exp A\n(Real+BT L3/4)",
        "b": "Exp B\n(Real+BT All)",
        "c": "Exp C\n(Real+BT L3)",
    }
    keys = [k for k in ["d", "a", "b", "c"] if k in exp_results]
    labels = [exp_labels[k] for k in keys]
    f1s  = [exp_results[k].get("f1_weighted", 0) for k in keys]
    accs = [exp_results[k].get("accuracy",    0) for k in keys]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, accs, w, label="Accuracy",      color="#4C72B0")
    ax.bar(x + w/2, f1s,  w, label="F1 (Weighted)", color="#DD8452")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.002, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, f + 0.002, f"{f:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.70, 1.00)
    ax.set_title(f"Phase 3 DAPT — {model_name} (20 epochs, mega-TAPT ~45k)", fontsize=13)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"phase3_dapt_{model_name.lower()}_comparison.png")
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

    # ---- Step 0: Build mega-TAPT corpus once ----
    build_mega_corpus()

    # ---- Load fine-tuning datasets ----
    print("\nLoading fine-tuning datasets...")
    df_base = pd.read_excel(BASE_TRAIN, engine="openpyxl")
    df_real = df_base[df_base["source"] == "original"].copy().reset_index(drop=True)
    print(f"  Real rows:            {len(df_real)}")

    df_bt = pd.read_excel(BT_PATH, engine="openpyxl")
    df_bt = df_bt.rename(columns={"backtranslated_post": "posts", "label": "labels"})
    df_bt["source"] = "synthetic"
    print(f"  Back-translated rows: {len(df_bt)}")

    df_val  = pd.read_excel(VAL_PATH,  engine="openpyxl")
    df_test = pd.read_excel(TEST_PATH, engine="openpyxl")

    val_texts   = df_val["posts"].astype(str).tolist()
    val_labels  = load_labels(df_val["labels"].tolist())
    test_texts  = df_test["posts"].astype(str).tolist()
    test_labels = load_labels(df_test["labels"].tolist())
    print(f"  Val: {len(val_texts)} | Test: {len(test_texts)}")

    # ---- Build experiment training sets (once, reused for all models) ----
    # Exp D: Real only — cleanest, mirrors fix_dapt winner
    df_exp_d = df_real.copy()

    # Exp A: Real + BT Label 3 & 4
    df_bt_34 = df_bt[df_bt["labels"].isin([3, 4])].copy()
    df_exp_a = pd.concat([df_real, df_bt_34], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Exp B: Real + BT All Labels
    df_exp_b = pd.concat([df_real, df_bt], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Exp C: Real + BT Label 3 only
    df_bt_3  = df_bt[df_bt["labels"] == 3].copy()
    df_exp_c = pd.concat([df_real, df_bt_3], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  Exp D rows: {len(df_exp_d)}  (Real only)")
    print(f"  Exp A rows: {len(df_exp_a)}  (Real + BT L3/L4)")
    print(f"  Exp B rows: {len(df_exp_b)}  (Real + BT All)")
    print(f"  Exp C rows: {len(df_exp_c)}  (Real + BT L3 only)")

    EXP_DATA = {
        "d": (df_exp_d, "Real only"),
        "a": (df_exp_a, "Real + BT L3/L4"),
        "b": (df_exp_b, "Real + BT All"),
        "c": (df_exp_c, "Real + BT L3 only"),
    }

    all_results = {}

    for model_name, dapt_path in MODELS.items():
        print(f"\n{'='*65}")
        print(f"  Model : {model_name}")
        print(f"  DAPT  : {dapt_path}")
        print(f"{'='*65}")

        if not os.path.isdir(dapt_path):
            print(f"  [ERROR] DAPT weights not found: {dapt_path}. Skipping.")
            continue

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        tapt_dir  = os.path.join(model_dir, "tapt_shared")
        bcfg      = BATCH_CFG[model_name]
        os.makedirs(model_dir, exist_ok=True)

        if all_exps_done(model_dir):
            print(f"  [SKIP] All 4 experiments done for {model_name}.")
            exp_results = {}
            for x in ["a", "b", "c", "d"]:
                mf = os.path.join(model_dir, f"exp_{x}", "metrics.json")
                if os.path.exists(mf):
                    with open(mf, "r", encoding="utf-8") as f:
                        exp_results[x] = json.load(f)["test_metrics"]
            all_results[model_name] = exp_results
            plot_model_comparison(model_name, exp_results)
            continue

        # Load tokenizer once per model
        tokenizer = AutoTokenizer.from_pretrained(dapt_path)

        # ---- TAPT ----
        if tapt_done(tapt_dir):
            print(f"  [SKIP] TAPT complete — reusing {tapt_dir}")
        else:
            print("  Running Mega-TAPT...")
            run_tapt(dapt_path, tokenizer, tapt_dir, model_name)

        # ---- Four experiments ----
        exp_results = {}
        for exp_id, (df_exp, desc) in EXP_DATA.items():
            print(f"\n  --- Experiment {exp_id.upper()} [{desc}] ({model_name}) ---")
            exp_dir      = os.path.join(model_dir, f"exp_{exp_id}")
            train_texts  = df_exp["posts"].astype(str).tolist()
            train_labels = load_labels(df_exp["labels"].tolist())

            test_metrics = run_experiment(
                exp_id       = exp_id,
                tapt_path    = tapt_dir,
                tokenizer    = tokenizer,
                train_texts  = train_texts,
                train_labels = train_labels,
                val_texts    = val_texts,
                val_labels   = val_labels,
                test_texts   = test_texts,
                test_labels  = test_labels,
                batch_cfg    = bcfg,
                model_name   = model_name,
                exp_dir      = exp_dir,
            )
            exp_results[exp_id] = test_metrics

        all_results[model_name] = exp_results
        plot_model_comparison(model_name, exp_results)
        flush_gpu()

    # ---- Final summary ----
    print(f"\n{'='*65}")
    print("  PHASE 3 SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<12} {'Exp':<8} {'Accuracy':>10} {'F1-Wt':>10} {'L3 F1':>8} {'L4 F1':>8}")
    print(f"  {'-'*60}")
    for mn, er in all_results.items():
        for x in ["d", "a", "b", "c"]:
            if x not in er:
                continue
            tm = er[x]
            l3 = tm.get("per_class", {}).get("Moderate", {}).get("f1", 0)
            l4 = tm.get("per_class", {}).get("Severe",   {}).get("f1", 0)
            print(f"  {mn:<12} Exp {x.upper():<4} "
                  f"{tm.get('accuracy',0):>10.4f} "
                  f"{tm.get('f1_weighted',0):>10.4f} "
                  f"{l3:>8.4f} {l4:>8.4f}")

    print(f"\n  Results -> {OUTPUT_DIR}")
    print(f"  Figures -> {FIG_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
