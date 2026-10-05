import os
import tempfile
import pandas as pd
import numpy as np
import torch
import warnings
import json
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from utils import compute_metrics, plot_confusion_matrix, plot_performance

# Ignore specific huggingface warnings for cleaner output
warnings.filterwarnings("ignore")

# Directories
DATA_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw/dataset.xlsx"))
DAPT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/dapt_pretrained"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/dapt_eval"))
FIG_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/figures"))

# Each model is loaded from its locally saved DAPT weights folder
MODELS = {
    "BanglaBERT": os.path.join(DAPT_DIR, "BanglaBERT"),
    "sahajBERT":  os.path.join(DAPT_DIR, "sahajBERT"),
    "mBERT":      os.path.join(DAPT_DIR, "mBERT"),
}

CLASSES  = ["Minimum", "Mild", "Moderate", "Severe"]
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

# ============================================================ #
#  TAPT CONFIGURATION
#
#  Task-Adaptive Pretraining (TAPT) is performed INSIDE each
#  CV fold, using ONLY that fold's training partition.
#
#  DATA LEAKAGE PREVENTION:
#  The held-out validation fold (val_texts) is NEVER passed to
#  run_tapt(). The model's weights are updated solely from
#  train_texts during both TAPT and subsequent fine-tuning.
#  val_texts first appear only in the final trainer.predict()
#  call, identical to how unseen deployment data would arrive.
#
#  This mirrors the methodology of Gururangan et al. (ACL 2020)
#  where TAPT uses task-adjacent training data, not held-out
#  evaluation data.
# ============================================================ #
TAPT_EPOCHS   = 30    # EarlyStoppingCallback will terminate before this if loss plateaus
TAPT_LR       = 3e-5
TAPT_BATCH    = 16    # Conservative for 256-token sequences on large GPU
TAPT_MAX_LEN  = 256   # Must match DAPT and fine-tuning — consistent across all stages
TAPT_PATIENCE = 5     # EarlyStopping patience epochs for TAPT MLM loss

# ============================================================ #
#  FINE-TUNING CONFIGURATION
# ============================================================ #
FINETUNE_MAX_LEN = 256


class BanglaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_fold_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute per-fold classification metrics from raw label lists.
    Kept separate from HuggingFace's compute_metrics() so it can be
    called outside of a Trainer context.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        'accuracy':  round(float(acc),       4),
        'f1':        round(float(f1),        4),
        'precision': round(float(precision), 4),
        'recall':    round(float(recall),    4),
    }


def run_tapt(dapt_model_path: str, tokenizer, train_texts: list[str], tapt_output_dir: str) -> str:
    """
    Runs Task-Adaptive Pretraining (TAPT) via MLM on the training-fold
    texts only. The validation fold is never touched here.

    DATA ISOLATION GUARANTEE:
    -------------------------
    - Input:  train_texts — the ~3,900 texts from the TRAINING partition
              of the current fold (no labels, no val_texts).
    - Output: tapt_output_dir — directory containing the best MLM
              checkpoint, used to initialise the classification head.

    The 90/10 split inside this function (tapt_train / tapt_val) is an
    internal split of train_texts only, used to drive EarlyStopping on
    the TAPT MLM loss. It does not involve any held-out fold data.

    Parameters
    ----------
    dapt_model_path : str
        Path to the completed DAPT weights (ForMaskedLM).
    tokenizer : PreTrainedTokenizer
        Tokenizer loaded from the DAPT model directory.
    train_texts : list[str]
        Raw Bangla text strings from this fold's training partition.
        MUST NOT include any validation or test samples.
    tapt_output_dir : str
        Directory to save the best TAPT checkpoint.

    Returns
    -------
    str
        Path to the saved TAPT model directory.
    """
    os.makedirs(tapt_output_dir, exist_ok=True)

    # Build a HuggingFace Dataset from the training texts (labels not needed)
    hf_dataset = Dataset.from_dict({"text": train_texts})

    # Internal 90/10 split of train_texts for TAPT MLM early stopping.
    # This is NOT the CV split — it is entirely within the training partition.
    split      = hf_dataset.train_test_split(test_size=0.1, seed=42)
    tapt_train = split["train"]
    tapt_val   = split["test"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=TAPT_MAX_LEN
        )

    tapt_train = tapt_train.map(tokenize_fn, batched=True, remove_columns=["text"])
    tapt_val   = tapt_val.map(tokenize_fn,   batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Load DAPT weights as a masked language model for continued MLM
    tapt_model = AutoModelForMaskedLM.from_pretrained(dapt_model_path)

    tapt_args = TrainingArguments(
        output_dir=tapt_output_dir,
        num_train_epochs=TAPT_EPOCHS,
        per_device_train_batch_size=TAPT_BATCH,
        per_device_eval_batch_size=TAPT_BATCH * 2,
        learning_rate=TAPT_LR,
        warmup_steps=100,  # ~10% of first 3 epochs (~3.5K TAPT texts / batch 16 = ~220 steps/epoch)
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=20,
        seed=42,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    tapt_trainer = Trainer(
        model=tapt_model,
        args=tapt_args,
        train_dataset=tapt_train,
        eval_dataset=tapt_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=TAPT_PATIENCE)]
    )

    tapt_trainer.train()

    # Save the best TAPT model and tokenizer to the fold-specific directory
    tapt_trainer.save_model(tapt_output_dir)
    tokenizer.save_pretrained(tapt_output_dir)

    # Free GPU memory before moving on to classification fine-tuning
    del tapt_model
    del tapt_trainer
    torch.cuda.empty_cache()
    import gc
    import time
    gc.collect()
    time.sleep(10)

    return tapt_output_dir


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)

    print("Using 'posts' for text and 'labels' for labels.")
    texts      = df['posts'].astype(str).tolist()
    labels_raw = df['labels'].tolist()

    # Convert 1-indexed labels (1, 2, 3, 4) to 0-indexed (0, 1, 2, 3)
    labels = []
    for l in labels_raw:
        if isinstance(l, str):
            labels.append(LABEL2ID.get(l.capitalize(), 0))
        else:
            labels.append(int(l) - 1)

    # Stratified 5-Fold CV — same seed as baseline for a fair apples-to-apples comparison
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    final_results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*60}\nFine-tuning DAPT+TAPT Model: {model_name}\nWeights: {model_path}\n{'='*60}")

        # Verify the DAPT weights exist before trying to load them
        if not os.path.isdir(model_path):
            print(f"ERROR: DAPT weights not found at {model_path}. Skipping {model_name}.")
            continue

        metrics_file = os.path.join(OUTPUT_DIR, f"{model_name}_dapt_metrics.json")
        if os.path.exists(metrics_file):
            print(f"Skipping {model_name} — already evaluated. Loading saved metrics.")
            with open(metrics_file, 'r') as f:
                final_results[model_name] = json.load(f)
            continue

        # Load tokenizer from the DAPT model folder (not from HuggingFace)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Accumulators for pooled metrics (all folds combined)
        all_y_true = []
        all_y_pred = []

        # Per-fold metric storage for mean ± std reporting
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            print(f"\n--- Fold {fold + 1}/5 ---")

            train_texts  = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts    = [texts[i] for i in val_idx]
            val_labels   = [labels[i] for i in val_idx]

            # ---------------------------------------------------------- #
            #  TAPT — isolated to train_texts of this fold only.
            #  val_texts are NOT passed here under any circumstances.
            # ---------------------------------------------------------- #
            tapt_output_dir   = os.path.join(OUTPUT_DIR, f"{model_name}_fold{fold + 1}_tapt")
            print(f"  [TAPT] Starting on {len(train_texts)} train texts "
                  f"({len(val_texts)} val texts withheld)...")
            tapt_weights_path = run_tapt(model_path, tokenizer, train_texts, tapt_output_dir)
            print(f"  [TAPT] Complete → {tapt_weights_path}")

            # ---------------------------------------------------------- #
            #  Fine-tuning — starts from TAPT weights (not raw DAPT)
            # ---------------------------------------------------------- #
            train_encodings = tokenizer(
                train_texts, truncation=True, padding='max_length', max_length=FINETUNE_MAX_LEN
            )
            val_encodings = tokenizer(
                val_texts, truncation=True, padding='max_length', max_length=FINETUNE_MAX_LEN
            )

            train_dataset = BanglaDataset(train_encodings, train_labels)
            val_dataset   = BanglaDataset(val_encodings,   val_labels)

            # Load classification model from TAPT weights.
            # ignore_mismatched_sizes=True is required: the saved TAPT model has an
            # MLM head (lm_head); we replace it with a fresh classification head.
            model = AutoModelForSequenceClassification.from_pretrained(
                tapt_weights_path,
                num_labels=4,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True
            )

            training_args = TrainingArguments(
                output_dir=os.path.join(OUTPUT_DIR, f"{model_name}_fold{fold + 1}"),
                num_train_epochs=15,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                learning_rate=2e-5,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                save_total_limit=1,
                logging_steps=50,
                seed=42,
                fp16=torch.cuda.is_available(),
                report_to="none"
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )

            trainer.train()

            # Predict on the held-out validation fold
            preds       = trainer.predict(val_dataset)
            predictions = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
            y_pred      = list(predictions.argmax(-1))

            # ---- Per-fold metrics (for mean ± std reporting) ----
            fold_m = compute_fold_metrics(val_labels, y_pred)
            fold_metrics.append(fold_m)
            print(f"  [Fold {fold + 1}] Accuracy={fold_m['accuracy']:.4f} | "
                  f"F1={fold_m['f1']:.4f} | "
                  f"Precision={fold_m['precision']:.4f} | "
                  f"Recall={fold_m['recall']:.4f}")

            # ---- Accumulate for pooled metrics ----
            all_y_true.extend(val_labels)
            all_y_pred.extend(y_pred)

            # Free GPU memory between folds
            del model
            del trainer
            torch.cuda.empty_cache()
            import gc
            import time
            gc.collect()
            time.sleep(10)

        # ---- Pooled metrics (computed on all 4,898 predictions at once) ----
        pooled = compute_fold_metrics(all_y_true, all_y_pred)

        # ---- Per-fold mean ± std ----
        def mean_std(key):
            vals = [fm[key] for fm in fold_metrics]
            return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

        acc_mean,  acc_std  = mean_std('accuracy')
        f1_mean,   f1_std   = mean_std('f1')
        prec_mean, prec_std = mean_std('precision')
        rec_mean,  rec_std  = mean_std('recall')

        # ---- Build the complete metrics payload ----
        overall_metrics = {
            # Pooled metrics — primary reporting numbers
            'accuracy':           pooled['accuracy'],
            'f1':                 pooled['f1'],
            'precision':          pooled['precision'],
            'recall':             pooled['recall'],
            # Per-fold distribution — for thesis mean ± std table
            'per_fold': {
                'accuracy':  {'mean': acc_mean,  'std': acc_std},
                'f1':        {'mean': f1_mean,   'std': f1_std},
                'precision': {'mean': prec_mean, 'std': prec_std},
                'recall':    {'mean': rec_mean,  'std': rec_std},
            },
            # Individual fold results — full traceability
            'fold_results': fold_metrics,
        }

        final_results[model_name] = overall_metrics

        # Save metrics immediately so reruns will skip this model
        with open(metrics_file, 'w') as f:
            json.dump(overall_metrics, f, indent=2)

        print(f"\n[DONE] DAPT+TAPT {model_name} — Results:")
        print(f"  Pooled  → Accuracy: {pooled['accuracy']:.4f} | F1: {pooled['f1']:.4f}")
        print(f"  Per-fold → Accuracy: {acc_mean:.4f} ± {acc_std:.4f} | "
              f"F1: {f1_mean:.4f} ± {f1_std:.4f}")

        # Save confusion matrix labelled as DAPT+TAPT
        plot_confusion_matrix(
            all_y_true,
            all_y_pred,
            CLASSES,
            f'DAPT+TAPT {model_name} Confusion Matrix',
            os.path.join(FIG_DIR, f'cm_dapt_{model_name.lower()}.png')
        )

        # For the performance chart, pass only top-level keys (pooled values)
        final_results[model_name]['_chart'] = pooled

    # Generate performance comparison chart using pooled values
    chart_data = {k: v['_chart'] for k, v in final_results.items() if '_chart' in v}
    if chart_data:
        plot_performance(chart_data, os.path.join(FIG_DIR, 'dapt_performance_comparison.png'))

    print(f"\nAll DAPT+TAPT evaluations finished! Figures saved in {FIG_DIR}.")

if __name__ == "__main__":
    main()
