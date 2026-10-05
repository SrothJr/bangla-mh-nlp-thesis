"""
Section 4 (Transformer Models) pipeline for the Bangla depression-severity benchmark.
Shared logic imported by the driver notebook / scripts. Kept as a plain module (not inline
notebook code) so that long-running training can be launched as background processes and
debugged independently of the notebook JSON.
"""
import os
import re
import gc
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, balanced_accuracy_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

LABEL_NAMES = ["Minimum", "Mild", "Moderate", "Severe"]
LABEL_REMAP = {1: 0, 2: 1, 3: 2, 4: 3}

CHECKPOINTS = {
    "roberta-base": "roberta-base",
    "xlm-roberta-base": "xlm-roberta-base",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "mental-bert-base-uncased": "mental/mental-bert-base-uncased",
    "mental-roberta-base": "mental/mental-roberta-base",
}

SEEDS = [42, 123, 2024, 3407, 9999]
LR_GRID = [1e-5, 2e-5, 3e-5, 5e-5]
MAX_LENGTH = 224
BATCH_SIZE = 32
PRIMARY_SEED = 42

# ---------------------------------------------------------------------------
# Text cleaning (must exactly match Sections 1-3)
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_SYMBOL_RE.sub("", t)
    t = REPEATED_CHAR_RE.sub(r"\1\1\1", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def load_and_split(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df["label"] = df["labels"].map(LABEL_REMAP)
    assert df["label"].isna().sum() == 0, "Unmapped label values found"
    df["clean_text"] = df["posts"].apply(clean_text)
    before = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    after = len(df)

    idx = np.arange(len(df))
    labels = df["label"].values

    train_val_idx, test_idx = train_test_split(
        idx, test_size=0.10, stratify=labels, random_state=42
    )
    relative_val_size = 0.10 / (1 - 0.10)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_size,
        stratify=labels[train_val_idx], random_state=42,
    )

    info = dict(rows_before_drop=before, rows_after_drop=after,
                n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx))
    return df, train_idx, val_idx, test_idx, info


# ---------------------------------------------------------------------------
# Tokenizer compatibility audit
# ---------------------------------------------------------------------------
def tokenizer_audit(name, hf_id, texts, hf_token=None):
    from transformers import AutoTokenizer
    result = {"checkpoint": name, "hf_id": hf_id}
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    except Exception as e:
        result.update(status="ACCESS RESTRICTED" if "gated" in str(e).lower() or "403" in str(e) else "LOAD FAILED",
                       error=str(e)[:500], tokens_per_word=None, unk_rate=None)
        return result

    total_tokens, total_words, total_unk = 0, 0, 0
    unk_id = tok.unk_token_id
    for t in texts:
        words = t.split()
        total_words += len(words)
        ids = tok(t, add_special_tokens=False)["input_ids"]
        total_tokens += len(ids)
        if unk_id is not None:
            total_unk += sum(1 for i in ids if i == unk_id)
    result.update(
        status="OK",
        tokens_per_word=total_tokens / max(total_words, 1),
        unk_rate=total_unk / max(total_tokens, 1),
        n_sample=len(texts),
        error=None,
    )
    return result


# ---------------------------------------------------------------------------
# Resampling (row-level, applied to training indices only)
# ---------------------------------------------------------------------------
def resample_indices(train_idx, labels, method, seed):
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    X = train_idx.reshape(-1, 1)
    y = labels[train_idx]
    if method == "original":
        return train_idx.copy()
    elif method == "oversampled":
        sampler = RandomOverSampler(random_state=seed)
    elif method == "undersampled":
        sampler = RandomUnderSampler(random_state=seed)
    else:
        raise ValueError(method)
    X_res, _ = sampler.fit_resample(X, y)
    return X_res.reshape(-1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_full_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    pcp, pcr, pcf1, pcs = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    diffs = np.abs(np.array(y_true) - np.array(y_pred))
    mae = diffs.mean()
    within1 = (diffs <= 1).mean()
    out = dict(
        accuracy=acc, macro_precision=mp, macro_recall=mr, macro_f1=mf1,
        weighted_precision=wp, weighted_recall=wr, weighted_f1=wf1,
        balanced_accuracy=bacc, severity_mae=mae, within_one_level_accuracy=within1,
    )
    for i, name in enumerate(LABEL_NAMES):
        out[f"{name.lower()}_precision"] = pcp[i]
        out[f"{name.lower()}_recall"] = pcr[i]
        out[f"{name.lower()}_f1"] = pcf1[i]
        out[f"{name.lower()}_support"] = int(pcs[i])
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training core
# ---------------------------------------------------------------------------
def build_model_and_tokenizer(hf_id, num_labels, hf_token=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    # Force fp32 master weights explicitly: some checkpoints (e.g. deberta-v3-base) declare
    # fp16 in their saved config, and from_pretrained honors that dtype by default. fp16 master
    # weights combined with autocast bf16 training caused DeBERTa's disentangled-attention
    # position-bias terms to overflow into NaN within the first optimizer step.
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=num_labels, token=hf_token, torch_dtype=torch.float32
    )
    return tok, model


def make_optimizer(model, lr, weight_decay=0.01):
    from torch.optim import AdamW
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    return AdamW(grouped, lr=lr)


def train_and_evaluate(
    hf_id, texts_train, y_train, texts_val, y_val, texts_test=None, y_test=None,
    lr=2e-5, seed=42, max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
    grad_accum=1, max_epochs=10, patience=3, hf_token=None, num_labels=4,
    verbose=False,
):
    """Full fine-tune. Returns (val_metrics, test_metrics_or_None, history, elapsed_sec)."""
    from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
    from torch.utils.data import Dataset, DataLoader

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    tok, model = build_model_and_tokenizer(hf_id, num_labels, hf_token=hf_token)
    model.to(device)

    class TextDS(Dataset):
        def __init__(self, texts, labels):
            self.texts = list(texts)
            self.labels = list(labels)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = tok(self.texts[i], truncation=True, max_length=max_length)
            enc["labels"] = int(self.labels[i])
            return enc

    collator = DataCollatorWithPadding(tok)
    train_ds = TextDS(texts_train, y_train)
    val_ds = TextDS(texts_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = make_optimizer(model, lr)
    total_steps = (len(train_loader) // grad_accum) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )

    best_val_f1 = -1
    best_state = None
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if bf16_ok else torch.float16,
                                 enabled=torch.cuda.is_available()):
                out = model(**batch)
                loss = out.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if bf16_ok else torch.float16,
                                     enabled=torch.cuda.is_available()):
                    out = model(**batch)
                p = out.logits.argmax(-1).cpu().numpy()
                preds.extend(p.tolist())
                trues.extend(labels.numpy().tolist())
        val_metrics = compute_full_metrics(trues, preds)
        history.append({"epoch": epoch + 1, "val_macro_f1": val_metrics["macro_f1"]})
        if verbose:
            print(f"  epoch {epoch+1}: val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            best_val_metrics = val_metrics
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    elapsed = time.time() - t0

    test_metrics = None
    y_pred_test = None
    if texts_test is not None:
        model.load_state_dict(best_state)
        model.eval()
        test_ds = TextDS(texts_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if bf16_ok else torch.float16,
                                     enabled=torch.cuda.is_available()):
                    out = model(**batch)
                p = out.logits.argmax(-1).cpu().numpy()
                preds.extend(p.tolist())
                trues.extend(labels.numpy().tolist())
        test_metrics = compute_full_metrics(trues, preds)
        y_pred_test = preds

    del model, optimizer, scheduler, best_state
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_metrics, test_metrics, history, elapsed, y_pred_test


def append_csv_row(path, row: dict):
    df_row = pd.DataFrame([row])
    if Path(path).exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)
