"""5-fold stratified CV LR selection per checkpoint, on the TRAINING split only (original
distribution). Logs every fold result to results/transformer_kfold_cv_log.csv incrementally.
Writes the selected best LR per checkpoint to results/_selected_lr.json.
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import (
    load_and_split, ROOT, RESULTS, CHECKPOINTS, LR_GRID, PRIMARY_SEED,
    train_and_evaluate, append_csv_row,
)

HF_TOKEN = os.environ.get("HF_TOKEN")
ACCESSIBLE = ["roberta-base", "xlm-roberta-base", "deberta-v3-base", "mental-roberta-base"]
CV_MAX_EPOCHS = 6
CV_PATIENCE = 2

df, train_idx, val_idx, test_idx, info = load_and_split(ROOT / "dataset.xlsx")
texts = df["clean_text"].values
labels = df["label"].values

log_path = RESULTS / "transformer_kfold_cv_log.csv"
selected_path = RESULTS / "_selected_lr.json"
selected = {}
if selected_path.exists():
    selected = json.loads(selected_path.read_text())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=PRIMARY_SEED)
folds = list(skf.split(train_idx, labels[train_idx]))

done_keys = set()
if log_path.exists():
    prev = pd.read_csv(log_path)
    done_keys = set(zip(prev.checkpoint, prev.lr.astype(float), prev.fold.astype(int)))

for name in ACCESSIBLE:
    if name in selected:
        print(f"{name}: already has selected LR = {selected[name]}, skipping CV")
        continue
    hf_id = CHECKPOINTS[name]
    lr_scores = {lr: [] for lr in LR_GRID}
    for lr in LR_GRID:
        for fold_i, (tr_rel, va_rel) in enumerate(folds):
            key = (name, float(lr), fold_i)
            if key in done_keys:
                prev_row = pd.read_csv(log_path)
                prior = prev_row[(prev_row.checkpoint == name) & (prev_row.lr == lr) & (prev_row.fold == fold_i)]
                lr_scores[lr].append(float(prior.iloc[0]["val_macro_f1"]))
                print(f"{name} lr={lr} fold={fold_i}: cached val_f1={prior.iloc[0]['val_macro_f1']:.4f}")
                continue
            tr_ids = train_idx[tr_rel]
            va_ids = train_idx[va_rel]
            t0 = time.time()
            val_metrics, _, history, elapsed, _ = train_and_evaluate(
                hf_id, texts[tr_ids].tolist(), labels[tr_ids].tolist(),
                texts[va_ids].tolist(), labels[va_ids].tolist(),
                lr=lr, seed=PRIMARY_SEED, max_epochs=CV_MAX_EPOCHS, patience=CV_PATIENCE,
                hf_token=HF_TOKEN,
            )
            wall = time.time() - t0
            print(f"{name} lr={lr} fold={fold_i}: val_f1={val_metrics['macro_f1']:.4f} "
                  f"({len(history)} epochs, {wall:.1f}s)", flush=True)
            lr_scores[lr].append(val_metrics["macro_f1"])
            append_csv_row(log_path, dict(
                checkpoint=name, lr=lr, fold=fold_i, epochs_run=len(history),
                val_macro_f1=val_metrics["macro_f1"], wall_sec=wall,
            ))

    mean_scores = {lr: float(np.mean(v)) for lr, v in lr_scores.items()}
    best_lr = max(mean_scores, key=mean_scores.get)
    print(f"== {name}: mean CV scores {mean_scores} -> best_lr={best_lr}")
    selected[name] = best_lr
    selected_path.write_text(json.dumps(selected, indent=2))

print("Final selected LRs:", selected)
