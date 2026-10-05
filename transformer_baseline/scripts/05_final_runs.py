"""Final fine-tuning grid: checkpoint x scenario x seed, using each checkpoint's CV-selected LR.
Saves incrementally to results/transformer_seed_results.csv (resumable) and confusion-matrix
arrays to results/figures/_cm_<checkpoint>_<scenario>_<seed>.npy (raw counts, order Min/Mild/Mod/Sev).
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import (
    load_and_split, ROOT, RESULTS, FIGURES, CHECKPOINTS, SEEDS,
    train_and_evaluate, resample_indices, append_csv_row, compute_full_metrics,
)
from sklearn.metrics import confusion_matrix

HF_TOKEN = os.environ.get("HF_TOKEN")
ACCESSIBLE = ["roberta-base", "xlm-roberta-base", "deberta-v3-base", "mental-roberta-base"]
SCENARIOS = ["original", "oversampled", "undersampled"]
MAX_EPOCHS = 10
PATIENCE = 3

df, train_idx, val_idx, test_idx, info = load_and_split(ROOT / "dataset.xlsx")
texts = df["clean_text"].values
labels = df["label"].values

selected_path = RESULTS / "_selected_lr.json"
selected = json.loads(selected_path.read_text())

results_path = RESULTS / "transformer_seed_results.csv"
done_keys = set()
if results_path.exists():
    prev = pd.read_csv(results_path)
    done_keys = set(zip(prev.checkpoint, prev.scenario, prev.seed.astype(int)))

texts_val = texts[val_idx].tolist()
labels_val = labels[val_idx].tolist()
texts_test = texts[test_idx].tolist()
labels_test = labels[test_idx].tolist()

for name in ACCESSIBLE:
    hf_id = CHECKPOINTS[name]
    lr = selected[name]
    for scenario in SCENARIOS:
        for seed in SEEDS:
            key = (name, scenario, seed)
            if key in done_keys:
                print(f"SKIP (done) {key}")
                continue
            res_idx = resample_indices(train_idx, labels, scenario, seed)
            tr_texts = texts[res_idx].tolist()
            tr_labels = labels[res_idx].tolist()
            n_train_scenario = len(res_idx)
            class_counts = pd.Series(tr_labels).value_counts().sort_index().to_dict()

            t0 = time.time()
            val_metrics, test_metrics, history, elapsed, y_pred_test = train_and_evaluate(
                hf_id, tr_texts, tr_labels, texts_val, labels_val,
                texts_test=texts_test, y_test=labels_test,
                lr=lr, seed=seed, max_epochs=MAX_EPOCHS, patience=PATIENCE, hf_token=HF_TOKEN,
            )
            wall = time.time() - t0
            print(f"{name} | {scenario} | seed={seed}: test_macro_f1={test_metrics['macro_f1']:.4f} "
                  f"severe_f1={test_metrics['severe_f1']:.4f} ({len(history)} epochs, {wall:.1f}s)", flush=True)

            cm = confusion_matrix(labels_test, y_pred_test, labels=[0, 1, 2, 3])
            np.save(FIGURES / f"_cm_{name}_{scenario}_{seed}.npy", cm)

            row = dict(checkpoint=name, scenario=scenario, seed=seed, lr=lr,
                       n_train=n_train_scenario, class_counts=json.dumps(class_counts),
                       epochs_run=len(history), train_sec=elapsed, wall_sec=wall,
                       val_macro_f1=val_metrics["macro_f1"])
            row.update({f"test_{k}": v for k, v in test_metrics.items()})
            append_csv_row(results_path, row)

print("Final runs complete.")
