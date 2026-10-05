import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import load_and_split, ROOT, RESULTS, CHECKPOINTS, train_and_evaluate

HF_TOKEN = os.environ.get("HF_TOKEN")
ACCESSIBLE = ["roberta-base", "xlm-roberta-base", "deberta-v3-base", "mental-roberta-base"]

df, train_idx, val_idx, test_idx, info = load_and_split(ROOT / "dataset.xlsx")
texts_train = df.loc[train_idx, "clean_text"].tolist()
y_train = df.loc[train_idx, "label"].tolist()
texts_val = df.loc[val_idx, "clean_text"].tolist()
y_val = df.loc[val_idx, "label"].tolist()

rows = []
for name in ACCESSIBLE:
    hf_id = CHECKPOINTS[name]
    print(f"Calibrating {name} ...", flush=True)
    t0 = time.time()
    val_metrics, _, history, elapsed, _ = train_and_evaluate(
        hf_id, texts_train, y_train, texts_val, y_val,
        lr=2e-5, seed=42, max_epochs=2, patience=3, hf_token=HF_TOKEN, verbose=True,
    )
    wall = time.time() - t0
    sec_per_epoch = elapsed / len(history)
    print(f" -> {len(history)} epochs, {elapsed:.1f}s train, {sec_per_epoch:.1f}s/epoch, val_f1={val_metrics['macro_f1']:.4f}")
    rows.append(dict(checkpoint=name, epochs_run=len(history), train_sec=elapsed,
                      sec_per_epoch=sec_per_epoch, wall_sec=wall, val_macro_f1=val_metrics["macro_f1"]))

import pandas as pd
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "_timing_calibration.csv", index=False)
print(out)
