import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import load_and_split, ROOT

df, train_idx, val_idx, test_idx, info = load_and_split(ROOT / "dataset.xlsx")
print(info)
assert info["n_train"] == 3917 and info["n_val"] == 490 and info["n_test"] == 490, "SPLIT MISMATCH"
print("Split sizes match expected 3917/490/490")

import numpy as np
for name, ids in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    vc = df.loc[ids, "label"].value_counts().sort_index()
    print(name, vc.to_dict())
