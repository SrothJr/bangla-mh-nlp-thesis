import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import load_and_split, ROOT, RESULTS, CHECKPOINTS, tokenizer_audit

HF_TOKEN = os.environ.get("HF_TOKEN")

df, train_idx, val_idx, test_idx, info = load_and_split(ROOT / "dataset.xlsx")
sample = df.loc[train_idx, "clean_text"].sample(n=200, random_state=42).tolist()

rows = []
for name, hf_id in CHECKPOINTS.items():
    print(f"Auditing {name} ({hf_id}) ...")
    r = tokenizer_audit(name, hf_id, sample, hf_token=HF_TOKEN)
    print(" ->", r.get("status"), r.get("tokens_per_word"), r.get("unk_rate"), r.get("error"))
    rows.append(r)

import pandas as pd
pd.DataFrame(rows).to_csv(RESULTS / "transformer_compatibility_audit.csv", index=False)
print("Saved", RESULTS / "transformer_compatibility_audit.csv")
