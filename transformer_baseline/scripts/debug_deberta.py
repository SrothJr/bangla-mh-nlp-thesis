import sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import load_and_split, ROOT, CHECKPOINTS, build_model_and_tokenizer, make_optimizer, MAX_LENGTH
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding

HF_TOKEN = os.environ.get("HF_TOKEN")
df, train_idx, val_idx, test_idx, info = load_and_split(ROOT / "dataset.xlsx")
texts = df["clean_text"].values
labels = df["label"].values

hf_id = CHECKPOINTS["deberta-v3-base"]
tok, model = build_model_and_tokenizer(hf_id, 4, hf_token=HF_TOKEN)
device = torch.device("cuda")
model.to(device)

class TextDS(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = list(texts), list(labels)
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = tok(self.texts[i], truncation=True, max_length=MAX_LENGTH)
        enc["labels"] = int(self.labels[i])
        return enc

collator = DataCollatorWithPadding(tok)
sub_idx = train_idx[:64]
ds = TextDS(texts[sub_idx], labels[sub_idx])
loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collator)

opt = make_optimizer(model, 2e-5)
model.train()
for step, batch in enumerate(loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(**batch)
    print(f"step {step}: loss={out.loss.item()}, logits sample={out.logits[0].float().detach().cpu().numpy()}")
    if torch.isnan(out.loss):
        print("NaN LOSS DETECTED")
    out.loss.backward()
    total_grad_norm = 0.0
    n_nan_grads = 0
    for n, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                n_nan_grads += 1
    print(f"  params with NaN grad: {n_nan_grads}")
    opt.step()
    opt.zero_grad()
    if step >= 4:
        break

# Also test fp32 (no autocast) to isolate precision issue
print("\n--- fp32 test ---")
tok2, model2 = build_model_and_tokenizer(hf_id, 4, hf_token=HF_TOKEN)
model2.to(device)
opt2 = make_optimizer(model2, 2e-5)
model2.train()
for step, batch in enumerate(loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model2(**batch)
    print(f"step {step}: loss={out.loss.item()}")
    out.loss.backward()
    opt2.step()
    opt2.zero_grad()
    if step >= 4:
        break
