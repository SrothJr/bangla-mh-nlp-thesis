import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import load_and_split, ROOT, CHECKPOINTS, build_model_and_tokenizer, make_optimizer, MAX_LENGTH
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

HF_TOKEN = os.environ.get("HF_TOKEN")
df, train_idx, val_idx, test_idx, info = load_and_split(ROOT / "dataset.xlsx")
texts = df["clean_text"].values
labels = df["label"].values

hf_id = CHECKPOINTS["deberta-v3-base"]
tok, model = build_model_and_tokenizer(hf_id, 4, hf_token=HF_TOKEN)
device = torch.device("cuda")
model.to(device)
print("transformers model dtype:", next(model.parameters()).dtype)

class TextDS(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = list(texts), list(labels)
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = tok(self.texts[i], truncation=True, max_length=MAX_LENGTH)
        enc["labels"] = int(self.labels[i])
        return enc

collator = DataCollatorWithPadding(tok)
sub_idx = train_idx[:320]
ds = TextDS(texts[sub_idx], labels[sub_idx])
loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collator)

opt = make_optimizer(model, 2e-5)
total_steps = len(loader) * 3
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

model.train()
for step, batch in enumerate(loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(**batch)
    loss = out.loss
    nan_before = torch.isnan(loss).item()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad()
    print(f"step {step}: loss={loss.item():.4f} grad_norm={float(gn):.4f} lr={sched.get_last_lr()[0]:.2e}")
    if step >= 15:
        break
