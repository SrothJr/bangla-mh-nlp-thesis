import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import copy, random
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 0. IDENTICAL SEEDING TO BASE SCRIPT
# ---------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------
# 1. IDENTICAL DATA SPLIT (same random_state, same stratify)
# ---------------------------------------------------------
print("1. Loading Data & Creating Strict Validation Split...")
raw_train_df = pd.read_csv("../lexicon_mining/data/processed/train_dataset.csv").dropna(subset=['posts', 'labels'])
test_df = pd.read_csv("../lexicon_mining/data/processed/test_dataset.csv").dropna(subset=['posts', 'labels'])
lexicon_df = pd.read_excel("../Bangla_Depression_Lexicon_latest.xlsx").dropna(subset=['term_surface', 'category_name'])

train_df, val_df = train_test_split(
    raw_train_df, test_size=0.10, stratify=raw_train_df['labels'], random_state=SEED
)

X_train_text = train_df['posts'].astype(str).values
y_train = (train_df['labels'].astype(int) - 1).values
X_val_text = val_df['posts'].astype(str).values
y_val = (val_df['labels'].astype(int) - 1).values
X_test_text = test_df['posts'].astype(str).values
y_test = (test_df['labels'].astype(int) - 1).values

# ---------------------------------------------------------
# 2. LEXICON FEATURES (fit on train only — unchanged logic)
# ---------------------------------------------------------
categories = sorted(lexicon_df['category_name'].unique())
cat_phrases = {cat: lexicon_df[lexicon_df['category_name'] == cat]['term_surface'].astype(str).tolist() for cat in categories}

PUNCT_RE = re.compile(r'[।\.,\?!;:"\'\(\)\[\]\{\}/]')
SEGMENT_SPLIT_RE = re.compile(r'[।\n]|[০-৯0-9]+[/\.]')

def clean_text(text): return " " + PUNCT_RE.sub(' ', text).strip() + " "
def split_segments(text):
    segments = [s.strip() for s in SEGMENT_SPLIT_RE.split(text) if s.strip()]
    return segments if segments else [text.strip()]

class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.scaler_ = MinMaxScaler().fit(self._raw_features(X))
        return self
    def transform(self, X): return self.scaler_.transform(self._raw_features(X))
    def _raw_features(self, X):
        features = []
        for text in X:
            text_clean = clean_text(text)
            word_count = max(len(text.split()), 1)
            cat_counts = [sum(1 for p in cat_phrases[cat] if f" {p} " in text_clean) for cat in categories]
            total = sum(cat_counts)
            row = cat_counts + [total, sum(1 for c in cat_counts if c > 0), 1.0 if total == 0 else 0.0]
            row += [(c / total) if total > 0 else 0.0 for c in cat_counts]
            row += [c / word_count for c in cat_counts]
            features.append(row)
        return np.array(features)

class SemanticLexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, cat_phrases, model_name='paraphrase-multilingual-mpnet-base-v2'):
        self.cat_phrases = cat_phrases
        self.model_name = model_name
    def fit(self, X, y=None):
        self.model_ = SentenceTransformer(self.model_name)
        self.categories_ = sorted(self.cat_phrases.keys())
        self.term_embeds_ = {cat: self.model_.encode(self.cat_phrases[cat], normalize_embeddings=True) for cat in self.categories_}
        self.scaler_ = MinMaxScaler().fit(self._raw_features(X))
        return self
    def transform(self, X): return self.scaler_.transform(self._raw_features(X))
    def _raw_features(self, X):
        features = []
        for text in X:
            segments = split_segments(str(text))
            seg_embeds = self.model_.encode(segments, normalize_embeddings=True)
            row = []
            for cat in self.categories_:
                sims = seg_embeds @ self.term_embeds_[cat].T
                row += [sims.max() if sims.size else 0.0, np.sort(sims.flatten())[-3:].mean() if sims.size else 0.0]
            features.append(row)
        return np.array(features)

print("2. Precomputing Lexicon Features (exact + semantic)...")
lex_extractor = LexiconFeatureExtractor().fit(X_train_text)
X_train_lex = lex_extractor.transform(X_train_text)
X_val_lex   = lex_extractor.transform(X_val_text)
X_test_lex  = lex_extractor.transform(X_test_text)

sem_extractor = SemanticLexiconFeatureExtractor(cat_phrases).fit(X_train_text)
X_train_sem = sem_extractor.transform(X_train_text)
X_val_sem   = sem_extractor.transform(X_val_text)
X_test_sem  = sem_extractor.transform(X_test_text)

lexicon_vec_train = np.hstack([X_train_lex, X_train_sem]).astype(np.float32)
lexicon_vec_val   = np.hstack([X_val_lex, X_val_sem]).astype(np.float32)
lexicon_vec_test  = np.hstack([X_test_lex, X_test_sem]).astype(np.float32)
LEXICON_DIM = lexicon_vec_train.shape[1]

# ---------------------------------------------------------
# 3. MODEL — IDENTICAL TO BASE BERT EXCEPT ONE CONCAT LINE
# ---------------------------------------------------------
print(f"3. Building STRICT ABLATION model (Lexicon Dim: {LEXICON_DIM})...")
MODEL_NAME = "sagorsarker/bangla-bert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME)   

class DepressionDataset(Dataset):
    def __init__(self, texts, lexicon_vecs, labels):
        self.texts = texts
        self.lexicon_vecs = lexicon_vecs
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(str(self.texts[idx]), truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'lexicon': torch.tensor(self.lexicon_vecs[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BertLexiconAblation(nn.Module):
    def __init__(self, bert_model, lexicon_dim, num_classes=4, use_lexicon=True):
        super().__init__()
        self.bert = bert_model
        self.use_lexicon = use_lexicon
        hidden_dim = self.bert.config.hidden_size
        in_dim = hidden_dim + lexicon_dim if use_lexicon else hidden_dim
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_dim, num_classes)  

    def forward(self, input_ids, attention_mask, lexicon):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        if self.use_lexicon:
            fused = torch.cat([cls, lexicon], dim=1)
        else:
            fused = cls
        return self.classifier(fused)

train_ds = DepressionDataset(X_train_text, lexicon_vec_train, y_train)
val_ds   = DepressionDataset(X_val_text, lexicon_vec_val, y_val)
test_ds  = DepressionDataset(X_test_text, lexicon_vec_test, y_test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)
test_loader  = DataLoader(test_ds, batch_size=16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BertLexiconAblation(bert, LEXICON_DIM, use_lexicon=True).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------
# 4. TRAINING — SAME EPOCH BUDGET, SAME PATIENCE, VAL-ONLY SELECTION
# ---------------------------------------------------------
print(f"\n4. Training Ablation Model (Max 10 Epochs) on {device.upper()}...")
EPOCHS = 10
PATIENCE = 2
best_macro_f1 = 0
best_model_weights = None
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['lexicon'].to(device))
        loss = criterion(logits, batch['label'].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['lexicon'].to(device))
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            val_true.extend(batch['label'].numpy())

    val_macro_f1 = f1_score(val_true, val_preds, average='macro')
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {total_loss/len(train_loader):.4f} | VAL Macro F1: {val_macro_f1:.4f}")

    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered! (Patience {PATIENCE} reached)")
            break

# ---------------------------------------------------------
# 5. FINAL BLIND TEST EVAL (touched once)
# ---------------------------------------------------------
print("\n5. Loading Best Epoch and Evaluating on LOCKED BLIND TEST SET...")
model.load_state_dict(best_model_weights)
model.eval()

final_preds, final_true = [], []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['lexicon'].to(device))
        final_preds.extend(logits.argmax(dim=1).cpu().numpy())
        final_true.extend(batch['label'].numpy())

final_preds = np.array(final_preds) + 1
final_true = np.array(final_true) + 1

print(f"\n=======================================================")
print(f"--- STRICT ABLATION: BASE BERT + RAW LEXICON CONCAT ---")
print(f"=======================================================")
print(classification_report(final_true, final_preds))
