import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. LOAD DATA & LOCK TEST SET
# ---------------------------------------------------------
print("1. Loading datasets (Test set is locked away)...")
train_df = pd.read_csv("train_dataset.csv").dropna(subset=['posts', 'labels'])
test_df = pd.read_csv("test_dataset.csv").dropna(subset=['posts', 'labels'])
lexicon_df = pd.read_excel("Bangla_Depression_Lexicon_latest.xlsx").dropna(subset=['term_surface', 'category_name'])

X_train = train_df['posts'].astype(str).values
y_train = train_df['labels'].astype(int).values
X_test = test_df['posts'].astype(str).values
y_test = test_df['labels'].astype(int).values

# We enforce the exact same custom weights everywhere. No mismatched parameters.
CUSTOM_WEIGHTS = {1: 1.0, 2: 2.0, 3: 4.0, 4: 2.5}

# ---------------------------------------------------------
# 2. LEXICON EXTRACTORS (No cheating, no overrides)
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
            wc = max(len(text.split()), 1)
            cat_counts = [sum(1 for p in cat_phrases[cat] if f" {p} " in text_clean) for cat in categories]
            total = sum(cat_counts)
            row = cat_counts + [total, sum(1 for c in cat_counts if c > 0), 1.0 if total == 0 else 0.0]
            row += [(c / total) if total > 0 else 0.0 for c in cat_counts]
            row += [c / wc for c in cat_counts]
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

# ---------------------------------------------------------
# 3. PRECOMPUTE ALL FEATURES
# ---------------------------------------------------------
print("\n2. Precomputing TF-IDF Features...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

print("3. Precomputing Exact-Match Lexicon Features...")
lex_ext = LexiconFeatureExtractor().fit(X_train)
X_train_lex = lex_ext.transform(X_train)
X_test_lex = lex_ext.transform(X_test)

print("4. Precomputing Semantic Lexicon Features... (This may take a minute)")
sem_ext = SemanticLexiconFeatureExtractor(cat_phrases).fit(X_train)
X_train_sem = sem_ext.transform(X_train)
X_test_sem = sem_ext.transform(X_test)

# Build the 3 Ablation Feature Sets
FEATURE_SETS = {
    "1. TF-IDF ONLY (Baseline)": (
        X_train_tfidf, 
        X_test_tfidf
    ),
    "2. TF-IDF + EXACT LEXICON": (
        np.hstack([X_train_tfidf, X_train_lex]), 
        np.hstack([X_test_tfidf, X_test_lex])
    ),
    "3. TF-IDF + EXACT + SEMANTIC LEXICON": (
        np.hstack([X_train_tfidf, X_train_lex, X_train_sem]), 
        np.hstack([X_test_tfidf, X_test_lex, X_test_sem])
    )
}

# ---------------------------------------------------------
# 4. INDEPENDENT GRID SEARCH & EVALUATION
# ---------------------------------------------------------
print("\n5. Running Independent Grid Searches for each condition (No leakage, no C-reuse)...")

# We use 5-Fold Stratified Cross Validation purely on the training set to find the best C.
# The Test set is never touched until the final predict.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'C': [0.1, 0.5, 1.0, 5.0, 10.0]}

for label, (X_tr, X_te) in FEATURE_SETS.items():
    print(f"\n=======================================================")
    print(f"--- {label} ---")
    print(f"=======================================================")
    
    # 1. INDEPENDENT GRID SEARCH (Finds optimal C for THIS specific feature set)
    # Wrap in a Pipeline with StandardScaler(with_mean=False) to ensure TF-IDF and Lexicon
    # features are perfectly balanced for L2 regularization, preventing "feature drowning".
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('lr', LogisticRegression(class_weight=CUSTOM_WEIGHTS, max_iter=2000, random_state=42))
    ])
    
    # Notice the 'lr__C' syntax to access the C parameter inside the pipeline
    param_grid_scaled = {'lr__C': [0.1, 0.5, 1.0, 5.0, 10.0]}
    
    grid = GridSearchCV(pipeline, param_grid_scaled, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_tr, y_train)
    
    best_c = grid.best_params_['lr__C']
    print(f"-> Selected optimal C: {best_c} (CV Macro F1: {grid.best_score_:.4f})")
    
    # 2. FINAL TEST SET EVALUATION
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_te)
    
    print("\nClassification Report (Locked Test Set):")
    print(classification_report(y_test, y_pred))
    
    w_f1 = f1_score(y_test, y_pred, average='weighted')
    m_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Final Weighted F1: {w_f1:.4f} | Final Macro F1: {m_f1:.4f}")
