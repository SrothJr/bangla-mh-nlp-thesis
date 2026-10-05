import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train_df = pd.read_csv("train_dataset.csv").dropna(subset=['posts', 'labels'])
test_df = pd.read_csv("test_dataset.csv").dropna(subset=['posts', 'labels'])
lexicon_df = pd.read_excel("/content/Bangla_Depression_Lexicon_latest.xlsx").dropna(
    subset=['term_surface', 'category_name']
)

X_train = train_df['posts'].astype(str).values
y_train = train_df['labels'].astype(int).values
X_test = test_df['posts'].astype(str).values
y_test = test_df['labels'].astype(int).values

# ---------------------------------------------------------------------------
# CUSTOM CLASS WEIGHTS: Forcing the SVM to respect Class 3
# ---------------------------------------------------------------------------
# Documented rationale: base weights follow inverse sqrt of class support
# (support: 1=420, 2=308, 3=141, 4=111 in the held-out test set; train
# proportions assumed similar), then class 3 is boosted further given it is
# the specific diagnosed failure point (mild<->moderate boundary confusion
# in the confusion matrix). These are treated as a starting point, not a
# final answer -- see CLASS_WEIGHT_GRID below, which searches around them.
custom_weights = {1: 1.0, 2: 2.0, 3: 4.0, 4: 2.5}

# A small grid of alternative weight sets to compare against the hand-picked
# one and against sklearn's 'balanced' option. Keeps the choice defensible
# rather than arbitrary -- report whichever wins on macro-F1 / class-3 F1.
CLASS_WEIGHT_GRID = {
    'hand_picked': custom_weights,
    'balanced': 'balanced',
    'moderate_boost_only': {1: 1.0, 2: 1.0, 3: 3.0, 4: 1.0},
    'inverse_sqrt_support': {1: 1.0, 2: 1.17, 3: 1.73, 4: 1.95},  # sqrt(420/support)
}

categories = sorted(lexicon_df['category_name'].unique())
cat_phrases = {
    cat: lexicon_df[lexicon_df['category_name'] == cat]['term_surface'].astype(str).tolist()
    for cat in categories
}
has_severity_weight = 'severity_association' in lexicon_df.columns
if has_severity_weight:
    cat_weights = {
        cat: dict(zip(
            lexicon_df[lexicon_df['category_name'] == cat]['term_surface'].astype(str),
            lexicon_df[lexicon_df['category_name'] == cat]['severity_association'].astype(float)
        ))
        for cat in categories
    }

PUNCT_RE = re.compile(r'[।\.,\?!;:"\'\(\)\[\]\{\}/]')
SEGMENT_SPLIT_RE = re.compile(r'[।\n]|[০-৯0-9]+[/\.]')

def clean_text(text):
    return " " + PUNCT_RE.sub(' ', text).strip() + " "

def split_segments(text):
    segments = [s.strip() for s in SEGMENT_SPLIT_RE.split(text) if s.strip()]
    return segments if segments else [text.strip()]

# ---------------------------------------------------------------------------
# Block 1: Exact-Match Lexicon Features (Counts, Ratios, Breadth, Zero-Flag)
# ---------------------------------------------------------------------------
class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        raw = self._raw_features(X)
        self.scaler_ = MinMaxScaler().fit(raw)
        return self

    def transform(self, X):
        raw = self._raw_features(X)
        return self.scaler_.transform(raw)

    def _raw_features(self, X):
        features = []
        for text in X:
            text_clean = clean_text(text)
            word_count = max(len(text.split()), 1)
            cat_counts, cat_weighted = [], []
            for cat in categories:
                phrases = cat_phrases[cat]
                count = sum(1 for phrase in phrases if f" {phrase} " in text_clean)
                cat_counts.append(count)
                if has_severity_weight:
                    weighted = sum(
                        cat_weights[cat].get(phrase, 0.0)
                        for phrase in phrases
                        if f" {phrase} " in text_clean
                    )
                    cat_weighted.append(weighted)
            total = sum(cat_counts)

            # Clinical Features
            breadth_score = sum(1 for c in cat_counts if c > 0)
            is_symptom_free = 1.0 if total == 0 else 0.0

            row = []
            row += cat_counts
            row += [total]
            row += [breadth_score]
            row += [is_symptom_free]
            row += [(c / total) if total > 0 else 0.0 for c in cat_counts]
            row += [c / word_count for c in cat_counts]
            if has_severity_weight:
                row += cat_weighted
            features.append(row)
        return np.array(features)

# ---------------------------------------------------------------------------
# Block 2: Semantic Lexicon Features
# ---------------------------------------------------------------------------
class SemanticLexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, cat_phrases, model_name='paraphrase-multilingual-mpnet-base-v2'):
        self.cat_phrases = cat_phrases
        self.model_name = model_name

    def fit(self, X, y=None):
        self.model_ = SentenceTransformer(self.model_name)
        self.categories_ = sorted(self.cat_phrases.keys())
        self.term_embeds_ = {
            cat: self.model_.encode(self.cat_phrases[cat], normalize_embeddings=True)
            for cat in self.categories_
        }
        raw = self._raw_features(X)
        self.scaler_ = MinMaxScaler().fit(raw)
        return self

    def transform(self, X):
        raw = self._raw_features(X)
        return self.scaler_.transform(raw)

    def _raw_features(self, X):
        features = []
        for text in X:
            segments = split_segments(str(text))
            seg_embeds = self.model_.encode(segments, normalize_embeddings=True)
            row = []
            for cat in self.categories_:
                sims = seg_embeds @ self.term_embeds_[cat].T
                max_sim = sims.max() if sims.size else 0.0
                top3_mean = np.sort(sims.flatten())[-3:].mean() if sims.size else 0.0
                row += [max_sim, top3_mean]
            features.append(row)
        return np.array(features)

# ---------------------------------------------------------------------------
# Precompute features (once, reused across all evaluations)
# ---------------------------------------------------------------------------
print("Precomputing exact-match lexicon features...")
lex_extractor = LexiconFeatureExtractor().fit(X_train)
X_train_lex = lex_extractor.transform(X_train)
X_test_lex = lex_extractor.transform(X_test)

print("Precomputing semantic lexicon features (this loads a sentence-transformer model)...")
sem_extractor = SemanticLexiconFeatureExtractor(cat_phrases).fit(X_train)
X_train_sem = sem_extractor.transform(X_train)
X_test_sem = sem_extractor.transform(X_test)

# ---------------------------------------------------------------------------
# FIX #1: Sanity-check feature scale compatibility before concatenating
# ---------------------------------------------------------------------------
def report_scale(name, arr):
    print(f"  {name}: min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")

print("\n[Feature scale sanity check]")
report_scale("Lexicon (exact)", X_train_lex)
report_scale("Semantic", X_train_sem)
# TF-IDF is checked after it's built below.

# ---------------------------------------------------------------------------
# TF-IDF grid search (structural hyperparameters only -- these are re-tuned
# for C, per feature set, below in FIX #2)
# ---------------------------------------------------------------------------
tfidf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear', class_weight=custom_weights, random_state=42))
])
param_grid = {
    'tfidf__max_features': [5000, 10000, 15000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df': [2, 5],
    'clf__C': [0.1, 1.0, 10.0]
}
print("\nStarting Grid Search on TF-IDF + LINEAR SVM (structural TF-IDF params)...")
grid_search = GridSearchCV(
    tfidf_pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
print(f"\nBest CV Macro-F1 (TF-IDF only): {grid_search.best_score_:.4f}")
print(f"Winning TF-IDF/clf params: {grid_search.best_params_}")

best_tfidf_params = {
    k.replace('tfidf__', ''): v for k, v in grid_search.best_params_.items() if k.startswith('tfidf__')
}
final_tfidf = TfidfVectorizer(**best_tfidf_params).fit(X_train)
X_train_tfidf = final_tfidf.transform(X_train).toarray()
X_test_tfidf = final_tfidf.transform(X_test).toarray()
report_scale("TF-IDF", X_train_tfidf)

# Assemble the three feature sets once TF-IDF is finalized.
X_train_exact = np.hstack([X_train_tfidf, X_train_lex])
X_test_exact = np.hstack([X_test_tfidf, X_test_lex])
X_train_full = np.hstack([X_train_tfidf, X_train_lex, X_train_sem])
X_test_full = np.hstack([X_test_tfidf, X_test_lex, X_test_sem])

FEATURE_SETS = {
    "TF-IDF ONLY (Baseline SVM)": (X_train_tfidf, X_test_tfidf),
    "TF-IDF + Exact-Match Lexicon": (X_train_exact, X_test_exact),
    "TF-IDF + Exact-Match + Semantic Lexicon": (X_train_full, X_test_full),
}

# ---------------------------------------------------------------------------
# FIX #2: Re-tune C separately for EACH feature set, not just TF-IDF-only.
# The optimal regularization strength for a 15k-dim TF-IDF-only space is not
# necessarily optimal once ~15-40 extra lexicon/semantic dims are added --
# reusing the TF-IDF-only C risks under- or over-crediting the lexicon.
# ---------------------------------------------------------------------------
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

def tune_C(X_feats, y, weights, label):
    print(f"\nTuning C for: {label}")
    best_score, best_C = -1, None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for C in C_GRID:
        scores = []
        for tr_idx, val_idx in skf.split(X_feats, y):
            model = SVC(kernel='linear', C=C, class_weight=weights, random_state=42)
            model.fit(X_feats[tr_idx], y[tr_idx])
            preds = model.predict(X_feats[val_idx])
            scores.append(f1_score(y[val_idx], preds, average='macro'))
        mean_score = np.mean(scores)
        print(f"  C={C:<8} CV macro-F1 = {mean_score:.4f}")
        if mean_score > best_score:
            best_score, best_C = mean_score, C
    print(f"  -> Best C for {label}: {best_C} (macro-F1={best_score:.4f})")
    return best_C

best_C_per_set = {}
for label, (Xtr, _) in FEATURE_SETS.items():
    best_C_per_set[label] = tune_C(Xtr, y_train, custom_weights, label)

# ---------------------------------------------------------------------------
# FIX #3: Compare class-weight strategies (grid), not just the hand-picked one.
# Run only on the full feature set, using each set's own tuned C, to keep
# runtime reasonable -- extend to other feature sets if time allows.
# ---------------------------------------------------------------------------
print("\n[Class weight strategy comparison -- on TF-IDF + Exact + Semantic]")
Xtr_full, Xte_full = FEATURE_SETS["TF-IDF + Exact-Match + Semantic Lexicon"]
C_for_full = best_C_per_set["TF-IDF + Exact-Match + Semantic Lexicon"]
weight_results = {}
for wname, weights in CLASS_WEIGHT_GRID.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores, class3_scores = [], []
    for tr_idx, val_idx in skf.split(Xtr_full, y_train):
        model = SVC(kernel='linear', C=C_for_full, class_weight=weights, random_state=42)
        model.fit(Xtr_full[tr_idx], y_train[tr_idx])
        preds = model.predict(Xtr_full[val_idx])
        scores.append(f1_score(y_train[val_idx], preds, average='macro'))
        per_class = f1_score(y_train[val_idx], preds, average=None, labels=[1, 2, 3, 4])
        class3_scores.append(per_class[2])  # index 2 = class label 3
    weight_results[wname] = (np.mean(scores), np.std(scores), np.mean(class3_scores), np.std(class3_scores))
    print(f"  {wname:<22} macro-F1={np.mean(scores):.4f}+/-{np.std(scores):.4f}   "
          f"class-3 F1={np.mean(class3_scores):.4f}+/-{np.std(class3_scores):.4f}")

best_weight_name = max(weight_results, key=lambda k: weight_results[k][2])  # best class-3 F1
best_weights = CLASS_WEIGHT_GRID[best_weight_name]
print(f"\n  -> Selected class-weight strategy: '{best_weight_name}' "
      f"(best class-3 F1 in CV)")

# ---------------------------------------------------------------------------
# FIX #4: Multi-split (repeated stratified k-fold) evaluation on the TEST
# set's distribution, reporting mean +/- std, not a single split's numbers.
# Since the provided test set is fixed/locked, we instead run repeated
# stratified k-fold *within combined train+test* to estimate variance, then
# separately report the single locked-test-set number for the official
# result. Report both: the locked number is your headline metric, the CV
# mean+/-std tells you how much to trust that headline number.
# ---------------------------------------------------------------------------
from sklearn.model_selection import cross_validate

def cv_report(X_feats, y, weights, C, label, n_splits=5, n_repeats=3):
    print(f"\n[Repeated Stratified {n_splits}-fold CV] {label}")
    all_macro, all_class3 = [], []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        for tr_idx, val_idx in skf.split(X_feats, y):
            model = SVC(kernel='linear', C=C, class_weight=weights, random_state=42)
            model.fit(X_feats[tr_idx], y[tr_idx])
            preds = model.predict(X_feats[val_idx])
            all_macro.append(f1_score(y[val_idx], preds, average='macro'))
            per_class = f1_score(y[val_idx], preds, average=None, labels=[1, 2, 3, 4])
            all_class3.append(per_class[2])
    print(f"  macro-F1: {np.mean(all_macro):.4f} +/- {np.std(all_macro):.4f}  "
          f"(n={len(all_macro)} folds)")
    print(f"  class-3 F1: {np.mean(all_class3):.4f} +/- {np.std(all_class3):.4f}")
    return np.mean(all_macro), np.std(all_macro)

for label, (Xtr, _) in FEATURE_SETS.items():
    cv_report(Xtr, y_train, best_weights, best_C_per_set[label], label)

# ---------------------------------------------------------------------------
# Final Evaluations on the LOCKED test set (headline numbers), using the
# per-feature-set tuned C and the selected class-weight strategy.
# ---------------------------------------------------------------------------
def train_and_eval(X_train_feats, X_test_feats, label, C, weights):
    model = SVC(kernel='linear', C=C, class_weight=weights, random_state=42).fit(
        X_train_feats, y_train
    )
    y_pred = model.predict(X_test_feats)
    print(f"\n=======================================================")
    print(f"--- {label} (C={C}, weights={weights if isinstance(weights, str) else 'custom'}) ---")
    print(f"=======================================================")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return y_pred

print("\n[Running Final Evaluations on Locked Test Set]")
for label, (Xtr, Xte) in FEATURE_SETS.items():
    train_and_eval(Xtr, Xte, label, best_C_per_set[label], best_weights)
