"""
Standalone, CPU-only figure generation for the thesis (Section 4 + dataset-level plots).
Deliberately does NOT import pipeline.py / torch, so this cannot touch the GPU under any
circumstance -- it only needs pandas / numpy / matplotlib / sklearn.

Produces 10 individually-saved PNGs (300 DPI) into results/figures/paper/.
"""
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["Minimum", "Mild", "Moderate", "Severe"]
LABEL_REMAP = {1: 0, 2: 1, 3: 2, 4: 3}

# ---------------------------------------------------------------------------
# Consistent professional style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "savefig.bbox": "tight",
})

PALETTE = {
    "Minimum": "#4C72B0",
    "Mild": "#55A868",
    "Moderate": "#DD8452",
    "Severe": "#C44E52",
}
CHECKPOINT_COLORS = {
    "roberta-base": "#4C72B0",
    "xlm-roberta-base": "#55A868",
    "deberta-v3-base": "#DD8452",
    "mental-roberta-base": "#8172B2",
}
CHECKPOINT_DISPLAY = {
    "roberta-base": "RoBERTa",
    "xlm-roberta-base": "XLM-R",
    "deberta-v3-base": "DeBERTa-v3",
    "mental-roberta-base": "MentalRoBERTa",
}
ACCESSIBLE = ["roberta-base", "xlm-roberta-base", "deberta-v3-base", "mental-roberta-base"]


def annotate_bars(ax, bars, fmt="{:.3f}", offset=0.01):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5)


# ---------------------------------------------------------------------------
# Reproduce the fixed data split (pure pandas/sklearn, no torch)
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text):
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_SYMBOL_RE.sub("", t)
    t = REPEATED_CHAR_RE.sub(r"\1\1\1", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


from sklearn.model_selection import train_test_split

df = pd.read_excel(ROOT / "dataset.xlsx")
df["label"] = df["labels"].map(LABEL_REMAP)
df["clean_text"] = df["posts"].apply(clean_text)
df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)

idx = np.arange(len(df))
labels = df["label"].values
train_val_idx, test_idx = train_test_split(idx, test_size=0.10, stratify=labels, random_state=42)
rel_val = 0.10 / 0.90
train_idx, val_idx = train_test_split(train_val_idx, test_size=rel_val,
                                       stratify=labels[train_val_idx], random_state=42)

df["word_len"] = df["clean_text"].str.split().apply(len)

print(f"Loaded {len(df)} rows. Train/Val/Test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

# ===========================================================================
# 1. Class Distribution (full dataset)
# ===========================================================================
counts = df["label"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar([LABEL_NAMES[i] for i in counts.index], counts.values,
              color=[PALETTE[LABEL_NAMES[i]] for i in counts.index], edgecolor="black", linewidth=0.6)
annotate_bars(ax, bars, fmt="{:.0f}")
ax.set_xlabel("Depression Severity Class")
ax.set_ylabel("Number of Posts")
ax.set_title("Class Distribution — Full Dataset")
total = counts.sum()
for b, c in zip(bars, counts.values):
    ax.annotate(f"{c/total*100:.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()/2),
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")
fig.savefig(OUT / "01_class_distribution.png")
plt.close(fig)

# ===========================================================================
# 2. Length Histograms (word-count length, full dataset)
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.hist(df["word_len"], bins=40, color="#4C72B0", edgecolor="black", linewidth=0.4, alpha=0.85)
ax.axvline(df["word_len"].median(), color="#C44E52", linestyle="--", linewidth=1.6,
           label=f"Median = {df['word_len'].median():.0f} words")
ax.axvline(df["word_len"].mean(), color="#333333", linestyle=":", linewidth=1.6,
           label=f"Mean = {df['word_len'].mean():.1f} words")
ax.set_xlabel("Post Length (words, after cleaning)")
ax.set_ylabel("Frequency (number of posts)")
ax.set_title("Distribution of Post Length — Full Dataset")
ax.legend()
fig.savefig(OUT / "02_length_histograms.png")
plt.close(fig)

# ===========================================================================
# 3. Length by Class (boxplot)
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 5))
data_by_class = [df.loc[df.label == i, "word_len"].values for i in range(4)]
bp = ax.boxplot(data_by_class, tick_labels=LABEL_NAMES, patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", linewidth=1.4), widths=0.55)
for patch, name in zip(bp["boxes"], LABEL_NAMES):
    patch.set_facecolor(PALETTE[name])
    patch.set_alpha(0.75)
    patch.set_edgecolor("black")
ax.set_xlabel("Depression Severity Class")
ax.set_ylabel("Post Length (words, after cleaning)")
ax.set_title("Post Length Distribution by Severity Class")
fig.savefig(OUT / "03_length_by_class.png")
plt.close(fig)

# ===========================================================================
# 4-8: Metric comparisons across checkpoints (Original scenario, mean +/- SD over 5 seeds)
# ===========================================================================
agg = pd.read_csv(RESULTS / "transformer_aggregated_mean_std.csv")
orig = agg[agg.scenario == "original"].set_index("checkpoint").reindex(ACCESSIBLE)

metric_specs = [
    ("test_macro_f1", "Macro-F1 Comparison Across Checkpoints (Original Scenario)", "Macro-F1 Score",
     "04_macro_f1_comparison.png"),
    ("test_accuracy", "Accuracy Comparison Across Checkpoints (Original Scenario)", "Accuracy",
     "05_accuracy_comparison.png"),
    ("test_balanced_accuracy", "Balanced Accuracy Comparison Across Checkpoints (Original Scenario)",
     "Balanced Accuracy", "06_balanced_accuracy_comparison.png"),
    ("test_severe_f1", "Severe-Class F1 Comparison Across Checkpoints (Original Scenario)",
     "Severe-Class F1 Score", "07_severe_f1_comparison.png"),
    ("test_severe_recall", "Severe-Class Recall Comparison Across Checkpoints (Original Scenario)",
     "Severe-Class Recall", "08_severe_recall_comparison.png"),
]

for metric, title, ylabel, fname in metric_specs:
    means = orig[f"{metric}_mean"].values
    stds = orig[f"{metric}_std"].values
    labels_x = [CHECKPOINT_DISPLAY[c] for c in ACCESSIBLE]
    colors = [CHECKPOINT_COLORS[c] for c in ACCESSIBLE]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(labels_x, means, yerr=stds, capsize=6, color=colors, edgecolor="black",
                  linewidth=0.6, error_kw=dict(elinewidth=1.3, ecolor="#333333"))
    annotate_bars(ax, bars, fmt="{:.3f}", offset=0.02)
    ax.set_xlabel("Transformer Checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, min(1.05, max(means + stds) * 1.2))
    ax.text(0.5, -0.16, "Error bars = ±1 SD across 5 random seeds (42, 123, 2024, 3407, 9999)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")
    fig.savefig(OUT / fname)
    plt.close(fig)

# ===========================================================================
# 9. Train Class Distribution Recap (post-split, pre-resampling training set)
# ===========================================================================
train_counts = df.loc[train_idx, "label"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar([LABEL_NAMES[i] for i in train_counts.index], train_counts.values,
              color=[PALETTE[LABEL_NAMES[i]] for i in train_counts.index], edgecolor="black", linewidth=0.6)
annotate_bars(ax, bars, fmt="{:.0f}")
ax.set_xlabel("Depression Severity Class")
ax.set_ylabel("Number of Training Posts")
ax.set_title(f"Training-Set Class Distribution Recap (n = {len(train_idx)}, Original / Natural Distribution)")
fig.savefig(OUT / "09_train_class_distribution_recap.png")
plt.close(fig)

# ===========================================================================
# 10. All Families — Macro-F1 (Original scenario) cross-family comparison
#     Classical ML / Deep Learning / Hybrid DL results are NOT available in this
#     project (produced in separate notebooks on different hardware). These bars
#     are explicit PLACEHOLDERS -- replace `family_macro_f1` values below with the
#     real numbers from Sections 1-3 before using this figure in the thesis.
# ===========================================================================
best_transformer_row = orig.sort_values("test_macro_f1_mean", ascending=False).iloc[0]
best_transformer_name = orig["test_macro_f1_mean"].idxmax()

# TODO(user): replace None with the actual mean Macro-F1 (Original scenario) from
# Sections 1-3. Leave as None to keep the bar visibly flagged as a placeholder.
family_macro_f1 = {
    "Classical ML": None,      # <-- INSERT REAL VALUE FROM SECTION 1
    "Deep Learning": None,     # <-- INSERT REAL VALUE FROM SECTION 2
    "Hybrid DL": None,         # <-- INSERT REAL VALUE FROM SECTION 3
    f"Transformer\n(best: {CHECKPOINT_DISPLAY[best_transformer_name]})": best_transformer_row["test_macro_f1_mean"],
}
family_macro_f1_std = {
    "Classical ML": None,
    "Deep Learning": None,
    "Hybrid DL": None,
    f"Transformer\n(best: {CHECKPOINT_DISPLAY[best_transformer_name]})": best_transformer_row["test_macro_f1_std"],
}

fig, ax = plt.subplots(figsize=(8, 5.5))
names = list(family_macro_f1.keys())
values = [v if v is not None else 0.0 for v in family_macro_f1.values()]
errs = [family_macro_f1_std[n] if family_macro_f1_std[n] is not None else 0.0 for n in names]
is_placeholder = [v is None for v in family_macro_f1.values()]

bar_colors = ["#BEBEBE" if p else "#55A868" for p in is_placeholder]
hatches = ["////" if p else None for p in is_placeholder]
bars = ax.bar(names, values, yerr=errs, capsize=6, color=bar_colors, edgecolor="black",
              linewidth=0.6, error_kw=dict(elinewidth=1.3, ecolor="#333333"))
for bar, hatch, p in zip(bars, hatches, is_placeholder):
    if hatch:
        bar.set_hatch(hatch)
    if p:
        ax.annotate("PLACEHOLDER\n(insert real value)", xy=(bar.get_x() + bar.get_width()/2, 0.05),
                    ha="center", va="bottom", fontsize=8.5, color="#7a0000", fontstyle="italic")
    else:
        ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10)

ax.set_xlabel("Model Family")
ax.set_ylabel("Macro-F1 Score")
ax.set_title("Cross-Family Macro-F1 Comparison — Original Scenario")
ax.set_ylim(0, 1.05)
fig.savefig(OUT / "10_all_families_macro_f1_original.png")
plt.close(fig)

print("\nAll 10 figures written to:", OUT)
for f in sorted(OUT.glob("*.png")):
    print(" -", f.name)
