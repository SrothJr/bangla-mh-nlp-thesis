"""
Comprehensive, CPU-only figure suite covering every performance/evaluation angle actually
computed in the Section 4 notebook: dataset EDA, tokenizer audit, CV hyperparameter selection,
per-checkpoint x per-scenario metrics (all metrics in the schema), per-class breakdowns,
cross-scenario deltas, seed stability, and rankings.

Does NOT import pipeline.py / torch -- pandas/numpy/matplotlib/sklearn only, so this cannot
touch the GPU regardless of what else is running on it.

Writes everything into results/figures/ (flat, numbered, descriptive filenames).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["Minimum", "Mild", "Moderate", "Severe"]
LABEL_REMAP = {1: 0, 2: 1, 3: 2, 4: 3}
ACCESSIBLE = ["roberta-base", "xlm-roberta-base", "deberta-v3-base", "mental-roberta-base"]
CHECKPOINT_DISPLAY = {
    "roberta-base": "RoBERTa", "xlm-roberta-base": "XLM-R",
    "deberta-v3-base": "DeBERTa-v3", "mental-roberta-base": "MentalRoBERTa",
}
CHECKPOINT_COLORS = {
    "roberta-base": "#4C72B0", "xlm-roberta-base": "#55A868",
    "deberta-v3-base": "#DD8452", "mental-roberta-base": "#8172B2",
}
SCENARIO_DISPLAY = {"original": "Original", "oversampled": "Oversampled", "undersampled": "Undersampled"}
SCENARIO_COLORS = {"original": "#4C72B0", "oversampled": "#55A868", "undersampled": "#C44E52"}
SCENARIOS = ["original", "oversampled", "undersampled"]
CLASS_COLORS = {"Minimum": "#4C72B0", "Mild": "#55A868", "Moderate": "#DD8452", "Severe": "#C44E52"}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 12, "font.family": "sans-serif",
    "axes.titlesize": 13.5, "axes.titleweight": "bold", "axes.labelsize": 12,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9, "axes.grid": True,
    "grid.color": "#d9d9d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.bbox": "tight",
})


def savefig(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def annotate_bars(ax, bars, fmt="{:.3f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)


# ===========================================================================
# Load data
# ===========================================================================
seed_results = pd.read_csv(RESULTS / "transformer_seed_results.csv")
agg = pd.read_csv(RESULTS / "transformer_aggregated_mean_std.csv")
audit = pd.read_csv(RESULTS / "transformer_compatibility_audit.csv")
kfold_log = pd.read_csv(RESULTS / "transformer_kfold_cv_log.csv")
cmp_df = pd.read_csv(RESULTS / "transformer_cross_scenario_comparison.csv")
rankings = pd.read_csv(RESULTS / "transformer_rankings.csv")

# Dataset EDA (pure pandas/sklearn, mirrors pipeline.load_and_split without importing torch)
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
labels_arr = df["label"].values
train_val_idx, test_idx = train_test_split(idx, test_size=0.10, stratify=labels_arr, random_state=42)
rel_val = 0.10 / 0.90
train_idx, val_idx = train_test_split(train_val_idx, test_size=rel_val,
                                       stratify=labels_arr[train_val_idx], random_state=42)
df["word_len"] = df["clean_text"].str.split().apply(len)

n = 0  # figure counter

# ===========================================================================
# GROUP A: Dataset EDA
# ===========================================================================
n += 1
counts = df["label"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar([LABEL_NAMES[i] for i in counts.index], counts.values,
              color=[CLASS_COLORS[LABEL_NAMES[i]] for i in counts.index], edgecolor="black", linewidth=0.6)
annotate_bars(ax, bars, "{:.0f}")
ax.set_xlabel("Depression Severity Class"); ax.set_ylabel("Number of Posts")
ax.set_title("Class Distribution — Full Dataset (n = %d)" % len(df))
savefig(fig, f"{n:02d}_class_distribution_full_dataset.png")

n += 1
split_counts = pd.DataFrame({
    "Train": df.loc[train_idx, "label"].value_counts().sort_index(),
    "Validation": df.loc[val_idx, "label"].value_counts().sort_index(),
    "Test": df.loc[test_idx, "label"].value_counts().sort_index(),
}).fillna(0)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
x = np.arange(4); width = 0.25
for i, split_name in enumerate(["Train", "Validation", "Test"]):
    ax.bar(x + (i - 1) * width, split_counts[split_name].values, width, label=split_name,
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(LABEL_NAMES)
ax.set_xlabel("Depression Severity Class"); ax.set_ylabel("Number of Posts")
ax.set_title("Class Distribution by Split (Train / Validation / Test)")
ax.legend()
savefig(fig, f"{n:02d}_class_distribution_by_split.png")

n += 1
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.hist(df["word_len"], bins=40, color="#4C72B0", edgecolor="black", linewidth=0.4, alpha=0.85)
ax.axvline(df["word_len"].median(), color="#C44E52", linestyle="--", linewidth=1.6,
           label=f"Median = {df['word_len'].median():.0f} words")
ax.axvline(df["word_len"].mean(), color="#333333", linestyle=":", linewidth=1.6,
           label=f"Mean = {df['word_len'].mean():.1f} words")
ax.set_xlabel("Post Length (words, after cleaning)"); ax.set_ylabel("Frequency")
ax.set_title("Distribution of Post Length — Full Dataset")
ax.legend()
savefig(fig, f"{n:02d}_post_length_histogram.png")

n += 1
fig, ax = plt.subplots(figsize=(7.5, 5))
data_by_class = [df.loc[df.label == i, "word_len"].values for i in range(4)]
bp = ax.boxplot(data_by_class, tick_labels=LABEL_NAMES, patch_artist=True, showfliers=False, widths=0.55,
                 medianprops=dict(color="black", linewidth=1.4))
for patch, name in zip(bp["boxes"], LABEL_NAMES):
    patch.set_facecolor(CLASS_COLORS[name]); patch.set_alpha(0.75); patch.set_edgecolor("black")
ax.set_xlabel("Depression Severity Class"); ax.set_ylabel("Post Length (words)")
ax.set_title("Post Length Distribution by Severity Class")
savefig(fig, f"{n:02d}_post_length_by_class.png")

n += 1
scen_counts = {}
for scen in SCENARIOS:
    row = seed_results[seed_results.scenario == scen].iloc[0]
    import json
    cc = json.loads(row["class_counts"])
    scen_counts[scen] = [cc.get(str(i), 0) for i in range(4)]
fig, ax = plt.subplots(figsize=(8.5, 5.5))
x = np.arange(4); width = 0.25
for i, scen in enumerate(SCENARIOS):
    ax.bar(x + (i - 1) * width, scen_counts[scen], width, label=SCENARIO_DISPLAY[scen],
           color=SCENARIO_COLORS[scen], edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(LABEL_NAMES)
ax.set_xlabel("Depression Severity Class"); ax.set_ylabel("Number of Training Posts")
ax.set_title("Training-Set Class Counts by Resampling Scenario")
ax.legend()
savefig(fig, f"{n:02d}_scenario_resampling_class_counts.png")

# ===========================================================================
# GROUP B: Tokenizer compatibility audit
# ===========================================================================
n += 1
audit_ok = audit[audit.status == "OK"].copy()
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = [CHECKPOINT_COLORS.get(c, "#888888") for c in audit_ok.checkpoint]
bars = ax.bar([CHECKPOINT_DISPLAY.get(c, c) for c in audit_ok.checkpoint], audit_ok.tokens_per_word,
              color=colors, edgecolor="black", linewidth=0.6)
annotate_bars(ax, bars, "{:.2f}")
ax.set_xlabel("Transformer Checkpoint"); ax.set_ylabel("Tokens per Word")
ax.set_title("Tokenizer Fragmentation Audit (Bangla Training Sample, n≈200 posts)")
savefig(fig, f"{n:02d}_tokenizer_fragmentation_comparison.png")

n += 1
fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar([CHECKPOINT_DISPLAY.get(c, c) for c in audit_ok.checkpoint], audit_ok.unk_rate,
              color=colors, edgecolor="black", linewidth=0.6)
annotate_bars(ax, bars, "{:.4f}")
ax.set_xlabel("Transformer Checkpoint"); ax.set_ylabel("UNK Token Rate")
ax.set_title("Tokenizer UNK-Token Rate on Bangla Training Sample")
savefig(fig, f"{n:02d}_tokenizer_unk_rate_comparison.png")

# ===========================================================================
# GROUP C: CV hyperparameter (learning-rate) selection
# ===========================================================================
n += 1
cv_summary = kfold_log.groupby(["checkpoint", "lr"])["val_macro_f1"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(8.5, 6))
for ckpt in ACCESSIBLE:
    sub = cv_summary[cv_summary.checkpoint == ckpt].sort_values("lr")
    ax.errorbar(sub.lr, sub["mean"], yerr=sub["std"], marker="o", capsize=4, linewidth=1.8,
                label=CHECKPOINT_DISPLAY[ckpt], color=CHECKPOINT_COLORS[ckpt])
ax.set_xscale("log")
ax.set_xlabel("Learning Rate (log scale)"); ax.set_ylabel("5-Fold CV Mean Validation Macro-F1")
ax.set_title("Learning-Rate Selection via 5-Fold Stratified CV (Training Split Only)")
ax.legend()
savefig(fig, f"{n:02d}_cv_learning_rate_selection.png")

# ===========================================================================
# GROUP D: Overall metrics, grouped bar (checkpoint x scenario), one figure per metric
# ===========================================================================
metric_specs = [
    ("test_macro_f1", "Macro-F1 Score", "Macro-F1 by Checkpoint and Scenario"),
    ("test_accuracy", "Accuracy", "Accuracy by Checkpoint and Scenario"),
    ("test_balanced_accuracy", "Balanced Accuracy", "Balanced Accuracy by Checkpoint and Scenario"),
    ("test_weighted_f1", "Weighted F1 Score", "Weighted F1 by Checkpoint and Scenario"),
    ("test_macro_precision", "Macro Precision", "Macro Precision by Checkpoint and Scenario"),
    ("test_macro_recall", "Macro Recall", "Macro Recall by Checkpoint and Scenario"),
    ("test_severe_f1", "Severe-Class F1", "Severe-Class F1 by Checkpoint and Scenario"),
    ("test_severe_recall", "Severe-Class Recall", "Severe-Class Recall by Checkpoint and Scenario"),
    ("test_severity_mae", "Severity MAE (lower is better)", "Severity MAE by Checkpoint and Scenario"),
    ("test_within_one_level_accuracy", "Within-One-Level Accuracy", "Within-One-Level Accuracy by Checkpoint and Scenario"),
]

for metric, ylabel, title in metric_specs:
    n += 1
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ACCESSIBLE)); width = 0.25
    for i, scen in enumerate(SCENARIOS):
        sub = agg[agg.scenario == scen].set_index("checkpoint").reindex(ACCESSIBLE)
        means = sub[f"{metric}_mean"].values
        stds = sub[f"{metric}_std"].values
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=SCENARIO_DISPLAY[scen],
               color=SCENARIO_COLORS[scen], edgecolor="black", linewidth=0.5,
               error_kw=dict(elinewidth=1.0, ecolor="#333333"))
    ax.set_xticks(x); ax.set_xticklabels([CHECKPOINT_DISPLAY[c] for c in ACCESSIBLE])
    ax.set_xlabel("Transformer Checkpoint"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.legend(title="Scenario")
    ax.text(0.5, -0.14, "Error bars = ±1 SD across 5 seeds (42, 123, 2024, 3407, 9999)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")
    savefig(fig, f"{n:02d}_{metric.replace('test_', '')}_by_checkpoint_scenario.png")

# ===========================================================================
# GROUP E: Per-class metrics (Original scenario)
# ===========================================================================
orig = seed_results[seed_results.scenario == "original"]
for metric_kind in ["precision", "recall", "f1"]:
    n += 1
    fig, ax = plt.subplots(figsize=(9.5, 6))
    x = np.arange(4); width = 0.2
    for i, ckpt in enumerate(ACCESSIBLE):
        sub = orig[orig.checkpoint == ckpt]
        means = [sub[f"test_{cls.lower()}_{metric_kind}"].mean() for cls in LABEL_NAMES]
        stds = [sub[f"test_{cls.lower()}_{metric_kind}"].std() for cls in LABEL_NAMES]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3, label=CHECKPOINT_DISPLAY[ckpt],
               color=CHECKPOINT_COLORS[ckpt], edgecolor="black", linewidth=0.4,
               error_kw=dict(elinewidth=0.9, ecolor="#333333"))
    ax.set_xticks(x); ax.set_xticklabels(LABEL_NAMES)
    ax.set_xlabel("Depression Severity Class"); ax.set_ylabel(f"Per-Class {metric_kind.capitalize()}")
    ax.set_title(f"Per-Class {metric_kind.capitalize()} by Checkpoint (Original Scenario)", fontsize=13)
    ax.legend(title="Checkpoint", fontsize=9)
    savefig(fig, f"{n:02d}_per_class_{metric_kind}_original.png")

# ===========================================================================
# GROUP F: Cross-scenario deltas (vs Original)
# ===========================================================================
delta_specs = [
    ("delta_test_macro_f1", "Δ Macro-F1 vs Original", "Cross-Scenario Change in Macro-F1"),
    ("delta_test_severe_f1", "Δ Severe-Class F1 vs Original", "Cross-Scenario Change in Severe-Class F1"),
    ("delta_test_severe_recall", "Δ Severe-Class Recall vs Original", "Cross-Scenario Change in Severe-Class Recall"),
    ("delta_test_severity_mae", "Δ Severity MAE vs Original", "Cross-Scenario Change in Severity MAE"),
]
for metric, ylabel, title in delta_specs:
    n += 1
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(ACCESSIBLE)); width = 0.3
    for i, scen in enumerate(["oversampled", "undersampled"]):
        sub = cmp_df[cmp_df.scenario == scen].set_index("checkpoint").reindex(ACCESSIBLE)
        vals = sub[metric].values
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=SCENARIO_DISPLAY[scen],
                      color=SCENARIO_COLORS[scen], edgecolor="black", linewidth=0.5)
        annotate_bars(ax, bars, "{:+.3f}")
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels([CHECKPOINT_DISPLAY[c] for c in ACCESSIBLE])
    ax.set_xlabel("Transformer Checkpoint"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.legend(title="Scenario")
    savefig(fig, f"{n:02d}_{metric.replace('delta_test_', 'delta_')}_cross_scenario.png")

# ===========================================================================
# GROUP G: Seed stability (variance across seeds)
# ===========================================================================
n += 1
fig, ax = plt.subplots(figsize=(9, 6))
data = [seed_results[seed_results.checkpoint == c]["test_macro_f1"].values for c in ACCESSIBLE]
bp = ax.boxplot(data, tick_labels=[CHECKPOINT_DISPLAY[c] for c in ACCESSIBLE], patch_artist=True,
                 widths=0.5, medianprops=dict(color="black", linewidth=1.4))
for patch, ckpt in zip(bp["boxes"], ACCESSIBLE):
    patch.set_facecolor(CHECKPOINT_COLORS[ckpt]); patch.set_alpha(0.75); patch.set_edgecolor("black")
for i, ckpt in enumerate(ACCESSIBLE):
    y = seed_results[seed_results.checkpoint == ckpt]["test_macro_f1"].values
    x = np.random.normal(i + 1, 0.04, size=len(y))
    ax.scatter(x, y, color="black", s=14, alpha=0.6, zorder=3)
ax.set_xlabel("Transformer Checkpoint"); ax.set_ylabel("Macro-F1 Score")
ax.set_title("Seed-to-Seed Stability of Macro-F1 (All Scenarios Pooled, 15 Runs/Checkpoint)")
savefig(fig, f"{n:02d}_seed_stability_macro_f1.png")

# ===========================================================================
# GROUP H: Rankings
# ===========================================================================
rank_specs = [
    ("test_macro_f1_mean", "Macro-F1", False),
    ("test_severe_f1_mean", "Severe-Class F1", False),
    ("test_severe_recall_mean", "Severe-Class Recall", False),
    ("test_severity_mae_mean", "Severity MAE (lower better)", True),
]
for metric, label, ascending in rank_specs:
    n += 1
    sub = rankings[rankings.metric == metric].sort_values("rank")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = [CHECKPOINT_COLORS.get(c, "#888888") for c in sub.checkpoint]
    bars = ax.barh([CHECKPOINT_DISPLAY.get(c, c) for c in sub.checkpoint], sub.value,
                   color=colors, edgecolor="black", linewidth=0.6)
    ax.invert_yaxis()
    for b, v in zip(bars, sub.value):
        ax.annotate(f"{v:.3f}", xy=(v, b.get_y() + b.get_height() / 2), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=10)
    ax.set_xlabel(label); ax.set_ylabel("Transformer Checkpoint")
    ax.set_title(f"Checkpoint Ranking by {label} (Original Scenario)", fontsize=13)
    savefig(fig, f"{n:02d}_ranking_by_{metric.replace('test_', '').replace('_mean','')}.png")

print(f"\nDone. {n} comprehensive figures written to {FIG}")
