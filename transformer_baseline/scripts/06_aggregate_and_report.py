"""Aggregation, comparison tables, rankings, confusion matrices, error analysis, and the
data-driven interpretation. Reads results/transformer_seed_results.csv (must be complete for
all checkpoint x scenario x seed combos) and produces all deliverable artifacts."""
import sys, os, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import (
    load_and_split, ROOT, RESULTS, FIGURES, CHECKPOINTS, SEEDS, LABEL_NAMES,
)

df, train_idx, val_idx, test_idx, info = load_and_split(ROOT / "dataset.xlsx")
seed_results = pd.read_csv(RESULTS / "transformer_seed_results.csv")
audit = pd.read_csv(RESULTS / "transformer_compatibility_audit.csv")

ACCESSIBLE = ["roberta-base", "xlm-roberta-base", "deberta-v3-base", "mental-roberta-base"]
SCENARIOS = ["original", "oversampled", "undersampled"]

metric_cols = [c for c in seed_results.columns if c.startswith("test_")]

# ---------------------------------------------------------------------
# Table D: per-run rows (matches schema for cross-family merge)
# ---------------------------------------------------------------------
table_d = seed_results.copy()
table_d.to_csv(RESULTS / "table_D_transformers.csv", index=False)

# ---------------------------------------------------------------------
# Mean +/- SD aggregation per checkpoint x scenario
# ---------------------------------------------------------------------
agg_rows = []
for name in ACCESSIBLE:
    for scenario in SCENARIOS:
        sub = seed_results[(seed_results.checkpoint == name) & (seed_results.scenario == scenario)]
        if len(sub) == 0:
            continue
        row = dict(checkpoint=name, scenario=scenario, n_seeds=len(sub))
        for mc in metric_cols:
            row[f"{mc}_mean"] = sub[mc].mean()
            row[f"{mc}_std"] = sub[mc].std()
        agg_rows.append(row)
agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv(RESULTS / "transformer_aggregated_mean_std.csv", index=False)

# ---------------------------------------------------------------------
# Cross-scenario comparison (deltas vs original)
# ---------------------------------------------------------------------
cmp_metrics = ["test_macro_f1", "test_accuracy", "test_balanced_accuracy",
               "test_severe_f1", "test_severe_recall", "test_severity_mae"]
cmp_rows = []
for name in ACCESSIBLE:
    base = agg_df[(agg_df.checkpoint == name) & (agg_df.scenario == "original")]
    if len(base) == 0:
        continue
    base = base.iloc[0]
    for scenario in ["oversampled", "undersampled"]:
        cur = agg_df[(agg_df.checkpoint == name) & (agg_df.scenario == scenario)]
        if len(cur) == 0:
            continue
        cur = cur.iloc[0]
        row = dict(checkpoint=name, scenario=scenario)
        for m in cmp_metrics:
            row[f"{m}_original"] = base[f"{m}_mean"]
            row[f"{m}_{scenario}"] = cur[f"{m}_mean"]
            row[f"delta_{m}"] = cur[f"{m}_mean"] - base[f"{m}_mean"]
        cmp_rows.append(row)
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(RESULTS / "transformer_cross_scenario_comparison.csv", index=False)

# ---------------------------------------------------------------------
# Rankings (on "original" scenario, mean across seeds)
# ---------------------------------------------------------------------
orig = agg_df[agg_df.scenario == "original"].copy()
rankings = {}
for m, asc in [("test_macro_f1_mean", False), ("test_accuracy_mean", False),
               ("test_severe_f1_mean", False), ("test_severe_recall_mean", False),
               ("test_severity_mae_mean", True)]:
    r = orig[["checkpoint", m]].sort_values(m, ascending=asc).reset_index(drop=True)
    rankings[m] = r
rank_out = []
for m, r in rankings.items():
    r = r.copy()
    r["metric"] = m
    r["rank"] = np.arange(1, len(r) + 1)
    rank_out.append(r.rename(columns={m: "value"}))
pd.concat(rank_out, ignore_index=True).to_csv(RESULTS / "transformer_rankings.csv", index=False)

# ---------------------------------------------------------------------
# Confusion matrices (Original scenario) per checkpoint -- averaged raw counts across seeds
# ---------------------------------------------------------------------
for name in ACCESSIBLE:
    cms = []
    for seed in SEEDS:
        p = FIGURES / f"_cm_{name}_original_{seed}.npy"
        if p.exists():
            cms.append(np.load(p))
    if not cms:
        continue
    cm_sum = np.sum(cms, axis=0)
    cm_norm = cm_sum / cm_sum.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title, fmt in [
        (axes[0], cm_sum, f"{name} — Raw counts (summed over {len(cms)} seeds)", "d"),
        (axes[1], cm_norm, f"{name} — Row-normalized", ".2f"),
    ]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_xticklabels(LABEL_NAMES, rotation=45)
        ax.set_yticks(range(4)); ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=9)
        for i in range(4):
            for j in range(4):
                val = mat[i, j]
                txt = f"{val:{fmt}}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val > mat.max() / 2 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / f"confusion_matrix_{name}_original.png", dpi=150)
    plt.close(fig)

print("Aggregation complete. Files written to results/.")
