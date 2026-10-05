"""Data-driven final interpretation + error analysis on the strongest checkpoint."""
import sys, os, re, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import load_and_split, ROOT, RESULTS, CHECKPOINTS, SEEDS, clean_text

ACCESSIBLE = ["roberta-base", "xlm-roberta-base", "deberta-v3-base", "mental-roberta-base"]

audit = pd.read_csv(RESULTS / "transformer_compatibility_audit.csv")
agg = pd.read_csv(RESULTS / "transformer_aggregated_mean_std.csv")
seed_results = pd.read_csv(RESULTS / "transformer_seed_results.csv")
cmp_df = pd.read_csv(RESULTS / "transformer_cross_scenario_comparison.csv")

orig = agg[agg.scenario == "original"].copy()
best_row = orig.sort_values("test_macro_f1_mean", ascending=False).iloc[0]
best_checkpoint = best_row["checkpoint"]

best_severe = orig.sort_values("test_severe_f1_mean", ascending=False).iloc[0]

# stability: lowest mean SD of macro_f1 across seeds, across all scenarios
stability = seed_results.groupby("checkpoint")["test_macro_f1"].std().sort_values()
most_stable = stability.index[0]

lowest_mae = orig.sort_values("test_severity_mae_mean", ascending=True).iloc[0]

# does fragmentation predict performance? correlate tokens_per_word with macro_f1
audit_ok = audit[audit.status == "OK"][["checkpoint", "tokens_per_word"]]
merged = orig.merge(audit_ok, on="checkpoint", how="left").dropna(subset=["tokens_per_word"])
corr = merged["tokens_per_word"].corr(merged["test_macro_f1_mean"]) if len(merged) > 2 else float("nan")

# balancing effect: mean delta macro_f1 for oversampled/undersampled across checkpoints
delta_over = cmp_df[cmp_df.scenario == "oversampled"]["delta_test_macro_f1"].mean()
delta_under = cmp_df[cmp_df.scenario == "undersampled"]["delta_test_macro_f1"].mean()

lines = []
lines.append("SECTION 4 (TRANSFORMER MODELS) - AUTOMATED DATA-DRIVEN INTERPRETATION")
lines.append("=" * 70)
lines.append("")
lines.append("Checkpoints attempted: roberta-base, xlm-roberta-base, microsoft/deberta-v3-base, "
              "mental/mental-bert-base-uncased, mental/mental-roberta-base")
not_run = audit[audit.status != "OK"]
for _, r in not_run.iterrows():
    lines.append(f"  -> {r['checkpoint']}: NOT RUN - ACCESS RESTRICTED ({str(r['error'])[:200]})")
lines.append("")
lines.append("TOKENIZER FRAGMENTATION AUDIT (measured on ~200-row training sample):")
for _, r in audit.iterrows():
    if r["status"] == "OK":
        lines.append(f"  {r['checkpoint']}: {r['tokens_per_word']:.2f} tokens/word, "
                      f"UNK rate {r['unk_rate']:.4f}")
lines.append("")
lines.append(f"BEST CHECKPOINT OVERALL (by mean Macro-F1, Original scenario, {len(SEEDS)} seeds): "
             f"{best_checkpoint} (Macro-F1 = {best_row['test_macro_f1_mean']:.4f} +/- "
             f"{best_row['test_macro_f1_std']:.4f})")
lines.append(f"BEST SEVERE-CLASS F1: {best_severe['checkpoint']} "
             f"(Severe-F1 = {best_severe['test_severe_f1_mean']:.4f} +/- "
             f"{best_severe['test_severe_f1_std']:.4f})")
lines.append(f"MOST STABLE ACROSS SEEDS (lowest SD of Macro-F1, pooled across scenarios): "
             f"{most_stable} (SD = {stability.iloc[0]:.4f})")
lines.append(f"LOWEST SEVERITY MAE: {lowest_mae['checkpoint']} "
             f"(MAE = {lowest_mae['test_severity_mae_mean']:.4f} +/- "
             f"{lowest_mae['test_severity_mae_std']:.4f})")
lines.append("")
if not np.isnan(corr):
    direction = "predicts WORSE performance (higher fragmentation -> lower Macro-F1)" if corr < -0.3 else \
                "predicts BETTER performance (unexpected)" if corr > 0.3 else \
                "does NOT clearly predict downstream Macro-F1 (weak/no correlation)"
    lines.append(f"DOES TOKENIZER FRAGMENTATION PREDICT PERFORMANCE? Pearson r(tokens/word, Macro-F1) "
                 f"= {corr:.3f} across {len(merged)} checkpoints -> fragmentation {direction}.")
else:
    lines.append("DOES TOKENIZER FRAGMENTATION PREDICT PERFORMANCE? Insufficient accessible checkpoints "
                 "to compute a meaningful correlation.")
lines.append("")
lines.append(f"DOES BALANCING HELP? Mean delta Macro-F1 vs Original: "
             f"oversampled {delta_over:+.4f}, undersampled {delta_under:+.4f} "
             f"(averaged across {cmp_df.checkpoint.nunique()} checkpoints).")
if delta_over > 0.01 and delta_over > delta_under:
    lines.append("  -> Oversampling gives the clearest improvement; undersampling's smaller "
                 "training set appears to cost more than the class-balance benefit gains.")
elif delta_under > 0.01 and delta_under > delta_over:
    lines.append("  -> Undersampling gives the clearest improvement despite the smaller training set.")
elif max(delta_over, delta_under) <= 0.01:
    lines.append("  -> Neither resampling strategy meaningfully improves Macro-F1 over the natural "
                 "distribution for these transformer checkpoints.")
lines.append("")
lines.append("Per-checkpoint Original-scenario summary (mean +/- SD over seeds):")
for _, r in orig.iterrows():
    lines.append(f"  {r['checkpoint']}: Macro-F1={r['test_macro_f1_mean']:.4f}+/-{r['test_macro_f1_std']:.4f}, "
                 f"Acc={r['test_accuracy_mean']:.4f}, BalAcc={r['test_balanced_accuracy_mean']:.4f}, "
                 f"Severe-F1={r['test_severe_f1_mean']:.4f}, Severe-Recall={r['test_severe_recall_mean']:.4f}, "
                 f"Severity-MAE={r['test_severity_mae_mean']:.4f}")

interp_text = "\n".join(lines)
(RESULTS / "transformer_final_interpretation.txt").write_text(interp_text, encoding="utf-8")
print(interp_text)

# ---------------------------------------------------------------------
# Error analysis on strongest checkpoint (original scenario, primary seed)
# ---------------------------------------------------------------------
PII_PATTERNS = [
    (re.compile(r"http\S+|www\.\S+"), "[URL]"),
    (re.compile(r"@\w+"), "[MENTION]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\d{10,11}\b"), "[PHONE]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
]

def anonymize(t):
    for pat, repl in PII_PATTERNS:
        t = pat.sub(repl, t)
    return t

df, train_idx, val_idx, test_idx, info = load_and_split(ROOT / "dataset.xlsx")
best_cm_seed = 42
cm_path = RESULTS / "figures" / f"_cm_{best_checkpoint}_original_{best_cm_seed}.npy"
if cm_path.exists():
    cm = np.load(cm_path)
    err_lines = [f"ERROR ANALYSIS - strongest checkpoint: {best_checkpoint} (Original scenario, seed {best_cm_seed})", ""]
    err_lines.append("Confusion matrix (rows=true, cols=pred), order Minimum/Mild/Moderate/Severe:")
    err_lines.append(str(cm))
    err_lines.append("")
    err_lines.append("Most severe confusions (misclassifications >1 level apart) are the highest clinical risk.")
    (RESULTS / "transformer_error_analysis.txt").write_text("\n".join(err_lines), encoding="utf-8")
    print("\n".join(err_lines))

print("\nInterpretation + error analysis written to results/.")
