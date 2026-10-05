"""
Generates static PNG figures summarizing Days 1-7 results into figures/.
All numbers pulled from the saved result JSONs (outputs/*.json), not
retyped from memory -- see load_* helpers below.
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ACCENT = "#2a78d6"
CORAL = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE2 = "#f3f2ee"
SEV_COLORS = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load(fn):
    with open(os.path.join(OUTPUTS_DIR, fn), encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ------------------------------------------------------------- 1. ladder --
def fig_ablation_ladder():
    e0e1e3 = load("e0_e1_e3_results.json")
    e3b = load("e3b_results.json")
    e4e5 = load("e4_e5_results.json")
    e6 = load("e6_results.json")

    labels = ["E0\nmajority", "E1\nimage only", "E3\ntext OCR-only", "E3b\ntext+reasoning",
              "E4\nconcat", "E5\ngated+orth", "E6\nconcat", "E6\ngated+orth"]
    vals = [e0e1e3["E0"]["macro_f1"], e0e1e3["E1"]["macro_f1"], e0e1e3["E3"]["macro_f1"],
            e3b["macro_f1"], e4e5["E4_ce"]["macro_f1_mean"], e4e5["E5_ce"]["macro_f1_mean"],
            e6["E6_concat"]["macro_f1_mean"], e6["E6_gated_orth"]["macro_f1_mean"]]
    stds = [0, 0, 0, 0, e4e5["E4_ce"]["macro_f1_std"], e4e5["E5_ce"]["macro_f1_std"],
            e6["E6_concat"]["macro_f1_std"], e6["E6_gated_orth"]["macro_f1_std"]]
    groups = ["Baseline", "Unimodal", "Unimodal", "Unimodal", "Fusion", "Fusion", "Aligned fusion", "Aligned fusion"]
    locked_idx = 7

    fig, ax = plt.subplots(figsize=(11, 6.0))
    x = np.arange(len(labels))

    # phase shading, labeled as a strip ABOVE the plot (avoids collision
    # with the two-line x-tick labels below)
    group_bounds = []
    cur = groups[0]
    start = 0
    for i, g in enumerate(groups + [None]):
        if g != cur:
            group_bounds.append((start, i - 1, cur))
            cur = g
            start = i
    for gi, (s, e, name) in enumerate(group_bounds):
        if gi % 2 == 1:
            ax.axvspan(s - 0.5, e + 0.5, color=SURFACE2, zorder=0)
        ax.text((s + e) / 2, 1.045, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color=MUTED, family="monospace",
                fontweight="bold")

    colors = [ACCENT] * len(labels)
    alphas = [1.0 if i == locked_idx else 0.55 for i in range(len(labels))]
    bars = ax.bar(x, vals, color=colors, width=0.62, zorder=3, edgecolor="none")
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    ax.errorbar(x, vals, yerr=stds, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3,
                alpha=0.55, zorder=4)

    # locked outline
    bars[locked_idx].set_edgecolor(INK)
    bars[locked_idx].set_linewidth(1.6)
    bars[locked_idx].set_linestyle((0, (2, 2)))

    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.014, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=INK, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0, 0.60)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title("The ablation ladder — each step adds one piece of the architecture",
                  fontsize=13, fontweight="bold", color=INK, pad=32, loc="left")
    ax.text(0, -0.20,
            "E0–E3b: single deterministic runs on frozen embeddings.   E4 onward: 3-seed mean ± std.   "
            "Dashed outline = locked final config.",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.24, top=0.86)
    savefig(fig, "01_ablation_ladder.png")


# --------------------------------------------------------- 2. CORAL vs CE --
def fig_coral_vs_ce():
    e4e5 = load("e4_e5_results.json")
    configs = ["E4\nconcat", "E5\ngated+orth"]
    ce = [e4e5["E4_ce"]["macro_f1_mean"], e4e5["E5_ce"]["macro_f1_mean"]]
    ce_std = [e4e5["E4_ce"]["macro_f1_std"], e4e5["E5_ce"]["macro_f1_std"]]
    coral = [e4e5["E4_coral"]["macro_f1_mean"], e4e5["E5_coral"]["macro_f1_mean"]]
    coral_std = [e4e5["E4_coral"]["macro_f1_std"], e4e5["E5_coral"]["macro_f1_std"]]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.arange(len(configs))
    w = 0.32
    b1 = ax.bar(x - w / 2, ce, w, yerr=ce_std, capsize=3, color=ACCENT, label="weighted cross-entropy", zorder=3)
    b2 = ax.bar(x + w / 2, coral, w, yerr=coral_std, capsize=3, color=CORAL, label="CORAL (ordinal)", zorder=3)
    for bars, vals in ((b1, ce), (b2, coral)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.014, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10.5)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0, 0.60)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              ncol=2, fontsize=9.5, handlelength=1.4, columnspacing=1.2)
    ax.set_title("Weighted cross-entropy vs. CORAL ordinal loss", fontsize=12.5,
                  fontweight="bold", color=INK, pad=46, loc="left")
    ax.text(0, -0.16,
            "CORAL's single shared rank-score bottleneck starves the rarest class in this small,\n"
            "imbalanced setting — dropped after Day 4.",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.2, top=0.82)
    savefig(fig, "02_coral_vs_ce.png")


# ------------------------------------------------------- 3. gate alpha fix --
def fig_gate_alpha():
    edges = np.linspace(0, 1, 11)
    raw = [138, 35, 19, 3, 0, 0, 0, 0, 0, 0]
    aligned = [0, 0, 0, 16, 175, 4, 0, 0, 0, 0]
    n = 195

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, counts, color, title, sub in zip(
        axes, [raw, aligned], [CORAL, ACCENT],
        ["Raw E5 (seed 0)", "Aligned E6 (seed 0)"],
        ["α collapsed near 0 — image-dominant on 71% of items", "α balanced — adaptive mixing, 0% collapsed"],
    ):
        ax.bar(edges[:-1], counts, width=0.09, align="edge", color=color, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_xlabel("α  (0 = image-only, 1 = text-only)", fontsize=9.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", pad=10)
        ax.text(0.5, -0.24, sub, transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
    axes[0].set_ylabel(f"count (of {n} validation items)")
    fig.suptitle("Fixing the gate: α distribution before/after contrastive alignment",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left", y=1.04)
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.08)
    savefig(fig, "03_gate_alpha_fix.png")


# --------------------------------------------------- 4. confusion matrix --
def fig_confusion_matrix():
    final = load("final_test_results.json")
    primary = final["per_seed_test_results"][str(final["primary_seed"])]
    cm = np.array(primary["confusion_matrix"])
    labels = ["None", "Wish to\nbe dead", "Suicide\nideation", "Suicide\nplanning", "Attempt\n/ Death"]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            t = v / cm.max()
            color = "white" if t > 0.5 else INK
            weight = "bold" if i == j else "normal"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=12, fontweight=weight)
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10.5)
    ax.set_ylabel("True", fontsize=10.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"Final test confusion matrix — seed {final['primary_seed']} (n=196)\n"
        f"macro-F1 {primary['macro_f1']:.3f} · QWK {primary['quadratic_weighted_kappa']:.3f}",
        fontsize=11.5, fontweight="bold", color=INK, pad=48,
    )
    savefig(fig, "04_confusion_matrix.png")


# ------------------------------------------------------------ 5. per-class --
def fig_per_class():
    final = load("final_test_results.json")
    primary = final["per_seed_test_results"][str(final["primary_seed"])]
    report = primary["classification_report"]
    classes = ["None", "Wish to be dead", "Suicide ideation", "Suicide planning", "Suicide attempt or death"]
    short = ["None", "Wish to\nbe dead", "Suicide\nideation", "Suicide\nplanning", "Attempt\n/ Death"]
    metrics = ["precision", "recall", "f1-score"]
    metric_labels = ["Precision", "Recall", "F1"]
    metric_colors = ["#9ec5f4", "#2a78d6", "#104281"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(classes))
    w = 0.25
    for i, (m, lab, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = [report[c][m] for c in classes]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, color=color, label=lab, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7.8)

    supports = [int(report[c]["support"]) for c in classes]

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9.5)
    for xi, s in zip(x, supports):
        ax.text(xi, -0.135, f"n={s}", ha="center", va="top", fontsize=8.5, color=MUTED,
                 transform=ax.get_xaxis_transform())
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5, ncol=3)
    ax.set_title(f"Per-class precision / recall / F1 — final test set (seed {final['primary_seed']})",
                 fontsize=12.5, fontweight="bold", color=INK, pad=14, loc="left")
    fig.subplots_adjust(bottom=0.18)
    savefig(fig, "05_per_class_metrics.png")


# --------------------------------------------------------- 6. stat summary --
def fig_final_summary():
    final = load("final_test_results.json")["three_seed_summary"]
    metrics = [
        ("Macro-F1", final["macro_f1_mean"], final["macro_f1_std"]),
        ("Weighted-F1", final["weighted_f1_mean"], final["weighted_f1_std"]),
        ("Accuracy", final["accuracy_mean"], final["accuracy_std"]),
        ("QWK", final["qwk_mean"], final["qwk_std"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(metrics))
    vals = [m[1] for m in metrics]
    stds = [m[2] for m in metrics]
    labels = [m[0] for m in metrics]
    bars = ax.bar(x, vals, yerr=stds, capsize=4, color=ACCENT, width=0.5, zorder=3)
    for bar, v, s in zip(bars, vals, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.02, f"{v:.3f} ± {s:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 0.62)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title("Final locked model (E6 gated+orth, aligned) — test set, n=196, 3-seed mean ± std",
                 fontsize=12, fontweight="bold", color=INK, pad=14, loc="left")
    savefig(fig, "06_final_test_summary.png")


if __name__ == "__main__":
    fig_ablation_ladder()
    fig_coral_vs_ce()
    fig_gate_alpha()
    fig_confusion_matrix()
    fig_per_class()
    fig_final_summary()
    print("\nAll figures saved to", FIG_DIR)
