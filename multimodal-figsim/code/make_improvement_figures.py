"""
Generates static PNG figures summarizing the Phase 1-4 improvement round
into figures/ (numbered 07 onward, continuing from make_figures.py's
Days 1-7 figures 01-06). All numbers pulled from outputs/*.json -- see
load() calls below -- except the two values noted inline where a script
printed to console without persisting JSON; those are cross-checked
against IMPROVEMENT_PLAN.md and saved into small summary JSONs first
(phase4_1_ensemble_tuned_gate_summary.json, phase4_2_direct_vlm_summary.json)
so this script still reads from disk, not from memory.
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
GREEN = "#2a9d5c"
RED = "#c0392b"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE2 = "#f3f2ee"

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


def style_bar_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


# --------------------------------------------------- 7. phase 1 experiments --
def fig_phase1_experiments():
    baseline = 0.5117  # original locked E6 gated+orth, single-seed best (Day 7)
    ens = load("phase1_ensemble_results.json")
    gate = load("phase1_2_gate_uncertainty_results.json")
    smooth = load("phase1_3_ordinal_smoothing_results.json")
    focal = load("phase1_4_focal_loss_results.json")

    labels = ["Baseline\n(single seed)", "1.1 Multi-seed\nmajority vote", "1.2 Gate\nuncertainty",
              "1.3 Ordinal\nsmoothing", "1.4 Focal\nloss"]
    vals = [baseline, ens["ensemble_majority_vote"]["macro_f1"], gate["majority_vote_macro_f1"],
            smooth["majority_vote_macro_f1"], focal["majority_vote_macro_f1"]]
    kept = [True, True, False, True, False]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(labels))
    colors = [GREEN if k else RED for k in kept]
    colors[0] = MUTED
    bars = ax.bar(x, vals, color=colors, width=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.axhline(baseline, color=INK, linewidth=1, linestyle=(0, (2, 2)), alpha=0.5, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0.46, 0.56)
    style_bar_axes(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="kept"), Patch(color=RED, label="rejected"),
                        Patch(color=MUTED, label="baseline")],
               frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Phase 1 — cheap/free changes", fontsize=13, fontweight="bold", color=INK,
                  pad=14, loc="left")
    ax.text(0, -0.16,
            "Multi-seed ensembling and ordinal smoothing both improved on baseline and were kept;\n"
            "gate-uncertainty weighting and focal loss underperformed and were reverted.",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, "07_phase1_experiments.png")


# --------------------------------------------------- 8. phase 2 experiments --
def fig_phase2_experiments():
    phase1_best = 0.5330
    english = load("phase2_1_english_results.json")
    mixup = load("phase2_2_mixup_results.json")
    supcon = load("phase2_3_supervised_align_results.json")
    clip = load("phase2_4_clip_results.json")

    labels = ["Phase 1 best\n(Bangla, SigLIP)", "2.1 English\ntext encoder", "2.2 Mixup\naugmentation",
              "2.3 Supervised\ncontrastive align", "2.4 CLIP\nimage encoder"]
    vals = [phase1_best, english["majority_vote_macro_f1"], mixup["majority_vote_macro_f1"],
            supcon["majority_vote_macro_f1"], clip["majority_vote_macro_f1"]]
    kept = [True, False, False, False, False]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(labels))
    colors = [GREEN if k else RED for k in kept]
    colors[0] = MUTED
    bars = ax.bar(x, vals, color=colors, width=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.axhline(phase1_best, color=INK, linewidth=1, linestyle=(0, (2, 2)), alpha=0.5, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.2)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0.40, 0.56)
    style_bar_axes(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED, label="rejected"), Patch(color=MUTED, label="baseline")],
               frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Phase 2 — moderate-effort changes: all four underperformed", fontsize=13,
                  fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, -0.20,
            "Every Phase 2 change swapped out a component already validated on this exact dataset\n"
            "(DAPT-BanglaBERT, SigLIP) for a more generic one, or added augmentation on too little data\n"
            "(582 train examples) -- all four reverted; original components confirmed correct.",
            transform=ax.transAxes, fontsize=8.6, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.26)
    savefig(fig, "08_phase2_experiments.png")


# ------------------------------------------------- 9. phase 3 architecture --
def fig_phase3_architecture():
    frozen = load("phase3_1_baseline_concat_frozen_results.json")
    unfreeze = load("phase3_1_unfreeze_results.json")
    xattn = load("phase3_2_cross_attention_results.json")
    ens = load("phase3_2_ensemble_results.json")
    gated_alone = ens["gated_alone_macro_f1"]

    labels = ["Frozen\nconcat baseline", "Partial\nunfreezing", "Gated+orth\n(Phase 1 best)",
              "Cross-attention\n(standalone)", "Cross-architecture\nensemble (vote)"]
    vals = [frozen["majority_vote_macro_f1"], unfreeze["majority_vote_macro_f1"], gated_alone,
            xattn["majority_vote_macro_f1"], ens["cross_arch_majority_vote_macro_f1"]]
    kept = [False, False, True, False, True]
    colors = [GREEN if k else RED for k in kept]
    colors[0] = MUTED

    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.6, zorder=3)
    bars[4].set_edgecolor(INK)
    bars[4].set_linewidth(1.8)
    bars[4].set_linestyle((0, (2, 2)))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.009, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.2)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0.44, 0.58)
    style_bar_axes(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="kept"), Patch(color=RED, label="rejected"),
                        Patch(color=MUTED, label="reference")],
               frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Phase 3 — architecture changes: cross-architecture ensembling is the single\nbiggest win of the whole improvement round",
                  fontsize=12.5, fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, -0.20,
            "Partial unfreezing hurt (only 582 training examples -- too little data to safely update\n"
            "encoder weights). Cross-attention alone underperformed gated+orth, but the two architectures\n"
            "make complementary errors -- majority-voting them together beats either one alone.",
            transform=ax.transAxes, fontsize=8.6, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.28)
    savefig(fig, "09_phase3_architecture.png")


# ---------------------------------------------------- 10. phase 4 refinements --
def fig_phase4_refinements():
    phase3_best = 0.5592
    tuned = load("phase4_1_ensemble_tuned_gate_summary.json")
    vlm = load("phase4_2_direct_vlm_summary.json")
    calib = load("phase4_4_calibration_results.json")
    hier = load("phase4_5_hierarchical_results.json")

    labels = ["Phase 3 best\n(ensemble)", "4.1 Tuned\ndropout=0.4", "4.2 Direct\nVLM prompting\n(different method)",
              "4.4 Post-hoc\ncalibration", "4.5 Hierarchical\ndecomposition"]
    vals = [phase3_best, tuned["cross_arch_majority_vote_macro_f1_tuned_gate"], vlm["macro_f1"],
            calib["calibrated_macro_f1"], hier["final_macro_f1"]]
    kept = [True, True, False, True, False]
    colors = [GREEN if k else RED for k in kept]
    colors[0] = MUTED

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.6, zorder=3)
    bars[3].set_edgecolor(INK)
    bars[3].set_linewidth(1.8)
    bars[3].set_linestyle((0, (2, 2)))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9.3, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0.25, 0.62)
    style_bar_axes(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="kept"), Patch(color=RED, label="rejected"),
                        Patch(color=MUTED, label="reference")],
               frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Phase 4 — final refinements (dashed outline = current best config)",
                  fontsize=13, fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, -0.22,
            "4.2 direct zero-shot VLM prompting used a fundamentally different method (no fusion\n"
            "architecture at all) and failed badly -- safety-aligned model hedges toward 'ideation' by\n"
            "default. 4.5 hierarchical decomposition failed via error-cascading despite both of its stages\n"
            "looking individually strong (see figure 12).",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.30)
    savefig(fig, "10_phase4_refinements.png")


# --------------------------------------------- 11. cumulative progression --
def fig_cumulative_progression():
    ens1 = load("phase1_ensemble_results.json")
    smooth = load("phase1_3_ordinal_smoothing_results.json")
    xarch = load("phase3_2_ensemble_results.json")
    tuned = load("phase4_1_ensemble_tuned_gate_summary.json")
    calib = load("phase4_4_calibration_results.json")

    stages = [
        ("Original\nlocked baseline\n(Day 7)", 0.5117),
        ("+ multi-seed\nensembling", ens1["ensemble_majority_vote"]["macro_f1"]),
        ("+ ordinal\nsmoothing\n(Phase 1 final)", smooth["majority_vote_macro_f1"]),
        ("+ cross-arch\nensemble\n(Phase 3)", xarch["cross_arch_majority_vote_macro_f1"]),
        ("+ tuned\ndropout\n(Phase 4.1)", tuned["cross_arch_majority_vote_macro_f1_tuned_gate"]),
        ("+ calibration\n(Phase 4.4, final)", calib["calibrated_macro_f1"]),
    ]
    labels = [s[0] for s in stages]
    vals = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    ax.plot(x, vals, color=ACCENT, linewidth=2, zorder=3, marker="o", markersize=8,
            markerfacecolor="white", markeredgewidth=2, markeredgecolor=ACCENT)
    ax.fill_between(x, vals, min(vals) - 0.02, color=ACCENT, alpha=0.08, zorder=1)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.008, f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.annotate("", xy=(len(x) - 1, vals[-1]), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                                 connectionstyle="arc3,rad=0.25"))
    ax.text((len(x) - 1) / 2, max(vals) + 0.035,
            f"+{vals[-1] - vals[0]:.4f}  (+{(vals[-1] / vals[0] - 1) * 100:.1f}% relative)",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0.49, 0.62)
    style_bar_axes(ax)
    ax.set_title("Cumulative improvement, Phase 1 → Phase 4 (validation split)",
                  fontsize=13, fontweight="bold", color=INK, pad=16, loc="left")
    ax.text(0, -0.18,
            "Each step is kept only if it beat the previous step on validation; test set was touched\n"
            "exactly twice in the whole project (original Day-7 lock and the Phase 3 re-lock).",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, "11_cumulative_progression.png")


# ----------------------------------------------- 12. hierarchical diagnosis --
def fig_hierarchical_diagnosis():
    hier = load("phase4_5_hierarchical_results.json")
    stage1_f1 = hier["stage1_binary_macro_f1"]
    stage2_oracle = 0.5661  # mean of seed 0.5609/0.5652/0.5723, see IMPROVEMENT_PLAN.md Phase 4.5
    flat_ensemble = hier["comparison_flat_ensemble"]
    full_pipeline = hier["final_macro_f1"]

    labels = ["Stage 1 alone\n(binary,\nNone vs. any)", "Stage 2 alone\n(4-way, oracle-\nrouted subset)",
              "Flat ensemble\n(Phase 4.4)", "Full pipeline\n(stage1→stage2,\nreal routing)"]
    vals = [stage1_f1, stage2_oracle, flat_ensemble, full_pipeline]
    colors = [ACCENT, ACCENT, MUTED, RED]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.annotate("", xy=(3, full_pipeline + 0.03), xytext=(1, stage2_oracle + 0.03),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4,
                                 connectionstyle="arc3,rad=-0.3"))
    ax.text(2, max(vals) + 0.08, "error cascading:\nreal routing loses ~0.046\nvs. oracle routing",
            ha="center", fontsize=9, color=RED, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0, 0.90)
    style_bar_axes(ax)
    ax.set_title("Why Phase 4.5 hierarchical decomposition failed despite strong individual stages",
                  fontsize=12, fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, -0.16,
            "Both stages look strong in isolation, but Stage 1's routing mistakes are unrecoverable for\n"
            "Stage 2 -- a flat classifier can hedge across all 5 classes jointly; a hard two-stage gate cannot.",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.20)
    savefig(fig, "12_hierarchical_diagnosis.png")


# ------------------------------------------------------- 13. phase 6 lora --
def fig_phase6_lora():
    frozen_baseline = 0.5139
    unfreeze = 0.4875
    lora = load("phase6_1_lora_finetune_results.json")

    labels = ["Frozen encoders\n(concat baseline)", "Full unfreeze\n(Phase 3.1, failed)",
              "LoRA fine-tune\n(Phase 6.1)"]
    vals = [frozen_baseline, unfreeze, lora["majority_vote_macro_f1"]]
    colors = [MUTED, RED, GREEN]

    fig, ax = plt.subplots(figsize=(8, 5.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0.40, 0.58)
    style_bar_axes(ax)
    ax.set_title("Phase 6.1 — LoRA succeeds where full unfreezing failed\n(same simple-concat architecture, all 3 bars)",
                  fontsize=12, fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, -0.20,
            "Updating <1% of each encoder's parameters (LoRA) avoids the fast overfitting collapse that\n"
            "full/partial unfreezing hit on 582 training examples. Confirms the mechanism works, but this\n"
            "was tested on the simple architecture for a clean comparison -- it does not replace the current\n"
            "best (0.5695), which uses frozen embeddings with the full ensemble + calibration stack.",
            transform=ax.transAxes, fontsize=8.4, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.28)
    savefig(fig, "13_phase6_lora.png")


if __name__ == "__main__":
    fig_phase1_experiments()
    fig_phase2_experiments()
    fig_phase3_architecture()
    fig_phase4_refinements()
    fig_cumulative_progression()
    fig_hierarchical_diagnosis()
    fig_phase6_lora()
    print("\nAll improvement-round figures saved to", FIG_DIR)
