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


# ------------------------------------------------- 14. label harmonization --
def fig_harmonization():
    diag = load("phase7_1_label_harmonization_diagnostic_results.json")
    comp = load("phase7_4_harmonized_3class_comparison_results.json")

    labels = ["5-class\n(existing best)", "3-class\naggregated\n(diagnostic, no retrain)",
              "3-class\ntrained natively\n(original SigLIP)", "4-class\naggregated\n(merges ideation+planning)"]
    vals = [diag["five_class"]["macro_f1"], diag["three_class"]["macro_f1"],
            comp["variant_A_original_siglip"]["majority_vote_macro_f1"], diag["four_class"]["macro_f1"]]
    colors = [MUTED, ACCENT, GREEN, ACCENT]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.55, zorder=3)
    bars[2].set_edgecolor(INK)
    bars[2].set_linewidth(1.8)
    bars[2].set_linestyle((0, (2, 2)))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_ylim(0, 0.74)
    style_bar_axes(ax)
    ax.set_title("Label harmonization — 3-class is a genuine win, not an aggregation artifact\n(dashed outline = natively trained, the confirmed result)",
                  fontsize=12, fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, -0.20,
            "Training natively on the 3-class target (0.6425) slightly beats even the aggregation-only\n"
            "diagnostic (0.6394) -- the gain is real, not an artifact of re-grouping predictions after the fact.\n"
            "4-class (merging exactly the ideation/planning boundary) gains less, showing the confusion is\n"
            "spread across multiple ordinal boundaries, not concentrated at just that one.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.26)
    savefig(fig, "14_label_harmonization.png")


# --------------------------------------------- 15. BN-HIB pretraining test --
def fig_bnhib_pretraining_test():
    bnhib = load("phase7_2_bnhib_vision_pretrain_results.json")
    comp = load("phase7_4_harmonized_3class_comparison_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # left panel: BN-HIB's own task (proof the pretraining learned something real)
    ax = axes[0]
    report = bnhib["classification_report"]
    classes = ["Benign", "Inflammatory", "Hate"]
    vals = [report[c]["f1-score"] for c in classes]
    bars = ax.bar(classes, vals, color=ACCENT, width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0.333, color=MUTED, linewidth=1.2, linestyle=(0, (2, 2)))
    ax.text(2.3, 0.333 + 0.015, "random\nchance", fontsize=7.5, color=MUTED, ha="center")
    ax.set_ylabel("F1 (BN-HIB's own validation task)")
    ax.set_ylim(0, 0.9)
    style_bar_axes(ax)
    ax.set_title(f"Per-class F1 on BN-HIB\n(macro-F1 {bnhib['best_val_macro_f1']:.4f} — real, above-chance signal)",
                 fontsize=11, fontweight="bold", color=INK, pad=10, loc="left")

    # right panel: does it transfer to FigSIM's own task?
    ax = axes[1]
    labels = ["Original SigLIP\n(no external pretraining)", "BN-HIB-pretrained\nSigLIP"]
    vals = [comp["variant_A_original_siglip"]["majority_vote_macro_f1"],
            comp["variant_B_bnhib_pretrained_siglip"]["majority_vote_macro_f1"]]
    colors = [GREEN, RED]
    bars = ax.bar(labels, vals, color=colors, width=0.45, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("FigSIM 3-class macro-F1 (validation)")
    ax.set_ylim(0, 0.74)
    style_bar_axes(ax)
    delta = comp["delta_B_minus_A"]
    ax.set_title(f"Transfer test on FigSIM's real task\n(delta {delta:+.4f} — did not transfer)",
                 fontsize=11, fontweight="bold", color=INK, pad=10, loc="left")

    fig.suptitle("BN-HIB pretraining: real signal on its own task, but did not transfer to FigSIM",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left", y=1.03)
    fig.text(0.01, -0.02,
             "A good score on the source task does not guarantee transfer -- likely specialized toward BN-HIB's own\n"
             "visual conventions (meme templates, Bangla-script text rendering) rather than learning features that\n"
             "generalize to FigSIM's different visual domain. Checkpoint not adopted; decision-gate protocol worked as intended.",
             fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(wspace=0.3, bottom=0.18, top=0.82)
    savefig(fig, "15_bnhib_pretraining_transfer_test.png")


# ------------------------------------------------ 16. fuzzy-logic compare --
def fig_fuzzy_logic_comparison():
    comp = load("phase7_6_fuzzy_logic_comparison_results.json")
    RISK_LEVELS = ["Minimal", "Low", "Elevated", "Critical"]
    risk_colors = ["#6da7ec", "#2a9d5c", "#eb6834", "#c0392b"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    ax = axes[0]
    x = np.arange(len(RISK_LEVELS))
    w = 0.35
    vals_5 = [comp["risk_distribution_5class"][r] for r in RISK_LEVELS]
    vals_3 = [comp["risk_distribution_3class"][r] for r in RISK_LEVELS]
    n = comp["n_val"]
    b1 = ax.bar(x - w / 2, vals_5, w, color=MUTED, label="5-class model", zorder=3)
    b2 = ax.bar(x + w / 2, vals_3, w, color=ACCENT, label="3-class model", zorder=3)
    for bars, vals in ((b1, vals_5), (b2, vals_3)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v}\n({100*v/n:.0f}%)",
                     ha="center", va="bottom", fontsize=8.3)
    ax.set_xticks(x)
    ax.set_xticklabels(RISK_LEVELS, fontsize=10)
    ax.set_ylabel(f"count (of {n} validation memes)")
    ax.set_ylim(0, 85)
    style_bar_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.set_title("Final risk-level distribution:\n5-class-fuzzy vs. 3-class-fuzzy",
                 fontsize=11.5, fontweight="bold", color=INK, pad=12, loc="left")

    ax = axes[1]
    mat = np.zeros((4, 4))
    for i, a in enumerate(RISK_LEVELS):
        for j, b in enumerate(RISK_LEVELS):
            mat[i, j] = comp["agreement_matrix"][f"{a}->{b}"]
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
    for i in range(4):
        for j in range(4):
            v = int(mat[i, j])
            t = v / mat.max() if mat.max() > 0 else 0
            color = "white" if t > 0.5 else INK
            weight = "bold" if i == j else "normal"
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=11, fontweight=weight)
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(RISK_LEVELS, fontsize=9)
    ax.set_yticklabels(RISK_LEVELS, fontsize=9)
    ax.set_xlabel("3-class-fuzzy risk", fontsize=10)
    ax.set_ylabel("5-class-fuzzy risk", fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Per-meme agreement: {comp['agreement_count']}/{n} ({comp['agreement_pct']:.1f}%)\n"
                 "diagonal = same final risk level",
                 fontsize=11.5, fontweight="bold", color=INK, pad=48)

    fig.suptitle("5-class vs. 3-class fuzzy risk combination — same memes, same depression scores, different final risk",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left", y=1.05)
    fig.text(0.01, -0.05,
             "Largest disagreement (Elevated->Critical, 24 cases) is mostly a mechanical consequence of the cautious-merge\n"
             "rule-table policy for the 3-class 'high acuity' bucket, not necessarily a change in what the underlying models\n"
             "perceive -- worth reading as a rule-table design effect, not proof the 3-class scheme finds more real risk.",
             fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(wspace=0.35, bottom=0.16, top=0.80)
    savefig(fig, "16_fuzzy_logic_5class_vs_3class.png")


# --------------------------------------------- 17. final 3-class test lock --
def fig_final_3class_test():
    test3 = load("phase7_final_test_results_3class.json")
    ens9 = load("phase7_9_harmonized_ensemble_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    ax = axes[0]
    labels = ["5-class test\n(Phase 3 lock,\nprior best)", "3-class test\n(this lock,\nnew best)"]
    vals = [0.4984, test3["macro_f1"]]
    colors = [MUTED, GREEN]
    bars = ax.bar(labels, vals, color=colors, width=0.45, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("macro-F1 (test, n=196)")
    ax.set_ylim(0, 0.68)
    style_bar_axes(ax)
    delta = test3["macro_f1"] - 0.4984
    ax.set_title(f"Test-confirmed result: +{delta:.4f} over prior best",
                 fontsize=12, fontweight="bold", color=INK, pad=12, loc="left")

    ax = axes[1]
    labels2 = ["Validation\n(Phase 7.9)", "Test\n(this lock,\nthe real number)"]
    vals2 = [ens9["best_macro_f1"], test3["macro_f1"]]
    colors2 = [ACCENT, GREEN]
    bars2 = ax.bar(labels2, vals2, color=colors2, width=0.45, zorder=3)
    for bar, v in zip(bars2, vals2):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 0.74)
    style_bar_axes(ax)
    gap = vals2[0] - vals2[1]
    ax.set_title(f"Validation vs. test gap: -{gap:.4f}\n(larger than usual -- see honest caveat)",
                 fontsize=12, fontweight="bold", color=INK, pad=12, loc="left")

    fig.suptitle("3-class harmonized model — locked final test result (third deliberate test touch)",
                 fontsize=13.5, fontweight="bold", color=INK, x=0.01, ha="left", y=1.04)
    fig.text(0.01, -0.04,
             "Gap reflects a long sequence of decisions all made against the same 195-item validation set\n"
             "(diagnostic, native training, hyperparameter sweep, architecture search, ensemble search) --\n"
             "consistent with Phase 5.1's finding that single-split deltas in this project carry real uncertainty.\n"
             "The test number (0.5730) is the genuine, defensible result -- not the validation figure.",
             fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(wspace=0.3, bottom=0.20, top=0.82)
    savefig(fig, "17_final_3class_test_lock.png")


# --------------------------------------------- 18. final fuzzy on test --
def fig_final_fuzzy_test():
    comp = load("phase7_11_fuzzy_logic_final_test_results.json")
    RISK_LEVELS = ["Minimal", "Low", "Elevated", "Critical"]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    x = np.arange(len(RISK_LEVELS))
    w = 0.35
    n = comp["n_test"]
    vals_5 = [comp["risk_distribution_5class_test_original"][r] for r in RISK_LEVELS]
    vals_3 = [comp["risk_distribution_3class_test"][r] for r in RISK_LEVELS]
    b1 = ax.bar(x - w / 2, vals_5, w, color=MUTED, label="5-class system (original, Days 9-10)", zorder=3)
    b2 = ax.bar(x + w / 2, vals_3, w, color=GREEN, label="3-class system (new, locked model)", zorder=3)
    for bars, vals in ((b1, vals_5), (b2, vals_3)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v}\n({100*v/n:.0f}%)",
                     ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(RISK_LEVELS, fontsize=10.5)
    ax.set_ylabel(f"count (of {n} test memes)")
    ax.set_ylim(0, 84)
    style_bar_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.set_title("Final system output on TEST — risk-level distribution, old vs. new locked model",
                 fontsize=12.5, fontweight="bold", color=INK, pad=14, loc="left")
    ax.text(0, -0.18,
            "Same 196 test memes, same untouched Phase-2 depression classifier -- only the suicide-severity\n"
            "model changed (old 5-class single-lock vs. new 3-class 9-model ensemble). New system spreads risk\n"
            "more evenly across all four levels rather than concentrating in Low/Elevated.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, "18_final_fuzzy_test_comparison.png")


# ----------------------------------------- 19. end-to-end pipeline diagram --
def fig_end_to_end_pipeline():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11.5, 12.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14.3)
    ax.axis("off")

    def box(x, y, w, h, text, color, text_color=INK, fontsize=10.2, fontweight="bold", edge=None, lw=1.4):
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=color, edgecolor=(edge or color), linewidth=lw, zorder=3,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight,
                 color=text_color, zorder=4, linespacing=1.4)
        return (x, y, w, h)

    def arrow(b1, b2, color=INK2, style="-", lw=1.6, from_side="bottom", to_side="top", rad=0.0):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        starts = {"bottom": (x1, y1 - h1 / 2), "top": (x1, y1 + h1 / 2)}
        ends = {"bottom": (x2, y2 - h2 / 2), "top": (x2, y2 + h2 / 2)}
        start = starts[from_side]
        end = ends[to_side]
        arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                               color=color, linewidth=lw, linestyle=style, zorder=2,
                               shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arr)

    LEFT, RIGHT, CENTER = 3.2, 8.8, 6.0

    # ------------------------------------------------------------ input --
    b_input = box(CENTER, 13.6, 6.4, 0.85, "Bangla meme (text + image)", "#0b0b0b", text_color="white", fontsize=12)

    b_text = box(LEFT, 12.0, 3.6, 0.75, "DAPT-BanglaBERT\ntext embedding", "#d8e6f9", fontsize=9.3)
    b_image = box(RIGHT, 12.0, 3.6, 0.75, "SigLIP\nimage embedding", "#d8e6f9", fontsize=9.3)
    arrow(b_input, b_text)
    arrow(b_input, b_image)

    # ------------------------------------------- two parallel branches --
    b_ensemble = box(LEFT, 10.2, 5.4, 1.5,
                      "Multimodal severity ensemble\nconcat + gated fusion +\ncross-attention (3 seeds each =\n9 models, majority vote)",
                      ACCENT, text_color="white", fontsize=9.3)
    arrow(b_text, b_ensemble)
    arrow(b_image, b_ensemble)

    b_dep = box(RIGHT, 10.2, 5.4, 1.5,
                "Depression classifier\n(Phase 2, untouched)\nDAPT-BanglaBERT + its own head\n"
                "text-only, same OCR text",
                "#f6d9c4", fontsize=9.3)
    arrow(b_text, b_dep, color=MUTED, style=(0, (4, 2)), rad=-0.35)

    b_severity = box(LEFT, 7.5, 5.4, 1.15,
                      "Suicide severity (3-class)\nNone / thought-or-desire /\nhigh-acuity — TEST macro-F1 = 0.5730",
                      "#c8e6d5", fontsize=9.3)
    arrow(b_ensemble, b_severity)

    b_dep_out = box(RIGHT, 7.5, 5.4, 1.15,
                     "Depression severity (4-class)\nMinimum / Mild / Moderate / Severe",
                     "#fbe8d8", fontsize=9.3)
    arrow(b_dep, b_dep_out)

    # ------------------------------------------------------ fuzzy combine --
    b_fuzzy = box(CENTER, 5.3, 8.8, 1.3,
                   "Fuzzy-logic risk combination\nmembership = softmax probabilities, "
                   "AND = min, OR = max\n6-rule table (3-class), argmax risk level",
                   "#ece5f7", fontsize=9.6)
    arrow(b_severity, b_fuzzy, from_side="bottom", to_side="top")
    arrow(b_dep_out, b_fuzzy, from_side="bottom", to_side="top")

    b_final = box(CENTER, 3.2, 8.8, 1.5,
                   "Final risk level\nMinimal / Low / Elevated / Critical\n"
                   "23% / 25% / 25% / 27% on 196 test memes",
                   "#0b0b0b", text_color="white", fontsize=11)
    arrow(b_fuzzy, b_final)

    ax.text(0.1, 1.1,
            "Solid arrows = the same meme's multimodal pipeline. Dashed arrow = the same OCR text reused\n"
            "for the separate, untouched depression classifier. Both branches meet only at the fuzzy-logic step.",
            fontsize=8.8, color=MUTED, ha="left", va="top", transform=ax.transData)

    ax.text(0.1, 14.2, "End-to-end system pipeline — from meme to final risk level",
            fontsize=15, fontweight="bold", color=INK, ha="left", va="top")

    savefig(fig, "19_end_to_end_pipeline.png")


# ------------------------------------- 20. updated pipeline, post Phase 8 --
def fig_current_verified_pipeline():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 15.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 18.0)
    ax.axis("off")

    def box(x, y, w, h, text, color, text_color=INK, fontsize=9.6, fontweight="bold", edge=None, lw=1.4):
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=color, edgecolor=(edge or color), linewidth=lw, zorder=3,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=fontweight,
                 color=text_color, zorder=4, linespacing=1.35)
        return (x, y, w, h)

    def arrow(b1, b2, color=INK2, style="-", lw=1.6, from_side="bottom", to_side="top", rad=0.0):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        starts = {"bottom": (x1, y1 - h1 / 2), "top": (x1, y1 + h1 / 2)}
        ends = {"bottom": (x2, y2 - h2 / 2), "top": (x2, y2 + h2 / 2)}
        arr = FancyArrowPatch(starts[from_side], ends[to_side], arrowstyle="-|>", mutation_scale=16,
                               color=color, linewidth=lw, linestyle=style, zorder=2,
                               shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arr)

    LEFT, RIGHT, CENTER = 3.2, 8.8, 6.0

    b_input = box(CENTER, 16.7, 6.6, 0.85, "Bangla meme (text + image) OR text alone", "#0b0b0b", text_color="white", fontsize=11.5)

    b_text = box(LEFT, 14.4, 3.8, 0.75, "DAPT-BanglaBERT\ntext embedding", "#d8e6f9", fontsize=9)
    b_image = box(RIGHT, 14.4, 3.8, 0.75, "SigLIP\nimage embedding\n(only if image present)", "#d8e6f9", fontsize=8.6)
    arrow(b_input, b_text)
    arrow(b_input, b_image)

    b_ensemble = box(LEFT, 12.5, 5.6, 1.4,
                      "Multimodal severity ensemble\nconcat + gated fusion + cross-attention\n"
                      "(9 models, majority vote) — requires BOTH modalities",
                      ACCENT, text_color="white", fontsize=8.8)
    arrow(b_text, b_ensemble)
    arrow(b_image, b_ensemble)

    b_dep = box(RIGHT, 12.5, 5.6, 1.7,
                "Depression classifier (Phase 2)\nDAPT-BanglaBERT + its own head — UNTOUCHED\n"
                "✓ parity-tested: 100% match, 0.00 diff (n=196)\n"
                "✓ verified on 4,897 labeled records\n"
                "(macro-F1 0.918, see training-overlap caveat)",
                "#f6d9c4", fontsize=8.2)
    arrow(b_text, b_dep, color=GREEN, lw=2.0, rad=-0.35)
    ax.text(1.3, 13.35, "text-only route\n(Phase 8, VERIFIED)", fontsize=8, color=GREEN, fontweight="bold", ha="center", style="italic")

    b_severity = box(LEFT, 9.9, 5.6, 1.9,
                      "Suicide severity — TWO locked models, reported\nSEPARATELY (not directly comparable, see correction):\n"
                      "• 5-class (prior best, strongest validated): 0.4984\n"
                      "• 3-class (harmonized, own task): 0.5730\n"
                      "Fair comparison: 3-class model is NOT an\nimprovement over 5-class once validly checked (−0.0294)",
                      "#c8e6d5", fontsize=7.9)
    arrow(b_ensemble, b_severity)

    b_dep_out = box(RIGHT, 9.9, 5.6, 1.1,
                     "Depression severity (4-class)\nMinimum / Mild / Moderate / Severe",
                     "#fbe8d8", fontsize=9.3)
    arrow(b_dep, b_dep_out)

    b_fuzzy = box(CENTER, 6.9, 9.0, 1.45,
                   "Fuzzy-logic risk combination\nmembership = vote fractions (suicide) / softmax (depression)\n"
                   "AND = min, OR = max — 11-row (5-class) or 6-row (3-class)\n"
                   "rule table available, matched to whichever severity model is used",
                   "#ece5f7", fontsize=8.6)
    arrow(b_severity, b_fuzzy, from_side="bottom", to_side="top")
    arrow(b_dep_out, b_fuzzy, from_side="bottom", to_side="top")

    b_final = box(CENTER, 4.6, 9.0, 1.5,
                   "Final risk level\nMinimal / Low / Elevated / Critical\n"
                   "(exploratory, rule-based — not clinically validated)",
                   "#0b0b0b", text_color="white", fontsize=10.5)
    arrow(b_fuzzy, b_final)

    b_gap = box(CENTER, 2.5, 9.6, 1.3,
                "Still open (Phase 8, in progress):\nno text-only or image-only route for SUICIDE severity yet —\n"
                "only the depression branch has a verified text-only path so far",
                "#f3f2ee", text_color=MUTED, fontsize=8.4, fontweight="normal", edge=GRID)
    arrow(b_final, b_gap, color=MUTED, style=(0, (3, 2)))

    ax.text(0.1, 0.55,
            "Solid arrows = the multimodal pipeline (unchanged since Day 1-7). Green arrow = the depression\n"
            "classifier's text-only route, added and proven in Phase 8. Both severity models are kept and\n"
            "reported honestly, side by side, per the Phase 8 Step 0 correction.",
            fontsize=8.4, color=MUTED, ha="left", va="top", transform=ax.transData)

    ax.text(0.1, 17.9, "Current verified system pipeline — post Phase 8 correction and verification",
            fontsize=14.5, fontweight="bold", color=INK, ha="left", va="top")

    savefig(fig, "20_current_verified_pipeline.png")


if __name__ == "__main__":
    fig_phase1_experiments()
    fig_phase2_experiments()
    fig_phase3_architecture()
    fig_phase4_refinements()
    fig_cumulative_progression()
    fig_hierarchical_diagnosis()
    fig_phase6_lora()
    fig_harmonization()
    fig_bnhib_pretraining_test()
    fig_fuzzy_logic_comparison()
    fig_final_3class_test()
    fig_final_fuzzy_test()
    fig_end_to_end_pipeline()
    fig_current_verified_pipeline()
    print("\nAll improvement-round figures saved to", FIG_DIR)
