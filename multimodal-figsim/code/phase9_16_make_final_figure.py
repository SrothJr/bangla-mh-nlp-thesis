"""
Phase 9, Step 9b: the final verdict figure.

Produces figures/22_phase9_final_verdict.png, which answers the question
"can Phase 9's configuration still beat what we already have?" in one page.

Three panels:
  LEFT    the Track A1 A/B -- the last untested idea, and it loses.
  MIDDLE  why it loses: the best epoch varies enormously between seeds, so
          the single fixed schedule A1 requires suits almost none of them.
  RIGHT   the resulting expected test accuracy, against the benchmarks it
          would have to beat.

Written as its own script so that Phase 9 continues to touch no existing
code. Figure 21 (the phase summary) is regenerated separately by
phase9_14_make_figures.py, which now includes A1 in its experiment list.
"""
import os
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PROJECT_ROOT, "outputs", "phase9_15_a1_refit_validation_results.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "figures")
COPY_DIRS = [
    os.path.join(PROJECT_ROOT, "final_report_figures"),
    os.path.join(PROJECT_ROOT, "supervisor_result", "final_report_figures"),
]
FNAME = "22_phase9_final_verdict.png"

# Established test benchmarks, all on the same 196 held-out memes.
NATIVE_3CLASS_TEST = 0.6071      # Phase 7.10 locked 3-class model
BEST_3CLASS_EQUIV_TEST = 0.6582  # Phase 8 valid collapse of the 5-class ensemble
TARGET = 0.70
PHASE9_CONFIG_GAIN = 0.0219      # measured in CV, Steps 8 + 11 (incl. depression features)


def main():
    with open(RESULTS, "r", encoding="utf-8") as f:
        d = json.load(f)
    a1 = d["arm1_current_protocol"]["accuracy"]
    a2 = d["arm2_a1_full_data_fixed_epochs"]["accuracy"]
    cmp = d["paired_comparison"]
    sched = d["epoch_schedules"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(17.5, 7.6), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})

    # ------------------------------------------------------ panel 1: the A/B --
    bars = ax1.bar(["current protocol\nearly stopping\n~527 rows",
                    "Track A1\nfixed epochs\n~621 rows (+18%)"],
                   [a1, a2], color=["#16a34a", "#dc2626"], alpha=0.88, width=0.58)
    for b, v in zip(bars, [a1, a2]):
        ax1.text(b.get_x() + b.get_width() / 2, v - 0.0035, f"{v:.4f}",
                 ha="center", va="top", fontsize=12.5, fontweight="bold", color="white")
    ax1.set_ylim(0.62, 0.695)
    ax1.set_ylabel("cross-validated accuracy", fontsize=11)
    ax1.set_title("Track A1 was the last untested idea.\nIt loses, despite 18% more data.",
                  fontsize=12.5, fontweight="bold")
    ax1.annotate("", xy=(1, a2), xytext=(1, a1),
                 arrowprops=dict(arrowstyle="<->", color="#374151", lw=1.6))
    ax1.text(1.28, (a1 + a2) / 2,
             f"{a2 - a1:+.4f}\nP(better)\n= {cmp['prob_accuracy_improved']:.3f}",
             fontsize=10, va="center", ha="center", color="#374151")
    ax1.set_xlim(-0.62, 1.62)
    ax1.grid(alpha=0.3, axis="y")

    # ------------------------------------------- panel 2: why -- epoch spread --
    archs = ["concat", "gated", "xattn"]
    colors = {"concat": "#1d4ed8", "gated": "#0891b2", "xattn": "#7c3aed"}
    for i, arch in enumerate(archs):
        rows = [e for e in sched if e["arch"] == arch]
        for r in rows:
            xs = np.full(len(r["arm1_best_epochs"]), i) + np.linspace(-0.17, 0.17,
                                                                     len(r["arm1_best_epochs"]))
            ax2.scatter(xs, r["arm1_best_epochs"], s=26, color=colors[arch], alpha=0.5,
                        edgecolors="none")
            ax2.scatter([i], [r["arm2_fixed_epochs"]], marker="_", s=420,
                        color="#dc2626", lw=2.4, zorder=5)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(archs, fontsize=11)
    ax2.set_ylabel("best epoch reached", fontsize=11)
    ax2.set_title("Why it loses: the best epoch varies hugely\n"
                  "between seeds, so one fixed schedule (red)\nfits almost none of them",
                  fontsize=12.5, fontweight="bold")
    ax2.scatter([], [], s=26, color="#6b7280", alpha=0.6, label="individual run's best epoch")
    ax2.plot([], [], marker="_", ls="none", ms=16, color="#dc2626", mew=2.4,
             label="fixed schedule A1 must use")
    ax2.legend(loc="upper left", fontsize=9.5)
    ax2.grid(alpha=0.3, axis="y")

    # --------------------------------- panel 3: expected test vs benchmarks ---
    lo = NATIVE_3CLASS_TEST + PHASE9_CONFIG_GAIN + (a2 - a1)   # A1 included, negative
    hi = NATIVE_3CLASS_TEST + PHASE9_CONFIG_GAIN               # A1 simply dropped

    ax3.axhspan(lo, hi, color="#f59e0b", alpha=0.35, zorder=1)
    ax3.text(0.42, hi + 0.0055, f"Phase 9 expected test accuracy  {lo:.3f} to {hi:.3f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color="#78350f", zorder=6)

    for y, label, color, style, dy, va in [
        (TARGET, f"target  {TARGET:.2f}", "#dc2626", ":", 0.0022, "bottom"),
        (BEST_3CLASS_EQUIV_TEST, f"best result you already have  {BEST_3CLASS_EQUIV_TEST:.4f}",
         "#16a34a", "-", 0.0022, "bottom"),
        (NATIVE_3CLASS_TEST, f"native 3-class lock  {NATIVE_3CLASS_TEST:.4f}",
         "#6b7280", "--", -0.0030, "top"),
    ]:
        ax3.axhline(y, color=color, ls=style, lw=2.2, zorder=4)
        ax3.text(0.02, y + dy, label, fontsize=10, color=color, fontweight="bold",
                 va=va, zorder=6)

    ax3.annotate("", xy=(0.86, BEST_3CLASS_EQUIV_TEST), xytext=(0.86, hi),
                 arrowprops=dict(arrowstyle="<->", color="#dc2626", lw=1.8), zorder=6)
    ax3.text(0.885, (hi + BEST_3CLASS_EQUIV_TEST) / 2,
             f"{BEST_3CLASS_EQUIV_TEST - hi:.3f}\nshort", fontsize=10.5,
             color="#dc2626", va="center", fontweight="bold")

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0.58, 0.715)
    ax3.set_xticks([])
    ax3.set_ylabel("test accuracy (196 held-out memes)", fontsize=11)
    ax3.set_title("The verdict: Phase 9 lands below the\nresult already on record. No path to 70%.",
                  fontsize=12.5, fontweight="bold")
    ax3.grid(alpha=0.3, axis="y")

    fig.suptitle("Phase 9 final verdict — can the new workflow beat the existing result?  No.",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.015,
             "Expected range = the base configuration's known test accuracy (0.6071) plus Phase 9's measured "
             "gains (+0.0219, including the depression features), with Track A1 either dropped or included "
             "at its measured value (-0.0103).\nThe test set was never evaluated in Phase 9; these are "
             "projections from cross-validation, not measurements on test.",
             ha="center", fontsize=9.5, style="italic", color="#374151")
    fig.tight_layout(rect=[0, 0.055, 1, 0.955])

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, FNAME)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    for dd in COPY_DIRS:
        if os.path.isdir(dd):
            shutil.copy2(out, os.path.join(dd, FNAME))
            print(f"Copied -> {os.path.join(dd, FNAME)}")


if __name__ == "__main__":
    main()
