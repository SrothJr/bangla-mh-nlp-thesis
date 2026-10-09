"""
Phase 9, Step 8b: summary figure.

Written as its own script rather than as an addition to
make_improvement_figures.py, so that Phase 9 touches no existing code at
all. Produces figures/21_phase9_summary.png (and copies to
final_report_figures/ and supervisor_result/final_report_figures/ if those
folders exist).

Two panels:
  LEFT   the learning curve -- the one large effect Phase 9 measured.
  RIGHT  every experiment's accuracy against the honest baseline, with the
         pre-registered gate marked, showing how far short the phase fell.
"""
import os
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "figures")
COPY_DIRS = [
    os.path.join(PROJECT_ROOT, "final_report_figures"),
    os.path.join(PROJECT_ROOT, "supervisor_result", "final_report_figures"),
]
FNAME = "21_phase9_summary.png"

BASELINE = 0.6654
GATE = 0.72

EXPERIMENTS = [
    ("Honest CV baseline (Step 1)", 0.6654, "baseline"),
    ("Stacking, class-balanced", 0.6384, "rejected"),
    ("Stacking meta-learner (5.2)", 0.6615, "rejected"),
    ("Full-data refit (Track A1)", 0.6641, "rejected"),
    ("External negatives (Track A2)", 0.6229, "rejected"),
    ("Reasoning streams, full (Track D)", 0.6126, "rejected"),
    ("CORN ordinal loss (5.5)", 0.6654, "rejected"),
    ("Per-class decision weights", 0.6654, "neutral"),
    ("5 classes collapsed to 3", 0.6692, "neutral"),
    ("Soft averaging", 0.6692, "kept"),
    ("5 seeds per architecture (5.4)", 0.6718, "kept"),
    ("Best assembled, no dep. features (Step 8)", 0.6744, "kept"),
    ("+ depression features (Step 11)", 0.6873, "final"),
]

COLORS = {
    "baseline": "#6b7280",
    "rejected": "#dc2626",
    "neutral": "#f59e0b",
    "kept": "#16a34a",
    "final": "#1d4ed8",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "outputs", "phase9_3_learning_curve_results.json")) as f:
        curve = json.load(f)["pooled_curve"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.2),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    # ------------------------------------------------------- learning curve --
    ns = [curve[k]["mean_n_train"] for k in sorted(curve, key=float)]
    accs = [curve[k]["accuracy"] for k in sorted(curve, key=float)]
    f1s = [curve[k]["macro_f1"] for k in sorted(curve, key=float)]

    ax1.plot(ns, accs, "o-", color="#1d4ed8", lw=2.5, ms=9, label="accuracy")
    ax1.plot(ns, f1s, "s--", color="#0891b2", lw=2, ms=8, label="macro-F1")
    ax1.axhline(GATE, color="#dc2626", ls=":", lw=2)
    ax1.text(ns[0], GATE + 0.004, "gate: 0.72 accuracy", color="#dc2626", fontsize=10, va="bottom")

    # Extrapolation of the final segment, marked clearly as speculative.
    slope = (accs[-1] - accs[-2]) / (ns[-1] - ns[-2])
    x_ext = np.array([ns[-1], 777])
    ax1.plot(x_ext, accs[-1] + slope * (x_ext - ns[-1]), ":", color="#1d4ed8", lw=2, alpha=0.6)
    ax1.annotate("linear extrapolation\nto all 777 examples\n(never tested)",
                 xy=(770, accs[-1] + slope * (770 - ns[-1])), xytext=(330, 0.695),
                 fontsize=9, color="#1d4ed8", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", alpha=0.6,
                                 connectionstyle="arc3,rad=-0.2"))

    ax1.set_xlabel("training examples", fontsize=11)
    ax1.set_ylabel("out-of-fold score", fontsize=11)
    ax1.set_title("The one large effect Phase 9 measured:\n"
                  "performance is still climbing steeply with data\n"
                  "(+0.030 accuracy per +100 examples)",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.52, 0.76)

    # --------------------------------------------------------- experiments --
    order = sorted(range(len(EXPERIMENTS)), key=lambda i: EXPERIMENTS[i][1])
    labels = [EXPERIMENTS[i][0] for i in order]
    vals = [EXPERIMENTS[i][1] for i in order]
    kinds = [EXPERIMENTS[i][2] for i in order]
    ypos = np.arange(len(labels))

    ax2.barh(ypos, vals, color=[COLORS[k] for k in kinds], alpha=0.88, height=0.68)
    ax2.axvline(BASELINE, color="#6b7280", ls="--", lw=2)
    ax2.axvline(GATE, color="#dc2626", ls=":", lw=2.5)
    ax2.text(BASELINE - 0.002, len(labels) - 0.2, "honest baseline 0.6654",
             rotation=90, va="top", ha="right", fontsize=9, color="#374151")
    ax2.text(GATE + 0.002, len(labels) - 0.2, "gate 0.72 (not reached)",
             rotation=90, va="top", ha="left", fontsize=9, color="#dc2626")

    for y, v in zip(ypos, vals):
        ax2.text(v + 0.0015, y, f"{v:.4f}", va="center", fontsize=9)

    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlim(0.60, 0.745)
    ax2.set_xlabel("cross-validated accuracy (777 examples, test set never touched)", fontsize=11)
    ax2.set_title("Eight ideas across four tracks.\n"
                  "Total honest gain: +0.0090 accuracy.",
                  fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3, axis="x")

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k], alpha=0.88)
               for k in ("kept", "neutral", "rejected", "final")]
    ax2.legend(handles, ["helped (kept)", "no effect", "hurt (rejected)", "best (with dep. features)"],
               loc="lower right", fontsize=9)

    fig.suptitle("Phase 9 — attempt to reach 70% test accuracy: outcome and evidence",
                 fontsize=14.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012,
             "Pre-registered gate was not cleared, so the test set was not evaluated. "
             "Best known 3-class-equivalent test accuracy remains 0.6582 (Phase 8), unchanged.",
             ha="center", fontsize=10, style="italic", color="#374151")
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])

    out = os.path.join(OUT_DIR, FNAME)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    for d in COPY_DIRS:
        if os.path.isdir(d):
            shutil.copy2(out, os.path.join(d, FNAME))
            print(f"Copied -> {os.path.join(d, FNAME)}")


if __name__ == "__main__":
    main()
