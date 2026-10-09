"""
Phase 9, Step 8: assemble everything that individually helped, measure the
combination as a whole, and apply the pre-registered gate.

WHY THE COMBINATION MUST BE RE-MEASURED
---------------------------------------
Components that each help on their own are not guaranteed to help together.
Steps 2 and 4 found two small gains -- soft probability averaging, and five
seeds per architecture instead of three -- plus a per-class weighting that
helped macro-F1 while leaving accuracy flat. All three act on the same
combination stage, so they overlap and may substitute for one another
rather than add. This script measures the assembled configuration directly
instead of adding up the individual deltas.

THE GATE, FIXED IN ADVANCE
--------------------------
PHASE9_BEYOND_70_PLAN.md Section 3 set it before any of this ran:

    proceed to the single test evaluation only if
    CV accuracy >= 0.72 AND the lower bootstrap CI bound >= 0.68

The 0.72 margin exists because of the nine-point validation-to-test gap
this project has already been bitten by once, and because several
combiners were compared on the same 777 rows, which carries some selection
optimism of its own. The CI condition exists so that a lucky point estimate
cannot pass on its own.

If the gate fails, the test set is NOT touched, the locked models stand,
and Phase 9 reports what it measured. That outcome was written down in
advance (plan Section 5) precisely so it could not be renegotiated
afterwards.

INPUTS
------
outputs/phase9_4_oof_probs.npz   15 cross-entropy models (5 seeds x 3 archs)

OUTPUT
------
outputs/phase9_13_gate_results.json
"""
import os
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, cohen_kappa_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OOF4_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_4_oof_probs.npz")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_13_gate_results.json")

NUM_3 = 3
CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
SEEDS_5 = [0, 1, 2, 3, 4]
ARCHS = ["concat", "gated", "xattn"]

N_FOLDS = 5
CV_RANDOM_STATE = 42
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 12345

WEIGHT_GRID = np.round(np.arange(0.50, 2.01, 0.05), 2)
COORD_PASSES = 3

GATE_MIN_ACCURACY = 0.72
GATE_MIN_LOWER_CI = 0.68

STEP1_BASELINE_ACC = 0.6654
STEP1_BASELINE_F1 = 0.6155
# Best 3-class-equivalent TEST accuracy this project has actually achieved,
# from Phase 8's valid collapse of the 5-class ensemble. Reference only.
BEST_KNOWN_TEST_ACC = 0.6582


def hard_votes(stack):
    preds = stack.argmax(axis=2)
    out = np.zeros((preds.shape[1], NUM_3))
    for c in range(NUM_3):
        out[:, c] = (preds == c).sum(axis=0) / preds.shape[0]
    return out


def fit_class_weights(scores, y, objective):
    metric = (lambda a, b: accuracy_score(a, b)) if objective == "accuracy" else \
             (lambda a, b: f1_score(a, b, average="macro", zero_division=0))
    w = np.ones(NUM_3)
    best = metric(y, (scores * w).argmax(axis=1))
    for _ in range(COORD_PASSES):
        changed = False
        for c in range(NUM_3):
            cur = w[c]
            for cand in WEIGHT_GRID:
                w[c] = cand
                s = metric(y, (scores * w).argmax(axis=1))
                if s > best + 1e-12:
                    best, cur, changed = s, cand, True
            w[c] = cur
        if not changed:
            break
    return w


def full_metrics(y, pred):
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
            zero_division=0, output_dict=True),
    }


def bootstrap_ci(y, pred, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    a, f = [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        a.append(accuracy_score(y[s], pred[s]))
        f.append(f1_score(y[s], pred[s], average="macro", zero_division=0))
    return {"accuracy_ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "macro_f1_ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]}


def main():
    print("=" * 74)
    print("PHASE 9 STEP 8 -- assemble the best configuration and apply the gate")
    print("=" * 74)

    d = np.load(OOF4_PATH, allow_pickle=True)
    y = d["y"]
    n = len(y)
    names15 = [f"{a}_seed{s}" for a in ARCHS for s in SEEDS_5]
    stack15 = np.stack([d[f"ce__{m}"] for m in names15])
    print(f"Loaded {len(names15)} models x {n} examples")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    folds = list(skf.split(np.zeros(n), y))

    candidates = {}
    candidates["locked_method_9models_hardvote"] = hard_votes(
        np.stack([d[f"ce__{a}_seed{s}"] for a in ARCHS for s in [0, 1, 2]])).argmax(axis=1)
    candidates["A_15models_hardvote"] = hard_votes(stack15).argmax(axis=1)
    candidates["B_15models_softavg"] = stack15.mean(axis=0).argmax(axis=1)

    # Soft average + per-class weights, weights fitted by nested CV so they
    # are never applied to the rows they were fitted on.
    soft15 = stack15.mean(axis=0)
    for objective in ("accuracy", "macro_f1"):
        pred = np.zeros(n, dtype=np.int64)
        for tr, te in folds:
            w = fit_class_weights(soft15[tr], y[tr], objective)
            pred[te] = (soft15[te] * w).argmax(axis=1)
        candidates[f"C_15models_softavg_weights_opt{objective}"] = pred

    results = {}
    print("\n" + "-" * 74)
    print(f"{'configuration':<46}{'acc':>8}{'macroF1':>9}{'loCI':>8}")
    print("-" * 74)
    for label, pred in candidates.items():
        m = full_metrics(y, pred)
        ci = bootstrap_ci(y, pred)
        m["bootstrap_ci95"] = ci
        results[label] = m
        print(f"{label:<46}{m['accuracy']:>8.4f}{m['macro_f1']:>9.4f}"
              f"{ci['accuracy_ci95'][0]:>8.4f}")
    print("-" * 74)

    best = max(results, key=lambda k: results[k]["accuracy"])
    b = results[best]
    lo = b["bootstrap_ci95"]["accuracy_ci95"][0]
    print(f"\nBest assembled configuration: {best}")
    print(f"  CV accuracy  {b['accuracy']:.4f}   95% CI "
          f"[{lo:.4f}, {b['bootstrap_ci95']['accuracy_ci95'][1]:.4f}]")
    print(f"  CV macro-F1  {b['macro_f1']:.4f}")
    print(f"  Step 1 baseline was accuracy {STEP1_BASELINE_ACC:.4f}, "
          f"macro-F1 {STEP1_BASELINE_F1:.4f}")
    print(f"  Total Phase 9 gain: {b['accuracy'] - STEP1_BASELINE_ACC:+.4f} accuracy, "
          f"{b['macro_f1'] - STEP1_BASELINE_F1:+.4f} macro-F1")

    print("\nPer-class (best configuration):")
    for c in CLASS_NAMES:
        r = b["classification_report"][c]
        print(f"  {c}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")

    passed = (b["accuracy"] >= GATE_MIN_ACCURACY) and (lo >= GATE_MIN_LOWER_CI)
    print("\n" + "=" * 74)
    print("GATE (fixed in advance, PHASE9_BEYOND_70_PLAN.md Section 3)")
    print("=" * 74)
    print(f"  required: CV accuracy >= {GATE_MIN_ACCURACY} and lower CI bound >= {GATE_MIN_LOWER_CI}")
    print(f"  actual:   CV accuracy  = {b['accuracy']:.4f}   lower CI bound = {lo:.4f}")
    print(f"\n  RESULT: {'PASS -- proceed to the single test evaluation' if passed else 'STOP -- the test set is NOT touched'}")
    if not passed:
        print("\n  The locked models stand. Phase 9 reports what it measured.")
        print("  This outcome was written down in advance (plan Section 5) so that it")
        print("  could not be renegotiated after the fact.")
        print(f"\n  For reference, the best 3-class-equivalent TEST accuracy this project")
        print(f"  has actually achieved remains {BEST_KNOWN_TEST_ACC} (Phase 8's valid collapse")
        print(f"  of the 5-class ensemble). Nothing in Phase 9 displaces it.")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Assembled configuration measured as a whole, not by summing "
                     "individual deltas. Per-class weights fitted by nested CV. "
                     "Test set not touched anywhere in Phase 9."),
            "n_examples": int(n),
            "configurations": results,
            "best_by_accuracy": best,
            "step1_baseline": {"accuracy": STEP1_BASELINE_ACC, "macro_f1": STEP1_BASELINE_F1},
            "total_phase9_gain": {
                "accuracy": float(b["accuracy"] - STEP1_BASELINE_ACC),
                "macro_f1": float(b["macro_f1"] - STEP1_BASELINE_F1),
            },
            "gate": {
                "required_accuracy": GATE_MIN_ACCURACY,
                "required_lower_ci": GATE_MIN_LOWER_CI,
                "actual_accuracy": b["accuracy"],
                "actual_lower_ci": lo,
                "passed": bool(passed),
                "fixed_in_advance": True,
            },
            "test_set_touched": False,
            "best_known_test_accuracy_unchanged": BEST_KNOWN_TEST_ACC,
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
