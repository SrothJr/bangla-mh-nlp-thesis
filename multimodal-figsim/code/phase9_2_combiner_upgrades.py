"""
Phase 9, Step 2: squeeze more out of the nine models that already exist.

No retraining of any base model happens here. This script only changes how
the nine models' predictions are COMBINED, using the out-of-fold ("OOF")
probabilities saved by phase9_1_cv_baseline_3class.py.

WHY THESE THREE
---------------
B1  Soft probability averaging instead of hard vote fractions.
    Phase 8's audit confirmed the locked ensemble combines HARD votes --
    each model contributes only its argmax and its confidence is thrown
    away. Soft averaging normally does better. This has never been
    compared on the 3-class task.

B2  A learned stacking meta-learner (IMPROVEMENT_PLAN.md item 5.2, open
    since Phase 5 and never attempted). A multinomial logistic regression
    over the nine models' probabilities can weight architectures per class
    -- e.g. trust cross-attention more on high-acuity, concat more on
    class 0 -- which a uniform vote cannot.

C2  Per-class decision weights fitted by coordinate ascent. Phase 4.4 did
    this for the 5-class task and it helped; it has never been redone for
    the 3-class task, whose class balance is completely different. Class 0
    is the measured weak point (Step 1: F1 0.469, recall 0.419), so this
    targets the actual failure.

LEAKAGE CONTROL -- the point that makes or breaks this step
----------------------------------------------------------
Every method here that FITS anything (B2's regression, C2's weights) is
evaluated with its own nested cross-validation over the OOF matrix, reusing
the exact same 5 folds as Step 1 (StratifiedKFold, shuffle=True,
random_state=42, over the same 777 labels in the same order). The combiner
is fitted on four folds' rows and applied to the fifth, never to rows it
was fitted on. Fitting a stacker and scoring it on the same rows is the
classic stacking error and would produce an inflated number that collapses
on test -- exactly the failure mode Phase 9 exists to avoid.

B1 and the baseline fit nothing, so they need no nesting.

A NOTE ON SELECTION
-------------------
Several methods are compared and the best is carried forward. With a
handful of candidates evaluated on the same 777 rows, a small amount of
selection optimism is unavoidable. It is not hidden: it is one of the
reasons Step 6's gate demands 0.72 CV accuracy rather than 0.70.

TEST SET: not touched.

OUTPUT
------
outputs/phase9_2_combiner_results.json
"""
import os
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, cohen_kappa_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OOF_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_1_oof_probs.npz")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_2_combiner_results.json")

NUM_CLASSES_3 = 3
THREE_CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]

# Must match phase9_1 exactly so the nested folds line up with the OOF rows.
N_FOLDS = 5
CV_RANDOM_STATE = 42

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 12345

# Coordinate-ascent search grid for per-class weights (C2).
WEIGHT_GRID = np.round(np.arange(0.50, 2.01, 0.05), 2)
COORD_PASSES = 3

BASELINE_NAME = "hard_vote_fractions_BASELINE"


# ================================================================ helpers ===
def hard_vote_fractions(prob_stack):
    """prob_stack: (n_models, n, 3) -> (n, 3) fractions of models voting each class."""
    preds = prob_stack.argmax(axis=2)                      # (n_models, n)
    n_models, n = preds.shape
    out = np.zeros((n, NUM_CLASSES_3))
    for c in range(NUM_CLASSES_3):
        out[:, c] = (preds == c).sum(axis=0) / n_models
    return out


def soft_average(prob_stack):
    return prob_stack.mean(axis=0)


def full_metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES_3))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(NUM_CLASSES_3)),
            target_names=THREE_CLASS_NAMES, zero_division=0, output_dict=True,
        ),
    }


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        accs.append(accuracy_score(y_true[s], y_pred[s]))
        f1s.append(f1_score(y_true[s], y_pred[s], average="macro", zero_division=0))
    return {
        "accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


def paired_bootstrap_vs_baseline(y_true, pred_a, pred_base, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Paired test: resample examples, recompute the DIFFERENCE. Shares the
    resample between the two methods, which is the right way to compare two
    predictors on the same rows."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    d_acc, d_f1 = [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        yt = y_true[s]
        d_acc.append(accuracy_score(yt, pred_a[s]) - accuracy_score(yt, pred_base[s]))
        d_f1.append(f1_score(yt, pred_a[s], average="macro", zero_division=0)
                    - f1_score(yt, pred_base[s], average="macro", zero_division=0))
    d_acc, d_f1 = np.array(d_acc), np.array(d_f1)
    return {
        "delta_accuracy_ci95": [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
        "delta_macro_f1_ci95": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
        "prob_accuracy_improved": float((d_acc > 0).mean()),
        "prob_macro_f1_improved": float((d_f1 > 0).mean()),
    }


# ================================================== fitted combiners (C2) ===
def fit_class_weights(scores, y, objective):
    """Coordinate ascent over per-class multiplicative weights on `scores`."""
    metric = (lambda yt, yp: accuracy_score(yt, yp)) if objective == "accuracy" else \
             (lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    w = np.ones(NUM_CLASSES_3)
    best = metric(y, (scores * w).argmax(axis=1))
    for _ in range(COORD_PASSES):
        changed = False
        for c in range(NUM_CLASSES_3):
            cur = w[c]
            for cand in WEIGHT_GRID:
                w[c] = cand
                sc = metric(y, (scores * w).argmax(axis=1))
                if sc > best + 1e-12:
                    best, cur, changed = sc, cand, True
            w[c] = cur
        if not changed:
            break
    return w


# ===================================================================== main ==
def main():
    print("=" * 74)
    print("PHASE 9 STEP 2 -- combiner upgrades on existing OOF predictions")
    print("No base model is retrained. TEST SET NOT TOUCHED.")
    print("=" * 74)

    d = np.load(OOF_PATH, allow_pickle=True)
    y = d["y"]
    model_names = [str(m) for m in d["model_names"]]
    prob_stack = np.stack([d[f"oof__{m}"] for m in model_names])   # (9, 777, 3)
    n_models, n, _ = prob_stack.shape
    print(f"Loaded OOF probabilities: {n_models} models x {n} examples")
    print(f"Models: {', '.join(model_names)}")
    print(f"Class distribution: {np.bincount(y).tolist()}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    folds = list(skf.split(np.zeros(n), y))

    predictions = {}

    # ---------------------------------------------------------- baseline ----
    predictions[BASELINE_NAME] = hard_vote_fractions(prob_stack).argmax(axis=1)

    # ------------------------------------------------- B1 soft averaging ----
    predictions["B1_soft_average"] = soft_average(prob_stack).argmax(axis=1)

    # Architecture-subset soft averages. Motivated by Phase 7.9, where
    # concat was the strongest single architecture and cross-attention the
    # weakest -- worth checking whether the weak member is dragging the
    # soft average down.
    groups = {
        "concat": [i for i, m in enumerate(model_names) if m.startswith("concat")],
        "gated": [i for i, m in enumerate(model_names) if m.startswith("gated")],
        "xattn": [i for i, m in enumerate(model_names) if m.startswith("xattn")],
    }
    for label, members in [
        ("concat+gated", groups["concat"] + groups["gated"]),
        ("concat+xattn", groups["concat"] + groups["xattn"]),
        ("gated+xattn", groups["gated"] + groups["xattn"]),
    ]:
        predictions[f"B1_soft_average_{label}"] = soft_average(prob_stack[members]).argmax(axis=1)

    # --------------------------------------- B2 stacking (nested CV fit) ----
    flat = prob_stack.transpose(1, 0, 2).reshape(n, n_models * NUM_CLASSES_3)  # (777, 27)
    for label, kwargs in [
        ("B2_stacker_lr", {"C": 1.0}),
        ("B2_stacker_lr_balanced", {"C": 1.0, "class_weight": "balanced"}),
    ]:
        pred = np.zeros(n, dtype=np.int64)
        for tr_pos, te_pos in folds:
            clf = LogisticRegression(max_iter=2000, **kwargs)
            clf.fit(flat[tr_pos], y[tr_pos])
            pred[te_pos] = clf.predict(flat[te_pos])
        predictions[label] = pred

    # ------------------------------ C2 per-class weights (nested CV fit) ----
    score_sources = {
        "on_hard_votes": hard_vote_fractions(prob_stack),
        "on_soft_average": soft_average(prob_stack),
    }
    fitted_weights = {}
    for src_label, scores in score_sources.items():
        for objective in ("macro_f1", "accuracy"):
            label = f"C2_classweights_{src_label}_opt{objective}"
            pred = np.zeros(n, dtype=np.int64)
            per_fold_w = []
            for tr_pos, te_pos in folds:
                w = fit_class_weights(scores[tr_pos], y[tr_pos], objective)
                per_fold_w.append(w.tolist())
                pred[te_pos] = (scores[te_pos] * w).argmax(axis=1)
            predictions[label] = pred
            fitted_weights[label] = per_fold_w

    # ------------------------------------------------------- evaluation ----
    base_pred = predictions[BASELINE_NAME]
    results = {}
    print("\n" + "-" * 74)
    print(f"{'method':<44}{'acc':>8}{'macroF1':>9}{'dAcc':>8}{'P(better)':>11}")
    print("-" * 74)
    for label, pred in predictions.items():
        m = full_metrics(y, pred)
        entry = {**m, "bootstrap_ci95": bootstrap_ci(y, pred)}
        if label != BASELINE_NAME:
            entry["paired_vs_baseline"] = paired_bootstrap_vs_baseline(y, pred, base_pred)
            d_acc = m["accuracy"] - results[BASELINE_NAME]["accuracy"]
            p_better = entry["paired_vs_baseline"]["prob_accuracy_improved"]
            print(f"{label:<44}{m['accuracy']:>8.4f}{m['macro_f1']:>9.4f}{d_acc:>+8.4f}{p_better:>11.3f}")
        else:
            print(f"{label:<44}{m['accuracy']:>8.4f}{m['macro_f1']:>9.4f}{'--':>8}{'--':>11}")
        results[label] = entry
    print("-" * 74)

    # ------------------------------------------------------------ winner ----
    best_acc_label = max(results, key=lambda k: results[k]["accuracy"])
    best_f1_label = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\nBest by accuracy : {best_acc_label}  ({results[best_acc_label]['accuracy']:.4f})")
    print(f"Best by macro-F1 : {best_f1_label}  ({results[best_f1_label]['macro_f1']:.4f})")

    print(f"\nPer-class for best-by-accuracy ({best_acc_label}):")
    for c in THREE_CLASS_NAMES:
        r = results[best_acc_label]["classification_report"][c]
        print(f"  {c}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f} n={int(r['support'])}")
    print("Confusion matrix (rows=true, cols=pred):")
    for row in results[best_acc_label]["confusion_matrix"]:
        print("  ", row)

    b = results[BASELINE_NAME]
    w = results[best_acc_label]
    print(f"\nBaseline (locked hard-vote method): acc={b['accuracy']:.4f} F1={b['macro_f1']:.4f}")
    print(f"Best combiner:                      acc={w['accuracy']:.4f} F1={w['macro_f1']:.4f}")
    print(f"Gain: {w['accuracy'] - b['accuracy']:+.4f} accuracy, {w['macro_f1'] - b['macro_f1']:+.4f} macro-F1")
    print(f"\nStep 6 gate needs CV accuracy >= 0.72 with lower CI bound >= 0.68.")
    print(f"Currently at {w['accuracy']:.4f}, lower CI bound {w['bootstrap_ci95']['accuracy_ci95'][0]:.4f}.")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Combiner-only comparison on phase9_1 OOF probabilities. No base "
                     "model retrained. Every fitted combiner (B2, C2) is evaluated by "
                     "nested CV over the same 5 folds as Step 1, so it never scores rows "
                     "it was fitted on. Test set not touched."),
            "n_examples": int(n),
            "n_models": int(n_models),
            "model_names": model_names,
            "n_folds": N_FOLDS,
            "cv_random_state": CV_RANDOM_STATE,
            "baseline_method": BASELINE_NAME,
            "results": results,
            "c2_fitted_weights_per_fold": fitted_weights,
            "best_by_accuracy": best_acc_label,
            "best_by_macro_f1": best_f1_label,
            "selection_caveat": ("Several combiners were compared on the same 777 rows, so "
                                 "the winner carries some selection optimism. This is one "
                                 "reason Step 6's gate is set at 0.72 rather than 0.70."),
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
