"""
Phase 9, Step 3: does more training data actually help?

WHY THIS EXISTS
---------------
Track A1 of PHASE9_BEYOND_70_PLAN.md assumes that refitting the final
models on all 777 train+val examples, instead of the 582 the locked models
were trained on, is worth +1 to +3 accuracy points. That is an assumption,
and this project's standing practice is to measure assumptions rather than
bank on them.

This script measures the learning curve directly. If accuracy is still
climbing at the largest training size available, more data helps and the
A1 refit is justified. If the curve has flattened, A1 is worth little, the
external-negatives work in Step 4 is unlikely to pay off either, and both
should be reported as such instead of being run on faith.

WHAT IT DOES
------------
Reuses the exact fold structure and training functions from Step 1. For
each outer fold, the nine models are retrained from scratch on nested
subsets of that fold's inner-training portion -- 40%, 60%, 80% and 100% --
with early stopping on the same inner-validation rows in every case, and
evaluated on the same untouched outer held-out rows.

Subsets are nested (the 40% rows are a subset of the 60% rows, and so on)
and stratified, so the curve reflects quantity of data rather than luck of
the draw. The inner-validation set and the outer held-out set are held
fixed across all sizes, so the only thing changing is training-set size.

TEST SET: not touched.

OUTPUT
------
outputs/phase9_3_learning_curve_results.json
"""
import os
import json
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

from train_e6 import SEEDS, DEVICE as GATED_DEVICE
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
from phase3_2_cross_attention import DEVICE as XATTN_DEVICE

import phase9_1_cv_baseline_3class as P9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_3_learning_curve_results.json")

FRACTIONS = [0.4, 0.6, 0.8, 1.0]
SUBSET_RANDOM_STATE = 99


def nested_stratified_subsets(positions, y, fractions, seed):
    """Nested stratified subsets: smaller ones are subsets of larger ones."""
    order = {}
    remaining = np.array(positions)
    # Build from largest down, each time carving a stratified subset out of
    # the previous one, so nesting is guaranteed by construction.
    fracs = sorted(fractions, reverse=True)
    order[fracs[0]] = remaining.copy()
    current = remaining
    for i in range(1, len(fracs)):
        target_frac = fracs[i] / fracs[i - 1]
        keep, _ = train_test_split(
            current, train_size=target_frac, random_state=seed, stratify=y[current]
        )
        order[fracs[i]] = keep
        current = keep
    return order


def main():
    t0 = time.time()
    print("=" * 74)
    print("PHASE 9 STEP 3 -- learning curve. Does more training data help?")
    print("TEST SET NOT TOUCHED.")
    print("=" * 74)

    combined_idx = P9.load_combined_indices()
    concat_pool = P9.build_concat_pool(combined_idx)
    gated_pool = P9.build_gated_pool(combined_idx)
    xattn_pool = P9.build_xattn_pool(combined_idx)
    y_all = concat_pool["y"]
    n_all = len(y_all)
    print(f"Pool: {n_all} examples, distribution {np.bincount(y_all).tolist()}")

    smoothing_cpu = torch.tensor(build_smoothing_matrix(P9.NUM_CLASSES_3, TAU), device=GATED_DEVICE)
    smoothing_gpu = torch.tensor(build_smoothing_matrix(P9.NUM_CLASSES_3, TAU), device=XATTN_DEVICE)

    skf = StratifiedKFold(n_splits=P9.N_FOLDS, shuffle=True, random_state=P9.CV_RANDOM_STATE)

    # frac -> pooled OOF predictions across folds
    pooled_pred = {f: np.zeros(n_all, dtype=np.int64) for f in FRACTIONS}
    per_fold = []

    for fold_i, (train_pos, outer_pos) in enumerate(skf.split(np.zeros(n_all), y_all)):
        inner_tr_pos, inner_val_pos = train_test_split(
            train_pos, test_size=P9.INNER_VAL_FRACTION,
            random_state=P9.INNER_RANDOM_STATE, stratify=y_all[train_pos],
        )
        subsets = nested_stratified_subsets(inner_tr_pos, y_all, FRACTIONS, SUBSET_RANDOM_STATE)
        print(f"\n=== Fold {fold_i + 1}/{P9.N_FOLDS} (outer held-out n={len(outer_pos)}) ===")

        fold_entry = {"fold": fold_i, "n_outer": int(len(outer_pos)), "by_fraction": {}}

        for frac in FRACTIONS:
            tr_pos = subsets[frac]
            cw = P9.class_weights_3(y_all[tr_pos])
            cw_gpu = cw.to(XATTN_DEVICE)

            def parts(pool, keys):
                return {
                    "inner_train": P9.slice_pool(pool, keys, tr_pos),
                    "inner_val": P9.slice_pool(pool, keys, inner_val_pos),
                    "outer": P9.slice_pool(pool, keys, outer_pos),
                }

            probs = []
            cparts = parts(concat_pool, P9.CONCAT_KEYS)
            for s in SEEDS:
                probs.append(P9.train_concat_fold(s, cparts, cw, smoothing_cpu)["oof_probs"])
            gparts = P9.normalize_gated_conf(parts(gated_pool, P9.GATED_KEYS), fit_on="inner_train")
            for s in SEEDS:
                probs.append(P9.train_gated_fold(s, gparts, cw, smoothing_cpu)["oof_probs"])
            xparts = parts(xattn_pool, P9.XATTN_KEYS)
            for s in SEEDS:
                probs.append(P9.train_xattn_fold(s, xparts, cw_gpu, smoothing_gpu)["oof_probs"])
            del xparts

            pred = P9.majority_vote_fractions(probs, P9.NUM_CLASSES_3).argmax(axis=1)
            pooled_pred[frac][outer_pos] = pred
            y_out = y_all[outer_pos]
            acc = accuracy_score(y_out, pred)
            f1 = f1_score(y_out, pred, average="macro", zero_division=0)
            fold_entry["by_fraction"][str(frac)] = {
                "n_train": int(len(tr_pos)), "accuracy": float(acc), "macro_f1": float(f1),
            }
            print(f"  frac={frac:.1f} n_train={len(tr_pos):4d}  acc={acc:.4f}  macroF1={f1:.4f}")

        per_fold.append(fold_entry)

    # ------------------------------------------------------ pooled curve ----
    print("\n" + "=" * 74)
    print("POOLED LEARNING CURVE (all 777 out-of-fold predictions)")
    print("=" * 74)
    print(f"{'train n':>9}{'accuracy':>11}{'macro-F1':>11}")
    curve = {}
    for frac in FRACTIONS:
        pred = pooled_pred[frac]
        n_tr = int(np.mean([per_fold[i]["by_fraction"][str(frac)]["n_train"] for i in range(len(per_fold))]))
        acc = accuracy_score(y_all, pred)
        f1 = f1_score(y_all, pred, average="macro", zero_division=0)
        curve[str(frac)] = {"mean_n_train": n_tr, "accuracy": float(acc), "macro_f1": float(f1)}
        print(f"{n_tr:>9}{acc:>11.4f}{f1:>11.4f}")

    # Slope over the last segment, expressed per +100 training examples.
    a_lo, a_hi = curve[str(FRACTIONS[-2])], curve[str(FRACTIONS[-1])]
    d_n = a_hi["mean_n_train"] - a_lo["mean_n_train"]
    slope_acc = (a_hi["accuracy"] - a_lo["accuracy"]) / d_n * 100.0
    slope_f1 = (a_hi["macro_f1"] - a_lo["macro_f1"]) / d_n * 100.0

    # A1 asks: the locked models trained on 582; a final refit would use 777.
    extra = 777 - 582
    proj_acc = slope_acc * extra / 100.0
    proj_f1 = slope_f1 * extra / 100.0

    print(f"\nSlope over the final segment (+{d_n} examples):")
    print(f"  accuracy {slope_acc:+.4f} per +100 examples")
    print(f"  macro-F1 {slope_f1:+.4f} per +100 examples")
    print(f"\nTrack A1 projection -- refitting on 777 instead of 582 (+{extra} examples):")
    print(f"  projected accuracy gain {proj_acc:+.4f}")
    print(f"  projected macro-F1 gain {proj_f1:+.4f}")
    print("  (linear extrapolation of a curve that is probably concave, so treat")
    print("   this as an optimistic upper bound, not a promise)")

    verdict = ("still climbing -- more data helps, A1 and A2 justified" if slope_acc > 0.005
               else "flattening -- more data buys little, A1/A2 expected value is low")
    print(f"\nVerdict: {verdict}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Learning curve over nested stratified subsets of each fold's "
                     "inner-training rows. Inner-validation and outer held-out rows are "
                     "identical across all sizes, so only training-set size varies. "
                     "Test set not touched."),
            "fractions": FRACTIONS,
            "pooled_curve": curve,
            "per_fold": per_fold,
            "final_segment_slope_per_100_examples": {
                "accuracy": float(slope_acc), "macro_f1": float(slope_f1),
            },
            "track_a1_projection_582_to_777": {
                "extra_examples": extra,
                "projected_accuracy_gain": float(proj_acc),
                "projected_macro_f1_gain": float(proj_f1),
                "caveat": ("Linear extrapolation of a concave curve. Optimistic upper "
                           "bound, not a prediction."),
            },
            "verdict": verdict,
            "runtime_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")
    print(f"Runtime: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
