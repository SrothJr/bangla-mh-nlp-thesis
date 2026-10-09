"""
Phase 9, Step 9: validate Track A1 -- the full-data refit -- without
touching the test set.

THE QUESTION
------------
A1 proposes that the final models be trained on ALL available non-test data
rather than holding a slice back for early stopping. Step 3's learning curve
strongly supports the "more data helps" half of that claim: +0.030 accuracy
per +100 training examples, still climbing.

But A1 has a second half that Step 3 did not test. To train on everything,
you must give up the early-stopping signal and instead run a FIXED number of
epochs chosen in advance. That could easily backfire: a fixed schedule can
overshoot into overfitting or stop short, and the gain from extra data could
be wiped out by the loss of per-run stopping.

Every projection of what a Phase 9 test evaluation would produce depends on
this untested half. So it gets measured.

THE DESIGN
----------
Same 5 folds, same 15-model configuration (5 seeds x 3 architectures, soft
probability averaging) that Step 8 assembled. Within each fold:

  ARM 1  CURRENT PROTOCOL. Train on the inner-training rows only (about
         528), early-stop on the inner-validation rows (about 94), predict
         the outer held-out rows. Record the best epoch reached.

  ARM 2  A1 PROTOCOL. Train on the ENTIRE fold-training portion (about 622,
         which is inner-train plus inner-val), for a fixed number of epochs
         taken from Arm 1's mean best epoch for that architecture. No early
         stopping, because there is nothing held back to stop on. Predict
         the same outer held-out rows.

Both arms are scored on identical outer rows that neither ever trained on,
so the comparison is clean. The difference between them is exactly the A1
procedure: +18% training data, minus the early-stopping signal.

WHAT THE RESULT MEANS
---------------------
Arm 2 better  -> A1 works. The projections hold, and a final full-data fit
                 on all 777 is justified.
Arm 2 worse   -> A1 backfires. The extrapolation behind every "Phase 9 might
                 still beat the previous result" argument collapses, and the
                 case for a test evaluation goes with it.

Note the scale: this measures a step from about 528 to about 622 training
rows (+18%). The real A1 step is 582 to 777 (+34%), so a positive result
here is directional evidence for a larger jump, not a measurement of it.

TEST SET: not touched.

OUTPUT
------
outputs/phase9_15_a1_refit_validation_results.json
"""
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
)

from train_e6 import (
    HIDDEN_DIM, MAX_EPOCHS as GATED_MAX_EPOCHS, PATIENCE as GATED_PATIENCE,
    DEVICE as GATED_DEVICE,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU
from phase4_1_hyperparam_sweep import E6GatedOrthTunable
from phase3_2_cross_attention import CrossAttentionFusion, DEVICE as XATTN_DEVICE

import phase9_1_cv_baseline_3class as P9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_15_a1_refit_validation_results.json")

SEEDS_5 = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 12345
K = 3

STEP8_BEST_CV_ACC = 0.6744


class ConcatN(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768 + 1152, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, t, i):
        return self.net(torch.cat([t, i], dim=-1))


def fw_concat(m, p, dev):
    return m(torch.tensor(p["text"], device=dev), torch.tensor(p["image"], device=dev))


def fw_gated(m, p, dev):
    return m(torch.tensor(p["text"], device=dev), torch.tensor(p["image"], device=dev),
             torch.tensor(p["raw_text"], device=dev), torch.tensor(p["conf"], device=dev))


def fw_xattn(m, p, dev):
    return m(torch.tensor(p["tokens"], device=dev), torch.tensor(p["mask"], device=dev),
             torch.tensor(p["patches"], device=dev), torch.tensor(p["raw_text"], device=dev))


ARCH = {
    "concat": (lambda: ConcatN(K), fw_concat, GATED_DEVICE, 300, 30, P9.CONCAT_KEYS),
    "gated": (lambda: E6GatedOrthTunable(K, P9.BEST_GATED_CFG["hidden_dim"],
                                         P9.BEST_GATED_CFG["dropout"]),
              fw_gated, GATED_DEVICE, GATED_MAX_EPOCHS, GATED_PATIENCE, P9.GATED_KEYS),
    "xattn": (lambda: CrossAttentionFusion(K), fw_xattn, XATTN_DEVICE,
              GATED_MAX_EPOCHS, GATED_PATIENCE, P9.XATTN_KEYS),
}


def train_early_stop(seed, parts, build, fwd, dev, max_ep, patience, cw, sm):
    """Arm 1: early stopping on inner-val. Returns outer probs and best epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y_tr = torch.tensor(parts["inner_train"]["y"], device=dev)
    y_iv = parts["inner_val"]["y"]
    best_f1, best_probs, best_epoch = -1.0, None, 0
    stale = 0
    for ep in range(max_ep):
        model.train()
        opt.zero_grad()
        soft_target_loss(fwd(model, parts["inner_train"], dev), y_tr, sm, cw).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            f1 = f1_score(y_iv, fwd(model, parts["inner_val"], dev).argmax(dim=1).cpu().numpy(),
                          average="macro", zero_division=0)
            probs = torch.softmax(fwd(model, parts["outer"], dev), dim=1).cpu().numpy()
        if f1 > best_f1:
            best_f1, best_probs, best_epoch = f1, probs.copy(), ep
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_probs, best_epoch


def train_fixed_epochs(seed, parts, build, fwd, dev, n_epochs, cw, sm):
    """Arm 2: train on the full fold-training portion for a fixed schedule."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y_tr = torch.tensor(parts["full_train"]["y"], device=dev)
    for _ in range(max(n_epochs, 1)):
        model.train()
        opt.zero_grad()
        soft_target_loss(fwd(model, parts["full_train"], dev), y_tr, sm, cw).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(fwd(model, parts["outer"], dev), dim=1).cpu().numpy()


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


def paired(y, pa, pb, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    da, df = [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        ys = y[s]
        da.append(accuracy_score(ys, pa[s]) - accuracy_score(ys, pb[s]))
        df.append(f1_score(ys, pa[s], average="macro", zero_division=0)
                  - f1_score(ys, pb[s], average="macro", zero_division=0))
    da, df = np.array(da), np.array(df)
    return {"delta_accuracy_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "delta_macro_f1_ci95": [float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))],
            "prob_accuracy_improved": float((da > 0).mean()),
            "prob_macro_f1_improved": float((df > 0).mean())}


def main():
    t0 = time.time()
    print("=" * 74)
    print("PHASE 9 STEP 9 -- does the Track A1 full-data refit actually work?")
    print("TEST SET NOT TOUCHED.")
    print("=" * 74)

    combined_idx = P9.load_combined_indices()
    pools = {
        "concat": P9.build_concat_pool(combined_idx),
        "gated": P9.build_gated_pool(combined_idx),
        "xattn": P9.build_xattn_pool(combined_idx),
    }
    y_all = pools["concat"]["y"]
    n_all = len(y_all)
    print(f"Pool {n_all}, distribution {np.bincount(y_all).tolist()}")

    sm_cpu = torch.tensor(build_smoothing_matrix(K, TAU), device=GATED_DEVICE)
    sm_gpu = torch.tensor(build_smoothing_matrix(K, TAU), device=XATTN_DEVICE)
    skf = StratifiedKFold(n_splits=P9.N_FOLDS, shuffle=True, random_state=P9.CV_RANDOM_STATE)

    names = [f"{a}_seed{s}" for a in ARCH for s in SEEDS_5]
    oof_arm1 = {m: np.zeros((n_all, K), dtype=np.float32) for m in names}
    oof_arm2 = {m: np.zeros((n_all, K), dtype=np.float32) for m in names}
    epoch_log = []
    size_log = []

    for fold_i, (train_pos, outer_pos) in enumerate(skf.split(np.zeros(n_all), y_all)):
        inner_tr, inner_val = train_test_split(
            train_pos, test_size=P9.INNER_VAL_FRACTION,
            random_state=P9.INNER_RANDOM_STATE, stratify=y_all[train_pos])
        print(f"\n=== Fold {fold_i + 1}/{P9.N_FOLDS}: "
              f"arm1 trains on {len(inner_tr)}, arm2 trains on {len(train_pos)} "
              f"(+{len(train_pos) - len(inner_tr)}), both scored on {len(outer_pos)} ===")
        size_log.append({"fold": fold_i, "arm1_n_train": int(len(inner_tr)),
                         "arm2_n_train": int(len(train_pos)), "n_outer": int(len(outer_pos))})

        for arch, (build, fwd, dev, max_ep, pat, keys) in ARCH.items():
            pool = pools[arch]
            parts = {
                "inner_train": P9.slice_pool(pool, keys, inner_tr),
                "inner_val": P9.slice_pool(pool, keys, inner_val),
                "full_train": P9.slice_pool(pool, keys, train_pos),
                "outer": P9.slice_pool(pool, keys, outer_pos),
            }
            if arch == "gated":
                # Confidence z-scoring fitted on each arm's own training rows,
                # so neither arm sees statistics from data it did not train on.
                mean = parts["inner_train"]["conf_raw"].mean(axis=0, keepdims=True)
                std = parts["inner_train"]["conf_raw"].std(axis=0, keepdims=True) + 1e-6
                for nm in ("inner_train", "inner_val", "outer"):
                    parts[nm]["conf"] = (parts[nm]["conf_raw"] - mean) / std
                fmean = parts["full_train"]["conf_raw"].mean(axis=0, keepdims=True)
                fstd = parts["full_train"]["conf_raw"].std(axis=0, keepdims=True) + 1e-6
                parts["full_train"]["conf"] = (parts["full_train"]["conf_raw"] - fmean) / fstd
                parts["outer_full"] = dict(parts["outer"])
                parts["outer_full"]["conf"] = (parts["outer"]["conf_raw"] - fmean) / fstd

            cw1 = P9.class_weights_3(y_all[inner_tr])
            cw2 = P9.class_weights_3(y_all[train_pos])
            sm = sm_gpu if dev == XATTN_DEVICE else sm_cpu
            if dev == XATTN_DEVICE:
                cw1, cw2 = cw1.to(dev), cw2.to(dev)

            best_epochs = []
            for s in SEEDS_5:
                probs, ep = train_early_stop(s, parts, build, fwd, dev, max_ep, pat, cw1, sm)
                oof_arm1[f"{arch}_seed{s}"][outer_pos] = probs
                best_epochs.append(ep)
            fixed = int(round(float(np.mean(best_epochs))))
            epoch_log.append({"fold": fold_i, "arch": arch,
                              "arm1_best_epochs": best_epochs, "arm2_fixed_epochs": fixed})

            arm2_parts = dict(parts)
            if arch == "gated":
                arm2_parts["outer"] = parts["outer_full"]
            for s in SEEDS_5:
                oof_arm2[f"{arch}_seed{s}"][outer_pos] = train_fixed_epochs(
                    s, arm2_parts, build, fwd, dev, fixed, cw2, sm)
            print(f"  {arch}: arm1 best epochs {best_epochs} -> arm2 fixed at {fixed}")
            if dev == XATTN_DEVICE:
                torch.cuda.empty_cache()

    pred1 = np.stack([oof_arm1[m] for m in names]).mean(axis=0).argmax(axis=1)
    pred2 = np.stack([oof_arm2[m] for m in names]).mean(axis=0).argmax(axis=1)

    def summarize(pred):
        return {
            "accuracy": float(accuracy_score(y_all, pred)),
            "macro_f1": float(f1_score(y_all, pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_all, pred, labels=[0, 1, 2]).tolist(),
            "classification_report": classification_report(
                y_all, pred, labels=[0, 1, 2], target_names=P9.THREE_CLASS_NAMES,
                zero_division=0, output_dict=True),
            "bootstrap_ci95": bootstrap_ci(y_all, pred),
        }

    a1, a2 = summarize(pred1), summarize(pred2)
    cmp = paired(y_all, pred2, pred1)

    print("\n" + "-" * 74)
    print(f"{'arm':<52}{'acc':>9}{'macroF1':>10}")
    print("-" * 74)
    print(f"{'ARM 1  current protocol (early stopping, ~528 rows)':<52}"
          f"{a1['accuracy']:>9.4f}{a1['macro_f1']:>10.4f}")
    print(f"{'ARM 2  A1 protocol (fixed epochs, ~622 rows)':<52}"
          f"{a2['accuracy']:>9.4f}{a2['macro_f1']:>10.4f}")
    print("-" * 74)
    print(f"\ndelta accuracy {a2['accuracy'] - a1['accuracy']:+.4f}  "
          f"CI [{cmp['delta_accuracy_ci95'][0]:+.4f}, {cmp['delta_accuracy_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_accuracy_improved']:.3f}")
    print(f"delta macro-F1 {a2['macro_f1'] - a1['macro_f1']:+.4f}  "
          f"CI [{cmp['delta_macro_f1_ci95'][0]:+.4f}, {cmp['delta_macro_f1_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_macro_f1_improved']:.3f}")

    works = cmp["prob_accuracy_improved"] > 0.5
    print("\n" + "=" * 74)
    print("VERDICT ON TRACK A1")
    print("=" * 74)
    if works:
        print("  The full-data refit HELPS at this scale (+18% training rows).")
        print("  Directional support for the real A1 step (582 -> 777, +34%),")
        print("  though the size of that larger gain is still not measured.")
    else:
        print("  The full-data refit does NOT help at this scale.")
        print("  Losing the early-stopping signal cancels the extra data. Every")
        print("  projection that Phase 9 might still beat the previous best")
        print("  rested on this, so that argument does not survive.")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Measures the actual A1 procedure -- train on the full fold-training "
                     "portion for a fixed epoch count instead of holding rows back for "
                     "early stopping. Both arms scored on identical outer rows. "
                     "Test set not touched."),
            "scale_caveat": ("This tests ~528 -> ~622 rows (+18%). The real A1 step is "
                             "582 -> 777 (+34%), so a positive result is directional "
                             "evidence, not a measurement of the larger jump."),
            "n_examples": int(n_all),
            "fold_sizes": size_log,
            "epoch_schedules": epoch_log,
            "arm1_current_protocol": a1,
            "arm2_a1_full_data_fixed_epochs": a2,
            "paired_comparison": cmp,
            "a1_helps_at_this_scale": bool(works),
            "step8_best_cv_accuracy_reference": STEP8_BEST_CV_ACC,
            "test_set_touched": False,
            "runtime_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")
    print(f"Runtime: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
