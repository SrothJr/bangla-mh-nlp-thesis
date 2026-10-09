"""
Phase 9, Step 4: two independent A/B tests under the Step 1 fold structure.

B3  MORE SEEDS -- 5 per architecture (15 models) instead of 3 (9 models).
    Pure variance reduction. Ensembling has been the most reliable lever in
    this project's history, and a large validation-to-test gap is exactly
    what variance reduction is for. No new ideas, just more of what works.

C1  CORN LOSS -- a genuine cumulative-link ordinal formulation
    (IMPROVEMENT_PLAN.md item 5.5, open since Phase 5, never attempted).
    The task is ordinal: no severity < thought or desire < high acuity.
    CORAL was tried and rejected on Day 4; ordinal label smoothing was tried
    and kept in Phase 1.3. A conditional-training cumulative-link approach
    is mechanistically different from both, not a rehash.

    CORN reformulates K-way ordinal classification as K-1 binary tasks:
      task 0: P(y > 0), trained on all examples
      task 1: P(y > 1), trained ONLY on examples with y > 0
    The conditional subsetting is what distinguishes CORN from CORAL and is
    what makes its probability estimates coherent. Prediction chains the
    conditional probabilities and counts how many exceed 0.5.

Each item is A/B'd INDEPENDENTLY against the Step 1 baseline, never bundled.
This project's experiment log is valuable precisely because every change is
individually attributed, and bundled changes cannot be.

TEST SET: not touched.

OUTPUT
------
outputs/phase9_4_seeds_corn_results.json
"""
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

from train_e6 import DEVICE as GATED_DEVICE, HIDDEN_DIM, MAX_EPOCHS as GATED_MAX_EPOCHS, PATIENCE as GATED_PATIENCE
from phase1_3_ordinal_smoothing import build_smoothing_matrix, TAU
from phase4_1_hyperparam_sweep import E6GatedOrthTunable
from phase3_2_cross_attention import CrossAttentionFusion, DEVICE as XATTN_DEVICE

import phase9_1_cv_baseline_3class as P9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_4_seeds_corn_results.json")

SEEDS_3 = [0, 1, 2]
SEEDS_5 = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 12345

BASELINE_ACC = 0.6654      # Step 1 pooled unbiased
BASELINE_F1 = 0.6155


# ==================================================================== CORN ==
def corn_loss(logits, y, num_classes):
    """CORN loss (Shi, Cao & Raschka). logits: (B, K-1).

    Task k predicts P(y > k), and is trained only on the conditional subset
    {y >= k}. Task 0's subset is everything, so the chain is well defined.
    """
    total = 0.0
    n_terms = 0
    for k in range(num_classes - 1):
        subset = y >= k
        if subset.sum() == 0:
            continue
        target = (y[subset] > k).float()
        total = total + F.binary_cross_entropy_with_logits(
            logits[subset, k], target, reduction="mean"
        )
        n_terms += 1
    return total / max(n_terms, 1)


def corn_probs_to_class(logits, num_classes):
    """Chain conditional probabilities into cumulative P(y > k), then count."""
    cond = torch.sigmoid(logits)                    # (B, K-1) = P(y>k | y>=k)
    cum = torch.cumprod(cond, dim=1)                # (B, K-1) = P(y > k)
    return (cum > 0.5).sum(dim=1)


def corn_class_probs(logits, num_classes):
    """Turn the cumulative chain into a proper K-way distribution, so CORN
    models can be ensembled with the same vote machinery as the others."""
    cond = torch.sigmoid(logits)
    cum = torch.cumprod(cond, dim=1)                # P(y>0), P(y>1)
    ones = torch.ones(logits.shape[0], 1, device=logits.device)
    zeros = torch.zeros(logits.shape[0], 1, device=logits.device)
    upper = torch.cat([ones, cum], dim=1)           # P(y >= k)
    lower = torch.cat([cum, zeros], dim=1)          # P(y >= k+1)
    return (upper - lower).clamp(min=1e-8)          # P(y == k)


class ConcatOrdinal(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768 + 1152, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, text_vec, image_vec):
        return self.net(torch.cat([text_vec, image_vec], dim=-1))


def _train_corn_generic(seed, parts, num_classes, build_model, forward, device, max_epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    y_tr = torch.tensor(parts["inner_train"]["y"], device=device)
    y_iv = parts["inner_val"]["y"]

    best_f1, best_probs = -1.0, None
    stale = 0
    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = corn_loss(forward(model, parts["inner_train"], device), y_tr, num_classes)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_iv = corn_probs_to_class(
                forward(model, parts["inner_val"], device), num_classes
            ).cpu().numpy()
            f1 = f1_score(y_iv, pred_iv, average="macro", zero_division=0)
            probs_out = corn_class_probs(
                forward(model, parts["outer"], device), num_classes
            ).cpu().numpy()
        if f1 > best_f1:
            best_f1, best_probs = f1, probs_out.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_probs


def fw_concat(model, p, device):
    return model(torch.tensor(p["text"], device=device), torch.tensor(p["image"], device=device))


def fw_gated(model, p, device):
    return model(torch.tensor(p["text"], device=device), torch.tensor(p["image"], device=device),
                 torch.tensor(p["raw_text"], device=device), torch.tensor(p["conf"], device=device))


def fw_xattn(model, p, device):
    return model(torch.tensor(p["tokens"], device=device), torch.tensor(p["mask"], device=device),
                 torch.tensor(p["patches"], device=device), torch.tensor(p["raw_text"], device=device))


# ================================================================ measuring =
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


def paired(y, pred_a, pred_b, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    da, df = [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        ys = y[s]
        da.append(accuracy_score(ys, pred_a[s]) - accuracy_score(ys, pred_b[s]))
        df.append(f1_score(ys, pred_a[s], average="macro", zero_division=0)
                  - f1_score(ys, pred_b[s], average="macro", zero_division=0))
    da, df = np.array(da), np.array(df)
    return {"delta_accuracy_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "delta_macro_f1_ci95": [float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))],
            "prob_accuracy_improved": float((da > 0).mean()),
            "prob_macro_f1_improved": float((df > 0).mean())}


# ===================================================================== main ==
def main():
    t0 = time.time()
    print("=" * 74)
    print("PHASE 9 STEP 4 -- B3 (more seeds) and C1 (CORN loss), A/B'd separately")
    print("TEST SET NOT TOUCHED.")
    print("=" * 74)

    combined_idx = P9.load_combined_indices()
    concat_pool = P9.build_concat_pool(combined_idx)
    gated_pool = P9.build_gated_pool(combined_idx)
    xattn_pool = P9.build_xattn_pool(combined_idx)
    y_all = concat_pool["y"]
    n_all = len(y_all)
    K = P9.NUM_CLASSES_3

    smoothing_cpu = torch.tensor(build_smoothing_matrix(K, TAU), device=GATED_DEVICE)
    smoothing_gpu = torch.tensor(build_smoothing_matrix(K, TAU), device=XATTN_DEVICE)
    skf = StratifiedKFold(n_splits=P9.N_FOLDS, shuffle=True, random_state=P9.CV_RANDOM_STATE)

    names_5 = ([f"concat_seed{s}" for s in SEEDS_5] + [f"gated_seed{s}" for s in SEEDS_5]
               + [f"xattn_seed{s}" for s in SEEDS_5])
    corn_names = ([f"concat_seed{s}" for s in SEEDS_3] + [f"gated_seed{s}" for s in SEEDS_3]
                  + [f"xattn_seed{s}" for s in SEEDS_3])
    oof_ce = {m: np.zeros((n_all, K), dtype=np.float32) for m in names_5}
    oof_corn = {m: np.zeros((n_all, K), dtype=np.float32) for m in corn_names}

    for fold_i, (train_pos, outer_pos) in enumerate(skf.split(np.zeros(n_all), y_all)):
        inner_tr_pos, inner_val_pos = train_test_split(
            train_pos, test_size=P9.INNER_VAL_FRACTION,
            random_state=P9.INNER_RANDOM_STATE, stratify=y_all[train_pos])
        print(f"\n=== Fold {fold_i + 1}/{P9.N_FOLDS} ===")
        cw = P9.class_weights_3(y_all[inner_tr_pos])
        cw_gpu = cw.to(XATTN_DEVICE)

        def parts(pool, keys):
            return {"inner_train": P9.slice_pool(pool, keys, inner_tr_pos),
                    "inner_val": P9.slice_pool(pool, keys, inner_val_pos),
                    "outer": P9.slice_pool(pool, keys, outer_pos)}

        cparts = parts(concat_pool, P9.CONCAT_KEYS)
        gparts = P9.normalize_gated_conf(parts(gated_pool, P9.GATED_KEYS), fit_on="inner_train")
        xparts = parts(xattn_pool, P9.XATTN_KEYS)

        # ---- B3: cross-entropy models, 5 seeds each
        for s in SEEDS_5:
            oof_ce[f"concat_seed{s}"][outer_pos] = P9.train_concat_fold(s, cparts, cw, smoothing_cpu)["oof_probs"]
            oof_ce[f"gated_seed{s}"][outer_pos] = P9.train_gated_fold(s, gparts, cw, smoothing_cpu)["oof_probs"]
            oof_ce[f"xattn_seed{s}"][outer_pos] = P9.train_xattn_fold(s, xparts, cw_gpu, smoothing_gpu)["oof_probs"]
        print(f"  B3: 15 cross-entropy models trained")

        # ---- C1: CORN models, 3 seeds each
        for s in SEEDS_3:
            oof_corn[f"concat_seed{s}"][outer_pos] = _train_corn_generic(
                s, cparts, K, lambda: ConcatOrdinal(K - 1), fw_concat, GATED_DEVICE, 300, 30)
            oof_corn[f"gated_seed{s}"][outer_pos] = _train_corn_generic(
                s, gparts, K, lambda: E6GatedOrthTunable(K - 1, P9.BEST_GATED_CFG["hidden_dim"],
                                                         P9.BEST_GATED_CFG["dropout"]),
                fw_gated, GATED_DEVICE, GATED_MAX_EPOCHS, GATED_PATIENCE)
            oof_corn[f"xattn_seed{s}"][outer_pos] = _train_corn_generic(
                s, xparts, K, lambda: CrossAttentionFusion(K - 1), fw_xattn, XATTN_DEVICE,
                GATED_MAX_EPOCHS, GATED_PATIENCE)
        print(f"  C1: 9 CORN models trained")
        del xparts
        if XATTN_DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ evaluate --
    def ens(probs_dict, names):
        return P9.majority_vote_fractions([probs_dict[m] for m in names], K).argmax(axis=1)

    names_3 = ([f"concat_seed{s}" for s in SEEDS_3] + [f"gated_seed{s}" for s in SEEDS_3]
               + [f"xattn_seed{s}" for s in SEEDS_3])
    pred_3seed = ens(oof_ce, names_3)
    pred_5seed = ens(oof_ce, names_5)
    pred_corn = ens(oof_corn, corn_names)
    pred_corn_plus_ce = P9.majority_vote_fractions(
        [oof_ce[m] for m in names_3] + [oof_corn[m] for m in corn_names], K).argmax(axis=1)

    entries = {
        "baseline_3seeds_9models": pred_3seed,
        "B3_5seeds_15models": pred_5seed,
        "C1_corn_9models": pred_corn,
        "C1_corn_plus_ce_18models": pred_corn_plus_ce,
    }

    print("\n" + "-" * 74)
    print(f"{'method':<34}{'acc':>9}{'macroF1':>10}{'dAcc':>9}{'P(better)':>11}")
    print("-" * 74)
    results = {}
    for label, pred in entries.items():
        acc = accuracy_score(y_all, pred)
        f1 = f1_score(y_all, pred, average="macro", zero_division=0)
        e = {"accuracy": float(acc), "macro_f1": float(f1), "bootstrap_ci95": bootstrap_ci(y_all, pred)}
        if label != "baseline_3seeds_9models":
            e["paired_vs_baseline"] = paired(y_all, pred, pred_3seed)
            print(f"{label:<34}{acc:>9.4f}{f1:>10.4f}{acc - results['baseline_3seeds_9models']['accuracy']:>+9.4f}"
                  f"{e['paired_vs_baseline']['prob_accuracy_improved']:>11.3f}")
        else:
            print(f"{label:<34}{acc:>9.4f}{f1:>10.4f}{'--':>9}{'--':>11}")
        results[label] = e
    print("-" * 74)
    print(f"\nStep 1 reference: acc={BASELINE_ACC:.4f} F1={BASELINE_F1:.4f}")

    best = max(results, key=lambda k: results[k]["accuracy"])
    print(f"Best by accuracy: {best} ({results[best]['accuracy']:.4f})")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("B3 (5 seeds vs 3) and C1 (CORN ordinal loss) A/B'd independently "
                     "under the same folds as Step 1. Not bundled. Test set not touched."),
            "n_examples": int(n_all),
            "results": results,
            "step1_reference": {"accuracy": BASELINE_ACC, "macro_f1": BASELINE_F1},
            "best_by_accuracy": best,
            "runtime_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    np.savez_compressed(
        os.path.join(PROJECT_ROOT, "outputs", "phase9_4_oof_probs.npz"),
        y=y_all,
        **{f"ce__{m}": oof_ce[m] for m in names_5},
        **{f"corn__{m}": oof_corn[m] for m in corn_names},
    )
    print(f"\nSaved -> {RESULTS_PATH}")
    print(f"Runtime: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
