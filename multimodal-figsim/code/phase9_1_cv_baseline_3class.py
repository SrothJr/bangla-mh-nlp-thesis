"""
Phase 9, Step 1: the honest cross-validation baseline for the 3-class
harmonized task.

WHY THIS EXISTS
---------------
The locked 3-class configuration scored 0.6974 accuracy / 0.6649 macro-F1
on the 195-example validation split (phase7_9) but only 0.6071 accuracy /
0.5730 macro-F1 on test (phase7_10) -- a ~9-point drop on both metrics.
Phase 9's whole premise is that this gap, not model capacity, is what
stands between the project and ~70% test accuracy.

A large part of that gap is suspected to be SELECTION OPTIMISM: in the
original protocol every model early-stops by picking the epoch with the
best macro-F1 *on the very split it is then reported on*. That makes the
reported validation number an upper bound on itself, not a held-out
estimate.

This script measures that directly, and produces the unbiased baseline
every later Phase 9 step is compared against.

WHAT IT DOES
------------
5-fold stratified CV over train+val combined (777 examples). The test set
is NOT touched anywhere in this file.

For each outer fold, the 9-model locked configuration is retrained from
scratch (3 concat seeds + 3 gated+orth seeds at the Phase 7.8 winning
config hidden_dim=512 + 3 cross-attention seeds), and TWO numbers are
recorded per model:

  1. UNBIASED (primary). The fold's training portion is split again into
     an inner-train / inner-val pair. Early stopping uses inner-val only.
     The outer fold's held-out portion is then predicted exactly once,
     having played no part in any selection decision. These are the
     out-of-fold ("OOF") probabilities.

  2. PEEKING (diagnostic only, never used for any decision). During the
     same run, the best-epoch score measured directly on the outer
     held-out portion is also tracked. This reproduces the original
     protocol's selection bias at no extra training cost.

The difference between (1) and (2) is a direct measurement of how much
optimism the original protocol carried. That is a thesis-relevant finding
regardless of whether Phase 9 ever reaches 70%.

WHAT IT PRESERVES
-----------------
Nothing existing is modified. Every model class and helper is imported
from the existing scripts, not edited. The three per-fold training
functions below are adapted copies of the 3-class functions in
phase7_10_final_test_eval_3class.py, changed ONLY to (a) take an
already-sliced fold dict instead of the fixed train/val split and (b)
return probabilities on the held-out portion. Hyperparameters,
architectures, losses, optimizers, epoch caps and patience are all
identical to the locked configuration.

OUTPUTS
-------
outputs/phase9_1_cv_baseline_results.json   -- metrics, per-fold + overall
outputs/phase9_1_oof_probs.npz              -- OOF probabilities, needed by Step 2
"""
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, cohen_kappa_score,
)

from train_e6 import (
    load_labels, load_split, load_embeddings, load_confidences,
    SIGLIP_DIR, BANGLABERT_E3B_DIR, HIDDEN_DIM, SEEDS,
    MAX_EPOCHS as GATED_MAX_EPOCHS, PATIENCE as GATED_PATIENCE,
    DEVICE as GATED_DEVICE,
)
from train_contrastive_align import AlignmentProjections, CKPT_DIR as ALIGN_CKPT_DIR
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU
from phase4_1_hyperparam_sweep import E6GatedOrthTunable
from phase3_2_cross_attention import (
    CrossAttentionFusion, load_patches, load_tokens, DEVICE as XATTN_DEVICE,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_1_cv_baseline_results.json")
OOF_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_1_oof_probs.npz")

NUM_CLASSES_3 = 3
THREE_CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
BEST_GATED_CFG = {"hidden_dim": 512, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4}  # Phase 7.8 winner

N_FOLDS = 5
CV_RANDOM_STATE = 42
INNER_VAL_FRACTION = 0.15
INNER_RANDOM_STATE = 7
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 12345

# Reference points, for reporting only -- never used to make a decision here.
REF_VAL_ACC_PHASE7_9 = 0.6974
REF_VAL_MACRO_F1_PHASE7_9 = 0.6649
REF_TEST_ACC_PHASE7_10 = 0.6071
REF_TEST_MACRO_F1_PHASE7_10 = 0.5730


def y3(y5_array):
    return np.array([THREE_CLASS_MAP[y] for y in y5_array], dtype=np.int64)


def class_weights_3(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES_3).astype(np.float32)
    weights = len(y_train) / (NUM_CLASSES_3 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================ feature pools ==
def load_combined_indices():
    return load_split("train") + load_split("val")


def build_concat_pool(idx):
    labels = load_labels()
    return {
        "text": load_embeddings(BANGLABERT_E3B_DIR, idx),
        "image": load_embeddings(SIGLIP_DIR, idx),
        "y": y3(np.array([labels[i] for i in idx], dtype=np.int64)),
    }


def build_gated_pool(idx):
    align_model = AlignmentProjections()
    ckpt = torch.load(os.path.join(ALIGN_CKPT_DIR, "current.pt"), map_location="cpu")
    align_model.load_state_dict(ckpt["state_dict"])
    align_model.eval()
    raw_text = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, idx))
    raw_image = torch.tensor(load_embeddings(SIGLIP_DIR, idx))
    with torch.no_grad():
        aligned_text, aligned_image = align_model(raw_text, raw_image)
    labels = load_labels()
    return {
        "text": aligned_text.numpy(),
        "image": aligned_image.numpy(),
        "raw_text": raw_text.numpy(),
        "conf_raw": load_confidences(idx),
        "y": y3(np.array([labels[i] for i in idx], dtype=np.int64)),
    }


def build_xattn_pool(idx):
    labels = load_labels()
    tokens, masks = load_tokens(idx)
    return {
        "tokens": tokens,
        "mask": masks,
        "patches": load_patches(idx),
        "raw_text": load_embeddings(BANGLABERT_E3B_DIR, idx),
        "y": y3(np.array([labels[i] for i in idx], dtype=np.int64)),
    }


def slice_pool(pool, keys, positions):
    return {k: pool[k][positions] for k in keys}


CONCAT_KEYS = ("text", "image", "y")
GATED_KEYS = ("text", "image", "raw_text", "conf_raw", "y")
XATTN_KEYS = ("tokens", "mask", "patches", "raw_text", "y")


def normalize_gated_conf(parts, fit_on):
    """Confidence-feature z-scoring, fitted ONLY on the inner-training rows."""
    mean = parts[fit_on]["conf_raw"].mean(axis=0, keepdims=True)
    std = parts[fit_on]["conf_raw"].std(axis=0, keepdims=True) + 1e-6
    for name in parts:
        parts[name]["conf"] = (parts[name]["conf_raw"] - mean) / std
    return parts


# ================================================================== models ==
class ConcatClassifier3Class(nn.Module):
    """Verbatim copy of the locked 3-class concat model (phase7_10)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768 + 1152, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES_3),
        )

    def forward(self, text_vec, image_vec):
        return self.net(torch.cat([text_vec, image_vec], dim=-1))


def _track(best, val_f1, probs_outer, y_outer):
    """Shared early-stopping bookkeeping for the unbiased + peeking numbers."""
    improved = val_f1 > best["inner_f1"]
    if improved:
        best["inner_f1"] = val_f1
        best["oof_probs"] = probs_outer.copy()
    peek_f1 = f1_score(y_outer, probs_outer.argmax(axis=1), average="macro", zero_division=0)
    if peek_f1 > best["peek_f1"]:
        best["peek_f1"] = peek_f1
        best["peek_probs"] = probs_outer.copy()
    return improved


def _new_best():
    return {"inner_f1": -1.0, "oof_probs": None, "peek_f1": -1.0, "peek_probs": None}


def train_concat_fold(seed, parts, class_weights, smoothing):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConcatClassifier3Class()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xt_tr = torch.tensor(parts["inner_train"]["text"])
    Xi_tr = torch.tensor(parts["inner_train"]["image"])
    y_tr = torch.tensor(parts["inner_train"]["y"])
    Xt_iv = torch.tensor(parts["inner_val"]["text"])
    Xi_iv = torch.tensor(parts["inner_val"]["image"])
    y_iv = parts["inner_val"]["y"]
    Xt_out = torch.tensor(parts["outer"]["text"])
    Xi_out = torch.tensor(parts["outer"]["image"])
    y_out = parts["outer"]["y"]

    best = _new_best()
    stale = 0
    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        loss = soft_target_loss(model(Xt_tr, Xi_tr), y_tr, smoothing, class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            f1_inner = f1_score(y_iv, model(Xt_iv, Xi_iv).argmax(dim=1).numpy(),
                                average="macro", zero_division=0)
            probs_out = torch.softmax(model(Xt_out, Xi_out), dim=1).numpy()
        if _track(best, f1_inner, probs_out, y_out):
            stale = 0
        else:
            stale += 1
            if stale >= 30:
                break
    return best


def train_gated_fold(seed, parts, class_weights, smoothing):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = E6GatedOrthTunable(NUM_CLASSES_3, BEST_GATED_CFG["hidden_dim"],
                               BEST_GATED_CFG["dropout"]).to(GATED_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_GATED_CFG["lr"],
                                 weight_decay=BEST_GATED_CFG["weight_decay"])

    def tens(p):
        return (torch.tensor(p["text"], device=GATED_DEVICE),
                torch.tensor(p["image"], device=GATED_DEVICE),
                torch.tensor(p["raw_text"], device=GATED_DEVICE),
                torch.tensor(p["conf"], device=GATED_DEVICE))

    tr = tens(parts["inner_train"])
    y_tr = torch.tensor(parts["inner_train"]["y"], device=GATED_DEVICE)
    iv = tens(parts["inner_val"])
    y_iv = parts["inner_val"]["y"]
    out = tens(parts["outer"])
    y_out = parts["outer"]["y"]

    best = _new_best()
    stale = 0
    for _ in range(GATED_MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = soft_target_loss(model(*tr), y_tr, smoothing, class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            f1_inner = f1_score(y_iv, model(*iv).argmax(dim=1).cpu().numpy(),
                                average="macro", zero_division=0)
            probs_out = torch.softmax(model(*out), dim=1).cpu().numpy()
        if _track(best, f1_inner, probs_out, y_out):
            stale = 0
        else:
            stale += 1
            if stale >= GATED_PATIENCE:
                break
    return best


def train_xattn_fold(seed, parts, class_weights, smoothing):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CrossAttentionFusion(NUM_CLASSES_3).to(XATTN_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def tens(p):
        return (torch.tensor(p["tokens"], device=XATTN_DEVICE),
                torch.tensor(p["mask"], device=XATTN_DEVICE),
                torch.tensor(p["patches"], device=XATTN_DEVICE),
                torch.tensor(p["raw_text"], device=XATTN_DEVICE))

    tr = tens(parts["inner_train"])
    y_tr = torch.tensor(parts["inner_train"]["y"], device=XATTN_DEVICE)
    iv = tens(parts["inner_val"])
    y_iv = parts["inner_val"]["y"]
    out = tens(parts["outer"])
    y_out = parts["outer"]["y"]

    best = _new_best()
    stale = 0
    for _ in range(GATED_MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = soft_target_loss(model(*tr), y_tr, smoothing, class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            f1_inner = f1_score(y_iv, model(*iv).argmax(dim=1).cpu().numpy(),
                                average="macro", zero_division=0)
            probs_out = torch.softmax(model(*out), dim=1).cpu().numpy()
        if _track(best, f1_inner, probs_out, y_out):
            stale = 0
        else:
            stale += 1
            if stale >= GATED_PATIENCE:
                break
    del tr, iv, out
    if XATTN_DEVICE == "cuda":
        torch.cuda.empty_cache()
    return best


# ================================================================ metrics ===
def majority_vote_fractions(prob_list, num_classes):
    """The locked ensemble's aggregation: HARD vote fractions (Phase 8 audit)."""
    preds = np.stack([p.argmax(axis=1) for p in prob_list])
    n = preds.shape[1]
    fractions = np.zeros((n, num_classes))
    for c in range(num_classes):
        fractions[:, c] = (preds == c).sum(axis=0) / preds.shape[0]
    return fractions


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
        yt, yp = y_true[s], y_pred[s]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    return {
        "accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


# =================================================================== main ===
def main():
    t0 = time.time()
    print("=" * 74)
    print("PHASE 9 STEP 1 -- honest CV baseline, 3-class. TEST SET NOT TOUCHED.")
    print("=" * 74)

    combined_idx = load_combined_indices()
    print(f"Combined train+val pool: {len(combined_idx)} examples")

    print("Building feature pools (Stage A alignment frozen and shared, as in Phase 5.1)...")
    concat_pool = build_concat_pool(combined_idx)
    gated_pool = build_gated_pool(combined_idx)
    xattn_pool = build_xattn_pool(combined_idx)

    y_all = concat_pool["y"]
    assert np.array_equal(y_all, gated_pool["y"]), "label mismatch: concat vs gated pool"
    assert np.array_equal(y_all, xattn_pool["y"]), "label mismatch: concat vs xattn pool"
    print(f"3-class distribution: {np.bincount(y_all).tolist()}  "
          f"({', '.join(THREE_CLASS_NAMES)})")

    model_names = ([f"concat_seed{s}" for s in SEEDS]
                   + [f"gated_seed{s}" for s in SEEDS]
                   + [f"xattn_seed{s}" for s in SEEDS])
    n_all = len(y_all)
    oof_probs = {m: np.zeros((n_all, NUM_CLASSES_3), dtype=np.float32) for m in model_names}
    peek_probs = {m: np.zeros((n_all, NUM_CLASSES_3), dtype=np.float32) for m in model_names}

    smoothing_cpu = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=GATED_DEVICE)
    smoothing_gpu = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=XATTN_DEVICE)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_records = []

    for fold_i, (train_pos, outer_pos) in enumerate(skf.split(np.zeros(n_all), y_all)):
        # Inner split for early stopping. The outer rows play no part in it.
        inner_tr_pos, inner_val_pos = train_test_split(
            train_pos, test_size=INNER_VAL_FRACTION, random_state=INNER_RANDOM_STATE,
            stratify=y_all[train_pos],
        )
        print(f"\n=== Fold {fold_i + 1}/{N_FOLDS} ===")
        print(f"  inner-train={len(inner_tr_pos)}  inner-val={len(inner_val_pos)}  "
              f"outer held-out={len(outer_pos)}")

        cw = class_weights_3(y_all[inner_tr_pos])
        cw_gpu = cw.to(XATTN_DEVICE)

        def three_parts(pool, keys):
            return {
                "inner_train": slice_pool(pool, keys, inner_tr_pos),
                "inner_val": slice_pool(pool, keys, inner_val_pos),
                "outer": slice_pool(pool, keys, outer_pos),
            }

        # ---- concat
        cparts = three_parts(concat_pool, CONCAT_KEYS)
        for s in SEEDS:
            b = train_concat_fold(s, cparts, cw, smoothing_cpu)
            oof_probs[f"concat_seed{s}"][outer_pos] = b["oof_probs"]
            peek_probs[f"concat_seed{s}"][outer_pos] = b["peek_probs"]
            print(f"  concat seed {s}: inner-val F1={b['inner_f1']:.4f}  (peek F1={b['peek_f1']:.4f})")

        # ---- gated
        gparts = normalize_gated_conf(three_parts(gated_pool, GATED_KEYS), fit_on="inner_train")
        for s in SEEDS:
            b = train_gated_fold(s, gparts, cw, smoothing_cpu)
            oof_probs[f"gated_seed{s}"][outer_pos] = b["oof_probs"]
            peek_probs[f"gated_seed{s}"][outer_pos] = b["peek_probs"]
            print(f"  gated  seed {s}: inner-val F1={b['inner_f1']:.4f}  (peek F1={b['peek_f1']:.4f})")

        # ---- cross-attention
        xparts = three_parts(xattn_pool, XATTN_KEYS)
        for s in SEEDS:
            b = train_xattn_fold(s, xparts, cw_gpu, smoothing_gpu)
            oof_probs[f"xattn_seed{s}"][outer_pos] = b["oof_probs"]
            peek_probs[f"xattn_seed{s}"][outer_pos] = b["peek_probs"]
            print(f"  xattn  seed {s}: inner-val F1={b['inner_f1']:.4f}  (peek F1={b['peek_f1']:.4f})")
        del xparts

        y_out = y_all[outer_pos]
        fold_unb = majority_vote_fractions([oof_probs[m][outer_pos] for m in model_names],
                                           NUM_CLASSES_3).argmax(axis=1)
        fold_peek = majority_vote_fractions([peek_probs[m][outer_pos] for m in model_names],
                                            NUM_CLASSES_3).argmax(axis=1)
        rec = {
            "fold": fold_i,
            "n_inner_train": len(inner_tr_pos),
            "n_inner_val": len(inner_val_pos),
            "n_outer": len(outer_pos),
            "unbiased_accuracy": float(accuracy_score(y_out, fold_unb)),
            "unbiased_macro_f1": float(f1_score(y_out, fold_unb, average="macro", zero_division=0)),
            "peeking_accuracy": float(accuracy_score(y_out, fold_peek)),
            "peeking_macro_f1": float(f1_score(y_out, fold_peek, average="macro", zero_division=0)),
        }
        fold_records.append(rec)
        print(f"  FOLD {fold_i + 1} ensemble -> unbiased acc={rec['unbiased_accuracy']:.4f} "
              f"F1={rec['unbiased_macro_f1']:.4f} | peeking acc={rec['peeking_accuracy']:.4f} "
              f"F1={rec['peeking_macro_f1']:.4f}")

    # ------------------------------------------------ pooled OOF evaluation --
    print("\n" + "=" * 74)
    print("POOLED OUT-OF-FOLD RESULTS (all 777 examples, each predicted by a")
    print("model that never saw it in training or selection)")
    print("=" * 74)

    oof_pred = majority_vote_fractions([oof_probs[m] for m in model_names], NUM_CLASSES_3).argmax(axis=1)
    peek_pred = majority_vote_fractions([peek_probs[m] for m in model_names], NUM_CLASSES_3).argmax(axis=1)

    unbiased = full_metrics(y_all, oof_pred)
    peeking = full_metrics(y_all, peek_pred)
    unbiased_ci = bootstrap_ci(y_all, oof_pred)

    print(f"\nUNBIASED (primary baseline for all later Phase 9 steps):")
    print(f"  accuracy    = {unbiased['accuracy']:.4f}  "
          f"95% CI [{unbiased_ci['accuracy_ci95'][0]:.4f}, {unbiased_ci['accuracy_ci95'][1]:.4f}]")
    print(f"  macro-F1    = {unbiased['macro_f1']:.4f}  "
          f"95% CI [{unbiased_ci['macro_f1_ci95'][0]:.4f}, {unbiased_ci['macro_f1_ci95'][1]:.4f}]")
    print(f"  weighted-F1 = {unbiased['weighted_f1']:.4f}   QWK = {unbiased['quadratic_weighted_kappa']:.4f}")
    print(f"\nPEEKING (diagnostic only -- reproduces the original protocol's bias):")
    print(f"  accuracy    = {peeking['accuracy']:.4f}")
    print(f"  macro-F1    = {peeking['macro_f1']:.4f}")
    print(f"\nMEASURED SELECTION OPTIMISM:")
    print(f"  accuracy  {peeking['accuracy'] - unbiased['accuracy']:+.4f}")
    print(f"  macro-F1  {peeking['macro_f1'] - unbiased['macro_f1']:+.4f}")

    print("\nPer-class (unbiased):")
    for c in THREE_CLASS_NAMES:
        r = unbiased["classification_report"][c]
        print(f"  {c}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f} n={int(r['support'])}")
    print("\nConfusion matrix (unbiased, rows=true, cols=pred):")
    for row in unbiased["confusion_matrix"]:
        print("  ", row)

    print("\nReference points (NOT comparable targets -- different splits/protocols):")
    print(f"  phase7_9  single-split VALIDATION: acc={REF_VAL_ACC_PHASE7_9} F1={REF_VAL_MACRO_F1_PHASE7_9}")
    print(f"  phase7_10 locked TEST:             acc={REF_TEST_ACC_PHASE7_10} F1={REF_TEST_MACRO_F1_PHASE7_10}")

    individual = {}
    for m in model_names:
        individual[m] = {
            "unbiased_macro_f1": float(f1_score(y_all, oof_probs[m].argmax(axis=1),
                                                average="macro", zero_division=0)),
            "unbiased_accuracy": float(accuracy_score(y_all, oof_probs[m].argmax(axis=1))),
        }
    print("\nIndividual models (unbiased OOF):")
    for m, v in individual.items():
        print(f"  {m}: acc={v['unbiased_accuracy']:.4f} F1={v['unbiased_macro_f1']:.4f}")

    np.savez_compressed(
        OOF_PATH,
        y=y_all,
        combined_idx=np.array(combined_idx),
        model_names=np.array(model_names),
        **{f"oof__{m}": oof_probs[m] for m in model_names},
        **{f"peek__{m}": peek_probs[m] for m in model_names},
    )

    results = {
        "note": ("5-fold stratified CV over train+val (777). Test set NOT touched. "
                 "'unbiased' uses an inner split for early stopping so the outer "
                 "held-out rows play no part in any selection. 'peeking' selects the "
                 "best epoch on the outer rows themselves, reproducing the original "
                 "protocol's bias -- diagnostic only, never used for a decision."),
        "n_examples": int(n_all),
        "n_folds": N_FOLDS,
        "cv_random_state": CV_RANDOM_STATE,
        "inner_val_fraction": INNER_VAL_FRACTION,
        "class_distribution": np.bincount(y_all).tolist(),
        "fold_results": fold_records,
        "pooled_unbiased": unbiased,
        "pooled_unbiased_bootstrap_ci95": unbiased_ci,
        "pooled_peeking": peeking,
        "measured_selection_optimism": {
            "accuracy": float(peeking["accuracy"] - unbiased["accuracy"]),
            "macro_f1": float(peeking["macro_f1"] - unbiased["macro_f1"]),
        },
        "individual_models_unbiased": individual,
        "reference_points_not_comparable": {
            "phase7_9_single_split_validation_accuracy": REF_VAL_ACC_PHASE7_9,
            "phase7_9_single_split_validation_macro_f1": REF_VAL_MACRO_F1_PHASE7_9,
            "phase7_10_locked_test_accuracy": REF_TEST_ACC_PHASE7_10,
            "phase7_10_locked_test_macro_f1": REF_TEST_MACRO_F1_PHASE7_10,
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics  -> {RESULTS_PATH}")
    print(f"Saved OOF probs -> {OOF_PATH}")
    print(f"Runtime: {results['runtime_seconds']}s")


if __name__ == "__main__":
    main()
