"""
Phase 9, Step 7: train on 5 classes, predict 3. Does finer supervision beat
training directly on the coarse target?

WHERE THIS CAME FROM
--------------------
Phase 8's correction produced an observation that was never followed up.
When the old 5-class models' test predictions were collapsed into the
3-class grouping, they scored BETTER than the natively-trained 3-class
model on the identical 196 test records:

    5-class ensemble, collapsed to 3   accuracy 0.6582   macro-F1 0.6024
    native 3-class ensemble            accuracy 0.6071   macro-F1 0.5730

That gap, about +5 accuracy points, was reported at the time as a reason
the 3-class model is not an improvement. It was never examined as a
TECHNIQUE. If training on the finer label set and collapsing afterwards is
genuinely better, it is the largest single lever found in Phase 9, and it
costs nothing at all -- the 5-class labels already exist and are already
used by the rest of the project.

There is a plausible mechanism. Collapsing 5 labels into 3 throws away
supervision: "wish to be dead" and "suicide ideation" become one bucket, so
the model is no longer told to separate them. Keeping the finer distinctions
during training gives the network more to learn from on a dataset where
Step 3 showed supervision is the binding constraint. The coarse decision can
then be recovered by summing probabilities, with no information lost.

Against that, the single-split test comparison it rests on could easily be
noise: 196 examples, and the two numbers came from different runs with
different architectures in the ensemble. That is exactly why this script
measures it properly under cross-validation instead of taking it on trust.

WHAT IT DOES
------------
Identical folds, seeds, architectures, protocol and evaluation rows as
Step 1. The only difference is the training target:

  ARM A (fine->coarse)  train on 5-class labels, then collapse the predicted
                        probability vector by summing groups
                        {0}, {1,2}, {3,4} into a 3-class distribution.
  ARM B (native coarse) train on 3-class labels directly. This is exactly
                        Step 1, re-run here so both arms come from the same
                        process and are directly comparable.

Both arms are scored on the same 3-class ground truth over the same outer
held-out rows, with the same unbiased inner-split early stopping.

TEST SET: not touched.

OUTPUT
------
outputs/phase9_12_coarse_vs_fine_results.json
outputs/phase9_12_oof_probs.npz
"""
import os
import json
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, cohen_kappa_score,
)

from train_e6 import (
    load_labels, load_embeddings, load_confidences, SIGLIP_DIR, BANGLABERT_E3B_DIR,
    HIDDEN_DIM, SEEDS, MAX_EPOCHS as GATED_MAX_EPOCHS, PATIENCE as GATED_PATIENCE,
    DEVICE as GATED_DEVICE,
)
from train_contrastive_align import AlignmentProjections, CKPT_DIR as ALIGN_CKPT_DIR
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU
from phase4_1_hyperparam_sweep import E6GatedOrthTunable
from phase3_2_cross_attention import CrossAttentionFusion, load_patches, load_tokens, DEVICE as XATTN_DEVICE

import phase9_1_cv_baseline_3class as P9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_12_coarse_vs_fine_results.json")
OOF_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_12_oof_probs.npz")

NUM_5 = 5
NUM_3 = 3
GROUPS = [[0], [1, 2], [3, 4]]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 12345

# Phase 8's single-split test observation, for reference only.
REF_COLLAPSED_5CLASS_TEST_ACC = 0.6582
REF_NATIVE_3CLASS_TEST_ACC = 0.6071


def collapse_probs(probs5):
    """Sum the 5-class probability vector into the 3-class grouping."""
    out = np.zeros((probs5.shape[0], NUM_3), dtype=np.float32)
    for gi, members in enumerate(GROUPS):
        out[:, gi] = probs5[:, members].sum(axis=1)
    return out


def class_weights(y, k):
    counts = np.bincount(y, minlength=k).astype(np.float32)
    return torch.tensor(len(y) / (k * np.maximum(counts, 1)), dtype=torch.float32)


class ConcatN(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768 + 1152, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, t, i):
        return self.net(torch.cat([t, i], dim=-1))


def _generic_train(seed, parts, k, build, fwd, device, max_epochs, patience, cw, smoothing):
    """One model. Early stops on inner-val macro-F1 measured in the model's
    OWN label space, then returns probabilities on the outer rows."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y_tr = torch.tensor(parts["inner_train"]["y"], device=device)
    y_iv = parts["inner_val"]["y"]

    best_f1, best_probs = -1.0, None
    stale = 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        soft_target_loss(fwd(model, parts["inner_train"], device), y_tr, smoothing, cw).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            f1 = f1_score(y_iv, fwd(model, parts["inner_val"], device).argmax(dim=1).cpu().numpy(),
                          average="macro", zero_division=0)
            probs = torch.softmax(fwd(model, parts["outer"], device), dim=1).cpu().numpy()
        if f1 > best_f1:
            best_f1, best_probs = f1, probs.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_probs


def fw_concat(m, p, dev):
    return m(torch.tensor(p["text"], device=dev), torch.tensor(p["image"], device=dev))


def fw_gated(m, p, dev):
    return m(torch.tensor(p["text"], device=dev), torch.tensor(p["image"], device=dev),
             torch.tensor(p["raw_text"], device=dev), torch.tensor(p["conf"], device=dev))


def fw_xattn(m, p, dev):
    return m(torch.tensor(p["tokens"], device=dev), torch.tensor(p["mask"], device=dev),
             torch.tensor(p["patches"], device=dev), torch.tensor(p["raw_text"], device=dev))


def build_pools(combined_idx):
    labels = load_labels()
    y5 = np.array([labels[i] for i in combined_idx], dtype=np.int64)

    concat = {"text": load_embeddings(BANGLABERT_E3B_DIR, combined_idx),
              "image": load_embeddings(SIGLIP_DIR, combined_idx)}

    align = AlignmentProjections()
    ckpt = torch.load(os.path.join(ALIGN_CKPT_DIR, "current.pt"), map_location="cpu")
    align.load_state_dict(ckpt["state_dict"])
    align.eval()
    raw_text = torch.tensor(concat["text"])
    raw_image = torch.tensor(concat["image"])
    with torch.no_grad():
        at, ai = align(raw_text, raw_image)
    gated = {"text": at.numpy(), "image": ai.numpy(), "raw_text": raw_text.numpy(),
             "conf_raw": load_confidences(combined_idx)}

    tokens, masks = load_tokens(combined_idx)
    xattn = {"tokens": tokens, "mask": masks, "patches": load_patches(combined_idx),
             "raw_text": concat["text"]}
    return concat, gated, xattn, y5


def full_metrics(y, pred):
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1, 2], target_names=P9.THREE_CLASS_NAMES,
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
    print("PHASE 9 STEP 7 -- train on 5 classes and collapse, vs train on 3.")
    print("TEST SET NOT TOUCHED.")
    print("=" * 74)

    combined_idx = P9.load_combined_indices()
    concat_pool, gated_pool, xattn_pool, y5_all = build_pools(combined_idx)
    y3_all = P9.y3(y5_all)
    n_all = len(y3_all)
    print(f"Pool {n_all}.  5-class {np.bincount(y5_all).tolist()}  ->  "
          f"3-class {np.bincount(y3_all).tolist()}")

    names = ([f"concat_seed{s}" for s in SEEDS] + [f"gated_seed{s}" for s in SEEDS]
             + [f"xattn_seed{s}" for s in SEEDS])
    oof_fine = {m: np.zeros((n_all, NUM_3), dtype=np.float32) for m in names}
    oof_coarse = {m: np.zeros((n_all, NUM_3), dtype=np.float32) for m in names}

    sm5_cpu = torch.tensor(build_smoothing_matrix(NUM_5, TAU), device=GATED_DEVICE)
    sm5_gpu = torch.tensor(build_smoothing_matrix(NUM_5, TAU), device=XATTN_DEVICE)
    sm3_cpu = torch.tensor(build_smoothing_matrix(NUM_3, TAU), device=GATED_DEVICE)
    sm3_gpu = torch.tensor(build_smoothing_matrix(NUM_3, TAU), device=XATTN_DEVICE)

    skf = StratifiedKFold(n_splits=P9.N_FOLDS, shuffle=True, random_state=P9.CV_RANDOM_STATE)
    for fold_i, (train_pos, outer_pos) in enumerate(skf.split(np.zeros(n_all), y3_all)):
        inner_tr, inner_val = train_test_split(
            train_pos, test_size=P9.INNER_VAL_FRACTION,
            random_state=P9.INNER_RANDOM_STATE, stratify=y3_all[train_pos])
        print(f"\n=== Fold {fold_i + 1}/{P9.N_FOLDS} ===")

        def parts_for(pool, keys, y):
            d = {}
            for name, pos in (("inner_train", inner_tr), ("inner_val", inner_val), ("outer", outer_pos)):
                d[name] = {k: pool[k][pos] for k in keys}
                d[name]["y"] = y[pos]
            return d

        ckeys = ("text", "image")
        gkeys = ("text", "image", "raw_text", "conf_raw")
        xkeys = ("tokens", "mask", "patches", "raw_text")

        for arm, y_arm, k, sm_cpu, sm_gpu in (
            ("fine", y5_all, NUM_5, sm5_cpu, sm5_gpu),
            ("coarse", y3_all, NUM_3, sm3_cpu, sm3_gpu),
        ):
            cw = class_weights(y_arm[inner_tr], k)
            cw_gpu = cw.to(XATTN_DEVICE)
            store = oof_fine if arm == "fine" else oof_coarse

            cp = parts_for(concat_pool, ckeys, y_arm)
            gp = P9.normalize_gated_conf(parts_for(gated_pool, gkeys, y_arm), fit_on="inner_train")
            xp = parts_for(xattn_pool, xkeys, y_arm)

            for s in SEEDS:
                p = _generic_train(s, cp, k, lambda: ConcatN(k), fw_concat,
                                   GATED_DEVICE, 300, 30, cw, sm_cpu)
                store[f"concat_seed{s}"][outer_pos] = collapse_probs(p) if k == NUM_5 else p
                p = _generic_train(s, gp, k,
                                   lambda: E6GatedOrthTunable(k, P9.BEST_GATED_CFG["hidden_dim"],
                                                              P9.BEST_GATED_CFG["dropout"]),
                                   fw_gated, GATED_DEVICE, GATED_MAX_EPOCHS, GATED_PATIENCE, cw, sm_cpu)
                store[f"gated_seed{s}"][outer_pos] = collapse_probs(p) if k == NUM_5 else p
                p = _generic_train(s, xp, k, lambda: CrossAttentionFusion(k), fw_xattn,
                                   XATTN_DEVICE, GATED_MAX_EPOCHS, GATED_PATIENCE, cw_gpu, sm_gpu)
                store[f"xattn_seed{s}"][outer_pos] = collapse_probs(p) if k == NUM_5 else p
            del xp
            if XATTN_DEVICE == "cuda":
                torch.cuda.empty_cache()
            print(f"  arm '{arm}' done ({k}-class training)")

    pred_fine = P9.majority_vote_fractions([oof_fine[m] for m in names], NUM_3).argmax(axis=1)
    pred_coarse = P9.majority_vote_fractions([oof_coarse[m] for m in names], NUM_3).argmax(axis=1)

    fine, coarse = full_metrics(y3_all, pred_fine), full_metrics(y3_all, pred_coarse)
    fine["bootstrap_ci95"] = bootstrap_ci(y3_all, pred_fine)
    coarse["bootstrap_ci95"] = bootstrap_ci(y3_all, pred_coarse)
    cmp = paired(y3_all, pred_fine, pred_coarse)

    print("\n" + "-" * 74)
    print(f"{'arm':<40}{'acc':>9}{'macroF1':>10}")
    print("-" * 74)
    print(f"{'B: native 3-class training':<40}{coarse['accuracy']:>9.4f}{coarse['macro_f1']:>10.4f}")
    print(f"{'A: 5-class training, collapsed to 3':<40}{fine['accuracy']:>9.4f}{fine['macro_f1']:>10.4f}")
    print("-" * 74)
    print(f"\ndelta accuracy {fine['accuracy'] - coarse['accuracy']:+.4f}  "
          f"CI [{cmp['delta_accuracy_ci95'][0]:+.4f}, {cmp['delta_accuracy_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_accuracy_improved']:.3f}")
    print(f"delta macro-F1 {fine['macro_f1'] - coarse['macro_f1']:+.4f}  "
          f"CI [{cmp['delta_macro_f1_ci95'][0]:+.4f}, {cmp['delta_macro_f1_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_macro_f1_improved']:.3f}")

    print("\nPer-class, fine->coarse arm:")
    for c in P9.THREE_CLASS_NAMES:
        r = fine["classification_report"][c]
        print(f"  {c}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")

    print(f"\nPhase 8's single-split TEST observation that motivated this, for reference:")
    print(f"  collapsed 5-class {REF_COLLAPSED_5CLASS_TEST_ACC} vs native 3-class "
          f"{REF_NATIVE_3CLASS_TEST_ACC}  (delta +{REF_COLLAPSED_5CLASS_TEST_ACC - REF_NATIVE_3CLASS_TEST_ACC:.4f})")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Fine-to-coarse vs native-coarse training, identical folds, seeds, "
                     "architectures and unbiased protocol. Both scored on the same 3-class "
                     "ground truth. Test set not touched."),
            "n_examples": int(n_all),
            "class_distribution_5": np.bincount(y5_all).tolist(),
            "class_distribution_3": np.bincount(y3_all).tolist(),
            "arm_A_fine_to_coarse": fine,
            "arm_B_native_coarse": coarse,
            "paired_comparison": cmp,
            "phase8_single_split_test_reference": {
                "collapsed_5class_accuracy": REF_COLLAPSED_5CLASS_TEST_ACC,
                "native_3class_accuracy": REF_NATIVE_3CLASS_TEST_ACC,
            },
            "runtime_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    np.savez_compressed(OOF_PATH, y3=y3_all, y5=y5_all,
                        **{f"fine__{m}": oof_fine[m] for m in names},
                        **{f"coarse__{m}": oof_coarse[m] for m in names})
    print(f"\nSaved -> {RESULTS_PATH}")
    print(f"Runtime: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
