"""
Phase 9, Step 6d (Track A2): does adding screened external negatives to the
training set improve the weakest class?

THE TARGET, AND WHY IT IS THIS ONE
----------------------------------
Step 1 measured class 0 ("no expressed severity") as the weak point:
F1 0.469, recall 0.419, from only 117 of 777 examples. Step 3 then measured
the learning curve and found it still climbing steeply, +0.030 accuracy per
+100 training examples. So the class-0 boundary is data-starved, and data is
the lever that actually moves this task.

DESIGN
------
Same folds, same seeds, same unbiased protocol as Step 1. Within each fold:

  1. Train baseline models on the fold's real inner-training rows only.
  2. LAYER 3 SCREEN. Use those baseline models -- which have seen only that
     fold's training data, so nothing leaks from the held-out rows -- to
     score every external candidate. Keep the ones most confidently
     predicted class 0, up to the 300 cap. Anything the model thinks might
     carry suicidal content is discarded rather than trusted.
  3. Train the treatment models on real inner-training rows PLUS those
     screened external rows, all labelled class 0.
  4. Early-stop both arms on the SAME real inner-validation rows, and score
     both on the SAME real outer held-out rows.

External rows appear only in training. They never enter validation and
never enter any held-out fold, so every number reported here is measured
purely on real in-domain data.

The architecture is the simple-concat model alone. The gated architecture
needs confidence features derived from the reasoning pass, and
cross-attention needs token and patch tensors, neither of which exists for
external memes. Scoping this to concat keeps the A/B clean and is stated as
a limitation rather than hidden.

Both arms use the OCR-only text feature, for the reason given in
phase9_10_external_features.py: the local vision model could not produce
reliable reasoning text for external memes, and giving external rows an
empty reasoning segment would hand the classifier a trivial shortcut.

TEST SET: not touched.

OUTPUT
------
outputs/phase9_11_external_negatives_ab_results.json
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

from train_e6 import load_labels, load_embeddings, SIGLIP_DIR, HIDDEN_DIM, SEEDS, DEVICE as DEV
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU

import phase9_1_cv_baseline_3class as P9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_ROOT = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
REAL_TEXT_DIR = os.path.join(EMB_ROOT, "banglabert_ocr")
EXT_TEXT_DIR = os.path.join(EMB_ROOT, "phase9_external_ocr")
EXT_IMAGE_DIR = os.path.join(EMB_ROOT, "phase9_external_siglip")
MANIFEST = os.path.join(PROJECT_ROOT, "outputs", "phase9_10_external_feature_manifest.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_11_external_negatives_ab_results.json")

MAX_EXTERNAL = 300
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 12345
TEXT_DIM, IMAGE_DIM = 768, 1152


class Concat3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_DIM + IMAGE_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, P9.NUM_CLASSES_3),
        )

    def forward(self, t, i):
        return self.net(torch.cat([t, i], dim=-1))


def train_model(seed, Xt, Xi, y, Xt_iv, Xi_iv, y_iv, Xt_out, Xi_out, class_weights, smoothing):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Concat3().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1, best_probs, best_state = -1.0, None, None
    stale = 0
    for _ in range(300):
        model.train()
        opt.zero_grad()
        soft_target_loss(model(Xt, Xi), y, smoothing, class_weights).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            f1 = f1_score(y_iv, model(Xt_iv, Xi_iv).argmax(dim=1).cpu().numpy(),
                          average="macro", zero_division=0)
            probs = torch.softmax(model(Xt_out, Xi_out), dim=1).cpu().numpy()
        if f1 > best_f1:
            best_f1, best_probs = f1, probs.copy()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 30:
                break
    return best_probs, best_state


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
    da, df, d0 = [], [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        ys = y[s]
        da.append(accuracy_score(ys, pa[s]) - accuracy_score(ys, pb[s]))
        df.append(f1_score(ys, pa[s], average="macro", zero_division=0)
                  - f1_score(ys, pb[s], average="macro", zero_division=0))
        d0.append(f1_score(ys, pa[s], average=None, labels=[0], zero_division=0)[0]
                  - f1_score(ys, pb[s], average=None, labels=[0], zero_division=0)[0])
    da, df, d0 = np.array(da), np.array(df), np.array(d0)
    return {"delta_accuracy_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "delta_macro_f1_ci95": [float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))],
            "delta_class0_f1_ci95": [float(np.percentile(d0, 2.5)), float(np.percentile(d0, 97.5))],
            "prob_accuracy_improved": float((da > 0).mean()),
            "prob_macro_f1_improved": float((df > 0).mean()),
            "prob_class0_f1_improved": float((d0 > 0).mean())}


def main():
    t0 = time.time()
    print("=" * 74)
    print("PHASE 9 STEP 6d (Track A2) -- external negatives A/B. TEST NOT TOUCHED.")
    print("=" * 74)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        stems = json.load(f)["stems"]
    ext_text = np.stack([np.load(os.path.join(EXT_TEXT_DIR, f"{s}.npy")) for s in stems]).astype(np.float32)
    ext_img = np.stack([np.load(os.path.join(EXT_IMAGE_DIR, f"{s}.npy")) for s in stems]).astype(np.float32)
    print(f"External pool available: {len(stems)}  (cap per fold: {MAX_EXTERNAL})")

    combined_idx = P9.load_combined_indices()
    labels = load_labels()
    y_all = P9.y3(np.array([labels[i] for i in combined_idx], dtype=np.int64))
    real_text = load_embeddings(REAL_TEXT_DIR, combined_idx).astype(np.float32)
    real_img = load_embeddings(SIGLIP_DIR, combined_idx).astype(np.float32)
    n_all = len(y_all)
    print(f"Real pool: {n_all}, distribution {np.bincount(y_all).tolist()}")

    smoothing = torch.tensor(build_smoothing_matrix(P9.NUM_CLASSES_3, TAU), device=DEV)
    skf = StratifiedKFold(n_splits=P9.N_FOLDS, shuffle=True, random_state=P9.CV_RANDOM_STATE)

    oof_base = {s: np.zeros((n_all, P9.NUM_CLASSES_3), dtype=np.float32) for s in SEEDS}
    oof_ext = {s: np.zeros((n_all, P9.NUM_CLASSES_3), dtype=np.float32) for s in SEEDS}
    screen_stats = []

    T = lambda a: torch.tensor(a, device=DEV)
    ext_text_t, ext_img_t = T(ext_text), T(ext_img)

    for fold_i, (train_pos, outer_pos) in enumerate(skf.split(np.zeros(n_all), y_all)):
        inner_tr, inner_val = train_test_split(
            train_pos, test_size=P9.INNER_VAL_FRACTION,
            random_state=P9.INNER_RANDOM_STATE, stratify=y_all[train_pos])
        print(f"\n=== Fold {fold_i + 1}/{P9.N_FOLDS} ===")

        Xt_tr, Xi_tr = T(real_text[inner_tr]), T(real_img[inner_tr])
        y_tr = T(y_all[inner_tr])
        Xt_iv, Xi_iv = T(real_text[inner_val]), T(real_img[inner_val])
        Xt_out, Xi_out = T(real_text[outer_pos]), T(real_img[outer_pos])
        cw = P9.class_weights_3(y_all[inner_tr])

        # ---- arm A: baseline, no external rows
        states = []
        for s in SEEDS:
            probs, state = train_model(s, Xt_tr, Xi_tr, y_tr, Xt_iv, Xi_iv, y_all[inner_val],
                                       Xt_out, Xi_out, cw, smoothing)
            oof_base[s][outer_pos] = probs
            states.append(state)

        # ---- layer 3 screen, using THIS fold's baseline models only
        with torch.no_grad():
            votes = []
            for state in states:
                m = Concat3().to(DEV)
                m.load_state_dict(state)
                m.eval()
                votes.append(torch.softmax(m(ext_text_t, ext_img_t), dim=1).cpu().numpy())
            ext_probs = np.mean(np.stack(votes), axis=0)
        # Most confidently "no expressed severity" first; anything the model
        # thinks might be suicidal content is dropped rather than trusted.
        risk = ext_probs[:, 1] + ext_probs[:, 2]
        order = np.argsort(risk)
        keep = order[:MAX_EXTERNAL]
        screen_stats.append({
            "fold": fold_i,
            "n_kept": int(len(keep)),
            "mean_risk_kept": float(risk[keep].mean()),
            "mean_risk_dropped": float(risk[order[MAX_EXTERNAL:]].mean()) if len(order) > MAX_EXTERNAL else None,
            "max_risk_kept": float(risk[keep].max()),
        })
        print(f"  screen kept {len(keep)} external rows "
              f"(mean predicted non-class-0 mass {risk[keep].mean():.3f}, max {risk[keep].max():.3f})")

        # ---- arm B: baseline + screened external rows, all labelled class 0
        Xt_aug = torch.cat([Xt_tr, ext_text_t[keep]], dim=0)
        Xi_aug = torch.cat([Xi_tr, ext_img_t[keep]], dim=0)
        y_aug = torch.cat([y_tr, torch.zeros(len(keep), dtype=torch.int64, device=DEV)], dim=0)
        cw_aug = P9.class_weights_3(y_aug.cpu().numpy())
        for s in SEEDS:
            probs, _ = train_model(s, Xt_aug, Xi_aug, y_aug, Xt_iv, Xi_iv, y_all[inner_val],
                                   Xt_out, Xi_out, cw_aug, smoothing)
            oof_ext[s][outer_pos] = probs
        print(f"  trained both arms (real {len(inner_tr)} vs real+external {len(inner_tr) + len(keep)})")

    pred_base = P9.majority_vote_fractions([oof_base[s] for s in SEEDS], P9.NUM_CLASSES_3).argmax(axis=1)
    pred_ext = P9.majority_vote_fractions([oof_ext[s] for s in SEEDS], P9.NUM_CLASSES_3).argmax(axis=1)

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

    base, ext = summarize(pred_base), summarize(pred_ext)
    cmp = paired(y_all, pred_ext, pred_base)

    print("\n" + "-" * 74)
    print(f"{'arm':<34}{'acc':>9}{'macroF1':>10}{'class0 F1':>11}")
    print("-" * 74)
    for label, r in (("baseline (real only)", base), ("+ external negatives", ext)):
        c0 = r["classification_report"][P9.THREE_CLASS_NAMES[0]]["f1-score"]
        print(f"{label:<34}{r['accuracy']:>9.4f}{r['macro_f1']:>10.4f}{c0:>11.4f}")
    print("-" * 74)
    print(f"\ndelta accuracy  {ext['accuracy'] - base['accuracy']:+.4f}  "
          f"CI [{cmp['delta_accuracy_ci95'][0]:+.4f}, {cmp['delta_accuracy_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_accuracy_improved']:.3f}")
    print(f"delta macro-F1  {ext['macro_f1'] - base['macro_f1']:+.4f}  "
          f"CI [{cmp['delta_macro_f1_ci95'][0]:+.4f}, {cmp['delta_macro_f1_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_macro_f1_improved']:.3f}")
    c0b = base["classification_report"][P9.THREE_CLASS_NAMES[0]]["f1-score"]
    c0e = ext["classification_report"][P9.THREE_CLASS_NAMES[0]]["f1-score"]
    print(f"delta class-0 F1 {c0e - c0b:+.4f}  "
          f"CI [{cmp['delta_class0_f1_ci95'][0]:+.4f}, {cmp['delta_class0_f1_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_class0_f1_improved']:.3f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Track A2 A/B. External BN-HIB rows are TRAINING-ONLY, never in "
                     "validation or any held-out fold. Layer-3 screen uses each fold's own "
                     "baseline models, so nothing leaks from held-out rows. Concat "
                     "architecture and OCR-only text for both arms. Test set not touched."),
            "n_real": int(n_all),
            "external_pool": len(stems),
            "max_external_per_fold": MAX_EXTERNAL,
            "architecture": "simple concat only (gated needs reasoning-derived confidence "
                            "features; cross-attention needs token/patch tensors -- neither "
                            "exists for external memes)",
            "text_feature": "banglabert_ocr (OCR-only) for BOTH arms",
            "baseline_real_only": base,
            "with_external_negatives": ext,
            "paired_comparison": cmp,
            "screen_stats_per_fold": screen_stats,
            "runtime_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")
    print(f"Runtime: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
