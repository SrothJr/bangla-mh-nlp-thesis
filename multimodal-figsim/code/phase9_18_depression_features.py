"""
Phase 9, Step 10: use the depression classifier's output as INPUT FEATURES
for the suicide-severity model.

THE IDEA
--------
The two halves of this system currently meet only at the very end, in the
fuzzy-logic layer. The depression classifier produces a 4-class distribution
per meme, that distribution is genuinely informative about suicide severity,
and the suicide model never sees it.

Wiring it in as four extra input features is cheap, has never been tried,
and is a deeper form of integration than a rule table applied after both
models have already decided. That makes it worth testing on its own merits
even if the gain is small.

WHAT IS AND IS NOT TOUCHED
--------------------------
The depression classifier is NOT retrained, modified, or fine-tuned. It is
called through the exact parity-tested wrapper from Phase 8 Step 2
(`phase8_2_depression_parity_wrapper.predict_depression`), which was
verified to reproduce previously saved outputs with 100.00% label agreement
and 0.00 maximum probability difference. Only the small downstream fusion
heads change, by taking four more input dimensions.

LEAKAGE
-------
The depression classifier was trained on its own separate 4,897-record text
dataset, not on these memes and not on their suicide-severity labels. Its
outputs are therefore a legitimate input feature here, not a leak of the
target. Worth stating explicitly in the thesis, because "model B's output as
model A's input" is the kind of thing a reader should want reassurance about.

STEP 1 of this script caches depression probabilities for all 973 memes from
their translated OCR text. Step 2 runs the A/B under the Step 1 folds.

TEST SET: not touched.

OUTPUT
------
outputs/phase9_18_depression_probs_all.json
outputs/phase9_18_depression_features_results.json
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
    load_labels, load_embeddings, SIGLIP_DIR, BANGLABERT_E3B_DIR, HIDDEN_DIM,
    DEVICE as DEV,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU

import phase9_1_cv_baseline_3class as P9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSLATION_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
PROBS_CACHE = os.path.join(PROJECT_ROOT, "outputs", "phase9_18_depression_probs_all.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_18_depression_features_results.json")

SEEDS_5 = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 12345
K = 3
TEXT_DIM, IMAGE_DIM, DEP_DIM = 768, 1152, 4


# ============================================================ step 1: cache ==
def build_depression_cache():
    if os.path.exists(PROBS_CACHE):
        with open(PROBS_CACHE, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f)["probs"].items()}

    from phase8_2_depression_parity_wrapper import predict_depression
    from fuzzy_logic_layer import DEPRESSION_CLASSES

    texts = {}
    with open(TRANSLATION_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                texts[r["image_index"]] = r.get("ocr_text_bn", "") or ""

    print(f"Running the parity-tested depression classifier over {len(texts)} memes "
          f"(inference only -- the checkpoint is never modified)...")
    probs = {}
    for n, (idx, t) in enumerate(sorted(texts.items()), 1):
        out = predict_depression(t)
        # "probabilities" is already in DEPRESSION_CLASSES order.
        probs[idx] = [float(p) for p in out["probabilities"]]
        if n % 200 == 0 or n == len(texts):
            print(f"  {n}/{len(texts)}", flush=True)

    with open(PROBS_CACHE, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Depression-classifier probability distributions for all 973 memes, "
                     "produced by the Phase 8 Step 2 parity-tested wrapper. Inference "
                     "only; the checkpoint was not modified. Aggregate probabilities "
                     "only -- no meme text is stored here."),
            "classes": DEPRESSION_CLASSES,
            "n": len(probs),
            "probs": {str(k): v for k, v in probs.items()},
        }, f, indent=2)
    print(f"Cached -> {PROBS_CACHE}")
    return probs


# ================================================================== models ===
class ConcatPlusDep(nn.Module):
    """The locked concat head, optionally widened by the 4 depression features."""

    def __init__(self, use_dep):
        super().__init__()
        extra = DEP_DIM if use_dep else 0
        self.use_dep = use_dep
        self.net = nn.Sequential(
            nn.Linear(TEXT_DIM + IMAGE_DIM + extra, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, K),
        )

    def forward(self, t, i, dep):
        z = torch.cat([t, i, dep], dim=-1) if self.use_dep else torch.cat([t, i], dim=-1)
        return self.net(z)


def train_one(seed, parts, use_dep, cw, sm):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConcatPlusDep(use_dep).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def T(p):
        return (torch.tensor(p["text"], device=DEV), torch.tensor(p["image"], device=DEV),
                torch.tensor(p["dep"], device=DEV))

    tr = T(parts["inner_train"])
    y_tr = torch.tensor(parts["inner_train"]["y"], device=DEV)
    iv = T(parts["inner_val"])
    y_iv = parts["inner_val"]["y"]
    out = T(parts["outer"])

    best_f1, best_probs = -1.0, None
    stale = 0
    for _ in range(300):
        model.train()
        opt.zero_grad()
        soft_target_loss(model(*tr), y_tr, sm, cw).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            f1 = f1_score(y_iv, model(*iv).argmax(dim=1).cpu().numpy(),
                          average="macro", zero_division=0)
            probs = torch.softmax(model(*out), dim=1).cpu().numpy()
        if f1 > best_f1:
            best_f1, best_probs = f1, probs.copy()
            stale = 0
        else:
            stale += 1
            if stale >= 30:
                break
    return best_probs


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
    print("PHASE 9 STEP 10 -- depression distribution as input features")
    print("TEST SET NOT TOUCHED.")
    print("=" * 74)

    dep = build_depression_cache()

    combined_idx = P9.load_combined_indices()
    labels = load_labels()
    y_all = P9.y3(np.array([labels[i] for i in combined_idx], dtype=np.int64))
    n_all = len(y_all)
    pool = {
        "text": load_embeddings(BANGLABERT_E3B_DIR, combined_idx).astype(np.float32),
        "image": load_embeddings(SIGLIP_DIR, combined_idx).astype(np.float32),
        "dep": np.stack([dep[i] for i in combined_idx]).astype(np.float32),
        "y": y_all,
    }
    print(f"\nPool {n_all}. Depression feature matrix {pool['dep'].shape}, "
          f"mean distribution {pool['dep'].mean(axis=0).round(3).tolist()}")

    # How informative are the depression features on their own?
    dep_pred = pool["dep"].argmax(axis=1)
    from scipy.stats import spearmanr
    rho = spearmanr(dep_pred, y_all).statistic
    print(f"Spearman correlation between predicted depression level and true "
          f"suicide severity: {rho:.4f}")

    sm = torch.tensor(build_smoothing_matrix(K, TAU), device=DEV)
    skf = StratifiedKFold(n_splits=P9.N_FOLDS, shuffle=True, random_state=P9.CV_RANDOM_STATE)

    oof = {arm: {s: np.zeros((n_all, K), dtype=np.float32) for s in SEEDS_5}
           for arm in ("baseline", "with_dep")}

    for fold_i, (train_pos, outer_pos) in enumerate(skf.split(np.zeros(n_all), y_all)):
        inner_tr, inner_val = train_test_split(
            train_pos, test_size=P9.INNER_VAL_FRACTION,
            random_state=P9.INNER_RANDOM_STATE, stratify=y_all[train_pos])
        parts = {
            nm: {k: pool[k][pos] for k in ("text", "image", "dep", "y")}
            for nm, pos in (("inner_train", inner_tr), ("inner_val", inner_val),
                            ("outer", outer_pos))
        }
        cw = P9.class_weights_3(y_all[inner_tr])
        for arm, use_dep in (("baseline", False), ("with_dep", True)):
            for s in SEEDS_5:
                oof[arm][s][outer_pos] = train_one(s, parts, use_dep, cw, sm)
        print(f"  fold {fold_i + 1}/{P9.N_FOLDS} done")

    preds = {arm: np.stack([oof[arm][s] for s in SEEDS_5]).mean(axis=0).argmax(axis=1)
             for arm in oof}

    def summarize(p):
        return {
            "accuracy": float(accuracy_score(y_all, p)),
            "macro_f1": float(f1_score(y_all, p, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_all, p, labels=[0, 1, 2]).tolist(),
            "classification_report": classification_report(
                y_all, p, labels=[0, 1, 2], target_names=P9.THREE_CLASS_NAMES,
                zero_division=0, output_dict=True),
            "bootstrap_ci95": bootstrap_ci(y_all, p),
        }

    base, wdep = summarize(preds["baseline"]), summarize(preds["with_dep"])
    cmp = paired(y_all, preds["with_dep"], preds["baseline"])

    print("\n" + "-" * 74)
    print(f"{'arm (concat architecture, 5 seeds, soft averaged)':<52}{'acc':>9}{'macroF1':>10}")
    print("-" * 74)
    print(f"{'baseline: text + image':<52}{base['accuracy']:>9.4f}{base['macro_f1']:>10.4f}")
    print(f"{'+ depression distribution (4 features)':<52}{wdep['accuracy']:>9.4f}{wdep['macro_f1']:>10.4f}")
    print("-" * 74)
    print(f"\ndelta accuracy {wdep['accuracy'] - base['accuracy']:+.4f}  "
          f"CI [{cmp['delta_accuracy_ci95'][0]:+.4f}, {cmp['delta_accuracy_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_accuracy_improved']:.3f}")
    print(f"delta macro-F1 {wdep['macro_f1'] - base['macro_f1']:+.4f}  "
          f"CI [{cmp['delta_macro_f1_ci95'][0]:+.4f}, {cmp['delta_macro_f1_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_macro_f1_improved']:.3f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Depression-classifier output distribution added as 4 input features "
                     "to the concat fusion head. The depression checkpoint is NOT retrained "
                     "or modified; it is called through the Phase 8 parity-tested wrapper. "
                     "Same folds, seeds and unbiased protocol as Step 1. Test not touched."),
            "leakage_note": ("The depression classifier was trained on a separate 4,897-record "
                             "text dataset, not on these memes and not on their suicide-severity "
                             "labels, so its outputs are a legitimate input feature rather than "
                             "a leak of the target."),
            "architecture": "simple concat only",
            "n_examples": int(n_all),
            "spearman_depression_vs_suicide_severity": float(rho),
            "baseline_text_image": base,
            "with_depression_features": wdep,
            "paired_comparison": cmp,
            "runtime_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")
    print(f"Runtime: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
