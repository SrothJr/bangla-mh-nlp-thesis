"""
Phase 9, Step 11: do the depression features survive into the full ensemble?

Step 10 found that adding the depression classifier's 4-class distribution
as input features improved the CONCAT sub-model by +0.0129 accuracy
(P(better) 0.927) -- the strongest positive signal anywhere in Phase 9.

That was one architecture. The locked system is a 15-model ensemble across
three architectures, and Phase 9 has repeatedly found that gains on one
component vanish once everything is combined (Step 2's combiner upgrades
being the clearest case). So this tests all three architectures, with and
without the depression features, and compares the assembled ensembles.

Each architecture gets a variant that takes four extra input dimensions
straight into its final classifier layer. The variants are written here
rather than by editing the originals, so no existing code changes:

  concat  -> extra dims appended to [text ; image]
  gated   -> extra dims appended to [gated ; orthogonal residual ; raw text]
  xattn   -> extra dims appended to [pooled attention output ; raw text]

Everything else -- hyperparameters, losses, optimizers, epoch caps,
patience, folds, seeds, the unbiased inner-split protocol -- is identical
between the two arms, so the four features are the only difference.

The depression checkpoint is NOT retrained or modified. Its cached outputs
come from the Phase 8 parity-tested wrapper via Step 10's cache.

TEST SET: not touched.

OUTPUT
------
outputs/phase9_19_depression_features_ensemble_results.json
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
    load_labels, load_embeddings, load_confidences, SIGLIP_DIR, BANGLABERT_E3B_DIR,
    HIDDEN_DIM, MAX_EPOCHS as GATED_MAX_EPOCHS, PATIENCE as GATED_PATIENCE,
    DEVICE as CPU_DEV,
)
from train_contrastive_align import (
    AlignmentProjections, SHARED_DIM as ALIGN_SHARED_DIM, CKPT_DIR as ALIGN_CKPT_DIR,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU
from phase3_2_cross_attention import (
    load_patches, load_tokens, DEVICE as GPU_DEV,
    PATCH_DIM, TOKEN_DIM, RAW_TEXT_DIM, SHARED_DIM as XATTN_SHARED, NUM_HEADS,
)

import phase9_1_cv_baseline_3class as P9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROBS_CACHE = os.path.join(PROJECT_ROOT, "outputs", "phase9_18_depression_probs_all.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs",
                            "phase9_19_depression_features_ensemble_results.json")

SEEDS_5 = [0, 1, 2, 3, 4]
K = 3
DEP_DIM = 4
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 12345
GATED_CFG = P9.BEST_GATED_CFG
STEP8_BEST = 0.6744   # best assembled configuration without depression features


# ================================================================== models ===
class ConcatV(nn.Module):
    def __init__(self, extra):
        super().__init__()
        self.extra = extra
        self.net = nn.Sequential(
            nn.Linear(768 + 1152 + extra, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, K))

    def forward(self, t, i, dep):
        z = [t, i] + ([dep] if self.extra else [])
        return self.net(torch.cat(z, dim=-1))


class GatedV(nn.Module):
    """Structure copied from phase4_1_hyperparam_sweep.E6GatedOrthTunable,
    with `extra` additional dimensions into the final classifier."""

    def __init__(self, extra, hidden_dim, dropout):
        super().__init__()
        self.extra = extra
        self.gate = nn.Sequential(
            nn.Linear(ALIGN_SHARED_DIM * 2 + 2, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))
        self.gated_norm = nn.LayerNorm(ALIGN_SHARED_DIM)
        self.orth_norm = nn.LayerNorm(ALIGN_SHARED_DIM)
        self.net = nn.Sequential(
            nn.Linear(ALIGN_SHARED_DIM * 2 + 768 + extra, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, K))

    def forward(self, t, i, raw_text, conf, dep):
        gate_in = torch.cat([t, i, conf], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in))
        gated = alpha * t + (1 - alpha) * i
        t_norm_sq = (t * t).sum(dim=-1, keepdim=True) + 1e-6
        i_orth = i - ((i * t).sum(dim=-1, keepdim=True) / t_norm_sq) * t
        z = [self.gated_norm(gated), self.orth_norm(i_orth), raw_text]
        if self.extra:
            z.append(dep)
        return self.net(torch.cat(z, dim=-1))


class XattnV(nn.Module):
    """Structure copied from phase3_2_cross_attention.CrossAttentionFusion,
    with `extra` additional dimensions into the final classifier."""

    def __init__(self, extra):
        super().__init__()
        self.extra = extra
        self.text_proj = nn.Linear(TOKEN_DIM, XATTN_SHARED)
        self.image_proj = nn.Linear(PATCH_DIM, XATTN_SHARED)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=XATTN_SHARED, num_heads=NUM_HEADS, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(XATTN_SHARED)
        self.head = nn.Sequential(
            nn.Linear(XATTN_SHARED + RAW_TEXT_DIM + extra, HIDDEN_DIM), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(HIDDEN_DIM, K))

    def forward(self, tokens, mask, patches, raw_text, dep):
        q = self.text_proj(tokens)
        kv = self.image_proj(patches)
        attended, _ = self.cross_attn(q, kv, kv)
        attended = self.norm(attended + q)
        m = mask.unsqueeze(-1)
        pooled = (attended * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        z = [pooled, raw_text]
        if self.extra:
            z.append(dep)
        return self.head(torch.cat(z, dim=-1))


# ================================================================ training ===
def train(seed, parts, build, fwd, dev, max_ep, patience, cw, sm):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    y_tr = torch.tensor(parts["inner_train"]["y"], device=dev)
    y_iv = parts["inner_val"]["y"]
    best_f1, best_probs, stale = -1.0, None, 0
    for _ in range(max_ep):
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
            best_f1, best_probs, stale = f1, probs.copy(), 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_probs


def fw_concat(m, p, d):
    return m(torch.tensor(p["text"], device=d), torch.tensor(p["image"], device=d),
             torch.tensor(p["dep"], device=d))


def fw_gated(m, p, d):
    return m(torch.tensor(p["atext"], device=d), torch.tensor(p["aimage"], device=d),
             torch.tensor(p["text"], device=d), torch.tensor(p["conf"], device=d),
             torch.tensor(p["dep"], device=d))


def fw_xattn(m, p, d):
    return m(torch.tensor(p["tokens"], device=d), torch.tensor(p["mask"], device=d),
             torch.tensor(p["patches"], device=d), torch.tensor(p["text"], device=d),
             torch.tensor(p["dep"], device=d))


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
    print("PHASE 9 STEP 11 -- depression features across the FULL ensemble")
    print("TEST SET NOT TOUCHED.")
    print("=" * 74)

    with open(PROBS_CACHE, "r", encoding="utf-8") as f:
        dep_raw = {int(k): v for k, v in json.load(f)["probs"].items()}

    combined_idx = P9.load_combined_indices()
    labels = load_labels()
    y_all = P9.y3(np.array([labels[i] for i in combined_idx], dtype=np.int64))
    n_all = len(y_all)

    text = load_embeddings(BANGLABERT_E3B_DIR, combined_idx).astype(np.float32)
    image = load_embeddings(SIGLIP_DIR, combined_idx).astype(np.float32)
    dep = np.stack([dep_raw[i] for i in combined_idx]).astype(np.float32)

    align = AlignmentProjections()
    ck = torch.load(os.path.join(ALIGN_CKPT_DIR, "current.pt"), map_location="cpu")
    align.load_state_dict(ck["state_dict"])
    align.eval()
    with torch.no_grad():
        atext, aimage = align(torch.tensor(text), torch.tensor(image))
    tokens, masks = load_tokens(combined_idx)

    pool = {
        "text": text, "image": image, "dep": dep,
        "atext": atext.numpy(), "aimage": aimage.numpy(),
        "conf_raw": load_confidences(combined_idx),
        "tokens": tokens, "mask": masks, "patches": load_patches(combined_idx),
        "y": y_all,
    }
    print(f"Pool {n_all}, distribution {np.bincount(y_all).tolist()}")

    sm_cpu = torch.tensor(build_smoothing_matrix(K, TAU), device=CPU_DEV)
    sm_gpu = torch.tensor(build_smoothing_matrix(K, TAU), device=GPU_DEV)
    skf = StratifiedKFold(n_splits=P9.N_FOLDS, shuffle=True, random_state=P9.CV_RANDOM_STATE)

    specs = {
        "concat": (("text", "image", "dep"), fw_concat, CPU_DEV, 300, 30,
                   lambda e: ConcatV(e)),
        "gated": (("atext", "aimage", "text", "conf_raw", "dep"), fw_gated, CPU_DEV,
                  GATED_MAX_EPOCHS, GATED_PATIENCE,
                  lambda e: GatedV(e, GATED_CFG["hidden_dim"], GATED_CFG["dropout"])),
        "xattn": (("tokens", "mask", "patches", "text", "dep"), fw_xattn, GPU_DEV,
                  GATED_MAX_EPOCHS, GATED_PATIENCE, lambda e: XattnV(e)),
    }

    names = [f"{a}_seed{s}" for a in specs for s in SEEDS_5]
    oof = {arm: {m: np.zeros((n_all, K), dtype=np.float32) for m in names}
           for arm in ("baseline", "with_dep")}

    for fold_i, (train_pos, outer_pos) in enumerate(skf.split(np.zeros(n_all), y_all)):
        inner_tr, inner_val = train_test_split(
            train_pos, test_size=P9.INNER_VAL_FRACTION,
            random_state=P9.INNER_RANDOM_STATE, stratify=y_all[train_pos])
        print(f"\n=== Fold {fold_i + 1}/{P9.N_FOLDS} ===")
        cw_cpu = P9.class_weights_3(y_all[inner_tr])
        cw_gpu = cw_cpu.to(GPU_DEV)

        for arch, (keys, fwd, dev, max_ep, pat, build) in specs.items():
            parts = {}
            for nm, pos in (("inner_train", inner_tr), ("inner_val", inner_val),
                            ("outer", outer_pos)):
                parts[nm] = {k: pool[k][pos] for k in keys}
                parts[nm]["y"] = y_all[pos]
            if arch == "gated":
                mean = parts["inner_train"]["conf_raw"].mean(axis=0, keepdims=True)
                std = parts["inner_train"]["conf_raw"].std(axis=0, keepdims=True) + 1e-6
                for nm in parts:
                    parts[nm]["conf"] = (parts[nm]["conf_raw"] - mean) / std
            cw = cw_gpu if dev == GPU_DEV else cw_cpu
            sm = sm_gpu if dev == GPU_DEV else sm_cpu

            for arm, extra in (("baseline", 0), ("with_dep", DEP_DIM)):
                for s in SEEDS_5:
                    oof[arm][f"{arch}_seed{s}"][outer_pos] = train(
                        s, parts, (lambda e=extra: build(e)), fwd, dev, max_ep, pat, cw, sm)
            print(f"  {arch}: both arms done")
            if dev == GPU_DEV:
                torch.cuda.empty_cache()

    preds = {arm: np.stack([oof[arm][m] for m in names]).mean(axis=0).argmax(axis=1)
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

    per_arch = {}
    for arch in specs:
        sub = [f"{arch}_seed{s}" for s in SEEDS_5]
        for arm in oof:
            p = np.stack([oof[arm][m] for m in sub]).mean(axis=0).argmax(axis=1)
            per_arch.setdefault(arch, {})[arm] = {
                "accuracy": float(accuracy_score(y_all, p)),
                "macro_f1": float(f1_score(y_all, p, average="macro", zero_division=0)),
            }

    print("\n" + "-" * 74)
    print(f"{'15-model ensemble (5 seeds x 3 architectures)':<52}{'acc':>9}{'macroF1':>10}")
    print("-" * 74)
    print(f"{'baseline: text + image':<52}{base['accuracy']:>9.4f}{base['macro_f1']:>10.4f}")
    print(f"{'+ depression distribution':<52}{wdep['accuracy']:>9.4f}{wdep['macro_f1']:>10.4f}")
    print("-" * 74)
    print(f"\ndelta accuracy {wdep['accuracy'] - base['accuracy']:+.4f}  "
          f"CI [{cmp['delta_accuracy_ci95'][0]:+.4f}, {cmp['delta_accuracy_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_accuracy_improved']:.3f}")
    print(f"delta macro-F1 {wdep['macro_f1'] - base['macro_f1']:+.4f}  "
          f"CI [{cmp['delta_macro_f1_ci95'][0]:+.4f}, {cmp['delta_macro_f1_ci95'][1]:+.4f}]  "
          f"P(better)={cmp['prob_macro_f1_improved']:.3f}")

    print("\nPer architecture (5 seeds each, soft averaged):")
    for arch, v in per_arch.items():
        print(f"  {arch:<8} baseline acc={v['baseline']['accuracy']:.4f}  "
              f"with-dep acc={v['with_dep']['accuracy']:.4f}  "
              f"delta={v['with_dep']['accuracy'] - v['baseline']['accuracy']:+.4f}")

    print(f"\nStep 8 best assembled configuration without these features: {STEP8_BEST:.4f}")
    print(f"This run's best: {max(base['accuracy'], wdep['accuracy']):.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Depression-classifier distribution as input features, across all three "
                     "architectures. Architecture variants written fresh here; no existing "
                     "code modified. Depression checkpoint not retrained. Same folds, seeds "
                     "and unbiased protocol as Step 1. Test set not touched."),
            "n_examples": int(n_all),
            "baseline_ensemble": base,
            "with_depression_ensemble": wdep,
            "paired_comparison": cmp,
            "per_architecture": per_arch,
            "step8_best_without_depression_features": STEP8_BEST,
            "runtime_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")
    print(f"Runtime: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
