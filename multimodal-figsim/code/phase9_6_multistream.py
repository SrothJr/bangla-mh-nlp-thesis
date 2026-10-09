"""
Phase 9, Step 5b (Track D): structured multi-stream text fusion.

THE HYPOTHESIS
--------------
The locked pipeline feeds BanglaBERT a single flattened string -- OCR text
and all three reasoning fields concatenated -- and uses one 768-d vector
(the "e3b" embedding) for everything textual. Structure that was generated
upstream is discarded before any model sees it.

Track D's claim is that keeping the textual sources APART, and letting the
model learn how much to weight each one per example, should help. Memes are
figurative by nature -- that is the premise of the FigSIM dataset -- so
being able to weigh "what this figuratively means" separately from "what it
literally says" is directly aligned with what makes the task hard.

HOW IT IS TESTED
----------------
A stream-attention head over S text streams:

    h_s = Dropout(ReLU(W_proj . x_s))          per stream, shared weights
    a_s = softmax_s( v . tanh(W_att . h_s) )   learned attention over streams
    t   = sum_s a_s h_s                        pooled text vector
    out = MLP([t ; image])                     fused with the SigLIP vector

The attention is a single small weight vector, not a transformer. That is
deliberate: Step 3 measured the learning curve and this dataset is only 777
examples, so added capacity is a real risk, not a free lunch. This is also
why Track D is ordered last in the plan.

STREAM SETS COMPARED
--------------------
Configurable via --streams. The two always available at zero extraction
cost are the OCR-only embedding and the combined e3b embedding. When
phase9_5_translate_reasoning_fields.py has finished, the three per-field
reasoning streams can be added, which is the full version of the
hypothesis.

Every configuration is A/B'd against the locked concat architecture under
the identical folds and the identical unbiased protocol from Step 1 (inner
split for early stopping, outer rows never used for selection).

If this does not beat the flattened baseline it is reported as a negative
result and dropped. A tested hypothesis that comes back negative is still a
genuine contribution; a quietly abandoned one is not.

TEST SET: not touched.

OUTPUT
------
outputs/phase9_6_multistream_results.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from train_e6 import load_labels, load_embeddings, SIGLIP_DIR, HIDDEN_DIM, SEEDS, DEVICE as CPU_DEVICE
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU

import phase9_1_cv_baseline_3class as P9

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_ROOT = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase9_6_multistream_results.json")

TEXT_DIM = 768
IMAGE_DIM = 1152
STREAM_DIM = 256
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 12345

# Stream name -> embedding directory. Extended automatically below if the
# per-field reasoning embeddings exist.
AVAILABLE_STREAMS = {
    "ocr": os.path.join(EMB_ROOT, "banglabert_ocr"),
    "e3b": os.path.join(EMB_ROOT, "banglabert_e3b"),
    "cause_effect": os.path.join(EMB_ROOT, "phase9_bn_cause_effect"),
    "figurative_meaning": os.path.join(EMB_ROOT, "phase9_bn_figurative_meaning"),
    "emotional_state": os.path.join(EMB_ROOT, "phase9_bn_emotional_state"),
}


class StreamAttentionFusion(nn.Module):
    def __init__(self, n_streams, num_classes, stream_dim=STREAM_DIM, dropout=0.2):
        super().__init__()
        self.n_streams = n_streams
        self.proj = nn.Sequential(
            nn.Linear(TEXT_DIM, stream_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.att = nn.Sequential(nn.Linear(stream_dim, stream_dim // 2), nn.Tanh())
        self.att_vec = nn.Linear(stream_dim // 2, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(stream_dim + IMAGE_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, streams, image_vec, return_attention=False):
        # streams: (B, S, 768)
        h = self.proj(streams)                       # (B, S, D)
        scores = self.att_vec(self.att(h))           # (B, S, 1)
        if self.n_streams == 1:
            alpha = torch.ones_like(scores)
        else:
            alpha = torch.softmax(scores, dim=1)     # (B, S, 1)
        pooled = (alpha * h).sum(dim=1)              # (B, D)
        out = self.head(torch.cat([pooled, image_vec], dim=-1))
        if return_attention:
            return out, alpha.squeeze(-1)
        return out


def build_pool(stream_names, combined_idx):
    labels = load_labels()
    streams = np.stack(
        [load_embeddings(AVAILABLE_STREAMS[s], combined_idx) for s in stream_names], axis=1
    ).astype(np.float32)                              # (n, S, 768)
    return {
        "streams": streams,
        "image": load_embeddings(SIGLIP_DIR, combined_idx),
        "y": P9.y3(np.array([labels[i] for i in combined_idx], dtype=np.int64)),
    }


def train_fold(seed, parts, n_streams, class_weights, smoothing, device=CPU_DEVICE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = StreamAttentionFusion(n_streams, P9.NUM_CLASSES_3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def tens(p):
        return (torch.tensor(p["streams"], device=device),
                torch.tensor(p["image"], device=device))

    tr_s, tr_i = tens(parts["inner_train"])
    y_tr = torch.tensor(parts["inner_train"]["y"], device=device)
    iv_s, iv_i = tens(parts["inner_val"])
    y_iv = parts["inner_val"]["y"]
    out_s, out_i = tens(parts["outer"])

    best_f1, best_probs, best_att = -1.0, None, None
    stale = 0
    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        loss = soft_target_loss(model(tr_s, tr_i), y_tr, smoothing, class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            f1 = f1_score(y_iv, model(iv_s, iv_i).argmax(dim=1).cpu().numpy(),
                          average="macro", zero_division=0)
            logits, att = model(out_s, out_i, return_attention=True)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        if f1 > best_f1:
            best_f1, best_probs = f1, probs.copy()
            best_att = att.mean(dim=0).cpu().numpy().copy()
            stale = 0
        else:
            stale += 1
            if stale >= 30:
                break
    return best_probs, best_att


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


def run_config(stream_names, combined_idx, y_all, folds, smoothing):
    pool = build_pool(stream_names, combined_idx)
    n_all = len(y_all)
    S = len(stream_names)
    oof = {s: np.zeros((n_all, P9.NUM_CLASSES_3), dtype=np.float32) for s in SEEDS}
    att_acc = []
    for train_pos, outer_pos in folds:
        inner_tr_pos, inner_val_pos = train_test_split(
            train_pos, test_size=P9.INNER_VAL_FRACTION,
            random_state=P9.INNER_RANDOM_STATE, stratify=y_all[train_pos])
        parts = {
            "inner_train": {k: pool[k][inner_tr_pos] for k in ("streams", "image", "y")},
            "inner_val": {k: pool[k][inner_val_pos] for k in ("streams", "image", "y")},
            "outer": {k: pool[k][outer_pos] for k in ("streams", "image", "y")},
        }
        cw = P9.class_weights_3(y_all[inner_tr_pos])
        for s in SEEDS:
            probs, att = train_fold(s, parts, S, cw, smoothing)
            oof[s][outer_pos] = probs
            att_acc.append(att)
    pred = P9.majority_vote_fractions([oof[s] for s in SEEDS], P9.NUM_CLASSES_3).argmax(axis=1)
    mean_att = np.mean(np.stack(att_acc), axis=0)
    return pred, mean_att, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="*", default=None,
                    help="stream sets, each comma-separated, e.g. ocr,e3b")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 74)
    print("PHASE 9 STEP 5b (Track D) -- structured multi-stream text fusion")
    print("TEST SET NOT TOUCHED.")
    print("=" * 74)

    present = {k: v for k, v in AVAILABLE_STREAMS.items() if os.path.isdir(v)}
    print(f"Streams available on disk: {', '.join(sorted(present))}")

    if args.streams:
        configs = [s.split(",") for s in args.streams]
    else:
        configs = [["e3b"], ["ocr", "e3b"]]
        full = ["ocr", "cause_effect", "figurative_meaning", "emotional_state"]
        if all(k in present for k in full):
            configs.append(full)
            configs.append(["ocr", "e3b", "cause_effect", "figurative_meaning", "emotional_state"])
    missing = {s for c in configs for s in c if s not in present}
    if missing:
        print(f"ERROR: requested streams not extracted yet: {sorted(missing)}")
        sys.exit(1)
    print(f"Configurations to evaluate: {configs}")

    combined_idx = P9.load_combined_indices()
    labels = load_labels()
    y_all = P9.y3(np.array([labels[i] for i in combined_idx], dtype=np.int64))
    n_all = len(y_all)
    skf = StratifiedKFold(n_splits=P9.N_FOLDS, shuffle=True, random_state=P9.CV_RANDOM_STATE)
    folds = list(skf.split(np.zeros(n_all), y_all))
    smoothing = torch.tensor(build_smoothing_matrix(P9.NUM_CLASSES_3, TAU), device=CPU_DEVICE)

    results = {}
    preds = {}
    print("\n" + "-" * 74)
    print(f"{'streams':<50}{'acc':>8}{'macroF1':>9}")
    print("-" * 74)
    for cfg in configs:
        label = "+".join(cfg)
        pred, mean_att, _ = run_config(cfg, combined_idx, y_all, folds, smoothing)
        preds[label] = pred
        acc = accuracy_score(y_all, pred)
        f1 = f1_score(y_all, pred, average="macro", zero_division=0)
        results[label] = {
            "streams": cfg,
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "mean_attention_per_stream": {s: float(a) for s, a in zip(cfg, mean_att)},
            "bootstrap_ci95": bootstrap_ci(y_all, pred),
            "confusion_matrix": confusion_matrix(y_all, pred,
                                                 labels=list(range(P9.NUM_CLASSES_3))).tolist(),
            "classification_report": classification_report(
                y_all, pred, labels=list(range(P9.NUM_CLASSES_3)),
                target_names=P9.THREE_CLASS_NAMES, zero_division=0, output_dict=True),
        }
        print(f"{label:<50}{acc:>8.4f}{f1:>9.4f}")
    print("-" * 74)

    ref = "e3b"
    if ref in preds:
        print(f"\nPaired comparisons against the single flattened stream ({ref}):")
        for label, pred in preds.items():
            if label == ref:
                continue
            p = paired(y_all, pred, preds[ref])
            results[label]["paired_vs_single_e3b"] = p
            print(f"  {label:<48} dAcc CI [{p['delta_accuracy_ci95'][0]:+.4f}, "
                  f"{p['delta_accuracy_ci95'][1]:+.4f}]  P(better)={p['prob_accuracy_improved']:.3f}")

    print("\nLearned mean attention per stream (what the model actually weighted):")
    for label, r in results.items():
        if len(r["streams"]) > 1:
            parts = "  ".join(f"{s}={v:.3f}" for s, v in r["mean_attention_per_stream"].items())
            print(f"  {label}: {parts}")

    print(f"\nStep 1 locked-architecture reference: acc=0.6654 F1=0.6155")
    best = max(results, key=lambda k: results[k]["accuracy"])
    print(f"Best here: {best} (acc={results[best]['accuracy']:.4f})")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "note": ("Stream-attention fusion over multiple text streams, A/B'd against "
                     "the single flattened e3b stream under the Step 1 folds and the "
                     "unbiased inner-split protocol. Test set not touched."),
            "n_examples": int(n_all),
            "configurations": results,
            "step1_reference": {"accuracy": 0.6654, "macro_f1": 0.6155},
            "best_by_accuracy": best,
            "runtime_seconds": round(time.time() - t0, 1),
        }, f, indent=2)
    print(f"\nSaved -> {RESULTS_PATH}")
    print(f"Runtime: {round(time.time() - t0, 1)}s")


if __name__ == "__main__":
    main()
