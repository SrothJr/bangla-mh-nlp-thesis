"""
Phase 7: final locked configuration for the 3-class harmonized target --
cross-architecture ensemble (simple concat, 3 seeds + gated+orth tuned
for 3-class [hidden_dim=512], 3 seeds + cross-attention, 3 seeds;
majority vote across all 9) -- evaluated on the test split EXACTLY ONCE,
per the same protocol discipline as every prior test lock in this
project (phase3_final_test_eval.py, run_final_test_eval.py).

This is the THIRD deliberate test-set touch in this project's history
(Day 7 original 5-class lock; Phase 3 5-class re-lock; this one -- the
first and only touch for the 3-class harmonized scheme). All 9 models
are trained and model-selected using TRAIN/VAL ONLY (early stopping on
validation macro-F1). Test is not touched until every model is already
fully trained and selected. Then, in a single non-interactive pass, all
9 models' predictions on test are combined by majority vote and every
requested metric is reported once. No re-runs, no tuning based on what
test shows.
"""
import os
import json
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, cohen_kappa_score,
)

from train_contrastive_align import AlignmentProjections, CKPT_DIR as ALIGN_CKPT_DIR
from train_e6 import (
    load_confidences, SEEDS, HIDDEN_DIM,
    MAX_EPOCHS as GATED_MAX_EPOCHS, PATIENCE as GATED_PATIENCE,
    DEVICE as GATED_DEVICE, load_labels, load_split, load_embeddings,
    SIGLIP_DIR, BANGLABERT_E3B_DIR,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU
from phase4_1_hyperparam_sweep import E6GatedOrthTunable
from phase3_2_cross_attention import CrossAttentionFusion, load_patches, load_tokens, DEVICE as XATTN_DEVICE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase7_final_test_results_3class.json")

NUM_CLASSES_3 = 3
THREE_CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
BEST_GATED_CFG = {"hidden_dim": 512, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4}  # Phase 7.8 winner


def y3(y5_array):
    return np.array([THREE_CLASS_MAP[y] for y in y5_array], dtype=np.int64)


def class_weights_3(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES_3).astype(np.float32)
    weights = len(y_train) / (NUM_CLASSES_3 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------- concat model ---
class ConcatClassifier3Class(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768 + 1152, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.2), nn.Linear(HIDDEN_DIM, NUM_CLASSES_3),
        )

    def forward(self, text_vec, image_vec):
        return self.net(torch.cat([text_vec, image_vec], dim=-1))


def load_concat_data_all_splits():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")
    test_idx = load_split("test")
    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        y5 = np.array([labels[i] for i in idx], dtype=np.int64)
        data[split_name] = {
            "text": load_embeddings(BANGLABERT_E3B_DIR, idx),
            "image": load_embeddings(SIGLIP_DIR, idx),
            "y": y3(y5),
        }
    return data


def train_concat_one(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConcatClassifier3Class()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X_text_tr = torch.tensor(data["train"]["text"])
    X_img_tr = torch.tensor(data["train"]["image"])
    y_tr = torch.tensor(data["train"]["y"])
    X_text_val = torch.tensor(data["val"]["text"])
    X_img_val = torch.tensor(data["val"]["image"])
    y_val = torch.tensor(data["val"]["y"])

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improve = 0
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(X_text_tr, X_img_tr)
        loss = soft_target_loss(out, y_tr, smoothing_matrix, class_weights)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out_val = model(X_text_val, X_img_val)
            pred_val = out_val.argmax(dim=1)
            val_f1 = f1_score(y_val.numpy(), pred_val.numpy(), average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= 30:
                break
    return best_val_f1, best_state


def concat_predict_test(state_dict, data):
    model = ConcatClassifier3Class()
    model.load_state_dict(state_dict)
    model.eval()
    X_text_t = torch.tensor(data["test"]["text"])
    X_img_t = torch.tensor(data["test"]["image"])
    y_t = data["test"]["y"]
    with torch.no_grad():
        out = model(X_text_t, X_img_t)
        probs = torch.softmax(out, dim=1).numpy()
    return probs, y_t


# --------------------------------------------------------- gated model ---
def load_gated_data_all_splits():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")
    test_idx = load_split("test")

    align_model = AlignmentProjections()
    ckpt = torch.load(os.path.join(ALIGN_CKPT_DIR, "current.pt"), map_location=GATED_DEVICE)
    align_model.load_state_dict(ckpt["state_dict"])
    align_model.eval()

    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        raw_text = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, idx))
        raw_image = torch.tensor(load_embeddings(SIGLIP_DIR, idx))
        with torch.no_grad():
            aligned_text, aligned_image = align_model(raw_text, raw_image)
        y5 = np.array([labels[i] for i in idx], dtype=np.int64)
        data[split_name] = {
            "text": aligned_text.numpy(), "image": aligned_image.numpy(),
            "raw_text": raw_text.numpy(), "conf": load_confidences(idx),
            "y": y3(y5),
        }
    conf_mean = data["train"]["conf"].mean(axis=0, keepdims=True)
    conf_std = data["train"]["conf"].std(axis=0, keepdims=True) + 1e-6
    for split_name in ("train", "val", "test"):
        data[split_name]["conf"] = (data[split_name]["conf"] - conf_mean) / conf_std
    return data


def gated_to_tensors(split_data):
    return (
        torch.tensor(split_data["text"], device=GATED_DEVICE),
        torch.tensor(split_data["image"], device=GATED_DEVICE),
        torch.tensor(split_data["raw_text"], device=GATED_DEVICE),
        torch.tensor(split_data["conf"], device=GATED_DEVICE),
        torch.tensor(split_data["y"], device=GATED_DEVICE),
    )


def train_gated_one(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = E6GatedOrthTunable(NUM_CLASSES_3, BEST_GATED_CFG["hidden_dim"], BEST_GATED_CFG["dropout"]).to(GATED_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_GATED_CFG["lr"], weight_decay=BEST_GATED_CFG["weight_decay"])

    X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr, y_tr = gated_to_tensors(data["train"])
    X_text_val, X_img_val, X_rawtext_val, X_conf_val, y_val = gated_to_tensors(data["val"])

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improve = 0
    for epoch in range(GATED_MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr)
        loss = soft_target_loss(out, y_tr, smoothing_matrix, class_weights)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out_val = model(X_text_val, X_img_val, X_rawtext_val, X_conf_val)
            pred_val = out_val.argmax(dim=1)
            val_f1 = f1_score(y_val.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= GATED_PATIENCE:
                break
    return best_val_f1, best_state


def gated_predict_test(state_dict, data):
    model = E6GatedOrthTunable(NUM_CLASSES_3, BEST_GATED_CFG["hidden_dim"], BEST_GATED_CFG["dropout"]).to(GATED_DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    X_text_t, X_img_t, X_rawtext_t, X_conf_t, y_t = gated_to_tensors(data["test"])
    with torch.no_grad():
        out = model(X_text_t, X_img_t, X_rawtext_t, X_conf_t)
        probs = torch.softmax(out, dim=1).cpu().numpy()
    return probs, y_t.cpu().numpy()


# ----------------------------------------------------- cross-attn model --
def load_xattn_data_all_splits():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")
    test_idx = load_split("test")
    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        tokens, masks = load_tokens(idx)
        y5 = np.array([labels[i] for i in idx], dtype=np.int64)
        data[split_name] = {
            "tokens": tokens, "mask": masks, "patches": load_patches(idx),
            "raw_text": load_embeddings(BANGLABERT_E3B_DIR, idx),
            "y": y3(y5),
        }
    return data


def xattn_to_tensors(split_data):
    return (
        torch.tensor(split_data["tokens"], device=XATTN_DEVICE),
        torch.tensor(split_data["mask"], device=XATTN_DEVICE),
        torch.tensor(split_data["patches"], device=XATTN_DEVICE),
        torch.tensor(split_data["raw_text"], device=XATTN_DEVICE),
        torch.tensor(split_data["y"], device=XATTN_DEVICE),
    )


def train_xattn_one(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CrossAttentionFusion(NUM_CLASSES_3).to(XATTN_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    tok_tr, mask_tr, patch_tr, rawtext_tr, y_tr = xattn_to_tensors(data["train"])
    tok_val, mask_val, patch_val, rawtext_val, y_val = xattn_to_tensors(data["val"])

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improve = 0
    for epoch in range(GATED_MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(tok_tr, mask_tr, patch_tr, rawtext_tr)
        loss = soft_target_loss(out, y_tr, smoothing_matrix, class_weights)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out_val = model(tok_val, mask_val, patch_val, rawtext_val)
            pred_val = out_val.argmax(dim=1)
            val_f1 = f1_score(y_val.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= GATED_PATIENCE:
                break
    return best_val_f1, best_state


def xattn_predict_test(state_dict, data):
    model = CrossAttentionFusion(NUM_CLASSES_3).to(XATTN_DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    tok_t, mask_t, patch_t, rawtext_t, y_t = xattn_to_tensors(data["test"])
    with torch.no_grad():
        out = model(tok_t, mask_t, patch_t, rawtext_t)
        probs = torch.softmax(out, dim=1).cpu().numpy()
    return probs, y_t.cpu().numpy()


def main():
    print("=" * 70)
    print("STEP 1: train all 9 models (3 concat + 3 gated + 3 cross-attention),")
    print("model-select on VALIDATION ONLY. Test set is not touched yet.")
    print("=" * 70)

    concat_data = load_concat_data_all_splits()
    concat_class_weights = class_weights_3(concat_data["train"]["y"])
    concat_smoothing = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device="cpu")
    concat_states = []
    for seed in SEEDS:
        val_f1, state = train_concat_one(seed, concat_data, concat_class_weights, concat_smoothing)
        concat_states.append(state)
        print(f"concat seed {seed}: best VAL macro-F1 = {val_f1:.4f}")

    gated_data = load_gated_data_all_splits()
    gated_class_weights = class_weights_3(gated_data["train"]["y"])
    smoothing_matrix_gated = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=GATED_DEVICE)
    gated_states = []
    for seed in SEEDS:
        val_f1, state = train_gated_one(seed, gated_data, gated_class_weights, smoothing_matrix_gated)
        gated_states.append(state)
        print(f"gated seed {seed}: best VAL macro-F1 = {val_f1:.4f}")

    xattn_data = load_xattn_data_all_splits()
    xattn_class_weights = class_weights_3(xattn_data["train"]["y"]).to(XATTN_DEVICE)
    smoothing_matrix_xattn = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=XATTN_DEVICE)
    xattn_states = []
    for seed in SEEDS:
        val_f1, state = train_xattn_one(seed, xattn_data, xattn_class_weights, smoothing_matrix_xattn)
        xattn_states.append(state)
        print(f"cross-attn seed {seed}: best VAL macro-F1 = {val_f1:.4f}")

    print("\n" + "=" * 70)
    print("STEP 2: unlocking the test set ONCE. Evaluating all 9 models.")
    print("=" * 70)

    all_probs = []
    y_test = None
    model_names = []
    for seed, state in zip(SEEDS, concat_states):
        probs, y_test = concat_predict_test(state, concat_data)
        all_probs.append(probs)
        model_names.append(f"concat_seed{seed}")
    for seed, state in zip(SEEDS, gated_states):
        probs, y_test_g = gated_predict_test(state, gated_data)
        assert (y_test == y_test_g).all(), "test label ordering mismatch (gated)"
        all_probs.append(probs)
        model_names.append(f"gated_seed{seed}")
    for seed, state in zip(SEEDS, xattn_states):
        probs, y_test_x = xattn_predict_test(state, xattn_data)
        assert (y_test == y_test_x).all(), "test label ordering mismatch (xattn)"
        all_probs.append(probs)
        model_names.append(f"xattn_seed{seed}")

    preds_per_model = np.stack([p.argmax(axis=1) for p in all_probs])  # (9, n_test)
    majority_pred = np.array([
        np.bincount(preds_per_model[:, i], minlength=NUM_CLASSES_3).argmax()
        for i in range(preds_per_model.shape[1])
    ])

    macro_f1 = f1_score(y_test, majority_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, majority_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_test, majority_pred)
    bal_acc = balanced_accuracy_score(y_test, majority_pred)
    qwk = cohen_kappa_score(y_test, majority_pred, weights="quadratic")
    cm = confusion_matrix(y_test, majority_pred, labels=list(range(NUM_CLASSES_3)))
    report = classification_report(
        y_test, majority_pred, labels=list(range(NUM_CLASSES_3)), target_names=THREE_CLASS_NAMES,
        zero_division=0, output_dict=True,
    )

    print(f"\n=== FINAL TEST METRICS (3-class, 9-model cross-architecture ensemble) ===")
    print(f"macro-F1:          {macro_f1:.4f}")
    print(f"weighted-F1:       {weighted_f1:.4f}")
    print(f"accuracy:          {acc:.4f}")
    print(f"balanced accuracy: {bal_acc:.4f}")
    print(f"QWK:               {qwk:.4f}")
    print(f"\nPer-class:")
    for cls in THREE_CLASS_NAMES:
        r = report[cls]
        print(f"  {cls}: precision={r['precision']:.4f} recall={r['recall']:.4f} "
              f"f1={r['f1-score']:.4f} support={int(r['support'])}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    for row in cm.tolist():
        print(" ", row)

    individual_f1s = {}
    for name, probs in zip(model_names, all_probs):
        f1 = f1_score(y_test, probs.argmax(axis=1), average="macro", zero_division=0)
        individual_f1s[name] = float(f1)
        print(f"  {name} test macro-F1: {f1:.4f}")

    print(f"\nComparison -- 5-class ensemble test result (Phase 3 lock): macro-F1=0.4984")
    print(f"Comparison -- 3-class validation result (Phase 7.9): macro-F1=0.6649")

    results = {
        "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
        "quadratic_weighted_kappa": float(qwk),
        "confusion_matrix": cm.tolist(), "classification_report": report,
        "individual_model_test_f1": individual_f1s,
        "n_test": len(y_test),
        "comparison_5class_test_phase3": 0.4984,
        "comparison_3class_validation_phase7_9": 0.6649,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
