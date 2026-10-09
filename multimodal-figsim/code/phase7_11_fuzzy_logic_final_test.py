"""
Phase 7, Step 2 of the "get ready for the panel" plan: reconnect the
fuzzy-logic risk-combination layer to the NEWLY LOCKED 3-class model
(Phase 7.10, test macro-F1 0.5730), using the actual TEST SET this time
-- not validation (Phase 7.6 was a validation-only comparison, useful
for exploration but not the final, presentable system output).

Re-runs the exact same 9-model ensemble (identical seeds, architecture,
and recipe as phase7_10_final_test_eval_3class.py) to obtain per-example
test predictions -- phase7_10 only persisted aggregate metrics, not
per-example probabilities, which fuzzy combination needs. This is NOT a
new test-set decision: same locked configuration, same deterministic
training recipe, reproduces the identical 0.5730 result (verified below
as a sanity check) -- just captures per-example detail from the same
already-finalized evaluation, exactly as the original Days 9-10 fuzzy
run used per-example detail from the original 5-class test lock.

Phase 2's depression classifier -- identical, untouched -- is run on
the same 196 test memes' translated OCR text (previously only run on
validation for the Phase 7.6 comparison; this is its first run on test
for the 3-class scheme). Combined via the 3-class rule table derived in
PHASE7_FUZZY_LOGIC_5CLASS_VS_3CLASS_PLAN.md. Compared directly against
the original Days 9-10 fuzzy result (5-class, also on test) for a
clean, final before/after comparison on the same 196 memes.
"""
import os
import json
import copy

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

from train_contrastive_align import AlignmentProjections, CKPT_DIR as ALIGN_CKPT_DIR
from train_e6 import (
    load_confidences, SEEDS, HIDDEN_DIM, MAX_EPOCHS as GATED_MAX_EPOCHS, PATIENCE as GATED_PATIENCE,
    DEVICE as GATED_DEVICE, load_labels, load_split, load_embeddings, SIGLIP_DIR, BANGLABERT_E3B_DIR,
)
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss, TAU
from phase4_1_hyperparam_sweep import E6GatedOrthTunable
from phase3_2_cross_attention import CrossAttentionFusion, load_patches, load_tokens, DEVICE as XATTN_DEVICE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAPT_ROOT = r"C:\Users\user6\T2520814\DAPT_models"
CKPT_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5", "checkpoint-735")
TOKENIZER_PATH = os.path.join(DAPT_ROOT, "results", "dapt_eval", "BanglaBERT_fold5_tapt")
TRANSLATION_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
OLD_FUZZY_RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "fuzzy_risk_results.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase7_11_fuzzy_logic_final_test_results.json")

NUM_CLASSES_3 = 3
THREE_CLASS_NAMES = ["No expressed severity", "Suicidal thought or desire", "High acuity suicidal content"]
THREE_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
BEST_GATED_CFG = {"hidden_dim": 512, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4}
DEPRESSION_CLASSES = ["Minimum", "Mild", "Moderate", "Severe"]
RISK_LEVELS = ["Minimal", "Low", "Elevated", "Critical"]
MAX_LEN = 256

RULES_3CLASS = [
    (2, [0, 1, 2, 3], "Critical"),
    (1, [1, 2, 3], "Elevated"),
    (1, [0], "Low"),
    (0, [3], "Elevated"),
    (0, [2], "Low"),
    (0, [0, 1], "Minimal"),
]


def y3(y5_array):
    return np.array([THREE_CLASS_MAP[y] for y in y5_array], dtype=np.int64)


def class_weights_3(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES_3).astype(np.float32)
    weights = len(y_train) / (NUM_CLASSES_3 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


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
    data = {}
    for split_name, idx in (("train", load_split("train")), ("val", load_split("val")), ("test", load_split("test"))):
        y5 = np.array([labels[i] for i in idx], dtype=np.int64)
        data[split_name] = {"text": load_embeddings(BANGLABERT_E3B_DIR, idx), "image": load_embeddings(SIGLIP_DIR, idx), "y": y3(y5)}
    return data


def train_concat_one(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ConcatClassifier3Class()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X_text_tr, X_img_tr, y_tr = torch.tensor(data["train"]["text"]), torch.tensor(data["train"]["image"]), torch.tensor(data["train"]["y"])
    X_text_val, X_img_val, y_val = torch.tensor(data["val"]["text"]), torch.tensor(data["val"]["image"]), torch.tensor(data["val"]["y"])
    best_val_f1, best_state, epochs_without_improve = -1.0, None, 0
    for epoch in range(300):
        model.train(); optimizer.zero_grad()
        loss = soft_target_loss(model(X_text_tr, X_img_tr), y_tr, smoothing_matrix, class_weights)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_val = model(X_text_val, X_img_val).argmax(dim=1)
            val_f1 = f1_score(y_val.numpy(), pred_val.numpy(), average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1, best_state, epochs_without_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= 30: break
    return best_state


def concat_predict_test(state_dict, data):
    model = ConcatClassifier3Class(); model.load_state_dict(state_dict); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(data["test"]["text"]), torch.tensor(data["test"]["image"])), dim=1).numpy()
    return probs, data["test"]["y"]


def load_gated_data_all_splits():
    labels = load_labels()
    align_model = AlignmentProjections()
    ckpt = torch.load(os.path.join(ALIGN_CKPT_DIR, "current.pt"), map_location=GATED_DEVICE)
    align_model.load_state_dict(ckpt["state_dict"]); align_model.eval()
    data = {}
    for split_name, idx in (("train", load_split("train")), ("val", load_split("val")), ("test", load_split("test"))):
        raw_text, raw_image = torch.tensor(load_embeddings(BANGLABERT_E3B_DIR, idx)), torch.tensor(load_embeddings(SIGLIP_DIR, idx))
        with torch.no_grad(): aligned_text, aligned_image = align_model(raw_text, raw_image)
        y5 = np.array([labels[i] for i in idx], dtype=np.int64)
        data[split_name] = {"text": aligned_text.numpy(), "image": aligned_image.numpy(), "raw_text": raw_text.numpy(), "conf": load_confidences(idx), "y": y3(y5)}
    conf_mean = data["train"]["conf"].mean(axis=0, keepdims=True)
    conf_std = data["train"]["conf"].std(axis=0, keepdims=True) + 1e-6
    for split_name in ("train", "val", "test"): data[split_name]["conf"] = (data[split_name]["conf"] - conf_mean) / conf_std
    return data


def gated_to_tensors(sd):
    return (torch.tensor(sd["text"], device=GATED_DEVICE), torch.tensor(sd["image"], device=GATED_DEVICE),
            torch.tensor(sd["raw_text"], device=GATED_DEVICE), torch.tensor(sd["conf"], device=GATED_DEVICE), torch.tensor(sd["y"], device=GATED_DEVICE))


def train_gated_one(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed); np.random.seed(seed)
    model = E6GatedOrthTunable(NUM_CLASSES_3, BEST_GATED_CFG["hidden_dim"], BEST_GATED_CFG["dropout"]).to(GATED_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_GATED_CFG["lr"], weight_decay=BEST_GATED_CFG["weight_decay"])
    Xt, Xi, Xr, Xc, ytr = gated_to_tensors(data["train"])
    Xtv, Xiv, Xrv, Xcv, yval = gated_to_tensors(data["val"])
    best_val_f1, best_state, epochs_without_improve = -1.0, None, 0
    for epoch in range(GATED_MAX_EPOCHS):
        model.train(); optimizer.zero_grad()
        loss = soft_target_loss(model(Xt, Xi, Xr, Xc), ytr, smoothing_matrix, class_weights)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_val = model(Xtv, Xiv, Xrv, Xcv).argmax(dim=1)
            val_f1 = f1_score(yval.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1, best_state, epochs_without_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= GATED_PATIENCE: break
    return best_state


def gated_predict_test(state_dict, data):
    model = E6GatedOrthTunable(NUM_CLASSES_3, BEST_GATED_CFG["hidden_dim"], BEST_GATED_CFG["dropout"]).to(GATED_DEVICE)
    model.load_state_dict(state_dict); model.eval()
    Xt, Xi, Xr, Xc, yt = gated_to_tensors(data["test"])
    with torch.no_grad(): probs = torch.softmax(model(Xt, Xi, Xr, Xc), dim=1).cpu().numpy()
    return probs, yt.cpu().numpy()


def load_xattn_data_all_splits():
    labels = load_labels()
    data = {}
    for split_name, idx in (("train", load_split("train")), ("val", load_split("val")), ("test", load_split("test"))):
        tokens, masks = load_tokens(idx)
        y5 = np.array([labels[i] for i in idx], dtype=np.int64)
        data[split_name] = {"tokens": tokens, "mask": masks, "patches": load_patches(idx), "raw_text": load_embeddings(BANGLABERT_E3B_DIR, idx), "y": y3(y5)}
    return data


def xattn_to_tensors(sd):
    return (torch.tensor(sd["tokens"], device=XATTN_DEVICE), torch.tensor(sd["mask"], device=XATTN_DEVICE),
            torch.tensor(sd["patches"], device=XATTN_DEVICE), torch.tensor(sd["raw_text"], device=XATTN_DEVICE), torch.tensor(sd["y"], device=XATTN_DEVICE))


def train_xattn_one(seed, data, class_weights, smoothing_matrix):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CrossAttentionFusion(NUM_CLASSES_3).to(XATTN_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tt, tm, tp, tr, ytr = xattn_to_tensors(data["train"])
    vt, vm, vp, vr, yval = xattn_to_tensors(data["val"])
    best_val_f1, best_state, epochs_without_improve = -1.0, None, 0
    for epoch in range(GATED_MAX_EPOCHS):
        model.train(); optimizer.zero_grad()
        loss = soft_target_loss(model(tt, tm, tp, tr), ytr, smoothing_matrix, class_weights)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_val = model(vt, vm, vp, vr).argmax(dim=1)
            val_f1 = f1_score(yval.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1, best_state, epochs_without_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= GATED_PATIENCE: break
    return best_state


def xattn_predict_test(state_dict, data):
    model = CrossAttentionFusion(NUM_CLASSES_3).to(XATTN_DEVICE)
    model.load_state_dict(state_dict); model.eval()
    tt, tm, tp, tr, yt = xattn_to_tensors(data["test"])
    with torch.no_grad(): probs = torch.softmax(model(tt, tm, tp, tr), dim=1).cpu().numpy()
    return probs, yt.cpu().numpy()


def majority_vote_fractions(prob_list, num_classes):
    preds = np.stack([p.argmax(axis=1) for p in prob_list])
    fractions = np.zeros((preds.shape[1], num_classes))
    for c in range(num_classes):
        fractions[:, c] = (preds == c).sum(axis=0) / preds.shape[0]
    return fractions


def combine_fuzzy(suicide_probs, depression_probs, rules, suicide_classes):
    risk_membership = {r: 0.0 for r in RISK_LEVELS}
    for suicide_idx, depression_idxs, risk in rules:
        depression_condition = float(max(depression_probs[i] for i in depression_idxs))
        firing_strength = float(min(suicide_probs[suicide_idx], depression_condition))
        risk_membership[risk] = max(risk_membership[risk], firing_strength)
    return max(risk_membership, key=risk_membership.get)


def main():
    print("=== Reproducing the locked 9-model ensemble to capture per-example test predictions ===")
    concat_data = load_concat_data_all_splits()
    cw = class_weights_3(concat_data["train"]["y"])
    sm = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device="cpu")
    concat_probs = []
    for seed in SEEDS:
        state = train_concat_one(seed, concat_data, cw, sm)
        probs, y_test = concat_predict_test(state, concat_data)
        concat_probs.append(probs)
        print(f"  concat seed {seed} done")

    gated_data = load_gated_data_all_splits()
    gcw = class_weights_3(gated_data["train"]["y"])
    gsm = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=GATED_DEVICE)
    gated_probs = []
    for seed in SEEDS:
        state = train_gated_one(seed, gated_data, gcw, gsm)
        probs, y_test_g = gated_predict_test(state, gated_data)
        assert (y_test == y_test_g).all()
        gated_probs.append(probs)
        print(f"  gated seed {seed} done")

    xattn_data = load_xattn_data_all_splits()
    xcw = class_weights_3(xattn_data["train"]["y"]).to(XATTN_DEVICE)
    xsm = torch.tensor(build_smoothing_matrix(NUM_CLASSES_3, TAU), device=XATTN_DEVICE)
    xattn_probs = []
    for seed in SEEDS:
        state = train_xattn_one(seed, xattn_data, xcw, xsm)
        probs, y_test_x = xattn_predict_test(state, xattn_data)
        assert (y_test == y_test_x).all()
        xattn_probs.append(probs)
        print(f"  cross-attn seed {seed} done")

    all_probs = concat_probs + gated_probs + xattn_probs
    vote_fractions = majority_vote_fractions(all_probs, NUM_CLASSES_3)
    pred = vote_fractions.argmax(axis=1)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"\nSanity check -- reproduced macro-F1: {f1:.4f} (expected 0.5730 from Phase 7.10)")

    test_idx = load_split("test")
    print("\n=== Running Phase-2 depression classifier on TEST OCR text ===")
    ocr_texts = {}
    with open(TRANSLATION_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ocr_texts[r["image_index"]] = r["ocr_text_bn"]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    dep_model = AutoModelForSequenceClassification.from_pretrained(CKPT_PATH)
    dep_model.eval()
    depression_probs = []
    for idx in test_idx:
        text = ocr_texts.get(idx, "") or ""
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = dep_model(**enc).logits
        depression_probs.append(torch.softmax(logits, dim=1)[0].numpy())
    depression_probs = np.stack(depression_probs)
    print(f"Depression classifier run on {len(test_idx)} test memes.")

    print("\n=== Combining via 3-class fuzzy rule table ===")
    risk_counts = {r: 0 for r in RISK_LEVELS}
    per_meme_results = []
    for i, idx in enumerate(test_idx):
        crisp_risk = combine_fuzzy(vote_fractions[i], depression_probs[i], RULES_3CLASS, THREE_CLASS_NAMES)
        risk_counts[crisp_risk] += 1
        per_meme_results.append({
            "image_index": idx,
            "suicide_pred_3class": THREE_CLASS_NAMES[int(vote_fractions[i].argmax())],
            "suicide_true_3class": THREE_CLASS_NAMES[int(y_test[i])],
            "depression_pred": DEPRESSION_CLASSES[int(depression_probs[i].argmax())],
            "crisp_risk": crisp_risk,
        })

    n = len(test_idx)
    print(f"\n=== Final risk distribution, 3-class-fuzzy on TEST (n={n}) ===")
    for r in RISK_LEVELS:
        print(f"  {r}: {risk_counts[r]} ({100 * risk_counts[r] / n:.1f}%)")

    old_dist = None
    if os.path.exists(OLD_FUZZY_RESULTS_PATH):
        with open(OLD_FUZZY_RESULTS_PATH, "r", encoding="utf-8") as f:
            old_results = json.load(f)
        old_counts = {r: 0 for r in RISK_LEVELS}
        for rec in old_results:
            old_counts[rec["crisp_risk"]] += 1
        old_dist = old_counts
        print(f"\n=== For comparison -- original 5-class-fuzzy on TEST (Days 9-10, n={len(old_results)}) ===")
        for r in RISK_LEVELS:
            print(f"  {r}: {old_counts[r]} ({100 * old_counts[r] / len(old_results):.1f}%)")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_test": n,
            "sanity_check_macro_f1": float(f1),
            "risk_distribution_3class_test": risk_counts,
            "risk_distribution_5class_test_original": old_dist,
            "per_meme_results": per_meme_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
