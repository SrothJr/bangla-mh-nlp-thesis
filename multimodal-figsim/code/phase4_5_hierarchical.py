"""
Phase 4, item 4.5 (IMPROVEMENT_PLAN.md): hierarchical decomposition.
"None" has looked more separable than the internal ordinal boundaries
throughout every experiment in this project -- this splits the problem
into (1) a coarse binary classifier, None vs. any suicide content, and
(2) a fine classifier among the 4 remaining ordinal severity levels,
trained only on non-None examples and applied only where stage 1 flags
"any suicide content." The hope is concentrating modeling capacity on the
genuinely hard 4-way ordinal split, instead of asking one 5-way classifier
to do both jobs simultaneously.

Uses the same aligned representations and E6GatedOrth architecture
(dropout=0.4, the Phase 4.1 tuned config) as the current best gated
component, majority vote across 3 seeds per stage.
"""
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

from train_e6 import load_aligned_data, class_weights_from, CLASSES, SEEDS, DEVICE, MAX_EPOCHS, PATIENCE, ALIGN_SHARED_DIM
from phase4_1_hyperparam_sweep import E6GatedOrthTunable, DEFAULT
from phase1_3_ordinal_smoothing import build_smoothing_matrix, soft_target_loss

TUNED_CFG = {**DEFAULT, "dropout": 0.4}


def class_weights_n(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = len(y_train) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def to_tensors(split_data, y_override=None):
    y = y_override if y_override is not None else split_data["y"]
    return (
        torch.tensor(split_data["text"], device=DEVICE),
        torch.tensor(split_data["image"], device=DEVICE),
        torch.tensor(split_data["raw_text"], device=DEVICE),
        torch.tensor(split_data["conf"], device=DEVICE),
        torch.tensor(y, device=DEVICE),
    )


def train_stage(seed, data, y_train, y_val, num_classes, class_weights, smoothing_matrix, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = E6GatedOrthTunable(num_classes, cfg["hidden_dim"], cfg["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr, y_tr = to_tensors(data["train"], y_train)
    X_text_val, X_img_val, X_rawtext_val, X_conf_val, y_val_t = to_tensors(data["val"], y_val)

    best_val_f1 = -1.0
    best_val_probs = None
    epochs_without_improve = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr)
        if smoothing_matrix is not None:
            loss = soft_target_loss(out, y_tr, smoothing_matrix, class_weights)
        else:
            loss = nn.functional.cross_entropy(out, y_tr, weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_text_val, X_img_val, X_rawtext_val, X_conf_val)
            probs_val = torch.softmax(out_val, dim=1)
            pred_val = probs_val.argmax(dim=1)
            val_f1 = f1_score(y_val_t.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_probs = probs_val.cpu().numpy().copy()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                break
    return best_val_f1, best_val_probs


def majority_vote(prob_list, num_classes):
    preds = np.stack([p.argmax(axis=1) for p in prob_list])
    return np.array([
        np.bincount(preds[:, i], minlength=num_classes).argmax()
        for i in range(preds.shape[1])
    ])


def main():
    data = load_aligned_data()
    y_train_full = data["train"]["y"]
    y_val_full = data["val"]["y"]

    # ---------------- Stage 1: None (0) vs. any suicide content (1) ----------------
    y_train_coarse = (y_train_full > 0).astype(np.int64)
    y_val_coarse = (y_val_full > 0).astype(np.int64)
    coarse_class_weights = class_weights_n(y_train_coarse, 2)

    print("=== Stage 1: None vs. any suicide content ===")
    coarse_probs = []
    for seed in SEEDS:
        val_f1, probs = train_stage(seed, data, y_train_coarse, y_val_coarse, 2,
                                     coarse_class_weights, None, TUNED_CFG)
        coarse_probs.append(probs)
        print(f"  seed {seed}: val macro-F1 (binary) = {val_f1:.4f}")
    coarse_pred = majority_vote(coarse_probs, 2)
    coarse_f1 = f1_score(y_val_coarse, coarse_pred, average="macro", zero_division=0)
    print(f"Stage 1 majority-vote binary macro-F1: {coarse_f1:.4f}")

    # ---------------- Stage 2: 4-way among non-None classes ----------------
    train_mask = y_train_full > 0
    val_mask_true = y_val_full > 0  # for TRAINING stage 2, use true non-None val examples only

    train_data_fine = {
        "train": {k: v[train_mask] if k != "y" else (y_train_full[train_mask] - 1) for k, v in data["train"].items()},
        "val": {k: v[val_mask_true] if k != "y" else (y_val_full[val_mask_true] - 1) for k, v in data["val"].items()},
    }
    fine_class_weights = class_weights_n(train_data_fine["train"]["y"], 4)
    fine_smoothing = torch.tensor(build_smoothing_matrix(4, 1.0), device=DEVICE)

    print("\n=== Stage 2: 4-way among non-None classes (trained/selected on true non-None subset) ===")
    fine_probs_on_true_nonnone = []
    for seed in SEEDS:
        val_f1, probs = train_stage(seed, train_data_fine,
                                     train_data_fine["train"]["y"], train_data_fine["val"]["y"],
                                     4, fine_class_weights, fine_smoothing, TUNED_CFG)
        fine_probs_on_true_nonnone.append(probs)
        print(f"  seed {seed}: val macro-F1 (4-way, true non-None subset) = {val_f1:.4f}")

    # Now apply stage 2 (using the same trained models is not directly possible here since
    # train_stage doesn't return models -- retrain once more and keep model objects for
    # full-pipeline inference on ALL val examples (not just true non-None ones), since at
    # deployment time stage 2 sees whatever stage 1 flagged, not an oracle subset.
    print("\n=== Retraining stage 2 models to apply to stage-1-flagged val examples ===")
    stage2_models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = E6GatedOrthTunable(4, TUNED_CFG["hidden_dim"], TUNED_CFG["dropout"]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=TUNED_CFG["lr"], weight_decay=TUNED_CFG["weight_decay"])
        X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr, y_tr = to_tensors(train_data_fine["train"])
        X_text_val, X_img_val, X_rawtext_val, X_conf_val, y_val_t = to_tensors(train_data_fine["val"])
        best_val_f1 = -1.0
        best_state = None
        epochs_without_improve = 0
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(X_text_tr, X_img_tr, X_rawtext_tr, X_conf_tr)
            loss = soft_target_loss(out, y_tr, fine_smoothing, fine_class_weights)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out_val = model(X_text_val, X_img_val, X_rawtext_val, X_conf_val)
                pred_val = out_val.argmax(dim=1)
                val_f1 = f1_score(y_val_t.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                import copy
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        stage2_models.append(model)

    # ---------------- Full pipeline: stage1 -> stage2, applied to ALL val examples ----------------
    X_text_val, X_img_val, X_rawtext_val, X_conf_val, _ = to_tensors(data["val"])
    fine_probs_all = []
    with torch.no_grad():
        for model in stage2_models:
            out = model(X_text_val, X_img_val, X_rawtext_val, X_conf_val)
            fine_probs_all.append(torch.softmax(out, dim=1).cpu().numpy())
    fine_pred_all = majority_vote(fine_probs_all, 4)  # 0..3, maps to original classes 1..4

    final_pred = np.where(coarse_pred == 0, 0, fine_pred_all + 1)
    final_f1 = f1_score(y_val_full, final_pred, average="macro", zero_division=0)

    print(f"\n=== Hierarchical pipeline (stage1 -> stage2), full 5-class macro-F1: {final_f1:.4f} ===")
    print(f"Comparison -- Phase 4.4 flat ensemble (calibrated): 0.5695")
    print(f"Comparison -- Phase 4.1 gated-alone (tuned, flat 5-way): 0.5418")
    print(f"Delta vs flat ensemble: {final_f1 - 0.5695:+.4f}")

    report = classification_report(
        y_val_full, final_pred, labels=list(range(5)), target_names=CLASSES, zero_division=0, output_dict=True,
    )
    print("\nPer-class F1:")
    for c in CLASSES:
        print(f"  {c}: {report[c]['f1-score']:.4f}")

    with open("../outputs/phase4_5_hierarchical_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "stage1_binary_macro_f1": float(coarse_f1),
            "final_macro_f1": float(final_f1),
            "comparison_flat_ensemble": 0.5695,
            "comparison_gated_alone": 0.5418,
            "classification_report": report,
        }, f, indent=2)
    print("\nSaved results to outputs/phase4_5_hierarchical_results.json")


if __name__ == "__main__":
    main()
