"""
Day 4: E4 (simple concatenation fusion) and E5 (gated fusion + orthogonal
feature), Section 5.4/5.5/8. Two loss variants per config (class-weighted
softmax cross-entropy, and CORAL-style ordinal loss), 3 seeds each.

Design decisions worth being explicit about (see PROGRESS_LOG.md for the
full reasoning):

1. Text vector source: the E3b (OCR + AI reasoning) BanglaBERT embeddings,
   not the E3 (OCR-only) ones. Section 5.1's "text branch" pipeline treats
   AI reasoning as CORE (step 2), not a stretch goal -- E3/E3b were an
   ablation to test whether reasoning helps *in isolation*; E4 onward builds
   the actual architecture, which uses the full text branch output.

2. E5's shared-dimension projection: Section 5.4's gate (`alpha*text +
   (1-alpha)*image`) and orthogonal-feature step both require text and image
   vectors in the SAME dimension, but they start at different dimensions
   (BanglaBERT 768 vs SigLIP 1152). Section 5.3's contrastive alignment
   projections (Stage A) would normally provide that shared space, but Stage
   A is E6, not yet run. So E5 uses its own small supervised-trained linear
   projections (trained jointly with the gate + classifier here), which
   Stage A's self-supervised projections can later replace/augment in E6.
   This is a necessary engineering decision to make E5 well-defined ahead of
   E6, not a brief requirement -- flagged here and in the log.

3. Gate inputs: OCR confidence + OCR-translation confidence (Section 5.4's
   named inputs), standardized using train-split statistics before use.
   Reasoning-agreement is explicitly a stretch-goal-only signal (needs the
   two-model ensemble, not built) so it is not included.

Protocol (Section 8): reported on the VALIDATION split. Test stays locked
until Day 7. Small trained pieces -- per Section 4.2, no incremental/resume
logic needed here (a restart just retrains from scratch, which finishes in
seconds); progress is printed per config/seed instead.
"""
import os
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGSIM_ROOT = os.path.join(PROJECT_ROOT, "i-h", "multimodal_figsim")
LEAKAGE_SAFE_DIR = os.path.join(FIGSIM_ROOT, "data", "leakage_safe")
INDEX_CSV = os.path.join(LEAKAGE_SAFE_DIR, "figsim_leakage_safe_index.csv")
SIGLIP_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "siglip")
BANGLABERT_E3B_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings", "banglabert_e3b")
OCR_JSONL = os.path.join(PROJECT_ROOT, "outputs", "ocr_results.jsonl")
TRANSLATION_JSONL = os.path.join(PROJECT_ROOT, "outputs", "translation_results.jsonl")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "e4_e5_results.json")
PRIOR_RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "e3b_results.json")

CLASSES = [
    "None", "Wish to be dead", "Suicide ideation",
    "Suicide planning", "Suicide attempt or death",
]
NUM_CLASSES = 5
RAW_LABEL_TO_ID = {
    "None": 0, "Wish to be dead": 1, "Suicide ideation": 2,
    "Suicide planning": 3, "Suicide attempts": 4, "Suicide death": 4,
}
TEXT_DIM = 768
IMAGE_DIM = 1152
SHARED_DIM = 512
HIDDEN_DIM = 256
SEEDS = [0, 1, 2]
MAX_EPOCHS = 300
PATIENCE = 30
LR = 1e-3
DEVICE = "cpu"  # tiny models on cached embeddings -- no need to touch the GPU


# ---------------------------------------------------------------- data ----

def load_labels():
    labels = {}
    with open(INDEX_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["include_in_leakage_safe_dataset"] == "True":
                labels[int(row["image_index"])] = RAW_LABEL_TO_ID[row["suicide_scale"]]
    return labels


def load_split(name):
    with open(os.path.join(LEAKAGE_SAFE_DIR, f"{name}.txt"), "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_embeddings(emb_dir, indices):
    return np.stack([np.load(os.path.join(emb_dir, f"{i}.npy")) for i in indices]).astype(np.float32)


def load_confidences(indices):
    ocr_conf = {}
    with open(OCR_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ocr_conf[r["image_index"]] = r["ocr_confidence"]
    trans_conf = {}
    with open(TRANSLATION_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            v = r["translation_confidence"]
            trans_conf[r["image_index"]] = v if v is not None else 0.0
    oc = np.array([ocr_conf[i] for i in indices], dtype=np.float32)
    tc = np.array([trans_conf[i] for i in indices], dtype=np.float32)
    return np.stack([oc, tc], axis=1)  # (N, 2)


def prepare_data():
    labels = load_labels()
    train_idx = load_split("train")
    val_idx = load_split("val")

    data = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx)):
        data[split_name] = {
            "text": load_embeddings(BANGLABERT_E3B_DIR, idx),
            "image": load_embeddings(SIGLIP_DIR, idx),
            "conf": load_confidences(idx),
            "y": np.array([labels[i] for i in idx], dtype=np.int64),
        }

    # Standardize confidences and embeddings using TRAIN statistics only.
    conf_mean = data["train"]["conf"].mean(axis=0, keepdims=True)
    conf_std = data["train"]["conf"].std(axis=0, keepdims=True) + 1e-6
    for split_name in ("train", "val"):
        data[split_name]["conf"] = (data[split_name]["conf"] - conf_mean) / conf_std

    return data


def class_weights_from(y_train):
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    weights = len(y_train) / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------- models ---

class E4Concat(nn.Module):
    """Simple concatenation fusion, no gate, no alignment (Section 8: E4)."""

    def __init__(self, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_DIM + IMAGE_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, text_vec, image_vec, conf):
        z = torch.cat([text_vec, image_vec], dim=-1)
        return self.net(z)


class E5GatedOrth(nn.Module):
    """Gated fusion + orthogonal feature (Section 5.4: Stage B/C), on top of
    small supervised projections into a shared dimension (see module
    docstring, design decision 2)."""

    def __init__(self, num_outputs):
        super().__init__()
        self.text_proj = nn.Linear(TEXT_DIM, SHARED_DIM)
        self.image_proj = nn.Linear(IMAGE_DIM, SHARED_DIM)
        self.gate = nn.Sequential(
            nn.Linear(SHARED_DIM * 2 + 2, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )
        self.text_norm = nn.LayerNorm(SHARED_DIM)
        self.orth_norm = nn.LayerNorm(SHARED_DIM)
        self.net = nn.Sequential(
            nn.Linear(SHARED_DIM * 2 + TEXT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, num_outputs),
        )

    def forward(self, text_vec, image_vec, conf):
        t = self.text_proj(text_vec)
        i = self.image_proj(image_vec)

        gate_in = torch.cat([t, i, conf], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_in))
        gated = alpha * t + (1 - alpha) * i

        # Orthogonal feature (Stage C, Section 5.4): project image onto text,
        # subtract the shared component, keep what's left. Fixed vector math,
        # not a trained step in itself -- only text_proj/image_proj are trained.
        t_norm_sq = (t * t).sum(dim=-1, keepdim=True) + 1e-6
        proj_coeff = (i * t).sum(dim=-1, keepdim=True) / t_norm_sq
        i_parallel = proj_coeff * t
        i_orth = i - i_parallel

        z = torch.cat([self.text_norm(gated), self.orth_norm(i_orth), text_vec], dim=-1)
        return self.net(z)


# ----------------------------------------------------------- CORAL loss ---

class CoralHead(nn.Module):
    """Rank-consistent ordinal head (Cao et al. 2020, CORAL). A single shared
    score plus K-1 monotonically-ordered thresholds, predicting P(y > k) for
    k = 0..K-2 via sigmoid."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_thresholds = num_classes - 1
        self.bias0 = nn.Parameter(torch.zeros(1))
        self.raw_deltas = nn.Parameter(torch.zeros(self.num_thresholds - 1))

    def thresholds(self):
        # bias[0] >= bias[1] >= ... enforced via non-negative softplus deltas,
        # so P(y>k) is guaranteed non-increasing in k.
        deltas = torch.nn.functional.softplus(self.raw_deltas)
        return self.bias0 - torch.cat([torch.zeros(1, device=deltas.device), torch.cumsum(deltas, dim=0)])

    def forward(self, score):  # score: (N, 1) shared rank score
        return score + self.thresholds().unsqueeze(0)  # (N, K-1) logits


def coral_targets(y, num_classes):
    # target_k = 1 if y > k else 0, for k = 0..K-2
    thresholds = torch.arange(num_classes - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > thresholds).float()


def coral_predict(logits):
    probs_gt = torch.sigmoid(logits)
    return (probs_gt > 0.5).sum(dim=1)  # rank = count of exceeded thresholds


# --------------------------------------------------------------- train ----

def make_model(fusion_type, loss_type):
    num_outputs = NUM_CLASSES if loss_type == "ce" else 1
    backbone_cls = E4Concat if fusion_type == "E4" else E5GatedOrth
    backbone = backbone_cls(num_outputs)
    coral_head = CoralHead(NUM_CLASSES) if loss_type == "coral" else None
    return backbone, coral_head


def train_one(fusion_type, loss_type, seed, data, class_weights):
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone, coral_head = make_model(fusion_type, loss_type)
    backbone.to(DEVICE)
    params = list(backbone.parameters())
    if coral_head is not None:
        coral_head.to(DEVICE)
        params += list(coral_head.parameters())
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=1e-4)

    X_text_tr = torch.tensor(data["train"]["text"], device=DEVICE)
    X_img_tr = torch.tensor(data["train"]["image"], device=DEVICE)
    X_conf_tr = torch.tensor(data["train"]["conf"], device=DEVICE)
    y_tr = torch.tensor(data["train"]["y"], device=DEVICE)

    X_text_val = torch.tensor(data["val"]["text"], device=DEVICE)
    X_img_val = torch.tensor(data["val"]["image"], device=DEVICE)
    X_conf_val = torch.tensor(data["val"]["conf"], device=DEVICE)
    y_val = torch.tensor(data["val"]["y"], device=DEVICE)

    sample_weights_tr = class_weights[y_tr]

    if loss_type == "coral":
        coral_targets_tr = coral_targets(y_tr, NUM_CLASSES)

    best_val_f1 = -1.0
    best_val_pred = None
    epochs_without_improve = 0

    for epoch in range(MAX_EPOCHS):
        backbone.train()
        optimizer.zero_grad()
        out = backbone(X_text_tr, X_img_tr, X_conf_tr)

        if loss_type == "ce":
            loss = nn.functional.cross_entropy(out, y_tr, weight=class_weights)
        else:
            logits = coral_head(out)  # (N, K-1)
            per_threshold = nn.functional.binary_cross_entropy_with_logits(
                logits, coral_targets_tr, reduction="none"
            ).mean(dim=1)  # (N,)
            loss = (per_threshold * sample_weights_tr).mean()

        loss.backward()
        optimizer.step()

        backbone.eval()
        with torch.no_grad():
            out_val = backbone(X_text_val, X_img_val, X_conf_val)
            if loss_type == "ce":
                pred_val = out_val.argmax(dim=1)
            else:
                logits_val = coral_head(out_val)
                pred_val = coral_predict(logits_val)
            val_f1 = f1_score(
                y_val.cpu().numpy(), pred_val.cpu().numpy(),
                average="macro", zero_division=0,
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_pred = pred_val.cpu().numpy().copy()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                break

    return best_val_f1, best_val_pred, y_val.cpu().numpy()


def main():
    print("Loading cached embeddings and confidences...")
    data = prepare_data()
    class_weights = class_weights_from(data["train"]["y"])
    print(f"Train n={len(data['train']['y'])}, Val n={len(data['val']['y'])}")
    print(f"Class weights (inverse frequency, train split): "
          f"{dict(zip(CLASSES, [round(w, 3) for w in class_weights.tolist()]))}")

    results = {}
    for fusion_type in ("E4", "E5"):
        for loss_type in ("ce", "coral"):
            key = f"{fusion_type}_{loss_type}"
            print(f"\n--- {key}: {len(SEEDS)} seeds ---")
            seed_f1s = []
            seed_preds = []
            for seed in SEEDS:
                val_f1, val_pred, y_val = train_one(fusion_type, loss_type, seed, data, class_weights)
                seed_f1s.append(val_f1)
                seed_preds.append(val_pred)
                print(f"  seed {seed}: macro-F1 = {val_f1:.4f}", flush=True)

            mean_f1 = float(np.mean(seed_f1s))
            std_f1 = float(np.std(seed_f1s))
            best_seed_idx = int(np.argmax(seed_f1s))
            best_pred = seed_preds[best_seed_idx]

            weighted_f1 = f1_score(y_val, best_pred, average="weighted", zero_division=0)
            acc = accuracy_score(y_val, best_pred)
            cm = confusion_matrix(y_val, best_pred, labels=list(range(NUM_CLASSES)))
            report = classification_report(
                y_val, best_pred, labels=list(range(NUM_CLASSES)),
                target_names=CLASSES, zero_division=0, output_dict=True,
            )

            print(f"  {key}: macro-F1 = {mean_f1:.4f} +/- {std_f1:.4f} "
                  f"(seeds: {[round(f, 4) for f in seed_f1s]})")

            results[key] = {
                "fusion_type": fusion_type,
                "loss_type": loss_type,
                "seeds": SEEDS,
                "macro_f1_per_seed": seed_f1s,
                "macro_f1_mean": mean_f1,
                "macro_f1_std": std_f1,
                "best_seed_weighted_f1": float(weighted_f1),
                "best_seed_accuracy": float(acc),
                "best_seed_confusion_matrix": cm.tolist(),
                "best_seed_classification_report": report,
                "n_val": len(y_val),
            }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")

    print("\n=== SUMMARY (macro-F1 mean +/- std across 3 seeds, validation split) ===")
    if os.path.exists(PRIOR_RESULTS_PATH):
        with open(PRIOR_RESULTS_PATH, "r", encoding="utf-8") as f:
            e3b = json.load(f)
        print(f"E3b (single run, no seeds): {e3b['macro_f1']:.4f}")
    for key, r in results.items():
        print(f"{key}: {r['macro_f1_mean']:.4f} +/- {r['macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()
