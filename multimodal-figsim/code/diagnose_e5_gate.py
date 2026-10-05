"""
Diagnostic: is E5's learned gate alpha varying per-meme, or collapsed near a
constant? Retrains E5 (weighted CE) for each of the 3 Day-4 seeds using the
exact same training loop as train_e4_e5.py, then reports the distribution of
alpha on the validation set for each.
"""
import numpy as np
import torch

from train_e4_e5 import (
    prepare_data, class_weights_from, E5GatedOrth, DEVICE, LR, MAX_EPOCHS,
    PATIENCE, SEEDS,
)
from sklearn.metrics import f1_score


def train_and_get_alpha(seed, data, class_weights):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = E5GatedOrth(NUM_OUTPUTS := 5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    X_text_tr = torch.tensor(data["train"]["text"], device=DEVICE)
    X_img_tr = torch.tensor(data["train"]["image"], device=DEVICE)
    X_conf_tr = torch.tensor(data["train"]["conf"], device=DEVICE)
    y_tr = torch.tensor(data["train"]["y"], device=DEVICE)

    X_text_val = torch.tensor(data["val"]["text"], device=DEVICE)
    X_img_val = torch.tensor(data["val"]["image"], device=DEVICE)
    X_conf_val = torch.tensor(data["val"]["conf"], device=DEVICE)
    y_val = torch.tensor(data["val"]["y"], device=DEVICE)

    best_val_f1 = -1.0
    best_alpha = None
    epochs_without_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_text_tr, X_img_tr, X_conf_tr)
        loss = torch.nn.functional.cross_entropy(out, y_tr, weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_text_val, X_img_val, X_conf_val)
            pred_val = out_val.argmax(dim=1)
            val_f1 = f1_score(
                y_val.cpu().numpy(), pred_val.cpu().numpy(), average="macro", zero_division=0
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Recompute alpha the same way forward() does, on the best epoch.
                t = model.text_proj(X_text_val)
                i = model.image_proj(X_img_val)
                gate_in = torch.cat([t, i, X_conf_val], dim=-1)
                alpha = torch.sigmoid(model.gate(gate_in)).squeeze(-1)
                best_alpha = alpha.cpu().numpy().copy()
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= PATIENCE:
                    break

    return best_val_f1, best_alpha


def main():
    data = prepare_data()
    class_weights = class_weights_from(data["train"]["y"])

    for seed in SEEDS:
        val_f1, alpha = train_and_get_alpha(seed, data, class_weights)
        print(f"\nseed {seed}: val macro-F1={val_f1:.4f}")
        print(f"  alpha: mean={alpha.mean():.4f} std={alpha.std():.4f} "
              f"min={alpha.min():.4f} max={alpha.max():.4f}")
        pct_near_0 = (alpha < 0.1).mean() * 100
        pct_near_1 = (alpha > 0.9).mean() * 100
        pct_mid = ((alpha >= 0.4) & (alpha <= 0.6)).mean() * 100
        print(f"  distribution: {pct_near_0:.1f}% < 0.1 (image-dominant), "
              f"{pct_mid:.1f}% in [0.4,0.6] (balanced), "
              f"{pct_near_1:.1f}% > 0.9 (text-dominant)")
        hist, edges = np.histogram(alpha, bins=10, range=(0, 1))
        print("  histogram (10 bins, 0->1):", hist.tolist())


if __name__ == "__main__":
    main()
