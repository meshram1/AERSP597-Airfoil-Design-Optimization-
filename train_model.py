"""
train_model.py
──────────────
Train AirfoilNet on the FULL_FEATURES set defined in airfoil_opt_utils.py
(currently 16 inputs: Re, AoA + 14 geometry features), targets [CL, log(CD), CM].

Run me when:
  • you change FULL_FEATURES in airfoil_opt_utils.py
  • you change AirfoilNet architecture
  • you have new data

Saves checkpoint to airfoil_surrogate.pt with:
  model_state_dict, in_dim, out_dim, X_mean, X_std, y_mean, y_std

Usage:
    python train_model.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from airfoil_opt_utils import AirfoilNet, FULL_FEATURES, CKPT_PATH, CSV_PATH


# ─── Hyper-parameters ──────────────────────────────────────────────────
EPOCHS       = 600
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-3
HIDDEN       = 256
N_BLOCKS     = 3
DROPOUT      = 0.10
SEED         = 42


def main():
    # ─── Load + preprocess data ──────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    targets = ["coefficientLift", "coefficientDrag", "coefficientMoment"]
    df = df.dropna(subset=FULL_FEATURES + targets).copy()
    df["coefficientDrag"] = np.log(df["coefficientDrag"])           # log-transform CD

    X = df[FULL_FEATURES].values.astype(np.float32)                  # (N, 16)
    y = df[targets].values.astype(np.float32)                        # (N, 3)
    print(f"data rows : {len(X)}    in_dim = {X.shape[1]}    out_dim = {y.shape[1]}")
    print(f"features  : {FULL_FEATURES}")

    # ─── Split (matches evaluate_model.py for direct R² comparison) ──
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=SEED)
    print(f"train     : {len(X_tr)}   test : {len(X_te)}")

    # ─── Normalisation (computed on training data only) ──────────────
    X_mean = X_tr.mean(axis=0, keepdims=True)
    X_std  = X_tr.std(axis=0,  keepdims=True)
    y_mean = y_tr.mean(axis=0, keepdims=True)
    y_std  = y_tr.std(axis=0,  keepdims=True)

    X_tr_n = (X_tr - X_mean) / X_std
    y_tr_n = (y_tr - y_mean) / y_std
    X_te_n = (X_te - X_mean) / X_std
    y_te_n = (y_te - y_mean) / y_std

    # ─── Build model + optimiser ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device    : {device}")

    model = AirfoilNet(in_dim=X.shape[1], out_dim=y.shape[1],
                       hidden=HIDDEN, n_blocks=N_BLOCKS, drop=DROPOUT).to(device)
    print(f"params    : {sum(p.numel() for p in model.parameters()):,}")

    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    crit  = nn.MSELoss()

    Xt = torch.from_numpy(X_tr_n).float().to(device)
    yt = torch.from_numpy(y_tr_n).float().to(device)
    Xv = torch.from_numpy(X_te_n).float().to(device)
    yv = torch.from_numpy(y_te_n).float().to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)

    # ─── Training loop ──────────────────────────────────────────────
    train_hist, test_hist = [], []
    print()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
        train_mse = ep_loss / len(Xt)

        model.eval()
        with torch.no_grad():
            test_mse = crit(model(Xv), yv).item()
        train_hist.append(train_mse); test_hist.append(test_mse)
        sched.step()

        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:4d}   train MSE {train_mse:.5f}   "
                  f"test MSE {test_mse:.5f}   lr {sched.get_last_lr()[0]:.2e}")

    # ─── R² on test set (physical units) ────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_n = model(Xv).cpu().numpy()
    pred_phys = pred_n * y_std + y_mean
    actual_phys = y_te                                          # already physical

    ss_res = ((actual_phys - pred_phys) ** 2).sum(axis=0)
    ss_tot = ((actual_phys - actual_phys.mean(axis=0)) ** 2).sum(axis=0)
    r2     = 1.0 - ss_res / ss_tot
    print()
    print("═" * 60)
    print("  Test R² (physical units)")
    print("═" * 60)
    for n, s in zip(["CL", "log(CD)", "CM"], r2):
        print(f"    {n:<10s}  R² = {s:.4f}")

    # also CL/CD R²
    cd_true = np.exp(actual_phys[:, 1]);  cd_pred = np.exp(pred_phys[:, 1])
    clcd_true = actual_phys[:, 0] / cd_true
    clcd_pred = pred_phys[:, 0]   / cd_pred
    r2_cd   = 1.0 - ((cd_true   - cd_pred  )**2).sum() / ((cd_true   - cd_true.mean()  )**2).sum()
    r2_clcd = 1.0 - ((clcd_true - clcd_pred)**2).sum() / ((clcd_true - clcd_true.mean())**2).sum()
    print(f"    {'CD':<10s}  R² = {r2_cd:.4f}")
    print(f"    {'CL/CD':<10s}  R² = {r2_clcd:.4f}")

    # ─── Save checkpoint ────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_dim":  X.shape[1],
        "out_dim": y.shape[1],
        "X_mean":  torch.tensor(X_mean),
        "X_std":   torch.tensor(X_std),
        "y_mean":  torch.tensor(y_mean),
        "y_std":   torch.tensor(y_std),
    }, CKPT_PATH)
    print(f"\nSaved checkpoint → {CKPT_PATH}")

    # ─── Loss curve ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_hist, label="train")
    ax.semilogy(test_hist,  label="test")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (normalised)")
    ax.set_title("Training curves"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=130)
    print("Saved → training_curves.png")


if __name__ == "__main__":
    main()
