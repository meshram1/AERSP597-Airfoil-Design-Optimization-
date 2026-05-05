"""
train_model.py
──────────────
Train AirfoilNet on the FULL_FEATURES set (currently 14 inputs:
Re + AoA + 12 geometry features), targets [CL, log(CD), CM].

Mirrors the pipeline from the original main.ipynb that produced the
best published R² values:
  • 70 / 20 / 10  train / val / test  split (random_state=42)
  • best-validation-MSE checkpointing (NOT final-epoch)
  • early stopping (patience = 40)
  • batch_size = 64
  • torch.manual_seed(0)  (deterministic weight init)
  • best hyper-params from the grid search:
        lr=1e-3, wd=1e-3, hidden=256, dropout=0.1, n_blocks=3

Run me when:
  • you change FULL_FEATURES in airfoil_opt_utils.py
  • you change AirfoilNet architecture
  • you have new data
  • you change hyper-parameters

Saves checkpoint to airfoil_surrogate.pt with:
  model_state_dict, in_dim, out_dim, X_mean, X_std, y_mean, y_std

Usage:
    python train_model.py
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from airfoil_opt_utils import AirfoilNet, FULL_FEATURES, CKPT_PATH, CSV_PATH


# ─── Hyper-parameters (winners of the original grid search) ────────────
EPOCHS       = 300
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-3
HIDDEN       = 256
N_BLOCKS     = 3
DROPOUT      = 0.10
PATIENCE     = 40                 # early-stop patience on val MSE
SEED         = 0                  # used for torch.manual_seed
SPLIT_SEED   = 42                 # used for train_test_split


def main():
    # ─── Load + preprocess data ──────────────────────────────────────
    # Match the original main.ipynb:
    #   df.iloc[:, 1:-4] drops airfoilName + the 4 trailing Unnamed cols
    #   then df.dropna() drops any remaining NaN rows
    df = pd.read_csv(CSV_PATH)
    df = df.iloc[:, 1:-4]                              # drop name + trailing unnamed
    df = df.dropna().copy()
    df["coefficientDrag"] = np.log(df["coefficientDrag"])

    targets = ["coefficientLift", "coefficientDrag", "coefficientMoment"]
    X = df[FULL_FEATURES].values.astype(np.float32)
    y = df[targets].values.astype(np.float32)
    print(f"data rows : {len(X)}    in_dim = {X.shape[1]}    out_dim = {y.shape[1]}")
    print(f"features  : {FULL_FEATURES}")

    # ─── 70 / 20 / 10 split ─────────────────────────────────────────
    # First 90/10 train+val / test, then split the 90 → 70 train, 20 val
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.10, random_state=SPLIT_SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=2/9, random_state=SPLIT_SEED)
    print(f"train     : {len(X_tr)}   val : {len(X_val)}   test : {len(X_te)}")

    # ─── Normalise on TRAIN stats only ──────────────────────────────
    X_mean = X_tr.mean(axis=0, keepdims=True)
    X_std  = X_tr.std(axis=0,  keepdims=True)
    y_mean = y_tr.mean(axis=0, keepdims=True)
    y_std  = y_tr.std(axis=0,  keepdims=True)

    def norm_x(a): return (a - X_mean) / X_std
    def norm_y(a): return (a - y_mean) / y_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device    : {device}")

    Xt = torch.from_numpy(norm_x(X_tr)).float().to(device)
    yt = torch.from_numpy(norm_y(y_tr)).float().to(device)
    Xv = torch.from_numpy(norm_x(X_val)).float().to(device)
    yv = torch.from_numpy(norm_y(y_val)).float().to(device)
    Xe = torch.from_numpy(norm_x(X_te)).float().to(device)
    ye = torch.from_numpy(norm_y(y_te)).float().to(device)

    # ─── Model ──────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    model = AirfoilNet(in_dim=X.shape[1], out_dim=y.shape[1],
                       hidden=HIDDEN, n_blocks=N_BLOCKS, drop=DROPOUT).to(device)
    print(f"params    : {sum(p.numel() for p in model.parameters()):,}")

    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    # ─── Training loop with best-val checkpointing + early stopping ─
    n          = Xt.shape[0]
    best_val   = float("inf")
    best_state = None
    bad        = 0
    train_hist, val_hist = [], []
    print()

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        running = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx  = perm[i:i + BATCH_SIZE]
            loss = F.mse_loss(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * idx.numel()
        sched.step()
        train_mse = running / n

        model.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(model(Xv), yv).item()
        train_hist.append(train_mse); val_hist.append(val_mse)

        if val_mse < best_val:
            best_val   = val_mse
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop @ epoch {ep} (no val improvement for {PATIENCE})")
                break

        if ep == 1 or ep % 25 == 0:
            print(f"  epoch {ep:4d}   train MSE {train_mse:.5f}   "
                  f"val MSE {val_mse:.5f}   lr {sched.get_last_lr()[0]:.2e}")

    model.load_state_dict(best_state)
    print(f"\nbest val MSE: {best_val:.5f}")

    # ─── R² on held-out TEST set (physical units) ──────────────────
    model.eval()
    with torch.no_grad():
        pn = model(Xe).cpu().numpy()
    pp = pn * y_std + y_mean
    ap = y_te
    ss_res = ((ap - pp) ** 2).sum(axis=0)
    ss_tot = ((ap - ap.mean(axis=0)) ** 2).sum(axis=0)
    r2     = 1.0 - ss_res / ss_tot

    print()
    print("═" * 60)
    print("  R² on held-out TEST set (10 %, n = {})".format(len(y_te)))
    print("═" * 60)
    for n_, s in zip(["CL", "log(CD)", "CM"], r2):
        print(f"    {n_:<10s}  R² = {s:.4f}")

    cd_t, cd_p = np.exp(ap[:, 1]), np.exp(pp[:, 1])
    cc_t, cc_p = ap[:, 0] / cd_t, pp[:, 0] / cd_p
    r2_cd   = 1 - ((cd_t - cd_p)**2).sum() / ((cd_t - cd_t.mean())**2).sum()
    r2_clcd = 1 - ((cc_t - cc_p)**2).sum() / ((cc_t - cc_t.mean())**2).sum()
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
    ax.semilogy(val_hist,   label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (normalised)")
    ax.set_title(f"Training curves  (best val = {best_val:.5f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=130)
    print("Saved → training_curves.png")


if __name__ == "__main__":
    main()
