"""
evaluate_model.py
─────────────────
Evaluate the trained surrogate (airfoil_surrogate.pt) on the same
train/test split that produced it (random_state=42, test_size=0.2).

Reports R² for each model output (CL, log CD, CM), for de-normalised
CD, and for the derived CL/CD which is the optimisation target.
Also prints MAE / RMSE and produces a 4-panel parity plot.

Two ways to use:
  •  python evaluate_model.py                    # CLI
  •  from evaluate_model import evaluate
     metrics = evaluate(plot=True, show=True)    # in a notebook
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from airfoil_opt_utils import (AirfoilNet, FULL_FEATURES, CKPT_PATH, CSV_PATH)


def _r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    ss_res = ((y - yhat) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def evaluate(
    plot: bool = True,
    show: bool = True,
    save_path: str | None = "model_parity.png",
    verbose: bool = True,
) -> dict:
    """
    Returns
    -------
    dict with keys:
        n_train, n_test
        r2_train, r2_test                   (each: dict cl/log_cd/cm)
        r2_cd_train, r2_cd_test
        r2_clcd_train, r2_clcd_test
        mae_test, rmse_test                 (each: dict cl/log_cd/cm/cl_cd)
    """
    # ─ data ──────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH).dropna(axis=1, how="all").dropna().copy()
    df["coefficientDrag"] = np.log(df["coefficientDrag"])
    X = df[FULL_FEATURES].values.astype(np.float32)
    y = df[["coefficientLift",
            "coefficientDrag",
            "coefficientMoment"]].values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # ─ model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model  = AirfoilNet(in_dim=ckpt["in_dim"], out_dim=ckpt["out_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    xm = ckpt["X_mean"].squeeze().cpu().numpy()
    xs = ckpt["X_std"].squeeze().cpu().numpy()
    ym = ckpt["y_mean"].squeeze().cpu().numpy()
    ys = ckpt["y_std"].squeeze().cpu().numpy()

    @torch.no_grad()
    def predict(X_phys):
        Xn = (X_phys - xm) / xs
        yn = model(torch.from_numpy(Xn.astype(np.float32)).to(device)).cpu().numpy()
        return yn * ys + ym

    yp_tr = predict(X_tr)
    yp_te = predict(X_te)

    # ─ metrics ───────────────────────────────────────────────────────
    r2_tr_v = _r2(y_tr, yp_tr)
    r2_te_v = _r2(y_te, yp_te)

    cd_true_tr = np.exp(y_tr[:, 1]);  cd_pred_tr = np.exp(yp_tr[:, 1])
    cd_true_te = np.exp(y_te[:, 1]);  cd_pred_te = np.exp(yp_te[:, 1])
    r2_cd_tr = float(_r2(cd_true_tr, cd_pred_tr))
    r2_cd_te = float(_r2(cd_true_te, cd_pred_te))

    clcd_true_tr = y_tr[:, 0]   / cd_true_tr
    clcd_pred_tr = yp_tr[:, 0] / cd_pred_tr
    clcd_true_te = y_te[:, 0]   / cd_true_te
    clcd_pred_te = yp_te[:, 0] / cd_pred_te
    r2_clcd_tr = float(_r2(clcd_true_tr, clcd_pred_tr))
    r2_clcd_te = float(_r2(clcd_true_te, clcd_pred_te))

    mae_te  = np.abs(y_te - yp_te).mean(axis=0)
    rmse_te = np.sqrt(((y_te - yp_te) ** 2).mean(axis=0))
    mae_clcd  = float(np.abs(clcd_true_te  - clcd_pred_te ).mean())
    rmse_clcd = float(np.sqrt(((clcd_true_te - clcd_pred_te) ** 2).mean()))

    # ─ pretty print ──────────────────────────────────────────────────
    if verbose:
        print(f"Rows total : {len(X)}   train: {len(X_tr)}   test: {len(X_te)}")
        print("\n" + "═" * 64)
        print("  R²  (model outputs in trained space)")
        print("═" * 64)
        print(f"               {'CL':>10s}  {'log(CD)':>10s}  {'CM':>10s}")
        print(f"  train        {r2_tr_v[0]:>10.4f}  {r2_tr_v[1]:>10.4f}  {r2_tr_v[2]:>10.4f}")
        print(f"  test         {r2_te_v[0]:>10.4f}  {r2_te_v[1]:>10.4f}  {r2_te_v[2]:>10.4f}")
        print("\n" + "═" * 64)
        print("  R²  (physical CD and derived CL/CD)")
        print("═" * 64)
        print(f"               {'CD':>10s}  {'CL/CD':>10s}")
        print(f"  train        {r2_cd_tr:>10.4f}  {r2_clcd_tr:>10.4f}")
        print(f"  test         {r2_cd_te:>10.4f}  {r2_clcd_te:>10.4f}")
        print("\n" + "═" * 64)
        print("  Test errors")
        print("═" * 64)
        print(f"  MAE   :  CL = {mae_te[0]:.4f}   log CD = {mae_te[1]:.4f}   CM = {mae_te[2]:.4f}")
        print(f"  RMSE  :  CL = {rmse_te[0]:.4f}   log CD = {rmse_te[1]:.4f}   CM = {rmse_te[2]:.4f}")
        print(f"  CL/CD :  MAE = {mae_clcd:.3f}   RMSE = {rmse_clcd:.3f}")

    # ─ parity plot ───────────────────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        panels = [
            ("CL",      y_te[:, 0],     yp_te[:, 0],     r2_te_v[0]),
            ("log(CD)", y_te[:, 1],     yp_te[:, 1],     r2_te_v[1]),
            ("CM",      y_te[:, 2],     yp_te[:, 2],     r2_te_v[2]),
            ("CL/CD",   clcd_true_te,   clcd_pred_te,    r2_clcd_te),
        ]
        for ax, (name, yt, yp, r2v) in zip(axes, panels):
            ax.scatter(yt, yp, s=8, alpha=0.55)
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.0)
            ax.set_xlabel(f"actual {name}")
            ax.set_ylabel(f"predicted {name}")
            ax.set_title(f"{name}   R² = {r2v:.4f}")
            ax.grid(alpha=0.3)
            ax.set_aspect("equal", adjustable="datalim")
        fig.suptitle(f"Parity – test set (n = {len(y_te)})", y=1.02)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=140)
            print(f"\nSaved → {save_path}")
        if show:
            plt.show()

    return {
        "n_train":       len(X_tr),
        "n_test":        len(X_te),
        "r2_train":      {"cl": float(r2_tr_v[0]), "log_cd": float(r2_tr_v[1]), "cm": float(r2_tr_v[2])},
        "r2_test":       {"cl": float(r2_te_v[0]), "log_cd": float(r2_te_v[1]), "cm": float(r2_te_v[2])},
        "r2_cd_train":   r2_cd_tr,    "r2_cd_test":   r2_cd_te,
        "r2_clcd_train": r2_clcd_tr,  "r2_clcd_test": r2_clcd_te,
        "mae_test":      {"cl": float(mae_te[0]), "log_cd": float(mae_te[1]),
                          "cm": float(mae_te[2]), "cl_cd": mae_clcd},
        "rmse_test":     {"cl": float(rmse_te[0]), "log_cd": float(rmse_te[1]),
                          "cm": float(rmse_te[2]), "cl_cd": rmse_clcd},
    }


if __name__ == "__main__":
    evaluate(plot=True, show=True)
