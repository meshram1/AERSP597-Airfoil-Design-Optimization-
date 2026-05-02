"""
plot_airfoil.py
───────────────
Reconstruct and visualise the airfoil shape from a 12-feature
geometry vector.

Reconstruction (PCHIP / monotone cubic):
    thickness(x):  (0, 0), (0.2, t02), (0.4, t04),
                   (0.6, t06), (0.8, t08), (1.0, 0)
    camber(x):     (0, 0), (0.25, c25), (0.5, c50),
                   (0.75, c75), (1.0, 0)

Surface coordinates:
    upper(x) = camber(x) + thickness(x) / 2
    lower(x) = camber(x) - thickness(x) / 2

Usage
─────
    python plot_airfoil.py                                  # all *.npy
    python plot_airfoil.py slsqp_result.npy                 # single file
    python plot_airfoil.py a.npy b.npy ...                  # several
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from airfoil_opt_utils import (KC135_GEOM, GEOM_NAMES, IDX_THK_PROFILE_GEOM,
                               N_GEOM)


# ─── Reconstruction ────────────────────────────────────────────────────
def reconstruct(z_geom, n_pts: int = 401):
    """
    Returns
    -------
    x       : (n_pts,)  chord-normalised positions (cosine-spaced)
    y_upper : (n_pts,)  upper-surface y/c
    y_lower : (n_pts,)  lower-surface y/c
    camber  : (n_pts,)  mean camber line
    thick   : (n_pts,)  full thickness y/c
    knots   : dict with the discrete sample points used
    """
    z = np.asarray(z_geom, dtype=np.float64).ravel()
    if z.size != N_GEOM:
        raise ValueError(f"Expected {N_GEOM} features, got {z.size}")

    t02, t04, t06, t08 = z[IDX_THK_PROFILE_GEOM]
    c25 = z[GEOM_NAMES.index("camber_x_0.25_y")]
    c50 = z[GEOM_NAMES.index("camber_x_0.5_y")]
    c75 = z[GEOM_NAMES.index("camber_x_0.75_y")]

    x_t = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y_t = np.array([0.0, t02, t04, t06, t08, 0.0])
    pchip_t = PchipInterpolator(x_t, y_t)

    x_c = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_c = np.array([0.0, c25, c50, c75, 0.0])
    pchip_c = PchipInterpolator(x_c, y_c)

    beta  = np.linspace(0, np.pi, n_pts)
    x     = 0.5 * (1.0 - np.cos(beta))                        # cosine spacing
    thick = np.maximum(pchip_t(x), 0.0)
    cam   = pchip_c(x)
    y_u   = cam + thick / 2.0
    y_l   = cam - thick / 2.0

    knots = {
        "thk_x":    x_t,  "thk_y":    y_t,
        "cam_x":    x_c,  "cam_y":    y_c,
        "le_r":     z[GEOM_NAMES.index("leading_edge_radius")],
        "te_angle": z[GEOM_NAMES.index("trailing_edge_angle_deg_y")],
        "max_camber":      z[GEOM_NAMES.index("max_camber")],
        "camber_position": z[GEOM_NAMES.index("camber_position")],
    }
    return x, y_u, y_l, cam, thick, knots


# ─── Plotting ──────────────────────────────────────────────────────────
def plot_airfoils(
    geometries: dict,
    save_path: str | None = "airfoil_shapes.png",
    show: bool = True,
):
    """
    geometries : {label: 12-feature vector}.  kc135 baseline auto-included.
    """
    if "kc135 baseline" not in geometries:
        geometries = {"kc135 baseline": KC135_GEOM, **geometries}

    fig, (ax_air, ax_prof) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]}
    )

    for label, z in geometries.items():
        x, yu, yl, cam, thick, k = reconstruct(z)
        line, = ax_air.plot(x, yu, "-", lw=1.6, label=label)
        col   = line.get_color()
        ax_air.plot(x, yl, "-", lw=1.6, color=col)
        ax_air.plot(x, cam, "--", lw=0.7, color=col, alpha=0.5)

        if label == "kc135 baseline":
            ax_air.plot(k["thk_x"][1:-1],  k["thk_y"][1:-1] / 2,
                        "o", ms=4, color=col, mfc="white",
                        label="thickness / 2 samples (kc135)")
            ax_air.plot(k["thk_x"][1:-1], -k["thk_y"][1:-1] / 2,
                        "o", ms=4, color=col, mfc="white")
            ax_air.plot(k["cam_x"][1:-1],  k["cam_y"][1:-1],
                        "s", ms=4, color=col, mfc="white",
                        label="camber samples (kc135)")

    ax_air.axhline(0, color="gray", lw=0.5)
    ax_air.set_xlabel("x / c");  ax_air.set_ylabel("y / c")
    ax_air.set_title("Airfoil shape — PCHIP reconstruction")
    ax_air.set_aspect("equal", adjustable="datalim")
    ax_air.grid(alpha=0.3); ax_air.legend(loc="best", fontsize=8)

    for label, z in geometries.items():
        x, yu, yl, cam, thick, k = reconstruct(z)
        ax_prof.plot(x, thick, "-", lw=1.4, label=label)
    ax_prof.set_xlabel("x / c");  ax_prof.set_ylabel("full thickness  t/c")
    ax_prof.set_title("Thickness profile")
    ax_prof.grid(alpha=0.3); ax_prof.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print(f"Saved → {save_path}")
    if show:
        plt.show()
    return fig


def print_summary(label, z):
    z = np.asarray(z).ravel()
    le_r  = z[GEOM_NAMES.index("leading_edge_radius")]
    te_a  = z[GEOM_NAMES.index("trailing_edge_angle_deg_y")]
    mc    = z[GEOM_NAMES.index("max_camber")]
    mcp   = z[GEOM_NAMES.index("camber_position")]
    t_max = max(z[i] for i in IDX_THK_PROFILE_GEOM)

    print(f"\n  {label}")
    print(f"    LE radius        : {le_r:+.6f}"
          + ("   ⚠ NEGATIVE / NON-PHYSICAL" if le_r <= 0 else ""))
    print(f"    TE angle (deg)   : {te_a:+.4f}")
    print(f"    max_camber       : {mc:+.6f}  at x = {mcp:.4f}")
    print(f"    peak thickness   : {t_max:.6f} (max of 4 sample stations)")


def main():
    args = sys.argv[1:]
    if not args:
        candidates = ["slsqp_result.npy", "cobyqa_result.npy",
                      "trust_constr_result.npy"]
        args = [c for c in candidates if os.path.exists(c)]
        if not args:
            print("No *_result.npy files found.  Run an optimiser first, "
                  "or pass paths on the command line.")
            return

    geoms = {}
    for path in args:
        if not os.path.exists(path):
            print(f"  ✖  {path} not found, skipping")
            continue
        z = np.load(path)
        if z.shape[-1] != N_GEOM:
            print(f"  ✖  {path}: shape {z.shape} (expected (..., {N_GEOM}))")
            continue
        label = os.path.splitext(os.path.basename(path))[0]
        geoms[label] = z

    if not geoms:
        print("Nothing to plot.")
        return

    print("=" * 70)
    print("  Airfoil reconstruction summary")
    print("=" * 70)
    print_summary("kc135 baseline", KC135_GEOM)
    for label, z in geoms.items():
        print_summary(label, z)
    print()

    plot_airfoils(geoms, save_path="airfoil_shapes.png", show=True)


if __name__ == "__main__":
    main()
