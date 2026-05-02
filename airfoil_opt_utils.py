"""
airfoil_opt_utils.py
────────────────────
Shared machinery for SLSQP / COBYQA / trust-constr optimisation of CL/CD.

The trained surrogate (airfoil_surrogate.pt) takes 14 inputs:

    full[ 0] reynoldsNumber                ← FIXED (RE_FIXED)
    full[ 1] alpha                         ← FIXED (AOA_FIXED)
    full[ 2] max_camber
    full[ 3] camber_position
    full[ 4] leading_edge_radius
    full[ 5] trailing_edge_angle_deg_y
    full[ 6] thickness_x_0.2_y             ← thickness profile
    full[ 7] thickness_x_0.4_y             ← thickness profile
    full[ 8] thickness_x_0.6_y             ← thickness profile
    full[ 9] thickness_x_0.8_y             ← thickness profile
    full[10] camber_x_0.25_y
    full[11] camber_x_0.5_y
    full[12] camber_x_0.75_y
    full[13] upper_slope_x_0.2_y

Re and AoA are flow conditions, not design variables.  The 12 geometry
features become the design vector  z ∈ R^12.  Optimisers run in
u = (z − lb)/(ub − lb) ∈ [0,1]^12  (min-max scaled, well-conditioned).

The surrogate predicts [CL, log(CD), CM] in normalised space.  We
denormalise then exponentiate to recover physical CD.

Note: the surrogate has no max_thickness input, so "reduce thickness vs
kc135" is enforced by constraining all four thickness_x_*_y features to
be ≤ (1 − reduction)·kc135 thickness profile.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─── paths ─────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(HERE, "airfoil_surrogate.pt")
CSV_PATH  = os.path.join(HERE, "airfoil_data.csv")

# ─── flow conditions (NOT optimised over) ──────────────────────────────
RE_FIXED  = 1_000_000.0
AOA_FIXED = 10.0

# ─── 14-feature ordering used at training time ─────────────────────────
FULL_FEATURES = [
    "reynoldsNumber",
    "alpha",
    "max_camber",
    "camber_position",
    "leading_edge_radius",
    "trailing_edge_angle_deg_y",
    "thickness_x_0.2_y",
    "thickness_x_0.4_y",
    "thickness_x_0.6_y",
    "thickness_x_0.8_y",
    "camber_x_0.25_y",
    "camber_x_0.5_y",
    "camber_x_0.75_y",
    "upper_slope_x_0.2_y",
]
N_FULL = len(FULL_FEATURES)                          # = 14

# Geometry-only design vector (drop Re & alpha)
GEOM_NAMES   = FULL_FEATURES[2:]                     # 12 names
N_GEOM       = len(GEOM_NAMES)                       # = 12

# Indices of the four thickness-profile features WITHIN the 12-d vector
IDX_THK_PROFILE_GEOM = [
    GEOM_NAMES.index("thickness_x_0.2_y"),           # 4
    GEOM_NAMES.index("thickness_x_0.4_y"),           # 5
    GEOM_NAMES.index("thickness_x_0.6_y"),           # 6
    GEOM_NAMES.index("thickness_x_0.8_y"),           # 7
]

# ─── kc135a baseline values for the 12 geometry features ───────────────
# Read straight from row 1 of airfoil_data.csv:
#   max_camber                = 0.015759598
#   camber_position           = 0.195979899
#   leading_edge_radius       = 0.020511475
#   trailing_edge_angle_deg_y = 16.34891452
#   thickness_x_0.2_y         = 0.136011176
#   thickness_x_0.4_y         = 0.153451829
#   thickness_x_0.6_y         = 0.114855638
#   thickness_x_0.8_y         = 0.057539035
#   camber_x_0.25_y           = 0.015513021
#   camber_x_0.5_y            = 0.012899623
#   camber_x_0.75_y           = 0.007071131
#   upper_slope_x_0.2_y       = 0.1199
KC135_GEOM = np.array([
    0.015759598,    # max_camber
    0.195979899,    # camber_position
    0.020511475,    # leading_edge_radius
    16.34891452,    # trailing_edge_angle_deg_y
    0.136011176,    # thickness_x_0.2_y
    0.153451829,    # thickness_x_0.4_y
    0.114855638,    # thickness_x_0.6_y
    0.057539035,    # thickness_x_0.8_y
    0.015513021,    # camber_x_0.25_y
    0.012899623,    # camber_x_0.5_y
    0.007071131,    # camber_x_0.75_y
    0.1199,         # upper_slope_x_0.2_y
], dtype=np.float64)


# ─── Physical (hard) bound overrides ───────────────────────────────────
# Applied on top of training-data bounds to prevent the optimiser from
# drifting into geometrically nonsensical regions when bounds_slack > 0.
#
# Example: training-data lb for leading_edge_radius is 5.66e-4.  With
# 2% slack on a 0.153-wide span the lb becomes -0.0024 (negative radius).
# That's how an early SLSQP run produced LE_r = -0.000317.  The override
# below floors the lb at the training-data minimum regardless of slack
# (so the optimiser cannot leave the distribution where the surrogate
# is reliable).
#
# Special string sentinel "DATASET_MIN" → use the un-slacked min from CSV.
PHYSICAL_LB = {
    "max_camber":                0.0,           # symmetric airfoil possible
    "camber_position":           0.0,           # x must be in [0, 1]
    "leading_edge_radius":       "DATASET_MIN", # ← floor at training minimum
    "trailing_edge_angle_deg_y": 0.0,           # angle ≥ 0 (sharp TE)
    "thickness_x_0.2_y":         1.0e-3,        # positive thickness
    "thickness_x_0.4_y":         1.0e-3,
    "thickness_x_0.6_y":         1.0e-3,
    "thickness_x_0.8_y":         1.0e-3,
}
PHYSICAL_UB = {
    "camber_position":           1.0,           # x must be in [0, 1]
}


# ─── AirfoilNet (matches the architecture saved in airfoil_surrogate.pt) ─
class AirfoilNet(nn.Module):
    """3 residual blocks of (Lin-BN-GELU-Dropout)*2 + Lin-BN-GELU, hidden=256."""
    def __init__(self, in_dim, out_dim, hidden=256, n_blocks=3, drop=0.1):
        super().__init__()
        self.stem   = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.blocks = nn.ModuleList(
            [self._res_block(hidden, drop) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    @staticmethod
    def _res_block(dim, drop=0.1):
        return nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
        )

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = h + blk(h)                  # residual
        return self.head(h)


# ─── Surrogate: load + predict + predict-with-grad ─────────────────────
def load_surrogate(device: torch.device | None = None):
    """Returns (predict_phys, predict_phys_grad, info)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    if ckpt["in_dim"] != N_FULL:
        raise RuntimeError(
            f"Checkpoint expects {ckpt['in_dim']} inputs but FULL_FEATURES "
            f"has {N_FULL}.  Update FULL_FEATURES to match training."
        )

    model = AirfoilNet(in_dim=ckpt["in_dim"], out_dim=ckpt["out_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    xm = ckpt["X_mean"].to(device).squeeze().to(torch.float32)
    xs = ckpt["X_std"].to(device).squeeze().to(torch.float32)
    ym = ckpt["y_mean"].to(device).squeeze().to(torch.float32)
    ys = ckpt["y_std"].to(device).squeeze().to(torch.float32)

    re_t  = torch.tensor(RE_FIXED,  dtype=torch.float32, device=device)
    aoa_t = torch.tensor(AOA_FIXED, dtype=torch.float32, device=device)

    def _build_x(z_geom_np, with_grad: bool):
        z = torch.tensor(np.asarray(z_geom_np, dtype=np.float32),
                         device=device, requires_grad=with_grad)
        # full vector = [Re, AoA, *geom]
        x = torch.cat([re_t.view(1), aoa_t.view(1), z]).view(1, -1)
        return z, x

    @torch.no_grad()
    def predict_phys(z_geom):
        _, x = _build_x(z_geom, with_grad=False)
        y_norm = model((x - xm) / xs)
        y_phys = y_norm * ys + ym
        cl  = float(y_phys[0, 0].item())
        cd  = float(torch.exp(y_phys[0, 1]).item())
        cm  = float(y_phys[0, 2].item())
        return cl, cd, cm

    def predict_phys_grad(z_geom):
        z, x = _build_x(z_geom, with_grad=True)
        y_norm = model((x - xm) / xs)
        y_phys = y_norm * ys + ym
        cl     = y_phys[0, 0]
        log_cd = y_phys[0, 1]
        cm     = y_phys[0, 2]
        cd     = torch.exp(log_cd)

        dcl = torch.autograd.grad(cl, z, retain_graph=True)[0]
        dcd = torch.autograd.grad(cd, z, retain_graph=True)[0]
        dcm = torch.autograd.grad(cm, z, retain_graph=False)[0]
        return {
            "cl": (float(cl.item()), dcl.detach().cpu().numpy().astype(np.float64)),
            "cd": (float(cd.item()), dcd.detach().cpu().numpy().astype(np.float64)),
            "cm": (float(cm.item()), dcm.detach().cpu().numpy().astype(np.float64)),
        }

    info = {"device": device, "X_mean": xm, "X_std": xs,
            "y_mean": ym, "y_std": ys, "model": model, "ckpt": ckpt}
    return predict_phys, predict_phys_grad, info


# ─── Bounds from training data ─────────────────────────────────────────
def get_geom_bounds(slack_frac: float = 0.0, apply_physical: bool = True):
    """
    (lb, ub) of shape (N_GEOM,), float64.

    slack_frac     : widen each interval by this fraction of its span
    apply_physical : enforce PHYSICAL_LB / PHYSICAL_UB on top of training bounds
                     (prevents e.g. negative leading-edge radius)
    """
    df = pd.read_csv(CSV_PATH).dropna(axis=1, how="all").dropna()
    lb_data = df[GEOM_NAMES].min().values.astype(np.float64)        # un-slacked
    ub_data = df[GEOM_NAMES].max().values.astype(np.float64)
    lb, ub  = lb_data.copy(), ub_data.copy()
    if slack_frac > 0:
        span = ub - lb
        lb -= slack_frac * span
        ub += slack_frac * span
    if apply_physical:
        for name, val in PHYSICAL_LB.items():
            if name in GEOM_NAMES:
                i = GEOM_NAMES.index(name)
                # "DATASET_MIN" → floor at the un-slacked training minimum
                floor = lb_data[i] if val == "DATASET_MIN" else float(val)
                lb[i] = max(lb[i], floor)
        for name, val in PHYSICAL_UB.items():
            if name in GEOM_NAMES:
                i = GEOM_NAMES.index(name)
                ceil_ = ub_data[i] if val == "DATASET_MAX" else float(val)
                ub[i] = min(ub[i], ceil_)
    return lb, ub


# ─── Geometric coherence constraints (linear in z) ─────────────────────
def build_coherence_matrix():
    """
    Returns A_z of shape (5, N_GEOM) such that  A_z @ z >= 0  is the
    coherence set:

        1) max_camber       ≥ camber_x_0.25_y
        2) max_camber       ≥ camber_x_0.5_y
        3) max_camber       ≥ camber_x_0.75_y
        4) thickness_x_0.4  ≥ thickness_x_0.6
        5) thickness_x_0.6  ≥ thickness_x_0.8

    Constraints 1–3 keep the camber peak above the local camber samples.
    Constraints 4–5 enforce monotone aft thinning (which most non-laminar
    airfoils — including kc135 — already satisfy).
    """
    i_mc  = GEOM_NAMES.index("max_camber")
    i_c25 = GEOM_NAMES.index("camber_x_0.25_y")
    i_c50 = GEOM_NAMES.index("camber_x_0.5_y")
    i_c75 = GEOM_NAMES.index("camber_x_0.75_y")
    i_t04 = GEOM_NAMES.index("thickness_x_0.4_y")
    i_t06 = GEOM_NAMES.index("thickness_x_0.6_y")
    i_t08 = GEOM_NAMES.index("thickness_x_0.8_y")

    A = np.zeros((5, N_GEOM), dtype=np.float64)
    A[0, i_mc]  =  1.0; A[0, i_c25] = -1.0
    A[1, i_mc]  =  1.0; A[1, i_c50] = -1.0
    A[2, i_mc]  =  1.0; A[2, i_c75] = -1.0
    A[3, i_t04] =  1.0; A[3, i_t06] = -1.0
    A[4, i_t06] =  1.0; A[4, i_t08] = -1.0
    return A

COHERENCE_LABELS = [
    "max_camber ≥ camber_x_0.25_y",
    "max_camber ≥ camber_x_0.5_y",
    "max_camber ≥ camber_x_0.75_y",
    "thickness_x_0.4_y ≥ thickness_x_0.6_y",
    "thickness_x_0.6_y ≥ thickness_x_0.8_y",
]


# ─── Min-max scaler u ↔ z ──────────────────────────────────────────────
class Scaler:
    """u = (z − lb)/(ub − lb), z = lb + u·(ub − lb)."""
    def __init__(self, lb, ub):
        self.lb   = np.asarray(lb, dtype=np.float64)
        self.ub   = np.asarray(ub, dtype=np.float64)
        self.span = self.ub - self.lb
        if (self.span <= 0).any():
            bad = np.where(self.span <= 0)[0].tolist()
            raise ValueError(f"Non-positive span at indices {bad}")

    def to_unit(self, z):
        return (np.asarray(z, dtype=np.float64) - self.lb) / self.span

    def to_phys(self, u):
        return self.lb + np.asarray(u, dtype=np.float64) * self.span

    def chain(self, dphys):
        """Convert ∂/∂z gradient to ∂/∂u via chain rule (∂z/∂u = span)."""
        return np.asarray(dphys, dtype=np.float64) * self.span


# ─── Problem builder ───────────────────────────────────────────────────
def build_problem(
    thickness_reduction: float = 0.10,
    cl_min: float              = 0.10,
    cd_max: float | None       = None,
    bounds_slack: float        = 0.02,
    enable_coherence: bool     = True,
    eps_cd: float              = 1e-8,
):
    """
    Builds the constrained problem in u-space.

    Constraints (all expressed as g(u) ≥ 0):
        cl_fun:           CL                      − cl_min
        thk_fun_i (×4):   (1 − reduction)·kc135_i − z_i        for the four
                          thickness_x_*_y features
        cd_fun (optional, if cd_max given): cd_max − CD

    Returns dict with everything optimisers need (objective, grads,
    constraints, bounds, warm start, baseline prediction, ...).
    """
    predict_phys, predict_phys_grad, info = load_surrogate()

    lb, ub = get_geom_bounds(slack_frac=bounds_slack)
    scaler = Scaler(lb, ub)

    THK_TARGETS = (1.0 - thickness_reduction) * KC135_GEOM[IDX_THK_PROFILE_GEOM]

    # ─── objective: −CL/CD ────────────────────────────────────────────
    def f(u):
        cl, cd, _ = predict_phys(scaler.to_phys(u))
        return float(-cl / (cd + eps_cd))

    def grad_f(u):
        z   = scaler.to_phys(u)
        res = predict_phys_grad(z)
        cl, dcl = res["cl"]
        cd, dcd = res["cd"]
        dphys = -(dcl * cd - cl * dcd) / (cd * cd + eps_cd)
        return scaler.chain(dphys).astype(np.float64)

    # ─── CL ≥ cl_min ─────────────────────────────────────────────────
    def cl_fun(u):
        cl, _, _ = predict_phys(scaler.to_phys(u))
        return float(cl - cl_min)

    def cl_jac(u):
        _, dcl = predict_phys_grad(scaler.to_phys(u))["cl"]
        return scaler.chain(dcl).astype(np.float64)

    # ─── thickness profile ≤ targets  (linear in z) ──────────────────
    # Pre-compute the constant Jacobian (in u-space) for the 4-vector
    # constraint  THK_TARGET_i − z_i  ≥ 0:
    #   ∂/∂z_j (THK_TARGET_i − z_i) = −δ_{ij},   ∂/∂u_j = −δ_{ij}·span_j
    thk_jac_u_const = np.zeros((4, N_GEOM), dtype=np.float64)
    for row, idx in enumerate(IDX_THK_PROFILE_GEOM):
        thk_jac_u_const[row, idx] = -scaler.span[idx]

    def thk_fun(u):
        z = scaler.to_phys(u)
        return (THK_TARGETS - z[IDX_THK_PROFILE_GEOM]).astype(np.float64)

    def thk_jac(u):
        return thk_jac_u_const.copy()

    # ─── optional CD ≤ cd_max ────────────────────────────────────────
    if cd_max is not None:
        def cd_fun(u):
            _, cd, _ = predict_phys(scaler.to_phys(u))
            return float(cd_max - cd)

        def cd_jac(u):
            _, dcd = predict_phys_grad(scaler.to_phys(u))["cd"]
            return scaler.chain(-dcd).astype(np.float64)
    else:
        cd_fun = cd_jac = None

    # ─── coherence: A_z @ z ≥ 0 (linear in z, affine in u) ────────────
    A_z = build_coherence_matrix() if enable_coherence else None
    if A_z is not None:
        A_u = A_z * scaler.span                 # broadcast: row i × span
        A_lb = A_z @ scaler.lb                  # constant offset

        def coh_fun(u):
            return (A_u @ np.asarray(u, dtype=np.float64) + A_lb).astype(np.float64)

        def coh_jac(u):
            return A_u
        n_coh = A_z.shape[0]
    else:
        coh_fun = coh_jac = None
        n_coh = 0

    # ─── coefficient predictors (diagnostics) ────────────────────────
    def cl_predict(u): return predict_phys(scaler.to_phys(u))[0]
    def cd_predict(u): return predict_phys(scaler.to_phys(u))[1]
    def cm_predict(u): return predict_phys(scaler.to_phys(u))[2]

    # ─── warm start: kc135 with thickness profile pre-clipped ────────
    z0 = KC135_GEOM.copy()
    for i, idx in enumerate(IDX_THK_PROFILE_GEOM):
        z0[idx] = min(z0[idx], THK_TARGETS[i]) - 1e-4
    # also clip into the physical bounds (e.g. LE radius floor)
    z0 = np.clip(z0, scaler.lb, scaler.ub)
    u0 = np.clip(scaler.to_unit(z0), 1e-6, 1.0 - 1e-6).astype(np.float64)

    # ─── Pre-built constraint lists ──────────────────────────────────
    # SLSQP wants list of dicts; COBYQA / trust-constr want NLCs.
    try:
        from scipy.optimize import NonlinearConstraint
        from scipy.sparse import csr_matrix
        zero_hess = lambda u, v: csr_matrix((N_GEOM, N_GEOM))
    except ImportError:
        NonlinearConstraint = None
        zero_hess = None

    constraints_dict = [
        {"type": "ineq", "fun": cl_fun,  "jac": cl_jac},
        {"type": "ineq", "fun": thk_fun, "jac": thk_jac},
    ]
    constraints_nlc = []
    if NonlinearConstraint is not None:
        try:
            from scipy.optimize import BFGS as _BFGS
            cl_hess = _BFGS()
        except ImportError:
            cl_hess = None
        constraints_nlc = [
            NonlinearConstraint(cl_fun,  0.0, np.inf, jac=cl_jac,  hess=cl_hess),
            NonlinearConstraint(thk_fun, 0.0, np.inf, jac=thk_jac, hess=zero_hess),
        ]
    if cd_fun is not None:
        constraints_dict.append({"type": "ineq", "fun": cd_fun, "jac": cd_jac})
        if NonlinearConstraint is not None:
            constraints_nlc.append(
                NonlinearConstraint(cd_fun, 0.0, np.inf, jac=cd_jac, hess=_BFGS())
            )
    if coh_fun is not None:
        constraints_dict.append({"type": "ineq", "fun": coh_fun, "jac": coh_jac})
        if NonlinearConstraint is not None:
            constraints_nlc.append(
                NonlinearConstraint(coh_fun, 0.0, np.inf, jac=coh_jac, hess=zero_hess)
            )

    return {
        "f":                f,
        "grad_f":           grad_f,
        "cl_fun":           cl_fun,
        "cl_jac":           cl_jac,
        "thk_fun":          thk_fun,
        "thk_jac":          thk_jac,
        "thk_targets":      THK_TARGETS,
        "cd_fun":           cd_fun,
        "cd_jac":           cd_jac,
        "cd_max":           cd_max,
        "coh_fun":          coh_fun,
        "coh_jac":          coh_jac,
        "n_coh":            n_coh,
        "cl_predict":       cl_predict,
        "cd_predict":       cd_predict,
        "cm_predict":       cm_predict,
        "scaler":           scaler,
        "bounds_lb":        np.zeros(N_GEOM, dtype=np.float64),
        "bounds_ub":        np.ones(N_GEOM,  dtype=np.float64),
        "u0":               u0,
        "kc135_pred":       predict_phys(KC135_GEOM),
        "cl_min":           float(cl_min),
        "thk_idx":          list(IDX_THK_PROFILE_GEOM),
        "predict_phys":     predict_phys,
        "predict_phys_grad": predict_phys_grad,
        "constraints_dict": constraints_dict,
        "constraints_nlc":  constraints_nlc,
    }


# ─── Result reporting ──────────────────────────────────────────────────
def report_result(name, res, prob, history=None):
    """Pretty-print an OptimizeResult and the airfoil it found."""
    u_opt = np.clip(res.x, 0.0, 1.0)
    z_opt = prob["scaler"].to_phys(u_opt)
    cl, cd, cm = prob["predict_phys"](z_opt)
    cl_b, cd_b, cm_b = prob["kc135_pred"]
    targets = prob["thk_targets"]

    print()
    print("═" * 78)
    print(f"  {name} result")
    print("═" * 78)
    print(f"  success        : {getattr(res, 'success', 'n/a')}")
    print(f"  message        : {getattr(res, 'message', 'n/a')}")
    print(f"  nfev / nit     : {getattr(res, 'nfev', '?')} / "
          f"{getattr(res, 'nit', '?')}")
    print(f"  −CL/CD (obj)   : {res.fun:+.6f}")
    print(f"   CL/CD         : {-res.fun:+.6f}    "
          f"(baseline kc135: {cl_b/cd_b:+.3f})")
    print(f"   CL            : {cl:+.6f}    (baseline {cl_b:+.4f})")
    print(f"   CD            : {cd:+.8f}    (baseline {cd_b:+.6f})")
    print(f"   CM            : {cm:+.6f}    (baseline {cm_b:+.4f})")
    print(f"   CL ≥ {prob['cl_min']:.3f}     : "
          f"{'OK' if cl >= prob['cl_min'] - 1e-6 else 'VIOLATED'}")
    if prob["coh_fun"] is not None:
        coh_vals = prob["coh_fun"](u_opt)
        print()
        print("  Coherence constraints (g(u) ≥ 0):")
        for label, v in zip(COHERENCE_LABELS, coh_vals):
            mark = "OK" if v >= -1e-6 else "VIOLATED"
            print(f"    {label:<42s}  g = {v:+.6f}   {mark}")

    print()
    print("  Thickness profile (target = 0.9 × kc135):")
    print(f"    {'station':<22s} {'kc135':>10s} {'target':>10s} "
          f"{'optimum':>10s} {'status'}")
    for k, idx in enumerate(prob["thk_idx"]):
        ok = "OK" if z_opt[idx] <= targets[k] + 1e-6 else "VIOLATED"
        print(f"    {GEOM_NAMES[idx]:<22s} {KC135_GEOM[idx]:>10.6f} "
              f"{targets[k]:>10.6f} {z_opt[idx]:>10.6f} {ok}")
    print()
    print("  Full geometry vs kc135:")
    print(f"    {'feature':<28s} {'kc135':>12s} {'optimum':>12s} {'Δ':>12s}")
    for i, n in enumerate(GEOM_NAMES):
        print(f"    {n:<28s} {KC135_GEOM[i]:>+12.6f} {z_opt[i]:>+12.6f} "
              f"{z_opt[i]-KC135_GEOM[i]:>+12.6f}")
    return z_opt, (cl, cd, cm)
