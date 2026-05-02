"""
tracked_optimisation.py
───────────────────────
Wraps build_problem() to record every objective evaluation and the
maximum constraint violation at that point, then runs SLSQP / COBYQA /
trust-constr while keeping a per-evaluation history.

Why per-evaluation, not per-iteration?
  An "iteration" is fuzzy across solvers (line search, trust-region
  trial steps, etc.).  "Function evaluation" is well-defined, identical
  in cost across solvers, and is what scipy's nfev counter tracks.

What gets recorded each time scipy calls our wrapped f(u):
  - the objective value −CL/CD
  - the maximum violation of any inequality constraint at that u:
        g_i(u) ≥ 0   →   violation_i = max(0, −g_i(u))
  - the design vector u itself

Public API
──────────
  TrackingProblem(prob)
  run_slsqp(prob)         -> (OptimizeResult, TrackingProblem)
  run_cobyqa(prob)        -> (OptimizeResult, TrackingProblem)
  run_trust_constr(prob)  -> (OptimizeResult, TrackingProblem)
  plot_objective_history(trackers, baseline_clcd, ...)
  plot_violation_history(trackers, ...)
  summary_table(prob, results)            -> prints table
  save_results(prob, results, out_dir)    -> dict[name -> z_geom]
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, BFGS


# ─── Tracker ───────────────────────────────────────────────────────────
class TrackingProblem:
    """Wrap a build_problem() dict and log every f() call."""

    def __init__(self, prob):
        self.prob = prob
        self.reset()

    def reset(self):
        self.f_calls:  list[float] = []
        self.cv_calls: list[float] = []
        self.u_calls:  list[np.ndarray] = []

    # ─── max constraint violation ────────────────────────────────────
    def _max_violation(self, u):
        u = np.asarray(u, dtype=np.float64)
        viols = [0.0]

        # CL ≥ cl_min:  g = cl_fun(u) ≥ 0
        viols.append(max(0.0, -float(self.prob["cl_fun"](u))))

        # thickness profile ≤ target:  g = thk_fun(u) ≥ 0  (vector of 4)
        thk_vals = np.atleast_1d(self.prob["thk_fun"](u))
        viols.extend([max(0.0, -float(v)) for v in thk_vals])

        # geometric coherence:  g = coh_fun(u) ≥ 0  (vector of 5 if enabled)
        if self.prob["coh_fun"] is not None:
            coh_vals = np.atleast_1d(self.prob["coh_fun"](u))
            viols.extend([max(0.0, -float(v)) for v in coh_vals])

        # optional CD ceiling
        if self.prob["cd_fun"] is not None:
            viols.append(max(0.0, -float(self.prob["cd_fun"](u))))

        # box bounds (usually 0; record so violation is honest)
        viols.append(max(0.0, float(np.max(self.prob["bounds_lb"] - u))))
        viols.append(max(0.0, float(np.max(u - self.prob["bounds_ub"]))))

        return max(viols)

    # ─── wrapped objective (this is what scipy calls) ────────────────
    def f(self, u):
        val = float(self.prob["f"](u))
        self.f_calls.append(val)
        self.cv_calls.append(self._max_violation(u))
        self.u_calls.append(np.asarray(u, dtype=np.float64).copy())
        return val

    def grad_f(self, u):
        return self.prob["grad_f"](u)


# ─── Solver runners ────────────────────────────────────────────────────
def run_slsqp(prob, max_iter: int = 500, ftol: float = 1e-9):
    tracker = TrackingProblem(prob)
    bounds  = Bounds(prob["bounds_lb"], prob["bounds_ub"])
    res = minimize(
        tracker.f, prob["u0"],
        jac    = tracker.grad_f,
        method = "SLSQP",
        bounds = bounds,
        constraints = prob["constraints_dict"],
        options = {"ftol": ftol, "maxiter": max_iter, "disp": False},
    )
    return res, tracker


def run_cobyqa(prob, max_eval_factor: int = 500, feasibility_tol: float = 1e-8):
    tracker = TrackingProblem(prob)
    bounds  = Bounds(prob["bounds_lb"], prob["bounds_ub"])
    res = minimize(
        tracker.f, prob["u0"],
        method = "COBYQA",
        bounds = bounds,
        constraints = prob["constraints_nlc"],
        options = {
            "maxfev":          max_eval_factor * len(prob["u0"]),
            "feasibility_tol": feasibility_tol,
            "disp":            False,
        },
    )
    return res, tracker


def run_trust_constr(prob, max_iter: int = 500,
                     xtol: float = 1e-8, gtol: float = 1e-7):
    tracker = TrackingProblem(prob)
    bounds  = Bounds(prob["bounds_lb"], prob["bounds_ub"])
    res = minimize(
        tracker.f, prob["u0"],
        jac    = tracker.grad_f,
        hess   = BFGS(),
        method = "trust-constr",
        bounds = bounds,
        constraints = prob["constraints_nlc"],
        options = {
            "xtol":    xtol,
            "gtol":    gtol,
            "maxiter": max_iter,
            "verbose": 0,
        },
    )
    return res, tracker


# ─── Plotting ──────────────────────────────────────────────────────────
def plot_objective_history(
    trackers: dict,
    baseline_clcd: float | None = None,
    save_path: str | None = "objective_history.png",
    show: bool = True,
):
    """One curve per solver: CL/CD = −f(u_k) at every f-evaluation k."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, tr in trackers.items():
        if not tr.f_calls:
            continue
        evals = np.arange(1, len(tr.f_calls) + 1)
        cl_cd = [-v for v in tr.f_calls]
        ax.plot(evals, cl_cd, "-", lw=1.4,
                label=f"{name}  (n_eval = {len(evals)})")
    if baseline_clcd is not None:
        ax.axhline(baseline_clcd, ls="--", c="k", lw=1.0,
                   label=f"kc135 baseline ({baseline_clcd:.2f})")
    ax.set_xlabel("function evaluation")
    ax.set_ylabel("CL / CD  (surrogate)")
    ax.set_title("Objective history – CL/CD vs function evaluations")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print(f"  saved → {save_path}")
    if show:
        plt.show()
    return fig


def plot_violation_history(
    trackers: dict,
    save_path: str | None = "violation_history.png",
    show: bool = True,
    eps: float = 1e-12,
):
    """One curve per solver: max constraint violation at every evaluation."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, tr in trackers.items():
        if not tr.cv_calls:
            continue
        evals = np.arange(1, len(tr.cv_calls) + 1)
        cv    = np.maximum(tr.cv_calls, eps)
        ax.semilogy(evals, cv, "-", lw=1.4, label=name)
    ax.axhline(1e-6, ls=":", c="gray", lw=0.8,
               label="feasibility tol (1e-6)")
    ax.set_xlabel("function evaluation")
    ax.set_ylabel("max constraint violation  (log scale)")
    ax.set_title("Constraint violation history")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
        print(f"  saved → {save_path}")
    if show:
        plt.show()
    return fig


# ─── Reporting helpers ─────────────────────────────────────────────────
def summary_table(prob, results: dict):
    """results = {name: (OptimizeResult, TrackingProblem)}"""
    cl_b, cd_b, _ = prob["kc135_pred"]
    KC = __import__("airfoil_opt_utils").KC135_GEOM
    peak_kc = max(KC[i] for i in prob["thk_idx"])

    print("=" * 86)
    print(f"  {'method':<14s} {'success':>8s} {'CL/CD':>10s} {'CL':>9s} "
          f"{'CD':>10s} {'peak_thk':>10s} {'LE_r':>9s} {'evals':>7s}")
    print("=" * 86)
    print(f"  {'kc135 baseline':<14s} {'-':>8s} {cl_b/cd_b:>10.3f} "
          f"{cl_b:>9.4f} {cd_b:>10.6f} {peak_kc:>10.6f} "
          f"{KC[2]:>9.4f} {'-':>7s}")
    for name, (res, tr) in results.items():
        u = np.clip(res.x, 0.0, 1.0)
        z = prob["scaler"].to_phys(u)
        cl, cd, _ = prob["predict_phys"](z)
        peak = max(z[i] for i in prob["thk_idx"])
        print(f"  {name:<14s} {str(bool(res.success)):>8s} "
              f"{-res.fun:>10.3f} {cl:>9.4f} {cd:>10.6f} "
              f"{peak:>10.6f} {z[2]:>9.4f} {len(tr.f_calls):>7d}")


def save_results(prob, results: dict, out_dir: str = "."):
    """Save each optimum as <method>_result.npy in out_dir.  Returns dict."""
    os.makedirs(out_dir, exist_ok=True)
    saved = {}
    for name, (res, _) in results.items():
        u = np.clip(res.x, 0.0, 1.0)
        z = prob["scaler"].to_phys(u)
        path = os.path.join(out_dir,
                            f"{name.lower().replace('-','_')}_result.npy")
        np.save(path, z)
        saved[name] = z
        print(f"  saved → {path}")
    return saved
