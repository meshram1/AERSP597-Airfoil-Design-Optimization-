"""
compare_optimisers.py
─────────────────────
Run SLSQP, COBYQA, and trust-constr on the SAME problem and tabulate
the results, plus a single plot with all convergence curves.

Usage:
    python compare_optimisers.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (minimize, Bounds, NonlinearConstraint, BFGS)
from scipy.sparse import csr_matrix

from airfoil_opt_utils import (build_problem, report_result,
                               GEOM_NAMES, KC135_GEOM, N_GEOM)


def _run(method, prob, bounds):
    history = {"f": [], "u": []}

    if method == "SLSQP":
        def cb(uk):
            history["f"].append(prob["f"](uk)); history["u"].append(uk.copy())
        res = minimize(prob["f"], prob["u0"],
                       jac=prob["grad_f"], method="SLSQP",
                       bounds=bounds, constraints=prob["constraints_dict"],
                       options={"ftol": 1e-9, "maxiter": 500, "disp": False},
                       callback=cb)

    elif method == "COBYQA":
        def cb(uk):
            history["f"].append(prob["f"](uk)); history["u"].append(uk.copy())
        res = minimize(prob["f"], prob["u0"],
                       method="COBYQA",
                       bounds=bounds, constraints=prob["constraints_nlc"],
                       options={"maxfev": 500 * len(prob["u0"]),
                                "feasibility_tol": 1e-8, "disp": False},
                       callback=cb)

    elif method == "trust-constr":
        def cb(uk, state):
            history["f"].append(prob["f"](uk)); history["u"].append(uk.copy())
            return False
        res = minimize(prob["f"], prob["u0"],
                       jac=prob["grad_f"], hess=BFGS(),
                       method="trust-constr",
                       bounds=bounds, constraints=prob["constraints_nlc"],
                       options={"xtol": 1e-8, "gtol": 1e-7,
                                "maxiter": 500, "verbose": 0},
                       callback=cb)
    else:
        raise ValueError(method)

    return res, history


def main():
    prob = build_problem(
        thickness_reduction = 0.10,
        cl_min              = 0.10,
        bounds_slack        = 0.02,
        enable_coherence    = True,
    )

    bounds = Bounds(prob["bounds_lb"], prob["bounds_ub"])

    cl_b, cd_b, cm_b = prob["kc135_pred"]
    print("=" * 72)
    print(f"  Baseline kc135  CL={cl_b:.4f}  CD={cd_b:.6f}  "
          f"CL/CD={cl_b/cd_b:.3f}")
    print(f"  Constraints: CL ≥ {prob['cl_min']},  thickness profile ≤ "
          f"{prob['thk_targets']}")
    print("=" * 72)

    summary = []
    convergence = {}
    for name in ("SLSQP", "COBYQA", "trust-constr"):
        print(f"\n>>> Running {name} ...")
        try:
            res, hist = _run(name, prob, bounds)
            z_opt, (cl, cd, cm) = report_result(name, res, prob, hist)
            summary.append({
                "method": name,
                "success": getattr(res, "success", "?"),
                "cl_cd":   -res.fun,
                "cl":      cl,
                "cd":      cd,
                "thk":     max(z_opt[i] for i in prob["thk_idx"]),
                "nfev":    getattr(res, "nfev", "?"),
            })
            convergence[name] = hist["f"]
            np.save(f"{name.lower().replace('-','_')}_result.npy", z_opt)
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            summary.append({"method": name, "success": False,
                            "cl_cd": np.nan, "cl": np.nan, "cd": np.nan,
                            "thk": np.nan, "nfev": "?"})

    # ─── Summary table ───────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  Summary")
    print("═" * 72)
    print(f"  {'method':<14s} {'CL/CD':>10s} {'CL':>9s} {'CD':>10s} "
          f"{'peak_thk':>10s} {'nfev':>6s}  {'success'}")
    kc135_max_profile = max(KC135_GEOM[i] for i in prob["thk_idx"])
    print(f"  kc135 baseline {cl_b/cd_b:>10.3f} {cl_b:>9.4f} "
          f"{cd_b:>10.6f} {kc135_max_profile:>10.6f} "
          f"{'-':>6s}  -")
    for s in summary:
        print(f"  {s['method']:<14s} {s['cl_cd']:>10.3f} {s['cl']:>9.4f} "
              f"{s['cd']:>10.6f} {s['thk']:>10.6f} {str(s['nfev']):>6s}"
              f"  {s['success']}")

    # ─── Convergence plot ────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    for name, hist in convergence.items():
        if hist:
            plt.plot([-v for v in hist], "-o", ms=3, label=name)
    plt.axhline(cl_b/cd_b, ls="--", c="k", lw=1, label="kc135 baseline")
    plt.xlabel("iteration"); plt.ylabel("CL/CD (surrogate)")
    plt.title("Optimiser comparison – maximise CL/CD with thickness reduction")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("optimiser_comparison.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
