"""
run_slsqp.py
────────────
Maximise CL/CD with SLSQP under:
  • CL  ≥ 0.10
  • max_thickness ≤ 0.90 · kc135_max_thickness   (10% reduction)
  • box bounds (training-data min/max, ±2% slack)

All gradients are analytic (autograd through the surrogate); the problem
is solved in u ∈ [0,1]^19 to keep the variables well-conditioned.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

from airfoil_opt_utils import build_problem, report_result, GEOM_NAMES


def main():
    prob = build_problem(
        thickness_reduction = 0.10,   # 10 % thinner than kc135
        cl_min              = 0.10,   # CL must stay ≥ 0.1
        bounds_slack        = 0.02,   # widen each [lb,ub] by 2 % of span
        enable_coherence    = True,   # add geometric-coherence constraints
    )

    bounds      = Bounds(prob["bounds_lb"], prob["bounds_ub"])
    constraints = prob["constraints_dict"]

    history = {"f": [], "u": []}
    def cb(uk):
        history["f"].append(prob["f"](uk))
        history["u"].append(uk.copy())

    cl_b, cd_b, cm_b = prob["kc135_pred"]
    print("─" * 60)
    print("  SLSQP – maximise CL/CD")
    print("─" * 60)
    print(f"  Baseline kc135   : CL={cl_b:.4f}  CD={cd_b:.6f}  "
          f"CL/CD={cl_b/cd_b:.3f}")
    print(f"  thickness target : ≤ {prob['thk_target']:.6f}")
    print(f"  CL constraint    : ≥ {prob['cl_min']:.3f}")
    print(f"  starting f(u0)   : {prob['f'](prob['u0']):.6f}")

    res = minimize(
        prob["f"], prob["u0"],
        jac     = prob["grad_f"],
        method  = "SLSQP",
        bounds  = bounds,
        constraints = constraints,
        options = {"ftol": 1e-9, "maxiter": 500, "disp": True},
        callback = cb,
    )

    z_opt, _ = report_result("SLSQP", res, prob, history)

    np.save("slsqp_result.npy", z_opt)
    print("\nSaved optimal geometry → slsqp_result.npy")

    if history["f"]:
        plt.figure(figsize=(7, 4))
        plt.plot([-v for v in history["f"]], "-o", ms=3)
        plt.xlabel("SLSQP iteration"); plt.ylabel("CL/CD (surrogate)")
        plt.title("SLSQP convergence")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("slsqp_convergence.png", dpi=130)
        plt.show()


if __name__ == "__main__":
    main()
