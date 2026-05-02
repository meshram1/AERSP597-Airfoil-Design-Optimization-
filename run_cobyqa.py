"""
run_cobyqa.py
─────────────
COBYQA (derivative-free trust-region SQP) for the same problem as
run_slsqp.py.  COBYQA does not use gradients, so we don't pass jac;
it will model the surrogate with quadratic interpolation inside its
trust region.

Constraints are passed as scipy.optimize.NonlinearConstraint objects,
which is the modern API COBYQA prefers.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint

from airfoil_opt_utils import build_problem, report_result


def main():
    prob = build_problem(
        thickness_reduction = 0.10,
        cl_min              = 0.10,
        bounds_slack        = 0.02,
        enable_coherence    = True,
    )

    bounds      = Bounds(prob["bounds_lb"], prob["bounds_ub"])
    constraints = prob["constraints_nlc"]

    history = {"f": [], "u": []}
    def cb(uk):
        history["f"].append(prob["f"](uk))
        history["u"].append(uk.copy())

    cl_b, cd_b, cm_b = prob["kc135_pred"]
    print("─" * 60)
    print("  COBYQA – maximise CL/CD")
    print("─" * 60)
    print(f"  Baseline kc135   : CL={cl_b:.4f}  CD={cd_b:.6f}  "
          f"CL/CD={cl_b/cd_b:.3f}")
    print(f"  thickness target : ≤ {prob['thk_target']:.6f}")
    print(f"  CL constraint    : ≥ {prob['cl_min']:.3f}")
    print(f"  starting f(u0)   : {prob['f'](prob['u0']):.6f}")

    res = minimize(
        prob["f"], prob["u0"],
        method  = "COBYQA",
        bounds  = bounds,
        constraints = constraints,
        options = {
            "maxfev": 500 * len(prob["u0"]),
            "feasibility_tol": 1e-8,
            "disp": True,
        },
        callback = cb,
    )

    z_opt, _ = report_result("COBYQA", res, prob, history)

    np.save("cobyqa_result.npy", z_opt)
    print("\nSaved optimal geometry → cobyqa_result.npy")

    if history["f"]:
        plt.figure(figsize=(7, 4))
        plt.plot([-v for v in history["f"]], "-o", ms=3)
        plt.xlabel("COBYQA iteration"); plt.ylabel("CL/CD (surrogate)")
        plt.title("COBYQA convergence")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("cobyqa_convergence.png", dpi=130)
        plt.show()


if __name__ == "__main__":
    main()
