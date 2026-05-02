"""
run_trust_constr.py
───────────────────
trust-constr (interior-point SQP, scipy) for the same problem.
Uses analytic gradients for objective AND for both constraints, plus a
finite-difference Hessian (BFGS approximation) since computing exact
Hessians of a NN is overkill.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint, BFGS
from scipy.sparse import csr_matrix

from airfoil_opt_utils import build_problem, report_result, N_GEOM


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
    def cb(uk, state):
        history["f"].append(prob["f"](uk))
        history["u"].append(uk.copy())
        return False                       # continue optimising

    cl_b, cd_b, cm_b = prob["kc135_pred"]
    print("─" * 60)
    print("  trust-constr – maximise CL/CD")
    print("─" * 60)
    print(f"  Baseline kc135   : CL={cl_b:.4f}  CD={cd_b:.6f}  "
          f"CL/CD={cl_b/cd_b:.3f}")
    print(f"  thickness target : ≤ {prob['thk_target']:.6f}")
    print(f"  CL constraint    : ≥ {prob['cl_min']:.3f}")
    print(f"  starting f(u0)   : {prob['f'](prob['u0']):.6f}")

    res = minimize(
        prob["f"], prob["u0"],
        jac     = prob["grad_f"],
        hess    = BFGS(),                  # quasi-Newton on the objective
        method  = "trust-constr",
        bounds  = bounds,
        constraints = constraints,
        options = {
            "xtol":      1e-8,
            "gtol":      1e-7,
            "maxiter":   500,
            "verbose":   2,
        },
        callback = cb,
    )

    z_opt, _ = report_result("trust-constr", res, prob, history)

    np.save("trust_constr_result.npy", z_opt)
    print("\nSaved optimal geometry → trust_constr_result.npy")

    if history["f"]:
        plt.figure(figsize=(7, 4))
        plt.plot([-v for v in history["f"]], "-o", ms=3)
        plt.xlabel("trust-constr iteration"); plt.ylabel("CL/CD (surrogate)")
        plt.title("trust-constr convergence")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("trust_constr_convergence.png", dpi=130)
        plt.show()


if __name__ == "__main__":
    main()
