"""
nn_surrogate.py
───────────────
Wraps the trained AirfoilNet into two callables:

  • surrogate_fn(x_phys)  → returns  -CL/CD   (scalar)
  • surrogate_grad(x_phys) → returns  d(-CL/CD)/dx   (same shape as x_phys)

x_phys is a 1-D numpy array (or list) of the design variables in PHYSICAL
units.  The exact ordering must match the columns of ``df_x`` built in
main.ipynb (reynoldsNumber, alpha, + 19 geometry features → 21 total).

NOTE: The network was trained with the *log* of CD (``np.log(cd)``),
so the second output column is log(CD).  When computing CL/CD,
this module first converts log(CD) → CD  via ``torch.exp()``.

These two functions are all you need to hand to any gradient-based
*or* derivative-free optimiser (scipy.optimize.minimize, COBYQA, etc.).

Usage (after running main.ipynb cells 0–4 so that model, X_mean, X_std,
y_mean, y_std, and device are all in scope):

    from nn_surrogate import build_surrogate
    surrogate_fn, surrogate_grad, surrogate_fn_and_grad = build_surrogate(
        model, X_mean, X_std, y_mean, y_std, device
    )

    # Now use them however you like:
    val  = surrogate_fn(x0)       # scalar  −CL/CD
    grad = surrogate_grad(x0)     # np array of shape (n_features,)
"""

import numpy as np
import torch


def build_surrogate(model, X_mean, X_std, y_mean, y_std, device):
    """
    Returns (surrogate_fn, surrogate_grad, surrogate_fn_and_grad).

    Both accept a 1-D numpy array of physical design variables and
    return numpy scalars / arrays, so they slot straight into
    scipy.optimize or any other framework.
    """

    # ── freeze the network weights ──────────────────────────────────────
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_features = X_mean.shape[-1]

    # ── cache normalisation constants on the right device ───────────────
    xm = X_mean.to(device).squeeze()   # shape (n_features,)
    xs = X_std.to(device).squeeze()
    ym = y_mean.to(device).squeeze()   # shape (3,)  → [CL_mean, logCD_mean, CM_mean]
    ys = y_std.to(device).squeeze()

    # ── helper: physical numpy → normalised tensor (with grad) ──────────
    def _to_normalised_tensor(x_phys):
        """Convert a 1-D numpy/list input to a normalised, grad-enabled tensor."""
        x_t = torch.tensor(np.asarray(x_phys, dtype=np.float32),
                           dtype=torch.float32, device=device)
        x_t = x_t.detach().requires_grad_(True)           # design variables need grad
        x_norm = (x_t - xm) / xs                          # z-score normalise
        return x_t, x_norm.unsqueeze(0)                    # add batch dim → (1, n_features)

    # ── helper: run model, denormalise, compute −CL/CD ──────────────────
    def _forward(x_phys):
        """
        Returns
        -------
        x_t    : the leaf tensor (for .grad after backward)
        obj    : scalar tensor  −CL/CD  (differentiable)
        """
        x_t, x_norm = _to_normalised_tensor(x_phys)
        y_norm = model(x_norm)                             # (1, 3) normalised
        y_phys = y_norm * ys + ym                          # de-normalise → (1, 3)

        cl     = y_phys[0, 0]
        log_cd = y_phys[0, 1]                              # the model predicts log(CD)
        cd     = torch.exp(log_cd)                         # convert to actual CD
        # small eps to avoid division by zero if CD ≈ 0
        obj = -(cl / (cd + 1e-8))
        return x_t, obj

    # ── public API ──────────────────────────────────────────────────────
    def surrogate_fn(x_phys):
        """Evaluate −CL/CD at a physical design point.  Returns a float."""
        with torch.enable_grad():
            _, obj = _forward(x_phys)
        return float(obj.detach().cpu())

    def surrogate_grad(x_phys):
        """Gradient of −CL/CD w.r.t. the physical design variables.
        Returns a numpy array of shape (n_features,)."""
        with torch.enable_grad():
            x_t, obj = _forward(x_phys)
            obj.backward()
        return x_t.grad.detach().cpu().numpy()

    def surrogate_fn_and_grad(x_phys):
        """
        Convenience function that returns BOTH value and gradient in one
        forward+backward pass (more efficient than calling the two above
        separately).  Useful for scipy L-BFGS-B which wants jac=True.

        Returns
        -------
        val  : float
        grad : np.ndarray of shape (n_features,)
        """
        with torch.enable_grad():
            x_t, obj = _forward(x_phys)
            obj.backward()
        return float(obj.detach().cpu()), x_t.grad.detach().cpu().numpy()

    return surrogate_fn, surrogate_grad, surrogate_fn_and_grad
