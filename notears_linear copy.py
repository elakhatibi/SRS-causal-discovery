# notears_linear.py
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


def _vec(W: np.ndarray) -> np.ndarray:
    return W.reshape(-1, order="F")


def _mat(w: np.ndarray, d: int) -> np.ndarray:
    # Reshape into (d, d) in Fortran order
    return w.reshape((d, d), order="F")


def _h(W: np.ndarray) -> float:
    """
    Acyclicity constraint:
      h(W) = tr(expm(W ◦ W)) - d
    where ◦ is elementwise product.
    """
    d = W.shape[0]
    return float(np.trace(expm(W * W)) - d)


def _grad_h(W: np.ndarray) -> np.ndarray:
    """
    Gradient of h(W):
      ∇h(W) = (expm(W◦W))^T ◦ (2W)
    """
    E = expm(W * W)
    return (E.T * W) * 2.0


def notears_linear(
    X: np.ndarray,
    lambda1: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.0,
) -> np.ndarray:
    """
    NOTEARS (linear, squared loss) with augmented Lagrangian.

    Parameters
    ----------
    X : np.ndarray
        (n, d) data matrix. SHOULD be standardized (zero mean, unit variance).
    lambda1 : float
        L1 sparsity coefficient.
    max_iter : int
        Max augmented Lagrangian outer iterations.
    h_tol : float
        Tolerance for acyclicity constraint h(W).
    rho_max : float
        Max penalty parameter.
    w_threshold : float
        Optional threshold applied after optimization.

    Returns
    -------
    W : np.ndarray
        (d, d) weighted adjacency matrix (diagonal = 0)
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # Precompute XtX/n for squared loss
    XtX = (X.T @ X) / n

    def loss(W: np.ndarray) -> float:
        # 0.5 * tr(XtX) - tr(XtX W) + 0.5 * tr(W^T XtX W)
        return 0.5 * np.trace(XtX) - np.trace(XtX @ W) + 0.5 * np.trace(W.T @ XtX @ W)

    def grad_loss(W: np.ndarray) -> np.ndarray:
        # derivative wrt W for squared loss
        return -XtX + XtX @ W

    # Optimize over vectorized W
    w_est = np.zeros(d * d)
    rho, alpha = 1.0, 0.0

    # Bounds enforce diagonal = 0
    bounds = []
    for j in range(d):
        for i in range(d):
            if i == j:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((None, None))

    def aug_lagrangian(w: np.ndarray) -> tuple[float, np.ndarray]:
        W = _mat(w, d)
        h_val = _h(W)

        f = loss(W)
        g = grad_loss(W)

        # Augmented Lagrangian objective
        obj = f + 0.5 * rho * (h_val ** 2) + alpha * h_val + lambda1 * np.sum(np.abs(W))

        # Gradient of smooth part + constraint part
        grad = g + (rho * h_val + alpha) * _grad_h(W)

        # L1 subgradient
        grad = grad + lambda1 * np.sign(W)

        return float(obj), _vec(grad)

    for _ in range(max_iter):
        sol = minimize(
            fun=lambda w: aug_lagrangian(w)[0],
            x0=w_est,
            jac=lambda w: aug_lagrangian(w)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        w_new = sol.x
        W_new = _mat(w_new, d)
        h_new = _h(W_new)

        # Check acyclicity
        if h_new <= h_tol:
            w_est = w_new
            break

        # Dual update
        alpha += rho * h_new
        rho *= 10.0
        if rho > rho_max:
            w_est = w_new
            break

        w_est = w_new

    # IMPORTANT FIX:
    # Make W_final writable before modifying it (some reshape/minimize outputs can be read-only views)
    W_final = _mat(w_est, d).copy()

    # Optional thresholding
    if w_threshold > 0.0:
        W_final[np.abs(W_final) < w_threshold] = 0.0

    # Force diagonal zeros
    np.fill_diagonal(W_final, 0.0)

    return W_final
