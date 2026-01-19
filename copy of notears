# notears_linear.py
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


def _vec(W: np.ndarray) -> np.ndarray:
    return W.reshape(-1, order="F")


def _mat(w: np.ndarray, d: int) -> np.ndarray:
    return w.reshape((d, d), order="F")


def _h(W: np.ndarray) -> float:
    # Acyclicity constraint: h(W)=tr(expm(W◦W)) - d
    d = W.shape[0]
    return float(np.trace(expm(W * W)) - d)


def _grad_h(W: np.ndarray) -> np.ndarray:
    # ∇h(W) = (expm(W◦W))^T ◦ (2W)
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
    Correct NOTEARS (linear, squared loss) with augmented Lagrangian.

    X: (n, d) data matrix. SHOULD be standardized (zero mean, unit variance).
    lambda1: L1 sparsity coefficient.
    Returns:
        W: (d, d) weighted adjacency matrix (diagonal = 0)
    """
    n, d = X.shape

    # Precompute
    XtX = X.T @ X / n

    # Objective and gradient for squared loss: 0.5/n ||X - XW||^2
    def loss(W: np.ndarray) -> float:
        # 0.5 * E[(x - xW)^2] = 0.5 * tr(XtX) - tr(XtX W) + 0.5 * tr(W^T XtX W)
        return 0.5 * np.trace(XtX) - np.trace(XtX @ W) + 0.5 * np.trace(W.T @ XtX @ W)

    def grad_loss(W: np.ndarray) -> np.ndarray:
        return -XtX + XtX @ W

    # We optimize over w = vec(W)
    w_est = np.zeros(d * d)
    rho, alpha = 1.0, 0.0

    # Bounds: enforce zero diagonal by fixing those entries to 0
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
        # smooth part
        f = loss(W)
        g = grad_loss(W)
        # augmented lagrangian
        obj = f + 0.5 * rho * h_val * h_val + alpha * h_val + lambda1 * np.sum(np.abs(W))
        # gradient of smooth part + constraint part
        grad = g + (rho * h_val + alpha) * _grad_h(W)
        # subgradient for L1 term
        grad += lambda1 * np.sign(W)
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

        if h_new <= h_tol:
            w_est = w_new
            break

        # Update dual variables
        alpha += rho * h_new
        rho *= 10.0
        if rho > rho_max:
            break

        w_est = w_new

    W_final = _mat(w_est, d)

    # Optional thresholding for numerical small values
    if w_threshold > 0:
        W_final[np.abs(W_final) < w_threshold] = 0.0

    # Always force diagonal 0
    np.fill_diagonal(W_final, 0.0)
    return W_final