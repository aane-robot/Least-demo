"""
================================================================================
 notears_baseline.py  —  NOTEARS (NeurIPS 2018, 论文 ref [38]) 线性版基线
================================================================================
 仅用于在 Demo 中与 LEAST 做「精度 / 耗时」对照（论文 §V-A 的对照方法）。

 NOTEARS 使用精确无环约束  h(W) = Tr(exp(W∘W)) - d
   - 计算 h 需要矩阵指数  →  O(d^3) 时间、O(d^2) 空间
   - ∇_W h = (exp(W∘W))^T ∘ 2W
 这正是 LEAST 要替换掉的东西（LEAST 的 δ 是近 O(d)）。
================================================================================
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as slin
from scipy.optimize import minimize

from least_core import h_fun


def _h_and_grad(W):
    """h(W) = Tr(exp(W∘W)) - d  以及  ∇_W h = (exp(W∘W))^T ∘ 2W"""
    S = W * W
    E = slin.expm(S)
    h = np.trace(E) - W.shape[0]
    G = E.T * W * 2.0
    return h, G


def notears_linear(X, lambda1: float = 0.5, max_iter: int = 100, h_tol: float = 1e-8,
                   rho_max: float = 1e16, w_threshold: float = 0.3,
                   loss_type: str = "l2", verbose: bool = False):
    """NOTEARS 线性 SEM 版（官方 notears.linear.notears_linear 的等价实现）。

    返回 dict: W, h_final, converged, elapsed, n_iter
    """
    import time
    t0 = time.time()
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    C = (X.T @ X) / n                     # 预计算，加速最小二乘部分
    rho, alpha, h = 1.0, 0.0, np.inf
    W_est = np.zeros((d, d))
    total_iter = 0

    def _loss(W):
        M = W.reshape(d, d)
        # 0.5/n * ||X - XW||^2 = 0.5*(tr(C) - 2 tr(W^T C) + tr(W^T C W))
        loss = 0.5 * (np.trace(C) - 2.0 * np.trace(M.T @ C) + np.trace(M.T @ C @ M))
        h_val = _h_and_grad(M)[0]
        return loss + 0.5 * rho * h_val * h_val + alpha * h_val + lambda1 * np.abs(M).sum()

    def _grad(W):
        M = W.reshape(d, d)
        G_loss = (C @ M - C)              # ∇ 0.5/n||X-XW||^2
        h_val, G_h = _h_and_grad(M)
        G = G_loss + (rho * h_val + alpha) * G_h + lambda1 * np.sign(M)
        return G.ravel()

    for it in range(int(max_iter)):
        sol = minimize(_loss, W_est.ravel(), jac=_grad, method="L-BFGS-B",
                       options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-10})
        W_new = sol.x.reshape(d, d)
        h_new = _h_and_grad(W_new)[0]
        total_iter += 1
        if h_new > 0.25 * h:
            rho *= 10.0
        else:
            W_est, h = W_new, h_new
            alpha += rho * h_new
            if h_new <= h_tol:
                break
            rho *= 10.0
        if rho > rho_max:
            break
        if verbose:
            print(f"  [notears iter {it}] h={h_new:.3e} rho={rho:.1e}")

    W_est[np.abs(W_est) < w_threshold] = 0.0
    return {"W": W_est, "h_final": float(_h_and_grad(W_est)[0]),
            "converged": bool(h <= h_tol), "n_iter": total_iter,
            "elapsed": time.time() - t0}
