"""验证 least_core.py 的每一步是否正确（对照论文公式 + 有限差分）"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from least_core import (forward, backward, delta_and_grad, h_fun, is_dag,
                        generate_dataset, count_accuracy, least, topological_order)

rng = np.random.default_rng(0)
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        ok = False


print("=" * 72)
print("[A] FORWARD: δ^(k) 的基本性质")
print("=" * 72)
W_zero = np.zeros((4, 4))
check("全零矩阵 δ=0", abs(forward(W_zero, 5, 0.9)[0]) < 1e-12, f"δ={forward(W_zero,5,0.9)[0]:.2e}")

W_dag = np.array([[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1.5], [0, 0, 0, 0]], dtype=float)
d_dag, _ = forward(W_dag, 5, 0.9)
check("DAG 的 δ=0", abs(d_dag) < 1e-12, f"δ={d_dag:.2e}")

W_cyc = np.array([[0, 1], [1, 0]], dtype=float)
d1, _ = forward(W_cyc, 5, 0.9)
W_cyc2 = np.array([[0, 2], [2, 0]], dtype=float)
d2, _ = forward(W_cyc2, 5, 0.9)
check("有环图 δ>0", d1 > 0, f"δ(w=1)={d1:.4f}")
check("环权重越大 δ 越大", d2 > d1, f"δ(w=2)={d2:.4f}")
# 谱半径上界性质: δ^(k) >= ρ(W∘W)
S = W_cyc * W_cyc
rho = max(abs(np.linalg.eigvals(S)))
check("δ 是谱半径上界 (δ>=ρ)", d1 >= rho - 1e-9, f"δ={d1:.4f} >= ρ={rho:.4f}")

# 随机 DAG 上验证 δ→0
for d in (10, 20):
    W, X, meta = generate_dataset(d=d, seed=7)
    dd, _ = forward(W, 5, 0.9)
    check(f"随机 ER DAG d={d} 的 δ≈0", dd < 1e-10, f"δ={dd:.3e}, h={h_fun(W):.3e}")

# δ 与 h 的相关性（论文观察 2：>0.8）
print()
print("=" * 72)
print("[B] δ 与 h 的相关性（论文 §V-A 观察 2：应 > 0.8）")
print("=" * 72)
_traj = []
res = least(X, 20, k=5, alpha=0.9, zeta=1e-4, lam=0.5, eps=1e-4, B=1.005,
            T_o=60, T_i=200, lr=0.01, seed=3, use_h_termination=True,
            callback=lambda o, dd, hh, r, e, W: _traj.append((dd, hh)))
ds = [t[0] for t in _traj]
hs = [t[1] for t in _traj]
corr = np.corrcoef(np.log10(np.maximum(ds, 1e-12)), np.log10(np.maximum(hs, 1e-12)))[0, 1]
check("LEAST 轨迹上 corr(log δ, log h) > 0.8", corr > 0.8, f"corr={corr:.4f} (n={len(ds)})")


print()
print("=" * 72)
print("[C] BACKWARD: 与有限差分对比（论文 Eq.7/9/10）")
print("=" * 72)


def fd_grad(W, k=5, alpha=0.9, h=1e-6):
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy(); Wp[i, j] += h
            Wm = W.copy(); Wm[i, j] -= h
            g[i, j] = (forward(Wp, k, alpha)[0] - forward(Wm, k, alpha)[0]) / (2 * h)
    return g


for trial, (d, dens) in enumerate([(5, 1.0), (6, 1.0), (8, 0.6), (10, 0.3)]):
    W = rng.normal(0, 0.6, (d, d))
    if dens < 1.0:
        W = W * (rng.random((d, d)) < dens)
    np.fill_diagonal(W, 0.0)
    if np.count_nonzero(W) < 3:
        continue
    delta, cache = forward(W, 5, 0.9)
    g_an = backward(W, cache)
    g_fd = fd_grad(W)
    mask = W != 0
    err = np.abs(g_an[mask] - g_fd[mask]).max() / max(np.abs(g_fd[mask]).max(), 1e-9)
    check(f"d={d} 密度={dens} 解析梯度==有限差分", err < 1e-4,
          f"最大相对误差={err:.2e}  (nnz={mask.sum()})")

print()
print("=" * 72)
print("[D] 数据生成：与 NOTEARS 官方流程一致性")
print("=" * 72)
for d, deg in ((10, 2), (20, 2), (50, 2)):
    W, X, meta = generate_dataset(d=d, seed=42, degree=deg)
    check(f"d={d} 是 DAG", is_dag(W))
    check(f"d={d} 边数≈{deg*d/2}", abs(meta["num_edges"] - deg * d / 2) <= max(2, 0.2 * d),
          f"实际={meta['num_edges']}")
    nz = W[W != 0]
    check(f"d={d} 权重在 ±[0.5,2]", np.all(np.abs(nz) >= 0.5) and np.all(np.abs(nz) <= 2.0),
          f"范围=[{np.abs(nz).min():.2f},{np.abs(nz).max():.2f}]")
# SF 图
W, X, meta = generate_dataset(d=30, seed=1, graph_type="SF", degree=4)
check("SF 图是 DAG", is_dag(W), f"边数={meta['num_edges']} (期望≈{4*30/2})")

# 检查 SEM 采样正确性: X ≈ XW + noise 的残差应为白噪声
W, X, meta = generate_dataset(d=20, n=5000, seed=11)
resid = X - X @ W
check("SEM 采样正确 (残差接近独立单位噪声)",
      abs(resid.std() - 1.0) < 0.15, f"残差 std={resid.std():.3f} (期望≈1.0)")

print()
print("=" * 72)
print("[E] 端到端：ER-2 Gaussian（论文 §V-A 协议：ε×τ 网格取最佳）")
print("=" * 72)
from least_core import PAPER_EPS_GRID, PAPER_TAU_GRID
for d in (10, 20, 50):
    W_true, X, meta = generate_dataset(d=d, seed=42)
    best = None
    for eps in PAPER_EPS_GRID:
        res = least(X, d, k=5, alpha=0.9, zeta=1e-4, lam=0.5, eps=eps,
                    B=1.005, T_o=1000, T_i=200, lr=0.01, seed=42,
                    warm_start=True, use_h_termination=True, max_seconds=60)
        m = count_accuracy(W_true, res["W"], taus=PAPER_TAU_GRID)
        if best is None or m["f1"] > best[0]["f1"]:
            best = (m, res, eps)
    m, res, eps = best
    print(f"  d={d}(真边={meta['num_edges']}): converged={res['converged']} "
          f"outer={res['n_outer']} δ={res['delta_final']:.2e} h={res['h_final']:.2e} "
          f"ε={eps} time={res['elapsed']:.1f}s")
    print(f"        F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
          f"SHD={m['shd']} τ={m['best_tau']}")
    check(f"d={d} F1 > 0.8", m["f1"] > 0.8, f"F1={m['f1']:.4f}")
    check(f"d={d} 输出是无环图", is_dag(np.abs(res["W"]) > m["best_tau"]))

print()
print("ALL PASS" if ok else "SOME CHECKS FAILED")
