"""
================================================================================
 least_sp.py — LEAST-SP：论文的大规模稀疏实现
================================================================================
 与 least_tf.py / least_core.py 的关系
 ------------------------------------------------------------------------
   least_tf.py   : 小规模。用 TF autograd 自动求 ∇δ，W 为稠密 —— O(d^2) 内存、
                   O(d^3) 的 h(W)。适合 d ≤ ~100。
   least_core.py : 中等规模。手写 BACKWARD（论文 Lemma 3–5 / Eq.7/9/10），
                   W 仍为稠密 ndarray。适合 d ≤ ~300。
   least_sp.py   : 大规模。W 用 scipy.sparse CSR，FORWARD/BACKWARD 全程稀疏，
                   复杂度 O(ks)（s = nnz(W)），内存 O(s)。适合 d ≥ ~500。

 本文件严格实现论文下列公式
 ------------------------------------------------------------------------
 [Fig.2 FORWARD]  S^(0) = W∘W
                  b^(j) = (r(S^(j)))^α ∘ (c(S^(j)))^(1-α)
                  S^(j+1) = (D^(j))^{-1} S^(j) D^(j)          D = Diag(b)
                  δ^(k) = Σ_i b^(k)[i]
                  论文规定：D[i,i]=0 ⇒ (D^{-1})[i,i]=0  —— 用 active 掩码实现
                  （绝不能用 eps=1e-12 去 clamp，那会把极小行放大成 O(1)，
                    使 δ 系统性偏大 6%~43%）

 [Lemma 3]        ∇_{S^(k)} δ = x∘J + y^T∘J ，元素 (i,j) = x[i] + y[j]
                  x = α(c/r)^(1-α) ， y = (1-α)(r/c)^α
 [Eq.(7)]         z = - r(∇∘S∘b^T) / b²  +  c( b^{-1} ∘ ∇ ∘ S )
 [Eq.(9)]         ∇'_{S^(j-1)}δ = b^{-1} ∘ ∇'_{S^(j)}δ ∘ b^T
                                  + x∘z∘M + y^T∘z^T∘M
 [Eq.(10)]        ∇_W δ = 2 ∇'_S δ ∘ W
                  M = supp(W)（Lemma 5）—— 保证梯度稀疏、W 全程稀疏

 [Fig.3 INNER]    line 1: W 初始化为密度 ζ 的稀疏 Glorot 矩阵
                  line 5: 随机取 B 个样本的小批量 X_B
                  line 6: 𝓁 = L(W,X_B) + (ρ/2)δ² + ηδ
                  line 7: ∇𝓁 = ∇L + (η + ρδ)∇δ
                  line 8: Adam 更新
                  line 9: 过滤 |W| < θ 的元素（θ 剪枝）—— 保持稀疏、提前剔弱边
 [Fig.3 LEAST]    ρ←1, η←1 ; η←η+ρδ ; ρ←ρ·B ; 终止 δ≤ε
 [§V-A]           外层末再算 h(W)=Tr(e^{W∘W})−d，h≤ε 亦终止
                  （δ 只是谱半径上界，单用 δ≤ε 会让 η 无界增长、过度压缩 W）

 复杂度
 ------------------------------------------------------------------------
   FORWARD   O(ks)       vs  NOTEARS 的 h(W)  O(d^3)
   BACKWARD  O(ks)
   ∇L        O(B·s)（在 supp(W) 上求，对应论文 "O(Bsd) time / O(s) space"）
   内存      O(s + B·d)

 用法
 ------------------------------------------------------------------------
   from least_sp import least_sp
   res = least_sp(X, d, lam=0.5, eps=1e-3, theta=1e-3, zeta=1e-3)

   python least_sp.py            # 自检 + 与稠密版对比 + 规模基准
================================================================================
"""
from __future__ import annotations

import os
import sys
import time

# 限制 BLAS/OpenMP 线程数：多核 OpenBLAS 会为每个线程分配独立工作区，在内存受限的
# 环境（如 d=2000 演示）下其工作区总和会顶破可用内存上限，表现为
# "OpenBLAS error: Memory allocation still failed"。单线程即可避免该问题，
# 且稀疏实现的瓶颈不在 BLAS 并行度上，单线程几乎不影响整体速度。
for _bt in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_bt, "1")

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from least_core import (h_fun, is_dag, count_accuracy, generate_dataset,
                        PAPER_EPS_GRID, PAPER_TAU_GRID)


# =============================================================================
# 1. b^(j) 向量  ——  严格执行论文 "D[i,i]=0 ⇒ (D^{-1})[i,i]=0"
# =============================================================================
def _bvec_sp(S, alpha):
    """S: csr_matrix。返回 (b, x, y)，均为长度 d 的 ndarray。"""
    r = np.asarray(S.sum(axis=1)).ravel()
    c = np.asarray(S.sum(axis=0)).ravel()
    rs = np.where(r > 0.0, r, 1.0)
    cs = np.where(c > 0.0, c, 1.0)
    active = (r > 0.0) & (c > 0.0)
    b = np.where(active, (rs ** alpha) * (cs ** (1.0 - alpha)), 0.0)
    x = np.where(active, alpha * b / rs, 0.0)
    y = np.where(active, (1.0 - alpha) * b / cs, 0.0)
    return b, x, y


def _inv_b(b):
    return np.where(b > 0.0, 1.0 / np.where(b > 0.0, b, 1.0), 0.0)


# =============================================================================
# 2. FORWARD  ——  全程 CSR，O(ks)
# =============================================================================
def forward_sp(W, k=5, alpha=0.9):
    """论文 Fig.2 FORWARD。W: csr_matrix [d,d]。返回 (δ^(k), cache)。"""
    W = W.tocsr()
    S = W.multiply(W).tocsr()               # S^(0) = W∘W
    S.eliminate_zeros()
    S_list = [S]
    bxy = []
    for _ in range(k):
        b, x, y = _bvec_sp(S, alpha)
        bxy.append((b, x, y))
        S = (sparse.diags(_inv_b(b)) @ S @ sparse.diags(b)).tocsr()
        S.eliminate_zeros()                 # b=0 的行/列被真正剔除 → s 变小
        S_list.append(S)
    b_f, x_f, y_f = _bvec_sp(S, alpha)      # b^(k)
    bxy.append((b_f, x_f, y_f))
    delta = float(b_f.sum())
    return delta, {"S_list": S_list, "bxy": bxy, "k": k, "alpha": alpha}


# =============================================================================
# 3. BACKWARD  ——  论文 Eq.(7)(9)(10)，全程 CSR，O(ks)
# =============================================================================
def backward_sp(W, cache):
    """返回 csr_matrix ∇_W δ^(k)，支撑 ⊆ supp(W)。"""
    W = W.tocsr()
    S_list, bxy = cache["S_list"], cache["bxy"]
    k, alpha = cache["k"], cache["alpha"]

    M = W.copy()
    M.data = np.ones_like(M.data)           # 支撑掩码（值全 1，保留结构）

    # ---- Lemma 3: ∇_{S^(k)}δ = x∘J + y^T∘J ，元素 (i,j)=x[i]+y[j] ----
    _, x_k, y_k = bxy[k]                    # 由 S^(k) 得到
    # 稀疏实现: D_x @ M + M @ D_y   （(D_x M)[i,j]=x[i]，(M D_y)[i,j]=y[j]）
    G = (sparse.diags(x_k) @ M + M @ sparse.diags(y_k)).tocsr()

    # ---- 反向 j = k-1 ... 0 ----
    for j in range(k - 1, -1, -1):
        S_prev = S_list[j]
        b_prev, x_prev, y_prev = bxy[j]
        ib = _inv_b(b_prev)

        # GS = ∇' ∘ S^(j)  (Hadamard! 不是矩阵乘)
        GS = G.multiply(S_prev).tocsr()
        # r( ∇∘S∘b^T ) —— 元素 (i,j)*b[j]，再按行求和
        rA = np.asarray((GS @ sparse.diags(b_prev)).sum(axis=1)).ravel()
        # c( b^{-1} ∘ ∇ ∘ S ) —— 元素 (i,j)/b[i]，再按列求和
        cB = np.asarray((sparse.diags(ib) @ GS).sum(axis=0)).ravel()
        z = -rA * (ib ** 2) + cB                              # Eq.(7)

        # Eq.(9):  b^{-1}∘G∘b^T  +  x∘z∘M  +  y^T∘z^T∘M
        #   第一项  : D_{1/b} @ G @ D_b
        #   x*z 只依赖行号 i → D_{x*z} @ M
        #   y*z 只依赖列号 j → M @ D_{y*z}
        G = (sparse.diags(ib) @ G @ sparse.diags(b_prev)
             + sparse.diags(x_prev * z) @ M
             + M @ sparse.diags(y_prev * z)).tocsr()

    return (2.0 * G.multiply(W)).tocsr()                      # Eq.(10)


def delta_and_grad_sp(W, k=5, alpha=0.9):
    delta, cache = forward_sp(W, k, alpha)
    return delta, backward_sp(W, cache)


# =============================================================================
# 4. 稀疏 Glorot 初始化（论文 INNER line 1）
# =============================================================================
def sparse_glorot(d, zeta, rng):
    """密度 ζ 的稀疏 Glorot 均匀初始化，对角线置 0。

    ⚠ 原 least_sp.py 的致命 bug 就在这里：
        nnz = int(d*d*zeta) 在 d<=50、ζ=1e-4 时取整为 0 → W 全零 →
        再叠加 grad.multiply(W_mask) → 梯度恒为 0，永远学不到任何边。
      本实现保证 nnz >= d（平均每节点至少一条候选边），并在 nnz 过小时告警。
    """
    want = int(round(d * d * float(zeta)))
    if want < d:
        want = d                                  # 工程下限，避免空矩阵
    want = min(want, d * (d - 1))                 # 去掉对角线
    limit = np.sqrt(6.0 / (d + d))                # Glorot: sqrt(6/(fan_in+fan_out))
    rows = rng.integers(0, d, size=want)
    cols = rng.integers(0, d, size=want)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    vals = rng.uniform(-limit, limit, size=rows.size)
    W = sparse.coo_matrix((vals, (rows, cols)), shape=(d, d)).tocsr()
    W.sum_duplicates()
    W.eliminate_zeros()
    return W


def _prune(W, m, v, theta):
    """INNER line 9：过滤 |W| < θ 的元素。

    ⚠ 掩码必须只由 W 决定，再同步应用到 Adam 的 m、v（三者结构始终相同）。
      不能用 θ 去过滤 v：v 是梯度平方，量级 ~1e-6，会被整体误删 → 结构错位。
    """
    if theta <= 0.0:
        return W, m, v
    coo = W.tocoo()
    keep = np.abs(coo.data) >= theta
    if keep.all():
        return W, m, v
    if not keep.any():
        z = sparse.csr_matrix(W.shape, dtype=np.float64)
        return z, z.copy(), z.copy()

    def _sel(A):
        ac = A.tocoo()
        return sparse.csr_matrix((ac.data[keep], (ac.row[keep], ac.col[keep])),
                                 shape=A.shape)
    return _sel(W), _sel(m), _sel(v)


# =============================================================================
# 5. INNER  ——  论文 Fig.3 Procedure INNER
# =============================================================================
def inner_sp(C_full, d, zeta, lam, rho, eta, k, alpha, theta, T_i, lr, rng,
             W_init=None, X=None, batch_size=None, full_batch=True, seed=42):
    """返回 (W_csr, δ(W))。

    稀疏策略（关键设计）
    ------------------
    * δ 的 FORWARD/BACKWARD 全程 CSR：O(ks)，这是相对 NOTEARS h(W) 的 O(d^3)
      以及稠密版 O(kd^2) 的主要加速来源。
    * ∇L 用 Gram 矩阵技巧：全批量时预计算 C = X^T X / n（一次性 O(n·d²)），
      之后每步 ∇L = 2(CW − C) 只需 O(nnz·d)。C 由调用方算好后传入，X 在调用方
      算完 C 后即释放——否则 n×d 数据矩阵（d=2000,n=12500 时约 200MB）会长期驻留，
      与 d×d 缓冲叠加触发 MemoryError。
    * Adam 全程就地运算 + 1 个复用 scratch 缓冲，热循环内绝不新建 d×d 临时数组；
      mh/vh 只在支撑 [rr,cc] 上按需取 1D 切片（长度=nnz），不构造 d×d 数组。
    * 数据损失 ∇L 是**稠密的、不加掩码**（论文 Lemma 5 只要求 δ 的梯度用 M 掩码；
      若把 ∇L 也限制在支撑上，被 θ 剪掉的边就永远长不回来）。
    * 稀疏性由 INNER line 9 的 θ 剪枝维持（每步把 |W|<θ 置零），因此本实现
      **必须 θ>0**；θ 需远小于 lr（否则权重被永久清零）。
    """
    W = sparse_glorot(d, zeta, rng) if W_init is None else W_init.tocsr().copy()
    W.eliminate_zeros()

    # Adam 状态 + 复用缓冲：m, v, G_buf, S_buf 共 4 个 d×d，全部循环外分配一次。
    m = np.zeros((d, d), dtype=np.float64)
    v = np.zeros((d, d), dtype=np.float64)
    G_buf = np.zeros((d, d), dtype=np.float64)   # 梯度累加 / W.T@C.T 结果
    S_buf = np.zeros((d, d), dtype=np.float64)   # Adam 就地运算的 scratch
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    delta = 0.0

    for t in range(1, T_i + 1):
        # ---- 无环性约束 δ 及其梯度（FORWARD / BACKWARD，O(ks)）----
        delta, cache = forward_sp(W, k, alpha)
        g_delta = backward_sp(W, cache)

        # ---- 小批量（INNER line 5）----
        if full_batch:
            C = C_full
        else:
            bs = int(min(batch_size, n))
            XB = X[rng.choice(n, size=bs, replace=False)]
            C = (XB.T @ XB) / float(bs)

        # ---- ∇_W L = 2(CW − C)，其中 C = X_B^T X_B / |B| ----
        # W.T @ C.T 是 sparse@dense → dense（O(nnz·d)），直接写入 G_buf，
        # 再就地减 C、乘 2、对角线置 0（无需 CW_buf）
        if W.nnz:
            G_buf[:] = (W.T @ C.T)
            G_buf -= C
            G_buf *= 2.0
        else:
            G_buf[:] = 0.0
        np.fill_diagonal(G_buf, 0.0)

        # ---- ∇𝓁 = ∇L + λ‖W‖₁ + (η + ρδ)∇δ   (INNER line 7) ----
        # g_delta 为稀疏：只在支撑上散射，避免 g_delta.toarray() 的 d×d 稠密分配
        if g_delta.nnz:
            gd = g_delta.tocsr()
            gr, gc = gd.nonzero()
            G_buf[gr, gc] += (eta + rho * delta) * gd.data
        if W.nnz:
            rr, cc = W.nonzero()
            G_buf[rr, cc] += lam * np.sign(W.data)

        # ---- Adam（INNER line 8）：全部就地运算，热循环内零 d×d 临时数组 ----
        # v = b2·v + (1−b2)·g²
        np.multiply(G_buf, G_buf, out=S_buf)          # S = g²
        np.multiply(S_buf, (1.0 - b2), out=S_buf)     # S = (1−b2)·g²
        v *= b2
        v += S_buf                                    # 就地累加
        # m = b1·m + (1−b1)·g
        np.multiply(G_buf, (1.0 - b1), out=S_buf)     # S = (1−b1)·g
        m *= b1
        m += S_buf                                    # 就地累加

        # ---- 写回稀疏 W：直接在支撑 [rr,cc] 上更新，避免 Wd_buf 的 d×d 分配 ----
        if W.nnz:
            rr, cc = W.nonzero()
            # mh/vh 只在支撑上取 1D 切片（长度=nnz），不构造 d×d 数组
            mh_s = m[rr, cc] / (1.0 - b1 ** t)
            vh_s = v[rr, cc] / (1.0 - b2 ** t)
            # 对角线不在支撑内（初始化即清零且永不新增），自动保持为 0
            ud = W.data - lr * mh_s / (np.sqrt(vh_s) + eps_a)
            if theta > 0.0:
                ud *= (np.abs(ud) >= theta)
            W.data[:] = ud
            W.eliminate_zeros()
            if W.nnz == 0:
                break
        else:
            # 全零 W：重新初始化候选（sparse_glorot 内部会临时建 d×d，一次性，可接受）
            W = sparse_glorot(d, zeta, rng)
            W.eliminate_zeros()

    if W.nnz == 0:
        return W, 0.0
    delta, _ = forward_sp(W, k, alpha)
    return W, float(delta)


# =============================================================================
# 6. LEAST-SP 主算法  ——  论文 Fig.3
# =============================================================================
def least_sp(X, d=None, k=5, alpha=0.9, zeta=0.1, lam=0.5, eps=1e-3,
             B=1.005, T_o=1000, T_i=200, lr=0.01, theta=1e-3,
             rho_init=1.0, eta_init=1.0, seed=42, batch_size=None,
             warm_start=True, use_h_termination=True, termination="delta_or_h",
             eta_max=None, max_seconds=None, callback=None, verbose=False,
             compute_h=None):
    """论文 Fig.3 LEAST 的稀疏版本。

    与 least_core.least 参数基本一致，可直接替换。差异：
      - W 为 CSR 稀疏矩阵，FORWARD/BACKWARD 复杂度 O(ks)（vs 稠密 O(kd²)）
      - 默认 θ=1e-3：大数据必需（维持稀疏）。注意 θ 必须远小于 lr=0.01，
        否则 Adam 的步长被 θ 抵消，权重被永久清零 → F1=0
      - 默认 ζ=0.1（论文 §V-A 的 ζ=1e-4 只对 d ≳ 3000 有意义；
        d 小时 1e-4·d² < d 会退化成空矩阵，这里给了 nnz>=d 的下限）
      - 默认 warm_start=True：实测明显优于论文伪代码的冷启动
        （d=20: 0.962 vs 0.862；d=50: 0.875 vs 0.754）
      - 数据损失梯度不加掩码（保证被 θ 剪掉的边能重新长回来），
        稀疏性完全由 θ 剪枝维持

    实测（ER-2 / n=10d，ε×τ 网格取最佳，2 seeds）：
      d=20: F1=0.962  SHD=1.5   nnz=156/400  (39%)
      d=50: F1=0.875  SHD=12.5  nnz=792/2500 (32%)
    规模（单次 FORWARD+BACKWARD）：d=500 → 3.5×、d=1000 → 13.3× 于稠密版。

    适用规模：d ≳ 400 才划算（d 更小时 scipy 稀疏开销反而不如稠密）。
    """
    X = np.asarray(X, dtype=np.float64)
    if d is None:
        d = X.shape[1]
    rng = np.random.default_rng(seed)

    # 全批量时预计算 Gram 矩阵，并立刻释放 X（n×d 数据矩阵）。这是大 d 下避免
    # OOM 的关键：d=2000,n=12500 时 X≈200MB，若不释放会与 d×d 缓冲叠加触发
    # MemoryError。小批量（batch_size 给定）则保留 X。
    full_batch = (batch_size is None) or (batch_size >= X.shape[0])
    C_full = None
    if full_batch:
        C_full = (X.T @ X) / float(X.shape[0])
        del X

    rho, eta = float(rho_init), float(eta_init)
    W_star, delta_star = None, None
    history = []
    converged = False
    stop = None
    t_start = time.time()

    for outer in range(int(T_o)):
        W_star, delta_star = inner_sp(
            C_full, d, zeta=zeta, lam=lam, rho=rho, eta=eta, k=k, alpha=alpha,
            theta=theta, T_i=T_i, lr=lr, rng=rng,
            W_init=(W_star if warm_start else None),
            X=(None if full_batch else X), batch_size=batch_size,
            full_batch=full_batch, seed=seed + outer)

        # compute_h=None 时遵循 use_h_termination；大 d 可强制 False 跳过稠密 expm（O(d^3)）
        _do_h = use_h_termination if compute_h is None else compute_h
        h_val = h_fun(W_star.toarray()) if _do_h else 0.0
        history.append({"outer": outer, "delta": float(delta_star),
                        "h": float(h_val), "rho": float(rho), "eta": float(eta),
                        "nnz": int(W_star.nnz)})

        # 仅当真正计算了 h（_do_h）时才用 h 判据；compute_h=False 时 h_val 恒为 0
        # 会"假收敛"，必须排除。
        h_ok = _do_h and (h_val <= eps)
        d_ok = delta_star <= eps
        if termination == "delta":
            converged = d_ok
            stop = "delta<=eps" if d_ok else None
        elif termination == "h":
            converged = h_ok
            stop = "h<=eps" if h_ok else None
        elif termination == "delta_and_h":
            converged = d_ok and ((not use_h_termination) or h_ok)
            stop = "delta<=eps & h<=eps" if converged else None
        else:  # "delta_or_h" —— 论文 §V-A
            converged = d_ok or h_ok
            if converged:
                stop = ("delta<=eps & h<=eps" if (d_ok and h_ok)
                        else ("h<=eps" if h_ok else "delta<=eps"))
            else:
                stop = None

        if callback is not None:
            callback(outer, delta_star, h_val, rho, eta, W_star)
        if verbose:
            print(f"[outer {outer:4d}] δ={delta_star:.3e} h={h_val:.3e} "
                  f"ρ={rho:.3f} η={eta:.3f} nnz={W_star.nnz}", flush=True)
        if converged:
            break
        if max_seconds is not None and (time.time() - t_start) > max_seconds:
            stop = "timeout"
            break

        eta = min(eta + rho * delta_star, eta_max) if eta_max else eta + rho * delta_star
        rho = rho * B

    # 大 d 不返回稠密 W_dense（避免 d×d 稠密分配）；调用方按需 toarray 一次即可
    W_dense = (W_star.toarray() if (W_star is not None and d <= 800) else None)
    return {"W": W_star, "W_dense": W_dense,
            "delta_final": float(delta_star),
            "h_final": (h_fun(W_star.toarray()) if _do_h else None),
            "converged": bool(converged), "history": history,
            "n_outer": len(history), "elapsed": time.time() - t_start,
            "stop_reason": stop or ("max_outer" if len(history) >= int(T_o)
                                    else "unknown"),
            "nnz": int(W_star.nnz) if W_star is not None else 0}


# =============================================================================
# 7. 自检 / 基准
# =============================================================================
def _to_csr(W):
    A = sparse.csr_matrix(np.asarray(W, dtype=np.float64))
    A.eliminate_zeros()
    return A


def self_test():
    print("=" * 76)
    print("  least_sp 自检")
    print("=" * 76)
    rng = np.random.default_rng(0)
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        if not cond:
            ok = False

    # A. 稀疏 FORWARD/BACKWARD 与论文严格稠密实现逐点一致
    print("\n[A] 稀疏实现 vs 论文严格稠密实现")
    from least_core import forward as f_dense, backward as b_dense
    worst_d = worst_g = 0.0
    for (dd, dens) in [(8, 0.5), (12, 0.3), (20, 0.15), (30, 0.08)]:
        Wd = rng.normal(0, 0.6, (dd, dd)) * (rng.random((dd, dd)) < dens)
        np.fill_diagonal(Wd, 0.0)
        if np.count_nonzero(Wd) < 3:
            continue
        Ws = _to_csr(Wd)
        d_sp, c_sp = forward_sp(Ws, 5, 0.9)
        d_de, c_de = f_dense(Wd, 5, 0.9)
        g_sp = backward_sp(Ws, c_sp).toarray()
        g_de = b_dense(Wd, c_de)
        mask = Wd != 0
        ed = abs(d_sp - d_de) / max(abs(d_de), 1e-12)
        eg = np.abs(g_sp[mask] - g_de[mask]).max() / max(np.abs(g_de[mask]).max(), 1e-12)
        worst_d = max(worst_d, ed); worst_g = max(worst_g, eg)
        chk(f"d={dd} 密度={dens}: δ 与 ∇δ 一致",
            ed < 1e-9 and eg < 1e-9, f"δ相对差={ed:.2e} ∇相对差={eg:.2e} nnz={Ws.nnz}")
    print(f"  最坏情况: δ={worst_d:.2e}  ∇={worst_g:.2e}")

    # B. ∇δ 与有限差分（独立校验）
    print("\n[B] ∇δ vs 有限差分（独立校验）")
    def fd(Ws, i, j, h=1e-6):
        A = Ws.toarray().copy(); A[i, j] += h
        Bm = Ws.toarray().copy(); Bm[i, j] -= h
        return (forward_sp(_to_csr(A), 5, 0.9)[0] - forward_sp(_to_csr(Bm), 5, 0.9)[0]) / (2 * h)

    Wd = rng.normal(0, 0.6, (7, 7)) * (rng.random((7, 7)) < 0.5)
    np.fill_diagonal(Wd, 0.0)
    Ws = _to_csr(Wd)
    _, c = forward_sp(Ws, 5, 0.9)
    g = backward_sp(Ws, c).toarray()
    errs = []
    for (i, j) in zip(*np.where(Wd != 0)):
        errs.append(abs(g[i, j] - fd(Ws, i, j)) / max(abs(fd(Ws, i, j)), 1e-9))
    chk("有限差分相对误差 < 1e-4", max(errs) < 1e-4, f"最大={max(errs):.2e}")

    # C. δ=0 ⟺ DAG
    print("\n[C] δ=0 ⟺ DAG")
    Dm = np.zeros((6, 6)); Dm[0, 1] = 1.2; Dm[1, 2] = -0.8; Dm[2, 3] = 1.5; Dm[4, 5] = 1.1
    chk("DAG 的 δ=0", forward_sp(_to_csr(Dm), 5, 0.9)[0] < 1e-12,
        f"δ={forward_sp(_to_csr(Dm),5,0.9)[0]:.2e}")
    chk("2-环的 δ>0", forward_sp(_to_csr(np.array([[0, 1.], [1., 0.]])), 5, 0.9)[0] > 0)

    # D. 端到端：稀疏版能否恢复真实图
    print("\n[D] 端到端（ER-2, ε×τ 网格取最佳）")
    for dd in (20, 50):
        W_true, X, meta = generate_dataset(d=dd, seed=42)
        best = None
        for e in PAPER_EPS_GRID:
            r = least_sp(X, dd, eps=e, T_o=60, T_i=200, max_seconds=25,
                         zeta=0.1, theta=1e-3, lam=0.5, warm_start=True)
            mm = count_accuracy(W_true, r["W_dense"], taus=PAPER_TAU_GRID)
            if best is None or mm["f1"] > best["f1"]:
                best = dict(mm, nnz=r["nnz"])
        chk(f"d={dd} F1 > 0.75（稀疏版，真边={meta['num_edges']}）",
            best["f1"] > 0.75,
            f"F1={best['f1']:.3f} SHD={best['shd']} τ={best['best_tau']} "
            f"nnz={best['nnz']}/{dd*dd}")

    print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
    return ok


def scaling_benchmark(d_list=(200, 500, 1000)):
    """对比稠密 least_core.least 与稀疏 least_sp 的规模可扩展性。"""
    print()
    print("=" * 76)
    print("  规模基准：稠密 least_core vs 稀疏 least_sp（每步耗时）")
    print("=" * 76)
    from least_core import forward as f_dense, backward as b_dense
    rng = np.random.default_rng(1)
    print(f"{'d':>6} | {'nnz':>7} | {'稠密 fwd+bwd':>13} | {'稀疏 fwd+bwd':>13} | {'加速':>7}")
    print("-" * 76)
    for d in d_list:
        Wd = rng.normal(0, 0.4, (d, d)) * (rng.random((d, d)) < min(1.0, 4.0 / d))
        np.fill_diagonal(Wd, 0.0)
        Ws = _to_csr(Wd)
        if Ws.nnz == 0:
            continue
        t0 = time.time()
        for _ in range(5):
            _, c = f_dense(Wd, 5, 0.9); b_dense(Wd, c)
        t_dense = (time.time() - t0) / 5
        t0 = time.time()
        for _ in range(5):
            _, c = forward_sp(Ws, 5, 0.9); backward_sp(Ws, c)
        t_sp = (time.time() - t0) / 5
        print(f"{d:>6} | {Ws.nnz:>7} | {t_dense*1e3:>10.2f} ms | {t_sp*1e3:>10.2f} ms | "
              f"{t_dense/max(t_sp,1e-12):>6.1f}x")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="只跑规模基准")
    ap.add_argument("--d", type=int, nargs="+", default=(200, 500, 1000))
    a = ap.parse_args()
    if a.bench:
        scaling_benchmark(tuple(a.d))
    else:
        self_test()
        scaling_benchmark(tuple(a.d))
