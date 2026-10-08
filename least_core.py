"""
================================================================================
 least_core.py  —  LEAST (ICDE 2021) 核心算法的严格、自洽实现
================================================================================
 Paper: "Efficient and Scalable Structure Learning for Bayesian Networks:
         Algorithms and Applications"  (Rong Zhu et al., ICDE 2021)

 本文件严格实现论文中的每一个公式：

  [1] §III-B  Eq.(4)(5)  —— 无环性上界 δ^(k) 的对角相似变换迭代
      S^(0)   = W ∘ W
      b^(j)   = (r(S^(j)))^α ∘ (c(S^(j)))^(1-α)
      S^(j+1) = (D^(j))^{-1} S^(j) D^(j) = (b^(j))^{-1} ∘ S^(j) ∘ (b^(j))^T
      δ^(k)   = Σ_i b^(k)[i]
      （论文明确规定 D[i,i]=0 时 (D^{-1})[i,i]=0 —— 本实现严格执行，
        这是数值稳定性的关键：见 _bvec()）

  [2] §III-C  Fig.2 FORWARD / BACKWARD
      Lemma 3: x = α (c/r)^(1-α),  y = (1-α)(r/c)^α,  ∂δ/∂S[i,j] = x_i + y_j
      Lemma 5: 用 W 的支撑掩码 M 提前屏蔽，∇'_S δ 与 W 同稀疏度
      Eq.(7):  z^(j-1) = -r(∇∘S∘b^T)/b² + c(b^{-1}∘∇∘S)
      Eq.(9):  ∇'_{S^(j-1)}δ = b^{-1}∘∇'_{S^(j)}δ∘b^T + x∘z∘M + y^T∘z^T∘M
      Eq.(10): ∇_W δ = 2 ∇'_S δ ∘ W

  [3] §IV Fig.3 LEAST / INNER
      增广拉格朗日: 𝓁(W) = (1/n)‖X-XW‖²_F + λ‖W‖₁ + (ρ/2)δ² + ηδ
      梯度:        ∇𝓁 = ∇L + (η + ρδ)·∇δ
      外层 ALM:    η ← η + ρδ ;  ρ ← ρ·B
      终止:        δ ≤ ε  (V-A 额外要求 h(W) ≤ ε)

  [4] NOTEARS 精确无环度量 (用于终止判据与基线对比)
      h(W) = Tr(exp(W∘W)) - d

 依赖: numpy, scipy (可选 sparse)。不依赖 TensorFlow / igraph。
================================================================================
"""
from __future__ import annotations

import os

# 限制 BLAS/OpenMP 线程数：多核 OpenBLAS 的工作区分配在内存受限环境下会顶破上限，
# 表现为 "OpenBLAS error: Memory allocation still failed"。单线程即可避免，
# 且对 LEAST 稀疏/稠密实现的速度影响可忽略。
for _bt in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_bt, "1")

import numpy as np
from scipy.linalg import expm

__all__ = [
    "set_random_seed", "simulate_dag", "simulate_parameter", "simulate_linear_sem",
    "generate_dataset",
    "forward", "backward", "delta_and_grad", "h_fun",
    "least", "count_accuracy", "is_dag", "topological_order", "prune_to_dag",
]


# =============================================================================
# 0. 基础图工具
# =============================================================================
def is_dag(W):
    """拓扑排序判环。"""
    d = W.shape[0]
    A = W != 0
    indeg = A.sum(axis=0).astype(int)
    queue = [i for i in range(d) if indeg[i] == 0]
    seen = 0
    while queue:
        v = queue.pop()
        seen += 1
        for j in np.where(A[v])[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    return seen == d


def topological_order(W):
    """返回拓扑序；若非 DAG 返回长度 < d 的列表。"""
    d = W.shape[0]
    A = W != 0
    indeg = A.sum(axis=0).astype(int)
    queue = [i for i in range(d) if indeg[i] == 0]
    order = []
    while queue:
        v = queue.pop()
        order.append(v)
        for j in np.where(A[v])[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    return order


# =============================================================================
# 1. 数据生成 —— 与 NOTEARS 官方 utils.py (论文 ref [38]) 同构
# =============================================================================
def set_random_seed(seed):
    np.random.seed(seed)


def _erdos_renyi_undirected(d, s0, rng):
    """均匀随机选 s0 条无向边 (等价于 igraph.Graph.Erdos_Renyi(n=d, m=s0))。"""
    max_e = d * (d - 1) // 2
    s0 = int(min(s0, max_e))
    idx = rng.choice(max_e, size=s0, replace=False)
    B = np.zeros((d, d))
    # 展开上三角索引 (i<j)
    triu_i, triu_j = np.triu_indices(d, k=1)
    ii, jj = triu_i[idx], triu_j[idx]
    B[ii, jj] = 1.0
    B[jj, ii] = 1.0
    return B


def _barabasi_albert_directed(d, m, rng):
    """有向 BA 图 (等价于 igraph.Graph.Barabasi(n=d, m=m, directed=True))。
    新节点 i 连向 m 个已有节点（度优先），故天然是 DAG。"""
    m = max(1, int(m))
    B = np.zeros((d, d))
    if d <= 1:
        return B
    seed_n = min(m + 1, d)
    # 初始种子团：i → j (j < i)，天然无环
    for i in range(1, seed_n):
        for j in range(i):
            B[i, j] = 1.0
    # 偏好连接词袋：只含 **已存在** 节点（严格 < 当前 i）
    bag = []
    for i in range(seed_n):
        for t in np.where((B[i] != 0) | (B[:, i] != 0))[0]:
            bag.append(int(t))
        bag.append(i)
    for i in range(seed_n, d):
        if len(bag) == 0:
            bag = list(range(i))
        k = min(m, len(bag))
        dest = np.atleast_1d(rng.choice(np.asarray(bag), size=k, replace=False))
        for t in dest:
            t = int(t)
            B[i, t] = 1.0
            bag.append(t)
            bag.append(i)
    return B


def simulate_dag(d, s0, graph_type="ER", rng=None):
    """生成随机 DAG 的 0/1 邻接矩阵 B (B[i,j]=1 表示 i→j)。"""
    if rng is None:
        rng = np.random.default_rng()
    if graph_type.upper() == "ER":
        B_und = _erdos_renyi_undirected(d, s0, rng)
        # 随机无环定向：随机排列 π，边从小序号指向大序号
        perm = rng.permutation(d)
        # rank[orig] = 位置
        rank = np.empty(d, dtype=int)
        rank[perm] = np.arange(d)
        iu, ju = np.where(np.triu(B_und, k=1) != 0)
        B = np.zeros((d, d))
        src = np.where(rank[iu] < rank[ju], iu, ju)
        dst = np.where(rank[iu] < rank[ju], ju, iu)
        B[src, dst] = 1.0
    elif graph_type.upper() == "SF":
        m = int(round(s0 / d))
        B = _barabasi_albert_directed(d, m, rng)
    else:
        raise ValueError(f"unknown graph type: {graph_type}")
    # 随机重标号 (NOTEARS: B_perm = P.T B P)。用花式索引实现，避免 d×d 稠密矩阵乘法
    # 产生的中间 32MB 临时数组（d 大时易触发 MemoryError）
    perm = rng.permutation(d)
    B_perm = B[perm][:, perm]
    assert is_dag(B_perm), "simulate_dag produced a cyclic graph"
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0)), rng=None):
    """为 DAG 的每条边随机分配权重（NOTEARS 默认区间，50% 负）。"""
    if rng is None:
        rng = np.random.default_rng()
    W = np.zeros(B.shape)
    S = rng.integers(len(w_ranges), size=B.shape)
    for i, (low, high) in enumerate(w_ranges):
        U = rng.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def _noise(sem_type, n, rng, scale=1.0):
    if sem_type == "gauss":
        return rng.normal(scale=scale, size=n)
    if sem_type == "exp":
        return rng.exponential(scale=scale, size=n) - scale
    if sem_type == "gumbel":
        return rng.gumbel(loc=0.0, scale=scale, size=n)
    if sem_type == "uniform":
        return rng.uniform(low=-scale, high=scale, size=n)
    if sem_type == "logistic":
        return rng.logistic(scale=scale, size=n)
    raise ValueError(f"unknown sem type: {sem_type}")


def simulate_linear_sem(W, n, sem_type="gauss", noise_scale=None, rng=None):
    """按 **拓扑序** 采样线性 SEM: X = X W + N。
    （注意：必须按拓扑序，否则父节点尚未生成 → 数据错误。
      原 notears_utils.py 用 `for j in range(d)` 是常见移植 bug。）"""
    if rng is None:
        rng = np.random.default_rng()
    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        scale_vec = np.asarray(noise_scale, dtype=float)
    X = np.zeros([n, d])
    order = topological_order(W)
    assert len(order) == d, "W is not a DAG"
    for j in order:
        parents = W[:, j] != 0
        if np.any(parents):
            X[:, j] = X[:, parents] @ W[parents, j] + _noise(sem_type, n, rng, scale_vec[j])
        else:
            X[:, j] = _noise(sem_type, n, rng, scale_vec[j])
    return X


_GRAPH_ALIAS = {"er": "ER", "erdos-renyi": "ER", "erdos_renyi": "ER",
                "sf": "SF", "scale-free": "SF", "barabasi-albert": "SF", "ba": "SF"}
_SEM_ALIAS = {"gaussian": "gauss", "gs": "gauss", "gauss": "gauss", "normal": "gauss",
              "exponential": "exp", "ex": "exp", "exp": "exp",
              "gumbel": "gumbel", "gb": "gumbel"}


def generate_dataset(d=20, n=None, graph_type="ER", noise_type="gaussian",
                     degree=2, seed=42, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """论文 V-A 设置: ER degree=2 / SF degree=4, n=10d, 权重 ±[0.5,2], 噪声 N(0,1)。"""
    rng = np.random.default_rng(seed)
    if n is None:
        n = 10 * d
    gt = _GRAPH_ALIAS.get(str(graph_type).lower(), str(graph_type).upper())
    sem = _SEM_ALIAS.get(str(noise_type).lower(), str(noise_type).lower())
    if degree is None:
        degree = 2 if gt == "ER" else 4
    s0 = int(round(degree * d / 2.0))       # NOTEARS: 期望边数 = 平均度 * d / 2
    B = simulate_dag(d, s0, gt, rng=rng)
    W_true = simulate_parameter(B, w_ranges, rng=rng)
    X = simulate_linear_sem(W_true, n, sem, rng=rng)
    meta = {"d": d, "n": int(n), "graph_type": gt, "degree": degree, "s0": s0,
            "noise_type": sem, "seed": seed,
            "num_edges": int(np.count_nonzero(W_true))}
    return W_true, X, meta


# =============================================================================
# 2. FORWARD  (论文 Fig.2)  ——  计算 δ^(k)
# =============================================================================
def _bvec(S, alpha):
    """给定 S，返回 (b, x, y)。
    严格执行论文 'D[i,i]=0 时 (D^{-1})[i,i]=0'：
      b_i = 0  当 r_i == 0 或 c_i == 0  （源点/汇点被"剥离"，不改变谱半径）
      x_i = ∂b_i/∂r_i = α (c/r)^(1-α),  y_i = ∂b_i/∂c_i = (1-α)(r/c)^α
      当 b_i == 0 时 x_i = y_i = 0（次梯度取 0，保证数值稳定）
    """
    r = S.sum(axis=1)
    c = S.sum(axis=0)
    rs = np.where(r > 0.0, r, 1.0)
    cs = np.where(c > 0.0, c, 1.0)
    active = (r > 0.0) & (c > 0.0)
    b = np.where(active, (rs ** alpha) * (cs ** (1.0 - alpha)), 0.0)
    x = np.where(active, alpha * b / rs, 0.0)
    y = np.where(active, (1.0 - alpha) * b / cs, 0.0)
    return b, x, y


def forward(W, k=5, alpha=0.9):
    """论文 Fig.2 FORWARD。返回 (δ^(k), cache)。
    W: np.ndarray [d,d]（稀疏 CSR 亦可，见 forward_sparse）"""
    W = np.asarray(W, dtype=np.float64)
    S = W * W                       # S^(0) = W ∘ W
    S_list = [S]
    b_list = []
    for j in range(k):              # j = 0 .. k-1
        b, _, _ = _bvec(S, alpha)
        b_list.append(b)
        inv_b = np.where(b > 0.0, 1.0 / np.where(b > 0.0, b, 1.0), 0.0)
        # S^(j+1)[i,j] = S[i,j] * b[j] / b[i]   （b[i]=0 或 b[j]=0 时置 0）
        S = (S * b[None, :]) * inv_b[:, None]
        S_list.append(S)
    b_final, _, _ = _bvec(S, alpha)     # b^(k)
    b_list.append(b_final)
    delta = float(b_final.sum())
    cache = {"S_list": S_list, "b_list": b_list, "k": k, "alpha": alpha}
    return delta, cache


def backward(W, cache):
    """论文 Fig.2 BACKWARD (Lemma 3-5, Eq.7/9/10)。返回 ∇_W δ^(k)。"""
    W = np.asarray(W, dtype=np.float64)
    S_list, b_list = cache["S_list"], cache["b_list"]
    k, alpha = cache["k"], cache["alpha"]

    # 支撑掩码 M（Lemma 5）—— 只屏蔽 δ 的梯度，绝不屏蔽数据损失梯度
    M = (W != 0)

    # ---- 初始化 ∇'_{S^(k)} δ = x∘M + y^T∘M  (元素 (i,j) = x_i + y_j) ----
    _, x_k, y_k = _bvec(S_list[k], alpha)
    G = (x_k[:, None] + y_k[None, :]) * M

    # ---- 反向 j = k-1 ... 0  (对应论文的 j-1) ----
    for j in range(k - 1, -1, -1):
        S_prev = S_list[j]
        b_prev = b_list[j]
        _, x_prev, y_prev = _bvec(S_prev, alpha)

        inv_b = np.where(b_prev > 0.0, 1.0 / np.where(b_prev > 0.0, b_prev, 1.0), 0.0)

        GS = G * S_prev                       # ∇' ∘ S^(j-1)   (Hadamard!)
        rA = (GS * b_prev[None, :]).sum(axis=1)      # r( ∇∘S∘b^T )
        cB = (GS * inv_b[:, None]).sum(axis=0)       # c( b^{-1}∘∇∘S )
        z = -rA * (inv_b ** 2) + cB                  # Eq.(7)

        # Eq.(9)
        G = (G * inv_b[:, None]) * b_prev[None, :] \
            + (x_prev[:, None] * z[:, None] + y_prev[None, :] * z[None, :]) * M

    return 2.0 * G * W                        # Eq.(10)


def delta_and_grad(W, k=5, alpha=0.9):
    delta, cache = forward(W, k, alpha)
    return delta, backward(W, cache)


# =============================================================================
# 3. h(W) —— NOTEARS 精确无环度量（终止判据 / 基线）
# =============================================================================
def h_fun(W):
    """h(W) = Tr(exp(W∘W)) - d ;  h(W)=0 ⟺ G(W) 是 DAG。"""
    W = np.asarray(W, dtype=np.float64)
    d = W.shape[0]
    if np.count_nonzero(W) == 0:
        return 0.0
    S = W * W
    # 对称化以稳定 expm（Tr(e^S) 对 S 的转置不变，且 S 非负）
    return float(np.trace(expm(S)) - d)


# =============================================================================
# 4. LEAST 主算法 (论文 Fig.3: INNER + LEAST)
# =============================================================================
def _sparse_glorot(d, zeta, rng):
    """稀疏 Glorot 均匀初始化，密度 ζ，对角线置 0。"""
    limit = np.sqrt(6.0 / (d + d))
    W = rng.uniform(-limit, limit, size=(d, d))
    if zeta < 1.0:
        W = W * (rng.random((d, d)) < zeta)
    np.fill_diagonal(W, 0.0)
    return W


def _mse_precompute(X):
    """预计算 C = X^T X / n，使全批量下每步梯度只需 O(d^2)。"""
    n = X.shape[0]
    return (X.T @ X) / float(n)


def _least_inner(X, C, d, zeta, lam, rho, eta, k, alpha, theta, T_i, lr, rng,
                 W_init=None, batch_size=None, progress=None):
    """论文 INNER 过程。返回 (W, δ(W))。"""
    W = _sparse_glorot(d, zeta, rng) if W_init is None else W_init.copy()
    n = X.shape[0]
    full_batch = (batch_size is None) or (batch_size >= n)

    m = np.zeros((d, d))
    v = np.zeros((d, d))
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    delta = 0.0

    for t in range(1, T_i + 1):
        # ---- 无环性约束 δ 及其梯度 (FORWARD / BACKWARD) ----
        delta, cache = forward(W, k, alpha)
        g_delta = backward(W, cache)

        # ---- 数据损失 L(W,X_B) = (1/|B|)‖X_B - X_B W‖² + λ‖W‖₁ ----
        if full_batch:
            g_mse = 2.0 * (C @ W - C)          # ∇_W (1/n)‖X-XW‖²
        else:
            bs = int(min(batch_size, n))
            idx = rng.choice(n, size=bs, replace=False)
            XB = X[idx]
            CB = (XB.T @ XB) / float(bs)
            g_mse = 2.0 * (CB @ W - CB)

        g = g_mse + lam * np.sign(W) + (eta + rho * delta) * g_delta
        np.fill_diagonal(g, 0.0)

        # ---- Adam ----
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        W = W - lr * mh / (np.sqrt(vh) + eps_adam)
        np.fill_diagonal(W, 0.0)

        # ---- 阈值剪枝 θ (论文 INNER line 9) ----
        if theta > 0:
            W[np.abs(W) < theta] = 0.0

        if progress is not None and (t % 25 == 0 or t == T_i):
            progress(t, delta)

    delta, _ = forward(W, k, alpha)
    return W, float(delta)


def least(X, d=None, k=5, alpha=0.9, zeta=1e-4, lam=0.5, eps=1e-4,
          B=1.005, T_o=1000, T_i=200, lr=0.01, theta=0.0,
          rho_init=1.0, eta_init=1.0, seed=42, batch_size=None,
          warm_start=False, use_h_termination=True, termination="delta_or_h",
          eta_max=None, max_seconds=None, callback=None, verbose=False):
    """论文 Fig.3 LEAST 主算法（ALM 外层 + Adam 内层）。

    termination:
      论文 Fig.3 原始条件为 δ(W*) ≤ ε；§V-A 为了与 NOTEARS 使用同一判据，
      额外在外层末尾计算 h(W) 并在 h(W) ≤ ε 时终止。可选：
      - "delta_or_h"  (默认) δ ≤ ε 或 h ≤ ε —— 即论文 §V-A 的修改版
      - "delta_and_h"                两者同时满足（最严格）
      - "delta"                      仅 δ ≤ ε（论文 Fig.3 原始）
      - "h"                          仅 h ≤ ε（与 NOTEARS 完全一致）
      ⚠ 由于 δ 只是谱半径的**上界**（k=5 时往往比 h 松 2~3 个数量级），
        单独用 δ ≤ ε 会让 ALM 的 η 无界增长，把 W 过度压缩从而损害结构精度。
        这也是原实现 F1 偏低的主因。

    返回 dict: W, delta_final, h_final, converged, history, n_outer, elapsed
    """
    X = np.asarray(X, dtype=np.float64)
    if d is None:
        d = X.shape[1]
    rng = np.random.default_rng(seed)
    C = _mse_precompute(X)

    rho, eta = float(rho_init), float(eta_init)
    W_star, delta_star, prev_W = None, None, None
    history = []
    converged = False
    stop = None
    import time as _time
    t_start = _time.time()

    for outer in range(int(T_o)):
        W_star, delta_star = _least_inner(
            X, C, d, zeta=zeta, lam=lam, rho=rho, eta=eta, k=k, alpha=alpha,
            theta=theta, T_i=T_i, lr=lr, rng=rng,
            W_init=(prev_W if warm_start else None),
            batch_size=batch_size,
        )
        prev_W = W_star.copy()

        h_val = h_fun(W_star) if use_h_termination else 0.0
        history.append({"outer": outer, "delta": float(delta_star),
                        "h": float(h_val), "rho": float(rho), "eta": float(eta)})

        h_ok = bool(use_h_termination) and (h_val <= eps)
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
        else:  # "delta_or_h" —— 论文 §V-A 的修改版
            converged = d_ok or h_ok
            if converged:
                stop = ("delta<=eps & h<=eps" if (d_ok and h_ok)
                        else ("h<=eps" if h_ok else "delta<=eps"))
            else:
                stop = None
        if callback is not None:
            callback(outer, delta_star, h_val, rho, eta, W_star)
        if verbose:
            print(f"[outer {outer:4d}] delta={delta_star:.3e} h={h_val:.3e} "
                  f"rho={rho:.3f} eta={eta:.3f} nnz={np.count_nonzero(W_star)}",
                  flush=True)
        if converged:
            break
        if max_seconds is not None and (_time.time() - t_start) > max_seconds:
            stop = "timeout"
            break

        eta = min(eta + rho * delta_star, eta_max) if eta_max else eta + rho * delta_star
        rho = rho * B

    return {"W": W_star, "delta_final": float(delta_star),
            "h_final": (h_fun(W_star) if use_h_termination else None),
            "converged": bool(converged), "history": history,
            "n_outer": len(history), "elapsed": _time.time() - t_start,
            "stop_reason": stop or ("max_outer" if len(history) >= int(T_o)
                                    else "unknown")}


# =============================================================================
# 5. 评估 —— NOTEARS / LEAST 标准阈值网格搜索
# =============================================================================
PAPER_TAU_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)     # 论文 §V-A: τ ∈ {0.1,...,0.5}
PAPER_EPS_GRID = (1e-1, 1e-2, 1e-3, 1e-4)      # 论文 §V-A: ε ∈ {10^-1,...,10^-4}


def count_accuracy(W_true, W_est, taus=PAPER_TAU_GRID):
    """在阈值 τ 网格上搜索，返回最佳 F1 及其 P/R/SHD。
    论文 §V-A: τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}，取最佳情形。"""
    W_true = np.asarray(W_true)
    W_est = np.asarray(W_est)
    G_true = np.abs(W_true) > 1e-8
    true_edges = set(zip(*np.where(G_true)))
    num_true = len(true_edges)

    w_abs = np.abs(W_est)
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "shd": int(num_true),
            "best_tau": 0.0, "best_graph": np.zeros_like(G_true)}

    for tau in taus:
        G_est = w_abs > tau
        est_edges = set(zip(*np.where(G_est)))
        tp = len(true_edges & est_edges)
        num_est = len(est_edges)
        prec = tp / num_est if num_est else 0.0
        rec = tp / num_true if num_true else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        # SHD: 多余边 + 缺失边 + 反向边，各计 1（NOTEARS 标准口径）
        fp = est_edges - true_edges
        reversed_edges = {(j, i) for (i, j) in fp} & true_edges
        shd = len(fp) + len(true_edges - est_edges)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "precision": float(prec), "recall": float(rec),
                    "shd": int(shd), "reverse": int(len(reversed_edges)),
                    "best_tau": float(tau),
                    "best_graph": G_est.copy(), "num_pred": int(num_est)}
    return best


def prune_to_dag(W):
    """后处理：把学到的 W 修剪成严格 DAG（按 |w| 从大到小贪心加边去环）。
    仅用于展示，不改变算法本身。"""
    W = np.asarray(W, dtype=np.float64).copy()
    d = W.shape[0]
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(W), axis=None), (d, d)))[0]
    out = np.zeros((d, d))
    for i, j in idx:
        if i == j or abs(W[i, j]) <= 0:
            continue
        out[i, j] = W[i, j]
        if not is_dag(out):
            out[i, j] = 0.0
    return out
