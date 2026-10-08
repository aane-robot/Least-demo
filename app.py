"""
================================================================================
 app.py — LEAST 交互式网页 Demo（Flask 后端）
================================================================================
 启动:  python app.py        浏览器打开 http://127.0.0.1:5000

 后端能力:
   /api/run            提交一次结构学习任务（后台线程 + 进度轮询）
   /api/status/<id>    轮询进度 / 中间收敛曲线
   /api/result/<id>    取回完整结果（指标 / 图 / 收敛轨迹）
   /api/paths/<id>     根因分析：枚举指向目标节点的所有因果路径
   /api/benchmark      复现论文 §V-A 基准（LEAST vs NOTEARS，含加速比）
   /api/download/<id>  下载学到的邻接矩阵 CSV
================================================================================
"""
from __future__ import annotations

import io, csv, os, sys, time, json, uuid, threading, traceback

# 限制 BLAS/OpenMP 线程数：必须在本文件任何 numpy/scipy 导入之前设置，否则无效。
# 多核 OpenBLAS 会为每个线程分配独立工作区，内存受限环境（如 d=2000 演示）下其
# 工作区总和会顶破可用内存上限，表现为 "OpenBLAS error: Memory allocation still failed"。
# 单线程即可避免该问题，且 LEAST 稀疏实现的瓶颈不在 BLAS 并行度上，速度几乎不受影响。
for _bt in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_bt, "1")

import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from least_core import (generate_dataset, least, count_accuracy, h_fun,
                        forward, backward, is_dag, PAPER_EPS_GRID, PAPER_TAU_GRID)
import demo_datasets
from bn_io import (parse_network, detect_format, bn_to_dag, sample_bn,
                   NET_FORMATS)

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE, "templates"),
            static_folder=os.path.join(BASE, "static"))
app.config["JSON_AS_ASCII"] = False

JOBS: dict = {}
JOBS_LOCK = threading.Lock()
DATA_CACHE: dict = {}          # job_id -> (X, W_true, node_names)
BENCH_STATE = {"status": "idle", "progress": 0.0, "log": [], "result": None}
SCAL_STATE = {"status": "idle", "progress": 0.0, "log": [], "result": None, "error": None}


# =============================================================================
# 工具
# =============================================================================
def _to_native(o):
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _parse_csv(text: str, has_header: bool = True):
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("CSV 为空")
    header = None
    if has_header:
        header = [c.strip() for c in rows[0]]
        rows = rows[1:]
    ncols = max(len(r) for r in rows)
    header = header if (header and len(header) == ncols) else None
    data = []
    for r in rows:
        row = []
        for c in r[:ncols]:
            c = c.strip()
            try:
                row.append(float(c))
            except ValueError:
                row.append(np.nan)
        data.append(row)
    X = np.asarray(data, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        # 用列均值填补缺失/非数值
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(col_mean, idx[1])
    return X, header


def _edge_list(W, mask=None, kind="tp", limit=None):
    W = np.asarray(W)
    idx = np.argwhere(mask if mask is not None else (np.abs(W) > 1e-8))
    out = [{"s": int(i), "t": int(j), "w": round(float(W[i, j]), 4), "type": kind}
           for i, j in idx]
    if limit:
        out = sorted(out, key=lambda e: -abs(e["w"]))[:limit]
    return out


def _classify(W_est, W_true, tau):
    """把预测边分为 tp / rev / fp，并给出 fn（漏掉的边）。"""
    G_est = np.abs(np.asarray(W_est)) > tau
    G_true = np.abs(np.asarray(W_true)) > 1e-8
    d = G_est.shape[0]
    edges = []
    for i in range(d):
        for j in range(d):
            if i == j or not G_est[i, j]:
                continue
            if G_true[i, j]:
                kind = "tp"
            elif G_true[j, i]:
                kind = "rev"
            else:
                kind = "fp"
            edges.append({"s": int(i), "t": int(j),
                          "w": round(float(W_est[i, j]), 4), "type": kind})
    fn = [{"s": int(i), "t": int(j), "w": round(float(W_true[i, j]), 4), "type": "fn"}
          for i, j in np.argwhere(G_true & ~G_est)]
    return edges, fn


# =============================================================================
# 任务执行
# =============================================================================
def _run_job(job_id: str, p: dict):
    job = JOBS[job_id]
    log = job["log"]

    def L(msg):
        log.append(str(msg))
        print(f"[{job_id[:8]}] {msg}", flush=True)

    try:
        t0 = time.time()
        # ---------- 1. 取数据 ----------
        mode = p.get("mode", "synthetic")
        node_names = None
        W_true = None
        has_gt = False
        if mode == "synthetic":
            d = int(p.get("d", 10)); n = int(p.get("n", 0)) or 10 * d
            # 内存保护：d 较大时限制样本量，使 X（n×d）不超过 ~200MB float64，
            # 否则稠密 d×d 分配会触发 MemoryError（d=2000 时 n=10d 即 ~320MB）
            NX_CAP = 25_000_000
            if n * d > NX_CAP:
                n_cap = max(2000, NX_CAP // d)
                L(f"⚠ 内存保护: d={d} 较大，样本量从 n={n} 裁剪到 n={n_cap}"
                  f"（保持 X≈{n_cap * d} 元素，避免 OOM）")
                n = n_cap
            W_true, X, meta = generate_dataset(
                d=d, n=n, graph_type=p.get("graph_type", "ER"),
                noise_type=p.get("noise_type", "gaussian"),
                degree=int(p.get("degree", 2)), seed=int(p.get("seed", 42)))
            node_names = [f"X{i + 1}" for i in range(d)]
            has_gt = True
            L(f"合成数据: {meta['graph_type']}-{meta['degree']}, {meta['noise_type']}, "
              f"d={d}, n={n}, 真实边数={meta['num_edges']}")
        elif mode == "builtin":
            name = p.get("dataset", "flight")
            W_true, X, meta = demo_datasets.get_dataset(
                name, n=int(p.get("n", 0)) or 2000, seed=int(p.get("seed", 7)),
                anomaly=bool(p.get("anomaly", False)))
            node_names = meta["node_names"]
            has_gt = True
            L(f"内置数据集 '{name}': {meta['desc']}, d={meta['d']}, n={meta['n']}, "
              f"真实边数={meta['num_edges']}")
            meta.setdefault("target_nodes", None)
        elif mode == "upload":
            text = p["csv_content"]
            fname = str(p.get("file_name", "") or "")
            fmt = str(p.get("file_format") or "").lower() or detect_format(text, fname)
            if fmt in NET_FORMATS:
                # ---- 贝叶斯网络定义文件（BIF / DSC / NET）----
                bn = parse_network(text, fmt, fname)
                bn_names, A = bn_to_dag(bn)
                d_bn = len(bn_names)
                if d_bn > 400:
                    raise ValueError(f"网络节点数 d={d_bn} 过大（Demo 上限 400）")
                n_req = int(p.get("sample_n", 0) or 0) or max(2000, 10 * d_bn)
                n_req = int(min(max(n_req, 100), 20000))
                if n_req < d_bn:
                    L(f"⚠ 警告: 采样数 n={n_req} < 节点数 d={d_bn}！每个节点的线性回归"
                      f"方程数少于未知数（欠定），结构完全不可辨识，F1 必然很差。"
                      f"建议把「网络采样数 n」调到 ≥ 10×d ≈ {10 * d_bn}（上限 20000）。")
                elif n_req < 5 * d_bn:
                    L(f"提示: 采样数 n={n_req} < 5×d={5 * d_bn}，样本偏少，"
                      f"F1 会偏低；建议 ≥ 10×d ≈ {10 * d_bn}。")
                X = sample_bn(bn, n_req, seed=int(p.get("seed", 42)))
                if p.get("standardize", False):
                    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
                node_names = bn_names
                W_true = A                      # 网络自身边表 = ground truth
                has_gt = True
                L(f"网络文件 '{fname}' [{(fmt or '').upper()}]: {bn.name}, "
                  f"d={d_bn}, 真实边数={int(A.sum())}, 按 CPT 采样 n={n_req} "
                  f"（离散状态编码为 0..k-1；LEAST 为线性 SEM，精度仅供参考）")
            else:
                # ---- 普通数据表 ----
                X, header = _parse_csv(text, bool(p.get("has_header", True)))
                n, d = X.shape
                if p.get("standardize", False):
                    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
                node_names = header or [f"V{i + 1}" for i in range(d)]
                has_gt = False
                L(f"上传数据: d={d}, n={n}" + (f", 列名: {header[:8]}..." if header else ""))
                if d > 400:
                    raise ValueError(f"变量数 d={d} 过大（Demo 上限 400），请先做特征筛选")
        else:
            raise ValueError(f"unknown mode: {mode}")

        n, d = X.shape
        with JOBS_LOCK:
            DATA_CACHE[job_id] = (X, W_true, node_names)

        # ---------- 2. 跑 LEAST ----------
        # 论文 §V-A: 对 ε∈{1e-1,1e-2,1e-3,1e-4} 与 τ∈{0.1..0.5} 做网格搜索并报告最佳
        lam = float(p.get("lam", 0.5))
        k = int(p.get("k", 5)); alpha = float(p.get("alpha", 0.9))
        zeta = float(p.get("zeta", 1e-4)); B = float(p.get("B", 1.005))
        T_o = int(p.get("T_o", 300)); T_i = int(p.get("T_i", 200))
        lr = float(p.get("lr", 0.01)); theta = float(p.get("theta", 0.0))
        warm = bool(p.get("warm_start", True))
        use_h = bool(p.get("use_h", True))
        max_sec = float(p.get("max_seconds", 120))
        batch = p.get("batch_size", None)
        batch = int(batch) if batch else None
        term = str(p.get("termination", "delta_or_h"))
        eps_grid = bool(p.get("eps_grid", True))
        eps_list = [float(e) for e in PAPER_EPS_GRID] if eps_grid \
            else [float(p.get("eps", 1e-3))]
        # 实测（d=400, ER-2, 5 外层轨迹）：F1 在 outer≈2 达峰后单调下降
        # （0.662 → 0.635 → 0.562 → 0.452），小 ε 只会过度惩罚且极耗时。
        if eps_grid and d >= 300 and len(eps_list) > 2:
            eps_list = eps_list[:2]
            L(f"d={d} ≥ 300：ε 网格截断为 {eps_list}（实测更小 ε 只会过度惩罚、"
              f"F1 下降且耗时大增）")

        # 引擎选择：d 大时自动切到稀疏实现 least_sp（FORWARD/BACKWARD 为 O(ks)）
        engine = str(p.get("engine", "auto"))
        if engine == "auto":
            engine = "sparse" if d >= int(p.get("sparse_threshold", 400)) else "dense"
        if engine == "sparse":
            from least_sp import least_sp as _solve
            # T_i 默认放大到 400（论文 200 是为 d≤100 调的；实测 d=400 时
            # T_i=400 的 F1=0.688 > T_i=200 的 0.662，其余超参仍用论文默认值）
            T_i = int(p.get("T_i_sp", 400))
            eng_kw = {"zeta": float(p.get("zeta_sp", 0.1)),
                      "theta": float(p.get("theta_sp", 1e-3))}
            # 大 d 自适应调参：单步 ∇L 是 O(nnz·d)（d=2000 时极贵），必须缩小 T_i
            # 以容纳更多外层循环；同时把时间上限抬到合理值（仅当用户未显式设更高时），
            # 否则默认 90s 只够 ~3 外层，训练严重不足、F1 接近 0。
            if d > 800:
                TI_CAP = 120
                if T_i > TI_CAP:
                    L(f"大 d={d}: T_i {T_i}→{TI_CAP}（单步昂贵，缩小内层步数→更多外层循环）")
                    T_i = TI_CAP
                if max_sec <= 120:
                    L(f"大 d={d}: max_seconds {max_sec:.0f}→240s（需更多时间才能收敛）")
                    max_sec = 240
            # 大 d：跳过稠密 expm（h(W)=Tr(e^{W∘W})−d 是 O(d³)，d=2000 时既极慢又
            # 额外分配多个 d×d 临时数组，是 d=2000 MemoryError 的元凶之一）。
            # 无环性改由谱半径 ρ(S) 卡与 is_dag 直接佐证（metrics.spec_r / is_dag）。
            H_SKIP = 1000
            if d > H_SKIP:
                eng_kw["compute_h"] = False
                L(f"引擎: least_sp（稀疏，O(ks)）  ζ={eng_kw['zeta']}, θ={eng_kw['theta']}  "
                  f"d={d}>{H_SKIP}→跳过稠密 expm（h 由 ρ(S)/is_dag 佐证）")
            else:
                L(f"引擎: least_sp（稀疏，O(ks)）  ζ={eng_kw['zeta']}, θ={eng_kw['theta']}")
        else:
            _solve = least
            eng_kw = {"zeta": zeta, "theta": theta}
            L(f"引擎: least_core（稠密）  ζ={zeta}, θ={theta}")

        L(f"LEAST 参数: k={k}, α={alpha}, λ={lam}, "
          f"ε={'网格' + str(eps_list) if eps_grid else eps_list[0]}, "
          f"B={B}, T_o={T_o}, T_i={T_i}, lr={lr}, 终止={term}, "
          f"{'热' if warm else '冷'}启动")

        def _one_run(eps):
            hist = []

            def cb(outer, delta, h_val, rho, eta, W):
                hist.append({"outer": outer, "delta": float(delta), "h": float(h_val),
                             "rho": float(rho), "eta": float(eta)})
                job["history"] = hist
                job["progress"] = min(0.95, 0.05 + 0.9 * (outer + 1) / max(T_o, 1))

            r = _solve(X, d, k=k, alpha=alpha, lam=lam, eps=eps, B=B,
                       T_o=T_o, T_i=T_i, lr=lr, seed=int(p.get("seed", 42)),
                       batch_size=batch, warm_start=warm, use_h_termination=use_h,
                       termination=term, max_seconds=max_sec, callback=cb, **eng_kw)
            # 统一成稠密矩阵，供下游 count_accuracy / _classify / is_dag / 矩阵展示使用。
            # 稀疏引擎在 d>800 时不返回 W_dense，这里一次性 toarray（仅 1 次 d×d 分配，
            # 不再像旧代码那样反复分配）。d 极大时也比反复稠密化省内存。
            W_out = r.get("W_dense")
            if W_out is None:
                Wsp = r["W"]
                W_out = (Wsp.toarray() if hasattr(Wsp, "toarray")
                         else np.asarray(Wsp, dtype=np.float64))
            r["W"] = W_out
            L(f"  ε={eps:<8g}: 收敛={r['converged']!s:5} 外层={r['n_outer']:<4} "
              f"δ={r['delta_final']:.2e} h={r['h_final']:.2e} t={r['elapsed']:.2f}s")
            return r, hist

        if has_gt:
            tau_list = [float(t) for t in p.get("taus", PAPER_TAU_GRID)]
            if mode == "upload" and min(tau_list) > 0.05:
                # 真实/离散数据的边权重远小于合成数据（实测 andes d=223：
                # τ=0.05 时 F1 0.159→0.367），τ 网格自动补充小阈值
                tau_list = [0.02, 0.05] + tau_list
                L("上传数据: τ 网格补充 0.02/0.05（真实数据边权重偏小，"
                  "默认 0.1 会丢掉大量弱边）")
            best = None
            least_t_grid = 0.0
            for eps in eps_list:
                r, hist = _one_run(eps)
                least_t_grid += r["elapsed"]
                m = count_accuracy(W_true, r["W"], taus=tuple(tau_list) or PAPER_TAU_GRID)
                if best is None or m["f1"] > best[1]["f1"]:
                    best = (r, m, hist, eps)
            res, m, hist, eps_used = best
            tau = m["best_tau"]
            L(f"网格最佳: ε={eps_used}, τ={tau} → F1={m['f1']:.4f} "
              f"(ε网格共 {len(eps_list)} 次, 总耗时 {least_t_grid:.1f}s)")
        else:
            # 无 ground truth：选 h 最小（最"无环"）的一次。
            # 大 d 关闭 expm 时 h_final 为 None，此时排到带 h 的运行之后；
            # 若全部都没有 h（大 d 上传数据），回退为取第一次（前端再用 ρ(S)/is_dag 佐证）。
            def _hkey(r):
                h = r["h_final"]
                return (0, h) if h is not None else (1, 0.0)
            best = None
            least_t_grid = 0.0
            for eps in eps_list:
                r, hist = _one_run(eps)
                least_t_grid += r["elapsed"]
                if best is None or _hkey(r) < _hkey(best[0]):
                    best = (r, None, hist, eps)
            res, _, hist, eps_used = best
            tau = float(p.get("tau_upload", 0.1))
            m = {"f1": None, "precision": None, "recall": None, "shd": None,
                 "best_tau": tau}
            L(f"无 ground truth：取 h 最小的 ε={eps_used} (h={res['h_final']:.2e})，τ={tau}")

        W_est = res["W"]
        # 稀疏引擎返回 CSR；后续索引/切片在稀疏矩阵上会出错，统一转稠密
        if hasattr(W_est, "toarray"):
            W_est = W_est.toarray()
        W_est = np.asarray(W_est)
        L(f"LEAST 完成: 收敛={res['converged']}, 外层={res['n_outer']}, "
          f"δ={res['delta_final']:.3e}, h={res['h_final']:.3e}, 用时={res['elapsed']:.1f}s")
        stop_reason = res.get("stop_reason", "unknown")
        WHY = {
            "h<=eps": "h(W)≤ε：学到的加权图已(近似)无环，约束满足，按论文 §V-A 判据提前终止"
                      "（外层数少是正常的，实测更多外层只会过度惩罚、降低 F1）",
            "delta<=eps": "δ(W)≤ε：无环性上界达到容差",
            "delta<=eps & h<=eps": "δ≤ε 且 h≤ε",
            "timeout": "达到 max_seconds 时间上限，返回当前最优 W",
            "max_outer": "跑满 T_o 轮外层",
        }
        L(f"终止原因: {stop_reason} —— {WHY.get(stop_reason, '')}")

        # ---------- 4. NOTEARS 对照（可选，小规模） ----------
        nt = None
        if p.get("compare_notears") and d <= 120:
            try:
                from benchmark import run_notears_grid
                L("运行 NOTEARS 作对照（λ×τ 网格取最佳，与 §V-A 一致）...")
                nt_best = run_notears_grid(W_true, X) if has_gt else None
                if nt_best is None:
                    from notears_baseline import notears_linear
                    r = notears_linear(X, lambda1=lam, max_iter=25, w_threshold=tau)
                    nt_best = {"f1": None, "precision": None, "recall": None,
                               "shd": None, "lam": lam, "tau": tau,
                               "time": r["elapsed"]}
                nt = {"f1": nt_best["f1"], "precision": nt_best["precision"],
                      "recall": nt_best["recall"], "shd": nt_best["shd"],
                      "time": round(nt_best["time"], 2), "lam": nt_best.get("lam"),
                      "tau": nt_best.get("tau"),
                      "time_single": round(nt_best.get("time_single", nt_best["time"]), 2)}
                sp_grid = nt_best["time"] / max(least_t_grid, 1e-9)
                sp_single = (nt_best.get("time_single", nt_best["time"])
                             / max(res["elapsed"], 1e-9))
                L(f"NOTEARS: F1={nt_best['f1']}, SHD={nt_best['shd']}, "
                  f"λ网格总耗时={nt_best['time']:.1f}s | LEAST ε网格总耗时={least_t_grid:.1f}s "
                  f"→ 加速(网格口径) {sp_grid:.1f}x; 单次最佳 {res['elapsed']:.1f}s vs "
                  f"{nt_best.get('time_single', nt_best['time']):.1f}s → {sp_single:.1f}x")
            except Exception as e:
                L(f"NOTEARS 对照失败: {e}")
        elif p.get("compare_notears"):
            # 勾选了对照但 d 超限：明确告知跳过，而不是静默不出数据
            reason = (f"d={d} 超过 NOTEARS 对照上限 120：NOTEARS 为稠密实现，"
                      f"d>120 时单次运行需数小时且内存溢出；LEAST 稀疏引擎不受此限。"
                      f"想验证大规模性能：打开「复现论文 §V-A 基准」弹窗下方的"
                      f"「大规模可扩展性实测」，可现场测量稀疏引擎在数千节点上的耗时，"
                      f"并对 NOTEARS 做 O(d³) 外推对比")
            L(f"NOTEARS 对照未运行 —— {reason}")
            nt = {"f1": None, "precision": None, "recall": None, "shd": None,
                  "time": None, "lam": None, "tau": None,
                  "skipped": True, "skip_reason": reason}

        # ---------- 5. 组装结果 ----------
        if has_gt:
            pred_edges, fn_edges = _classify(W_est, W_true, tau)
            true_edges = _edge_list(W_true)
        else:
            G = np.abs(W_est) > tau
            pred_edges = _edge_list(W_est, G, kind="pred")
            fn_edges, true_edges = [], []

        # 谱半径实测：ρ(S)≈0 是无环的直接证据。δ 只是它的松上界（k=5 约松 20 倍），
        # 所以 "δ 停在 1e-1~1e0" 不代表有环——以本值与 h(W) 为准。
        def _spec_radius(S):
            """S=W∘W 的谱半径。d≤800 用稠密特征值直算；否则用稀疏幂迭代，
            全程不产生 d×d 稠密矩阵（避免大 d 的 MemoryError）。"""
            if d <= 800:
                return float(np.max(np.abs(np.linalg.eigvals(np.asarray(S)))))
            import scipy.sparse as _sps
            Mc = S if _sps.issparse(S) else _sps.csr_matrix(np.asarray(S))
            v = np.random.default_rng(0).normal(size=d)
            lam = 0.0
            for _ in range(300):
                w = Mc @ v
                nrm = float(np.linalg.norm(w))
                if nrm == 0.0:
                    return 0.0                     # 幂零（DAG）→ 精确为 0
                v = w / nrm
                lam = float(np.linalg.norm(Mc @ v))
            return lam
        try:
            # 阈值化图（界面展示的那张）：直接在稀疏上阈值，不产生稠密 d×d
            if hasattr(W_est, "toarray"):
                S_thr = W_est.multiply(W_est)            # 稀疏 ⊗ 稀疏
                S_thr = S_thr.astype(float)
                spec_r = _spec_radius(S_thr)
                S_raw = W_est.multiply(W_est)
                spec_r_raw = _spec_radius(S_raw)
            else:
                Wn = np.asarray(W_est)
                W_thr = np.where(np.abs(Wn) > tau, Wn, 0.0)
                spec_r = _spec_radius(W_thr * W_thr)
                spec_r_raw = _spec_radius(Wn * Wn)
        except Exception:
            spec_r = spec_r_raw = None
        if spec_r is not None:
            L(f"谱半径实测: ρ(S|τ)={spec_r:.2e}, ρ(S raw)={spec_r_raw:.2e} "
              f"—— ≈0 即无环的直接证据（δ={res['delta_final']:.2e} 只是松上界）")

        d_show = d if d <= 80 else 80
        result = {
            "meta": {"d": d, "n": n, "mode": mode, "has_gt": has_gt,
                     "node_names": node_names,
                     "num_true_edges": int(np.count_nonzero(W_true)) if has_gt else 0,
                     "num_pred_edges": len(pred_edges),
                     "time": round(time.time() - t0, 2),
                     "algo_time": round(res["elapsed"], 2),
                     "algo_time_grid": round(least_t_grid, 2),
                     "algo_runs": len(eps_list)},
            "metrics": {"f1": m["f1"], "precision": m["precision"],
                        "recall": m["recall"], "shd": m["shd"], "tau": tau,
                        "eps": eps_used, "eps_grid": eps_grid, "lam": lam,
                        "k": k, "alpha": alpha, "theta": theta,
                        "engine": engine,
                        "warm_start": warm, "termination": term,
                        "delta": res["delta_final"], "h": res["h_final"],
                        "converged": res["converged"], "outer": res["n_outer"],
                        "stop_reason": stop_reason, "stop_why": WHY.get(stop_reason, ""),
                        "spec_r": spec_r, "spec_r_raw": spec_r_raw,
                        "is_dag": bool(is_dag(np.abs(W_est) > tau))},
            "history": {"outer": [x["outer"] for x in hist],
                        "delta": [x["delta"] for x in hist],
                        "h": [x["h"] for x in hist],
                        "rho": [x["rho"] for x in hist],
                        "eta": [x["eta"] for x in hist]},
            "graph": {"nodes": [{"id": i, "label": node_names[i]} for i in range(d)],
                      "true_edges": true_edges, "pred_edges": pred_edges,
                      "fn_edges": fn_edges},
            "matrix": (np.round(W_est[:d_show, :d_show], 4).tolist()
                       if p.get("return_matrix", True) else None),
            "top_edges": sorted(pred_edges, key=lambda e: -abs(e["w"]))[:200],
            "notears": nt,
            "log": list(log),
        }
        job["result"] = _to_native(result)
        job["status"] = "done"
        job["progress"] = 1.0
        L(f"全部完成，总耗时 {time.time() - t0:.1f}s")

    except Exception as e:
        tb = traceback.format_exc()
        L(f"[ERROR] {e}\n{tb}")
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job["traceback"] = tb
        job["progress"] = 1.0


# =============================================================================
# 路由
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    p = request.get_json(force=True)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "progress": 0.0, "log": [],
                    "history": [], "result": None, "created": time.time()}
    threading.Thread(target=_run_job, args=(job_id, p), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "job not found"}), 404
    return jsonify({"status": j["status"], "progress": j["progress"],
                    "log": j["log"][-40:],
                    "history": j["history"][-400:],
                    "error": j.get("error"), "traceback": j.get("traceback")})


@app.route("/api/result/<job_id>")
def api_result(job_id):
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "job not found"}), 404
    if j["status"] != "done":
        return jsonify({"status": j["status"], "progress": j["progress"]})
    return jsonify(j["result"])


@app.route("/api/paths/<job_id>", methods=["POST"])
def api_paths(job_id):
    """根因分析：枚举指向 target 的因果路径（论文 §VI-A）。"""
    j = JOBS.get(job_id)
    if not j or j["status"] != "done":
        return jsonify({"error": "job not ready"}), 400
    body = request.get_json(force=True)
    target = int(body.get("target", 0))
    max_len = int(body.get("max_len", 4))
    top_k = int(body.get("top_k", 30))
    res = j["result"]
    names = res["meta"]["node_names"]
    tau = res["metrics"]["tau"]
    edges = res["graph"]["pred_edges"]
    adj = {}
    for e in edges:
        adj.setdefault(int(e["s"]), []).append((int(e["t"]), float(e["w"])))

    paths = []
    def dfs(node, path, seen, wprod):
        if len(path) > max_len:
            return
        if node == target and len(path) >= 2:
            paths.append({"path": list(path), "weight": float(wprod),
                          "len": len(path)})
            return
        for (nx, w) in adj.get(node, []):
            if nx in seen:
                continue
            seen.add(nx)
            dfs(nx, path + [nx], seen, wprod * abs(w))
            seen.discard(nx)
    for src in list(adj.keys()):
        dfs(src, [src], {src}, 1.0)
    # 去重 + 排序
    uniq, seen_p = [], set()
    for pp in sorted(paths, key=lambda x: -x["weight"]):
        key = tuple(pp["path"])
        if key in seen_p:
            continue
        seen_p.add(key)
        pp["labels"] = [names[i] for i in pp["path"]]
        uniq.append(pp)
        if len(uniq) >= top_k:
            break
    # 直接父节点
    parents = sorted([{"node": s, "label": names[s], "w": w}
                      for s, lst in adj.items() for (t, w) in lst if t == target],
                     key=lambda x: -abs(x["w"]))
    return jsonify({"target": target, "target_label": names[target],
                    "parents": parents, "paths": uniq})


@app.route("/api/download/<job_id>")
def api_download(job_id):
    j = JOBS.get(job_id)
    if not j or j["status"] != "done":
        return "job not ready", 400
    X, W_true, names = DATA_CACHE.get(job_id, (None, None, None))
    r = j["result"]
    tau = r["metrics"]["tau"]
    d = r["meta"]["d"]
    W = np.zeros((d, d))
    for e in r["graph"]["pred_edges"]:
        W[int(e["s"]), int(e["t"])] = e["w"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([""] + names)
    for i in range(d):
        w.writerow([names[i]] + [f"{W[i, j]:.6f}" for j in range(d)])
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":
                             f"attachment; filename=least_adjacency_{job_id[:8]}.csv"})


@app.route("/api/benchmark", methods=["POST", "GET"])
def api_benchmark():
    """复现论文 §V-A。POST 启动，GET 轮询。"""
    if request.method == "GET":
        return jsonify(BENCH_STATE)
    if BENCH_STATE["status"] == "running":
        return jsonify({"status": "already_running"})
    body = request.get_json(force=True) or {}
    args = {"d_list": body.get("d", [10, 20]), "seeds": body.get("seeds", [42, 43]),
            "graph": body.get("graph", "ER"), "noise": body.get("noise", "gaussian"),
            "T_o": body.get("T_o", 300), "T_i": body.get("T_i", 200),
            "max_seconds": body.get("max_seconds", 45),
            "with_notears": bool(body.get("with_notears", True))}
    BENCH_STATE.update({"status": "running", "progress": 0.0, "log": [],
                        "result": None})
    threading.Thread(target=_run_bench, args=(args,), daemon=True).start()
    return jsonify({"status": "started"})


def _run_bench(a):
    log = BENCH_STATE["log"]
    def L(m):
        log.append(str(m)); print("[bench]", m, flush=True)
    try:
        from benchmark import run_least_grid, run_notears_grid
        from least_core import generate_dataset
        rows = []
        d_list = a["d_list"]; seeds = a["seeds"]
        degree = 2 if a["graph"] == "ER" else 4
        total = len(d_list) * len(seeds)
        done = 0
        for d in d_list:
            lr_rows, nt_rows = [], []
            for seed in seeds:
                W_true, X, meta = generate_dataset(d=d, seed=seed, graph_type=a["graph"],
                                                   noise_type=a["noise"], degree=degree)
                b = run_least_grid(W_true, X, d, seed=seed, T_o=a["T_o"],
                                   T_i=a["T_i"], max_seconds=a["max_seconds"])
                lr_rows.append(b)
                L(f"d={d} seed={seed}: LEAST F1={b['f1']:.3f} SHD={b['shd']} "
                  f"(ε={b['eps']}, τ={b['tau']}, {b['time']:.1f}s)")
                if a["with_notears"] and d <= 100:
                    nt_best = run_notears_grid(W_true, X)
                    nt_rows.append(nt_best)
                    L(f"d={d} seed={seed}: NOTEARS F1={nt_best['f1']:.3f} "
                      f"SHD={nt_best['shd']} ({nt_best['time']:.1f}s)")
                done += 1
                BENCH_STATE["progress"] = done / total
            def agg(rs, key):
                v = [r[key] for r in rs]
                return [float(np.mean(v)), float(np.std(v))]
            row = {"d": d, "n": 10 * d,
                   "f1": agg(lr_rows, "f1"), "precision": agg(lr_rows, "precision"),
                   "recall": agg(lr_rows, "recall"), "shd": agg(lr_rows, "shd"),
                   "time": agg(lr_rows, "time")[0],
                   "time_grid": agg(lr_rows, "time_grid")[0],
                   "converged": int(sum(r["converged"] for r in lr_rows)),
                   "n_seed": len(lr_rows)}
            if nt_rows:
                row["nt_f1"] = agg(nt_rows, "f1")
                row["nt_shd"] = agg(nt_rows, "shd")
                row["nt_time"] = agg(nt_rows, "time")[0]
                row["nt_time_single"] = agg(nt_rows, "time_single")[0]
                # 对称口径：双方网格总耗时之比（主指标）
                row["speedup"] = row["nt_time"] / max(row["time_grid"], 1e-9)
                # 参考口径：双方最佳单次耗时之比
                row["speedup_single"] = row["nt_time_single"] / max(row["time"], 1e-9)
            rows.append(row)
        BENCH_STATE["result"] = {"rows": rows, "graph": a["graph"],
                                 "noise": a["noise"], "degree": degree}
        BENCH_STATE["status"] = "done"
        BENCH_STATE["progress"] = 1.0
        L("基准实验完成")
    except Exception as e:
        import traceback as tb
        BENCH_STATE["status"] = "error"
        BENCH_STATE["error"] = f"{type(e).__name__}: {e}"
        L(tb.format_exc())


@app.route("/api/scaling", methods=["POST", "GET"])
def api_scaling():
    """大规模可扩展性实测。POST 启动，GET 轮询。

    回答的问题：NOTEARS 在 d>120 跑不动，参会者如何验证 LEAST 的大规模优势？
    三部分证据（全部现场实测，外推部分明确标注）：
      1. 算子级缩放：稀疏 δ fwd+bwd 实测到任意 d，稠密 δ 实测到时间预算内，
         更大 d 用 log-log 拟合外推 → 近线性 vs 二次方，给出交叉点
      2. NOTEARS 外推：在小 d_ref 实测单次端到端与单次 h(W)，按 O(d³) 外推
      3. LEAST-SP 端到端：在中等 d 完整跑通，报告耗时/F1/收敛
    """
    if request.method == "GET":
        return jsonify({k: SCAL_STATE.get(k) for k in
                        ("status", "progress", "log", "result", "error")})
    if SCAL_STATE["status"] == "running":
        return jsonify({"status": "already_running"})
    body = request.get_json(force=True) or {}
    args = {"d_list": body.get("d", [100, 200, 400, 800, 1600, 3200]),
            "e2e_d": body.get("e2e_d", [200, 400]),
            "nt_ref_d": int(body.get("nt_ref_d", 64)),
            "with_notears": bool(body.get("with_notears", True)),
            "with_e2e": bool(body.get("with_e2e", True)),
            "dense_budget": float(body.get("dense_budget", 8.0)),
            "max_seconds": float(body.get("max_seconds", 90))}
    SCAL_STATE.update({"status": "running", "progress": 0.0, "log": [],
                       "result": None, "error": None})
    threading.Thread(target=_run_scaling, args=(args,), daemon=True).start()
    return jsonify({"status": "started"})


def _time_op(fn, min_reps=3, min_time=0.25, hard_cap=30.0):
    """重复计时：至少 min_reps 次且总时长 >= min_time 秒；单次太久则提前停。"""
    t0 = time.time()
    reps = 0
    while True:
        fn()
        reps += 1
        el = time.time() - t0
        if (reps >= min_reps and el >= min_time) or el > hard_cap:
            break
    return el / reps


def _loglog_fit(ds, ts):
    """log(t) = p·log(d) + log(a) 的最小二乘拟合，返回 (阶数 p, 系数 a)。"""
    x, y = np.log(np.asarray(ds, float)), np.log(np.asarray(ts, float))
    if len(x) < 2:
        return None, None
    p, la = np.polyfit(x, y, 1)
    return float(p), float(np.exp(la))


def _run_scaling(a):
    log = SCAL_STATE["log"]
    def L(m):
        log.append(str(m)); print("[scaling]", m, flush=True)
    try:
        from least_core import forward as f_dense, backward as b_dense, h_fun
        from least_sp import forward_sp, backward_sp, _to_csr, least_sp
        rng = np.random.default_rng(1)

        d_list = sorted({max(8, int(d)) for d in a["d_list"]})
        e2e_d = sorted({max(8, int(d)) for d in a["e2e_d"]}) if a["with_e2e"] else []
        steps = len(d_list) + len(e2e_d) + (1 if a["with_notears"] else 0)
        done = 0

        def tick():
            nonlocal done
            done += 1
            SCAL_STATE["progress"] = done / max(steps, 1)

        # ---------- 1. 算子级缩放：稀疏 δ 全程实测，稠密 δ 预算内实测 ----------
        L(f"算子级缩放实测: d ∈ {d_list}（ER-2 稀疏 W，nnz≈4d，k=5, α=0.9）")
        rows = []
        dense_ds, dense_ts, sparse_ds, sparse_ts = [], [], [], []
        dense_alive = True
        for d in d_list:
            Wd = rng.normal(0, 0.4, (d, d)) * (rng.random((d, d)) < min(1.0, 4.0 / d))
            np.fill_diagonal(Wd, 0.0)
            Ws = _to_csr(Wd)

            def run_sp():
                _, c = forward_sp(Ws, 5, 0.9)
                backward_sp(Ws, c)

            def run_de():
                _, c = f_dense(Wd, 5, 0.9)
                b_dense(Wd, c)

            t_sp = _time_op(run_sp)
            sparse_ds.append(d); sparse_ts.append(t_sp)
            # 稠密是否还值得实测：按已有点的二次方估计本次耗时
            t_de = None
            if dense_alive:
                est = (dense_ts[-1] * (d / dense_ds[-1]) ** 2) if dense_ds else 0.0
                if est <= a["dense_budget"]:
                    t_de = _time_op(run_de)
                    dense_ds.append(d); dense_ts.append(t_de)
                else:
                    dense_alive = False
                    L(f"  d={d}: 稠密预计 {est:.1f}s > 预算 {a['dense_budget']:.0f}s，"
                      f"改用拟合外推")
            rows.append({"d": d, "nnz": int(Ws.nnz), "t_sparse": t_sp, "t_dense": t_de})
            L(f"  d={d:<5} nnz={Ws.nnz:<7} 稀疏δ={t_sp*1e3:8.2f} ms"
              + (f"  稠密δ={t_de*1e3:9.2f} ms  加速={t_de/max(t_sp,1e-12):6.1f}x"
                 if t_de is not None else "  稠密δ=外推"))
            tick()

        p_sp, a_sp = _loglog_fit(sparse_ds, sparse_ts)
        p_de, a_de = _loglog_fit(dense_ds, dense_ts)
        if p_de is None:                      # 稠密点太少，按理论阶数 2 外推
            p_de, a_de = 2.0, (dense_ts[0] / dense_ds[0] ** 2 if dense_ds else None)
        for r in rows:                        # 外推列：缺实测的用拟合值
            r["t_dense_fit"] = (a_de * r["d"] ** p_de) if a_de else None
            r["dense_measured"] = r["t_dense"] is not None
            base = r["t_dense"] if r["t_dense"] is not None else r["t_dense_fit"]
            r["speedup_op"] = (base / max(r["t_sparse"], 1e-12)) if base else None
        crossover = None
        if p_sp and p_de and a_sp and a_de and p_de > p_sp:
            dc = (a_sp / a_de) ** (1.0 / (p_de - p_sp))
            if dc > 0:
                crossover = float(dc)
        L(f"拟合阶数: 稀疏δ ~ O(d^{p_sp:.2f})（理论 O(d)），"
          f"稠密δ ~ O(d^{p_de:.2f})（理论 O(d²)）"
          + (f"，交叉点 d*≈{crossover:.0f}" if crossover else ""))

        # ---------- 2. NOTEARS 小 d 实测 + O(d³) 外推 ----------
        nt = None
        if a["with_notears"]:
            from notears_baseline import notears_linear
            d_ref = min(int(a["nt_ref_d"]), 120)
            L(f"NOTEARS 参考实测: d_ref={d_ref}（λ=0.5, max_iter=30，单次）...")
            _, Xr, _ = generate_dataset(d=d_ref, seed=42, graph_type="ER",
                                        noise_type="gaussian")
            r_nt = notears_linear(Xr, lambda1=0.5, max_iter=30, w_threshold=0.0)
            Wh = rng.normal(0, 0.3, (d_ref, d_ref)) * (rng.random((d_ref, d_ref)) < 0.1)
            np.fill_diagonal(Wh, 0.0)
            t_h = _time_op(lambda: h_fun(Wh))
            nt = {"d_ref": d_ref, "t_ref": r_nt["elapsed"], "t_h": t_h,
                  "extrap": [{"d": d,
                              "t_e2e": r_nt["elapsed"] * (d / d_ref) ** 3,
                              "t_h": t_h * (d / d_ref) ** 3}
                             for d in d_list]}
            L(f"  d={d_ref} 端到端单次={r_nt['elapsed']:.2f}s，h(W) 单次={t_h*1e3:.1f}ms；"
              f"按 O(d³) 外推: d={d_list[-1]} 端到端≈{nt['extrap'][-1]['t_e2e']:.0f}s")
            tick()

        # ---------- 3. LEAST-SP 端到端实测 ----------
        e2e = []
        for d in e2e_d:
            L(f"LEAST-SP 端到端实测: d={d}（ER-2, n=10d, ε∈{{1e-1,1e-2}}）...")
            W_true, X, meta = generate_dataset(d=d, seed=42, graph_type="ER",
                                               noise_type="gaussian")
            best, t_grid = None, 0.0
            for eps in (1e-1, 1e-2):
                res = least_sp(X, d, eps=eps, T_o=100, T_i=200,
                               max_seconds=a["max_seconds"], warm_start=True,
                               use_h_termination=True, seed=42,
                               **({"compute_h": False} if d > 1000 else {}))
                t_grid += res["elapsed"]
                Wo = res.get("W_dense")
                Wo = res["W"] if Wo is None else Wo
                Wo = Wo.toarray() if hasattr(Wo, "toarray") else np.asarray(Wo)
                m = count_accuracy(W_true, Wo, taus=PAPER_TAU_GRID)
                if best is None or m["f1"] > best[1]["f1"]:
                    best = (res, m, Wo, eps)
            res, m, Wo, eps_used = best
            nnz = int(np.count_nonzero(Wo))
            nt_est = (nt["t_ref"] * (d / nt["d_ref"]) ** 3) if nt else None
            e2e.append({"d": d, "time": t_grid, "time_best": res["elapsed"],
                        "f1": m["f1"], "shd": m["shd"], "tau": m["best_tau"],
                        "eps": eps_used, "outer": res["n_outer"],
                        "converged": bool(res["converged"]), "nnz": nnz,
                        "fill": nnz / (d * d), "nt_est": nt_est,
                        "speedup_est": (nt_est / t_grid) if nt_est else None})
            L(f"  d={d}: {t_grid:.1f}s, F1={m['f1']:.3f}, SHD={m['shd']}, "
              f"外层={res['n_outer']}, nnz={nnz} ({nnz/(d*d):.1%})"
              + (f"，NOTEARS 外推≈{nt_est:.0f}s → 预计加速 {nt_est/t_grid:.1f}x"
                 if nt_est else ""))
            tick()

        SCAL_STATE["result"] = {
            "rows": rows,
            "fit": {"sparse_order": p_sp, "dense_order": p_de,
                    "crossover_d": crossover,
                    "dense_budget": a["dense_budget"]},
            "notears": nt, "e2e": e2e}
        SCAL_STATE["status"] = "done"
        SCAL_STATE["progress"] = 1.0
        L("可扩展性实测完成")
    except Exception as e:
        SCAL_STATE["status"] = "error"
        SCAL_STATE["error"] = f"{type(e).__name__}: {e}"
        L(traceback.format_exc())


@app.route("/api/health")
def api_health():
    import scipy
    return jsonify({"ok": True, "numpy": np.__version__, "scipy": scipy.__version__})


if __name__ == "__main__":
    print("=" * 70)
    print("  LEAST — Interactive Web Demo for BN Structure Learning")
    print("  http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
