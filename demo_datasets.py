"""
================================================================================
 demo_datasets.py — Demo 内置示例数据集（对应论文 §VI 的两类应用）
================================================================================
  1) flight_booking : 论文 §VI-A 的"机票预订故障 / 根因分析"场景
     节点 = 监控指标（航司、票务源、机场、各环节错误率、基础设施延迟…）
     有 ground-truth DAG —— 可现场演示"从数据自动挖出故障传播链"

  2) movie_ratings  : 论文 §VI-C 的"可解释推荐"场景
     节点 = 电影（每列一部电影，每个用户一条样本），4 个潜在 genre
     学到的 BN 中同类型电影会互相连接 —— 可作"因为喜欢 A 所以推荐 B"的解释
================================================================================
"""
from __future__ import annotations
import os
import numpy as np

from least_core import simulate_linear_sem


# -----------------------------------------------------------------------------
# 1. 机票预订监控（论文 §VI-A）
# -----------------------------------------------------------------------------
FLIGHT_NODES = [
    "traffic_volume",   # 0  业务流量
    "cache_hit_rate",   # 1  缓存命中率
    "db_latency",       # 2  数据库延迟
    "cdn_latency",      # 3  CDN 延迟
    "airline_X",        # 4  航司 X 系统状态
    "airline_Y",        # 5  航司 Y 系统状态
    "fare_source_A",    # 6  票务源 A（航司直连）
    "fare_source_B",    # 7  票务源 B（中台/Amadeus）
    "airport_HGH",      # 8  杭州机场
    "airport_PEK",      # 9  北京机场
    "err_query_seat",   # 10 步骤1 查询座位 错误率
    "err_query_price",  # 11 步骤2 查询价格 错误率
    "err_reserve",      # 12 步骤3 占座     错误率
    "err_payment",      # 13 步骤4 支付     错误率
]

# (cause, effect, weight)  —— 人工构造的、可解释的故障传播 DAG
FLIGHT_EDGES = [
    (0, 2, 1.2), (0, 3, 1.0),          # 流量 ↑ → DB/CDN 延迟 ↑
    (1, 2, -1.4),                      # 缓存命中率 ↑ → DB 延迟 ↓
    (4, 6, 0.9), (5, 7, 0.8),          # 航司系统 → 票务源
    (4, 10, 1.1), (5, 10, 0.7),        # 航司 → 查座位错误
    (6, 11, 1.3), (7, 11, 1.0),        # 票务源 → 查价格错误
    (2, 11, 0.9), (2, 12, 0.8),        # DB 延迟 → 查价/占座错误
    (3, 10, 0.6),                      # CDN 延迟 → 查座位错误
    (8, 10, 0.7), (9, 12, 0.9),        # 机场 → 查座位 / 占座错误
    (10, 12, 1.1),                     # 查座位失败 → 占座失败
    (11, 13, 1.2), (12, 13, 1.0),      # 前序失败 → 支付失败
]


def flight_booking(n=2000, seed=7, anomaly=False):
    """机票预订监控数据。anomaly=True 时注入一次故障（某航司异常）。"""
    d = len(FLIGHT_NODES)
    W = np.zeros((d, d))
    for i, j, w in FLIGHT_EDGES:
        W[i, j] = w
    rng = np.random.default_rng(seed)
    X = simulate_linear_sem(W, n, "gauss", rng=rng)
    if anomaly:
        # 注入：航司 X 系统异常 → 抬高其自身及下游
        X[:, 4] += 3.0
    return W, X, {"d": d, "n": n, "node_names": list(FLIGHT_NODES),
                  "num_edges": int(np.count_nonzero(W)),
                  "target_nodes": ["err_query_seat", "err_query_price",
                                   "err_reserve", "err_payment"],
                  "desc": "机票预订全链路监控（论文 §VI-A 场景）"}


# -----------------------------------------------------------------------------
# 2. 电影评分（论文 §VI-C 可解释推荐）
# -----------------------------------------------------------------------------
MOVIE_GENRES = ["Action", "Romance", "Sci-Fi", "Comedy"]


def movie_ratings(n=3000, seed=11, d=32):
    """用户-电影评分矩阵。每列一部电影，每行一个用户（已去用户均值，同论文）。

    潜在结构: rating[u,i] = 3.5 + 1.3 * pref[u, genre(i)] + noise
    因此同类型电影之间会呈现强相关 —— LEAST 应能学出"同类电影团簇"。
    """
    rng = np.random.default_rng(seed)
    genre_of = np.array([i % 4 for i in range(d)])
    names = [f"{MOVIE_GENRES[g]}-{i + 1:02d}" for i, g in enumerate(genre_of)]
    pref = rng.normal(0, 1.0, size=(n, 4))
    R = 3.5 + 1.3 * pref[:, genre_of] + rng.normal(0, 0.6, size=(n, d))
    R = np.clip(R, 0.5, 5.0)
    X = R - R.mean(axis=1, keepdims=True)      # 减去用户均值（论文 §V-B 做法）
    # ground-truth: 同 genre 的电影互为"相关"，用块内全连接作为参考结构
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j and genre_of[i] == genre_of[j] and i < j:
                W[i, j] = 1.0
    return W, X, {"d": d, "n": n, "node_names": names,
                  "num_edges": int(np.count_nonzero(W)),
                  "genres": [MOVIE_GENRES[g] for g in genre_of],
                  "desc": "电影评分（论文 §VI-C 可解释推荐场景）"}


# -----------------------------------------------------------------------------
# 3. Sachs 蛋白信号网络（真实连续数据 + 公认真值图）
# -----------------------------------------------------------------------------
# 来源：Sachs et al. (2005) Science 308:523；Zenodo 10.5281/zenodo.7681811（CC BY 4.0）
# 数据：流式细胞术，853 个细胞 × 11 种磷酸化蛋白/磷脂（观测子集 cd3cd28）
# 真值：原文共识图 20 条边（含 PKA<->PIP3 反向对）；另提供去环的 19 条边版本
SACHS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "example_data", "sachs")


def sachs_protein(log=True, acyclic=True, seed=None):
    """真实生物数据：Sachs 蛋白信号网络（有 ground truth，可算 F1/P/R/SHD）。

    log     : 是否对表达量取 log(1+x)（文献常见做法，缓解长尾）
    acyclic : True 用去环真值（19 边）；False 用原始共识图（20 边，含 1 个 2-环）
    """
    data_f = os.path.join(SACHS_DIR, "sachs_cd3cd28.csv")
    gt_f = os.path.join(SACHS_DIR, "sachs_ground_truth_dag.csv" if acyclic
                        else "sachs_ground_truth.csv")
    import csv as _csv
    with open(data_f, encoding="utf-8") as f:
        rdr = _csv.reader(f)
        names = next(rdr)
        X = np.array([[float(v) for v in row] for row in rdr if row])
    with open(gt_f, encoding="utf-8") as f:
        rdr = _csv.reader(f)
        next(rdr)
        edges = [(r[0], r[1]) for r in rdr if len(r) >= 2]
    if log:
        X = np.log1p(np.maximum(X, 0.0))
    d = len(names)
    W = np.zeros((d, d))
    for a, b in edges:
        if a in names and b in names:
            W[names.index(a), names.index(b)] = 1.0
    n = X.shape[0]
    return W, X, {"d": d, "n": n, "node_names": names,
                  "num_edges": int(np.count_nonzero(W)),
                  "target_nodes": ["Erk", "Akt", "Jnk", "P38"],
                  "desc": f"Sachs 蛋白信号网络（真实流式细胞术数据，853×11，"
                          f"真值 {int(np.count_nonzero(W))} 边"
                          f"{'，log 变换' if log else ''}）"}


# -----------------------------------------------------------------------------
# 4. Causal Chambers — Light Tunnel（真实物理装置，真值由物理决定）
# -----------------------------------------------------------------------------
# 来源：Gamella, Peters & Bühlmann (2025), Nature Machine Intelligence,
#       10.1038/s42256-024-00964-x；数据 CC BY 4.0；取自 lt_walks_v1/actuators_white
# 规模：10,000 × 20（保留本实验中实际变动的 20 个变量），真值 39 边（原图诱导子图）
CHAMBERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "example_data", "chambers")


def chambers_light_tunnel(n=10000, seed=None, log=False):
    """真实物理装置数据：Causal Chambers 光隧道（真值 DAG 由装置物理决定）。

    真值不是专家共识、也不是统计反推，而是由"哪个传感器读哪个物理量"直接确定，
    因此是本项目里可信度最高的真实数据基准。
    """
    import csv as _csv
    with open(os.path.join(CHAMBERS_DIR,
                           "chambers_lt_actuators_white.csv"), encoding="utf-8") as f:
        rdr = _csv.reader(f)
        names = next(rdr)
        X = np.array([[float(v) for v in row] for row in rdr if row])
    with open(os.path.join(CHAMBERS_DIR,
                           "chambers_lt_ground_truth.csv"), encoding="utf-8") as f:
        rdr = _csv.reader(f)
        next(rdr)
        edges = [(r[0], r[1]) for r in rdr if len(r) >= 2]
    if log:
        X = np.log1p(np.maximum(X, 0.0))
    d = len(names)
    W = np.zeros((d, d))
    for a, b in edges:
        if a in names and b in names:
            W[names.index(a), names.index(b)] = 1.0
    n = int(min(max(n, 100), X.shape[0]))
    if n < X.shape[0]:
        idx = np.linspace(0, X.shape[0] - 1, n).round().astype(int)
        X = X[idx]
    return W, X, {"d": d, "n": X.shape[0], "node_names": names,
                  "num_edges": int(np.count_nonzero(W)),
                  "target_nodes": ["ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3"],
                  "desc": f"Causal Chambers 光隧道（真实物理装置，{X.shape[0]}×{d}，"
                          f"真值 {int(np.count_nonzero(W))} 边，由物理决定）"}


DATASETS = {
    "flight": flight_booking,
    "movie": movie_ratings,
    "sachs": sachs_protein,
    "chambers": chambers_light_tunnel,
}


def get_dataset(name, **kw):
    """按数据集签名过滤参数，避免前端统一透传时报错。"""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}, available: {list(DATASETS)}")
    fn = DATASETS[name]
    import inspect
    supported = set(inspect.signature(fn).parameters)
    kw = {k: v for k, v in kw.items() if k in supported}
    return fn(**kw)
