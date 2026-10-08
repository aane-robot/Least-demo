# -*- coding: utf-8 -*-
"""
bn_io.py —— 贝叶斯网络定义文件（BIF / DSC / NET）的解析、采样与真值 DAG 导出。

用途：让网页 Demo 支持上传标准 BN 基准网络（如 bnlearn repository 的
asia.bif / alarm.bif / hepar2.net / *.dsc），系统按 CPT 前向采样生成观测数据，
并把网络自身的边表作为 ground truth，从而可以计算 F1 / P / R / SHD。

支持格式
  BIF  —— Bayesian Interchange Format（bnlearn、UAI 仓库常用）
  DSC  —— bnlearn 的 .dsc 文本格式
  NET  —— Hugin .net 格式

限制：这三种格式只描述**离散**网络；采样得到的列是状态索引（0..k-1），
数据会被当作数值喂给 LEAST（线性 SEM），因此精度仅供参考。
"""
from __future__ import annotations

import re
import numpy as np

__all__ = ["BN", "parse_network", "detect_format", "bn_to_dag",
           "sample_bn", "load_network", "NET_FORMATS"]

NET_FORMATS = ("bif", "dsc", "net")


class BN:
    """离散贝叶斯网络：节点、状态、父集、条件概率表。"""

    def __init__(self, name="unknown"):
        self.name = name
        self.names = []                 # 节点名，按声明顺序
        self.states = {}                # name -> [state labels]
        self.parents = {}               # name -> [parent names]
        self.cpt = {}                   # name -> np.ndarray，形状 (parent_dims..., k)

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        return f"<BN {self.name}: {len(self.names)} nodes, {self.num_edges()} arcs>"

    def num_edges(self):
        return sum(len(v) for v in self.parents.values())

    def index(self, name):
        return self.names.index(name)


# --------------------------------------------------------------------------
# 文本预处理与工具
# --------------------------------------------------------------------------
def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"(?m)//.*$", " ", text)
    text = re.sub(r"(?m)%.*$", " ", text)
    return text


def _split_states(raw: str):
    """解析状态列表：'yes, no' / '"yes" "no"' / 'yes no'"""
    raw = raw.strip()
    if not raw:
        return []
    items = re.findall(r'"([^"]*)"|\'([^\']*)\'|([^,\s]+)', raw)
    out = []
    for a, b, c in items:
        v = (a or b or c).strip()
        if v and v != ",":
            out.append(v)
    return out


def _nums(raw: str):
    return np.array([float(x) for x in
                     re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)], dtype=float)


def _blocks(text: str, keyword: str):
    """提取 `keyword ... { ... }` 的 (头部, 主体) 列表，支持嵌套一层花括号。"""
    out = []
    pat = re.compile(r"\b" + keyword + r"\b\s*\(([^)]*)\)\s*\{", re.I)
    for m in pat.finditer(text):
        i = m.end() - 1            # 指向 '{'
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((m.group(1).strip(), text[i + 1:j]))
    return out


def _node_blocks(text: str):
    """提取 `variable|node X { ... }` 的 (名字, 主体)。"""
    out = []
    pat = re.compile(r"\b(?:variable|node)\s+([A-Za-z_][\w\-\.]*)\s*\{", re.I)
    for m in pat.finditer(text):
        i = m.end() - 1
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((m.group(1), text[i + 1:j]))
    return out


# --------------------------------------------------------------------------
# 三种格式各自的节点 / CPT 解析
# --------------------------------------------------------------------------
def _parse_nodes_bif(bn: BN, text: str):
    for name, body in _node_blocks(text):
        m = re.search(r"type\s+discrete\s*\[\s*(\d+)\s*\]\s*\{(.*?)\}", body, re.S | re.I)
        if m:
            st = _split_states(m.group(2))
            if len(st) != int(m.group(1)):
                st = [f"s{i}" for i in range(int(m.group(1)))]
        else:
            st = ["0", "1"]
        bn.names.append(name)
        bn.states[name] = st


def _parse_nodes_dsc(bn: BN, text: str):
    for name, body in _node_blocks(text):
        m = re.search(r"type\s*:\s*discrete\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\}", body, re.S | re.I)
        if m:
            st = _split_states(m.group(2))
            if len(st) != int(m.group(1)):
                st = [f"s{i}" for i in range(int(m.group(1)))]
        else:
            st = ["0", "1"]
        bn.names.append(name)
        bn.states[name] = st


def _parse_nodes_net(bn: BN, text: str):
    for name, body in _node_blocks(text):
        m = re.search(r"states\s*=\s*\((.*?)\)", body, re.S | re.I)
        st = _split_states(m.group(1)) if m else ["0", "1"]
        bn.names.append(name)
        bn.states[name] = st


def _rows_from_body(body: str):
    """把 probability / potential 主体解析成 [(父组合元组|None, 数值数组)]。

    支持：
      BIF  `(yes) 0.1, 0.9;` / `table 0.5, 0.5;`
      DSC  `(0) : 0.05, 0.95;` 以及无父时直接 `0.01, 0.99;`
      NET  `data = ( 0.01 0.99 );` 与多父时 `data = ((0.05 0.95)(0.01 0.99));`
    """
    # ---- NET：data = ( ... )，多父时为嵌套括号，每个内层括号 = 一个父组合
    m = re.search(r"data\s*=\s*\((.*)\)\s*;", body, re.S | re.I)
    if m:
        inner = m.group(1)
        groups = re.findall(r"\(([^()]*)\)", inner)
        if groups:
            return [(None, _nums(g)) for g in groups]
        return [(None, _nums(inner))]

    # ---- BIF / DSC：按父组合分行
    rows = []
    for mm in re.finditer(r"\(([^)]*)\)\s*:?\s*([0-9eE\.\+\-\s,]+);", body):
        key = mm.group(1).strip()
        if not key:
            continue
        parts = [p.strip().strip('"\'') for p in key.split(",")] if "," in key \
            else [p.strip().strip('"\'') for p in key.split()]
        rows.append((parts, _nums(mm.group(2))))
    if rows:
        return rows

    # ---- table 形式
    m = re.search(r"table\s*([0-9eE\.\+\-\s,]+);", body, re.I)
    if m:
        return [(None, _nums(m.group(1)))]

    # ---- 无父节点且没有任何关键字：整块就是概率向量（DSC 常见）
    stripped = body.strip()
    if stripped and re.fullmatch(r"[0-9eE\.\+\-\s,;]+", stripped):
        for seg in stripped.split(";"):
            arr = _nums(seg)
            if arr.size:
                return [(None, arr)]
    return []


def _parse_cpts(bn: BN, text: str, keyword="probability"):
    for head, body in _blocks(text, keyword):
        if "|" in head:
            child, par = [s.strip() for s in head.split("|", 1)]
            parents = [p.strip() for p in par.replace(",", " ").split()] if "," not in par \
                else [p.strip() for p in par.split(",")]
        else:
            child, parents = head.strip(), []
        if child not in bn.names:
            continue
        rows = _rows_from_body(body)
        if not rows:
            continue
        # 父状态名 → 索引
        def to_idx(vals):
            out = []
            for p, v in zip(parents, vals):
                st = bn.states.get(p, [])
                out.append(st.index(v) if v in st else int(v) if v.lstrip("-").isdigit() else 0)
            return tuple(out)

        pdims = [len(bn.states.get(p, ["0", "1"])) for p in parents]
        k = len(bn.states[child])
        table = np.zeros(pdims + [k] if pdims else [k], dtype=float)
        flat = table.reshape(-1, k)
        row_i = 0
        for parts, arr in rows:
            if arr.size != k:            # 容错：长度不符就截断/补齐
                arr = np.resize(arr, k)
            if parts is None or not parents:
                if row_i < flat.shape[0]:      # 无键的行按顺序填充（NET / table）
                    flat[row_i] = arr
                    row_i += 1
            else:
                table[to_idx(parts)] = arr
        # 归一化，防止文件里概率和不为 1
        s = table.sum(axis=-1, keepdims=True)
        table = np.where(s > 0, table / np.where(s > 0, s, 1.0), 1.0 / k)
        bn.parents[child] = parents
        bn.cpt[child] = table
    # 没有显式 CPT 的节点：均匀分布
    for nm in bn.names:
        if nm not in bn.parents:
            bn.parents[nm] = []
            k = len(bn.states[nm])
            bn.cpt[nm] = np.full(k, 1.0 / k)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def detect_format(text: str, filename: str = "") -> str:
    """根据内容/扩展名判断格式，返回 'bif' | 'dsc' | 'net' | 'csv'。

    数据表（.csv/.txt）优先按 CSV 处理，只有同时出现 CPT 块与节点声明、
    且含花括号结构时才判为网络文件，避免列名里带 "variable/probability" 的误判。
    """
    low = text[:4000].lower()
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    has_cpt = ("probability" in low) or ("potential" in low)
    has_node = ("type discrete" in low) or ("states =" in low) \
        or ("variable" in low) or ("node " in low)
    if "belief network" in low or ("node " in low and "type :" in low):
        return "dsc"
    if "potential" in low and "states =" in low:
        return "net"
    if ext in ("csv", "txt", "tsv", "dat"):
        return "bif" if (has_cpt and has_node and "{" in low) else "csv"
    if has_cpt and has_node:
        return "bif"
    if ext in NET_FORMATS:
        return ext
    return "csv"


def parse_network(text: str, fmt: str | None = None, filename: str = "") -> BN:
    if fmt is None:
        fmt = detect_format(text, filename)
    if fmt not in NET_FORMATS:
        raise ValueError(f"不支持的网络格式: {fmt}（支持 {NET_FORMATS}）")
    text = _strip_comments(text)
    bn = BN()
    m = re.search(r"(?:network|belief network|net)\s*\"?([^\"{\n]*)\"?\s*\{", text, re.I)
    if m and m.group(1).strip():
        bn.name = m.group(1).strip().strip('"')
    if fmt == "bif":
        _parse_nodes_bif(bn, text); _parse_cpts(bn, text, "probability")
    elif fmt == "dsc":
        _parse_nodes_dsc(bn, text); _parse_cpts(bn, text, "probability")
    else:                                  # net (Hugin)
        _parse_nodes_net(bn, text); _parse_cpts(bn, text, "potential")
    if not bn.names:
        raise ValueError("未解析到任何节点，请确认文件格式（BIF / DSC / NET）")
    return bn


def load_network(path: str, fmt: str | None = None) -> BN:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    return parse_network(text, fmt, path)


# --------------------------------------------------------------------------
# 真值 DAG 与采样
# --------------------------------------------------------------------------
def _topo(bn: BN):
    indeg = {n: len(bn.parents.get(n, [])) for n in bn.names}
    children = {n: [] for n in bn.names}
    for c, ps in bn.parents.items():
        for p in ps:
            children[p].append(c)
    ready = [n for n in bn.names if indeg[n] == 0]
    order = []
    while ready:
        v = ready.pop(0)
        order.append(v)
        for c in children[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != len(bn.names):
        raise ValueError("网络存在环，无法采样（BN 应为 DAG）")
    return order


def bn_to_dag(bn: BN):
    """返回 (节点名列表, A)，A[i, j] = 1 表示 i -> j。"""
    d = len(bn.names)
    A = np.zeros((d, d), dtype=float)
    for child, ps in bn.parents.items():
        j = bn.index(child)
        for p in ps:
            A[bn.index(p), j] = 1.0
    return list(bn.names), A


def sample_bn(bn: BN, n: int, seed: int = 0) -> np.ndarray:
    """按拓扑序前向采样，返回 (n, d) 的整数矩阵（值为状态索引）。"""
    rng = np.random.default_rng(seed)
    d = len(bn.names)
    X = np.zeros((n, d), dtype=float)
    for name in _topo(bn):
        j = bn.index(name)
        parents = bn.parents.get(name, [])
        table = bn.cpt[name]
        if not parents:
            probs = np.broadcast_to(table, (n, table.shape[-1]))
        else:
            pidx = [bn.index(p) for p in parents]
            pdims = [len(bn.states[p]) for p in parents]
            combos = np.ravel_multi_index(
                tuple(X[:, pidx].astype(int).T), dims=pdims)
            probs = table.reshape(-1, table.shape[-1])[combos]
        # 向量化按行多项式采样
        cum = probs.cumsum(axis=1)
        cum[:, -1] = 1.0
        u = rng.random(n)[:, None]
        X[:, j] = (cum < u).sum(axis=1)
    return X
