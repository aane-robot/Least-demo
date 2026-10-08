"""
================================================================================
 benchmark.py — 复现论文 §V-A 的基准实验
================================================================================
 论文协议：
   - 图模型: ER (平均度 2) / SF (平均度 4)
   - 噪声:   Gaussian / Exponential / Gumbel
   - d ∈ {10, 20, 50, 100},  n = 10d,  5 个随机种子
   - 网格搜索 ε ∈ {1e-1,1e-2,1e-3,1e-4} 与 τ ∈ {0.1,...,0.5}，报告最佳情形
   - LEAST: k=5, α=0.9, lr=0.01, ζ=1e-4, λ=0.5, B=1.005, T_o=1000, T_i=200
   - 终止: δ(W) ≤ ε 且 h(W) ≤ ε  (与 NOTEARS 用同一终止判据)
   - 对照: NOTEARS (ref [38])

 用法:
   python benchmark.py --quick                 # d=10,20 快速验证
   python benchmark.py --d 10 20 50 100        # 完整
   python benchmark.py --d 10 20 --no-notears  # 跳过 NOTEARS 对照
================================================================================
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from least_core import (generate_dataset, least, count_accuracy, h_fun,
                        PAPER_EPS_GRID, PAPER_TAU_GRID)


def run_least_grid(W_true, X, d, eps_grid=PAPER_EPS_GRID, lam=0.5, seed=42,
                   T_o=1000, T_i=200, max_seconds=90, warm_start=True,
                   verbose=False):
    """在 ε×τ 网格上跑 LEAST，返回最佳结果（论文 "report the best case"）。

    计时口径（与 run_notears_grid 对称）：
      time       = 最佳那一次的单次耗时
      time_grid  = 整个 ε 网格的总耗时（公平比较加速比时应使用此值）
    """
    best = None
    t_grid = 0.0
    for eps in eps_grid:
        res = least(X, d, k=5, alpha=0.9, zeta=1e-4, lam=lam, eps=eps, B=1.005,
                    T_o=T_o, T_i=T_i, lr=0.01, seed=seed, warm_start=warm_start,
                    use_h_termination=True, max_seconds=max_seconds)
        t_grid += res["elapsed"]
        m = count_accuracy(W_true, res["W"], taus=PAPER_TAU_GRID)
        rec = {**{k: m[k] for k in ("f1", "precision", "recall", "shd")},
               "eps": eps, "tau": m["best_tau"], "delta": res["delta_final"],
               "h": res["h_final"], "converged": res["converged"],
               "n_outer": res["n_outer"], "time": res["elapsed"]}
        if best is None or rec["f1"] > best["f1"]:
            best = rec
        if verbose:
            print(f"    eps={eps:<8} conv={res['converged']!s:5} outer={res['n_outer']:<4} "
                  f"δ={res['delta_final']:.2e} h={res['h_final']:.2e} "
                  f"F1={m['f1']:.3f} SHD={m['shd']}", flush=True)
    best["time_grid"] = t_grid
    return best


def run_notears_grid(W_true, X, lam_grid=(0.1, 0.5), w_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """NOTEARS 对照：λ 网格取最佳。

    计时口径（与 run_least_grid 对称）：
      time        = 网格总耗时（λ 网格各次累加）
      time_single = 最佳那一次的单次耗时
    """
    from notears_baseline import notears_linear
    best, best_t = None, 0.0
    for lam in lam_grid:
        r = notears_linear(X, lambda1=lam, max_iter=30, w_threshold=0.0)
        best_t += r["elapsed"]
        for tau in w_thresholds:
            m = count_accuracy(W_true, r["W"], taus=(tau,))
            rec = {**{k: m[k] for k in ("f1", "precision", "recall", "shd")},
                   "lam": lam, "tau": tau, "h": r["h_final"],
                   "converged": r["converged"], "time": r["elapsed"]}
            if best is None or rec["f1"] > best["f1"]:
                best = rec
    best["time_single"] = best["time"]
    best["time"] = best_t
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=(10, 20, 50, 100))
    ap.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44, 45, 46))
    ap.add_argument("--graph", default="ER", choices=["ER", "SF"])
    ap.add_argument("--noise", default="gaussian",
                    choices=["gaussian", "exponential", "gumbel"])
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--T_o", type=int, default=1000)
    ap.add_argument("--T_i", type=int, default=200)
    ap.add_argument("--max_seconds", type=float, default=90)
    ap.add_argument("--no-notears", action="store_true")
    ap.add_argument("--quick", action="store_true", help="只跑 d=10,20 且 seed=42,43")
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    if args.quick:
        args.d, args.seeds = (10, 20), (42, 43)
    degree = args.degree if args.degree else (2 if args.graph == "ER" else 4)

    results = {}
    print("=" * 92)
    print(f"  LEAST §V-A 基准复现   graph={args.graph}-{degree}, noise={args.noise}, "
          f"n=10d, seeds={list(args.seeds)}")
    print("=" * 92)

    for d in args.d:
        least_rows, nt_rows = [], []
        for seed in args.seeds:
            W_true, X, meta = generate_dataset(d=d, seed=seed, graph_type=args.graph,
                                               noise_type=args.noise, degree=degree)
            print(f"\n--- d={d}, seed={seed}, 真实边数={meta['num_edges']} ---", flush=True)
            b = run_least_grid(W_true, X, d, seed=seed, T_o=args.T_o, T_i=args.T_i,
                               max_seconds=args.max_seconds, verbose=True)
            least_rows.append(b)
            print(f"    LEAST   BEST: F1={b['f1']:.4f} P={b['precision']:.3f} "
                  f"R={b['recall']:.3f} SHD={b['shd']} (ε={b['eps']}, τ={b['tau']}, "
                  f"t={b['time']:.1f}s)", flush=True)
            if not args.no_notears:
                n = run_notears_grid(W_true, X)
                nt_rows.append(n)
                print(f"    NOTEARS BEST: F1={n['f1']:.4f} P={n['precision']:.3f} "
                      f"R={n['recall']:.3f} SHD={n['shd']} (λ={n['lam']}, τ={n['tau']}, "
                      f"t={n['time']:.1f}s)", flush=True)

        def agg(rows, key):
            v = [r[key] for r in rows]
            return float(np.mean(v)), float(np.std(v))
        entry = {"d": d, "n": 10 * d,
                 "least": {k: agg(least_rows, k) for k in ("f1", "precision", "recall", "shd")},
                 "least_time": agg(least_rows, "time")[0],
                 "least_time_grid": agg(least_rows, "time_grid")[0],
                 "least_conv": int(sum(r["converged"] for r in least_rows)),
                 "n_seed": len(least_rows)}
        if nt_rows:
            entry["notears"] = {k: agg(nt_rows, k) for k in ("f1", "precision", "recall", "shd")}
            entry["notears_time"] = agg(nt_rows, "time")[0]
            entry["notears_time_single"] = agg(nt_rows, "time_single")[0]
            # 对称口径：网格总耗时 ÷ 网格总耗时（论文双方都是 "best case over grid"）
            entry["speedup"] = entry["notears_time"] / max(entry["least_time_grid"], 1e-9)
            # 参考口径：单次 ÷ 单次
            entry["speedup_single"] = (entry["notears_time_single"]
                                       / max(entry["least_time"], 1e-9))
        results[f"{args.graph}-{degree}_{args.noise}_d{d}"] = entry

    print("\n" + "=" * 100)
    print(f"{'d':>5} | {'LEAST F1':>16} | {'LEAST SHD':>14} | {'NOTEARS F1':>16} | "
          f"{'NOTEARS SHD':>13} | {'加速比(网格)':>10} | {'加速比(单次)':>10}")
    print("-" * 100)
    for k, e in results.items():
        nt = e.get("notears")
        nt_f1 = f"{nt['f1'][0]:.3f}±{nt['f1'][1]:.3f}" if nt else "  —  "
        nt_shd = f"{nt['shd'][0]:.1f}±{nt['shd'][1]:.1f}" if nt else "  —  "
        sp = f"{e['speedup']:.1f}x" if "speedup" in e else "  —  "
        sp1 = f"{e['speedup_single']:.1f}x" if "speedup_single" in e else "  —  "
        print(f"{e['d']:>5} | {e['least']['f1'][0]:.3f}±{e['least']['f1'][1]:.3f}       | "
              f"{e['least']['shd'][0]:.1f}±{e['least']['shd'][1]:.1f}       | {nt_f1}       | "
              f"{nt_shd:>13} | {sp:>10} | {sp1:>10}")
    print("=" * 100)
    print("注: 加速比(网格) = NOTEARS λ网格总耗时 ÷ LEAST ε网格总耗时（对称口径，主指标）")
    print("    加速比(单次) = 双方各自最佳配置的单次耗时之比（参考口径）")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 -> {args.out}")


if __name__ == "__main__":
    main()
