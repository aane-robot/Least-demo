"""把下载的 UCI 原始数据转成 Demo 期望的标准 CSV 格式。

标准格式（Demo 上传要求）：
  - 逗号分隔 CSV，UTF-8
  - 第一行为列名（每个列名 = 因果图里的一个节点）
  - 第 2 行起为样本：每行一条观测，每列一个变量
  - 全部为数值；缺失值留空或写 NaN（后端按列均值填补）
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- 1) Auto MPG (UCI, 真实) ----------------
# 原始为空格分隔、无表头、horsepower 含 "?" 缺失值
cols = ["mpg", "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model_year", "origin"]
raw = []
with open(os.path.join(HERE, "_auto_mpg_raw.data"), "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 车名含空格且可能带引号 -> 只取前 8 个字段
        parts = line.replace('"', "").split()
        raw.append(parts[:8])
df = pd.DataFrame(raw, columns=cols)
df = df.replace("?", np.nan).astype(float)
print(f"auto_mpg: 原始 {len(df)} 行, horsepower 缺失 {int(df['horsepower'].isna().sum())} 个")
df = df.fillna(df.mean(numeric_only=True))          # 列均值填补
df.to_csv(os.path.join(HERE, "auto_mpg.csv"), index=False)
print(f"  -> auto_mpg.csv  {df.shape[0]} 行 x {df.shape[1]} 列")

# ---------------- 2) Wine Quality - red (UCI, 真实) ----------------
w = pd.read_csv(os.path.join(HERE, "_wine_raw.csv"), sep=";")
w.columns = [c.replace(" ", "_") for c in w.columns]   # 列名去空格
w.to_csv(os.path.join(HERE, "wine_quality_red.csv"), index=False)
print(f"  -> wine_quality_red.csv  {w.shape[0]} 行 x {w.shape[1]} 列")

# 清理中间文件
for f in ("_auto_mpg_raw.data", "_wine_raw.csv"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        os.remove(p)

print("\n生成的示例文件：")
for f in sorted(os.listdir(HERE)):
    if f.endswith(".csv"):
        p = os.path.join(HERE, f)
        with open(p, encoding="utf-8") as fh:
            head = fh.readline().strip()
        nrow = sum(1 for _ in open(p, encoding="utf-8")) - 1
        print(f"  {f:26s} {nrow:5d} 行   列名: {head[:90]}")
