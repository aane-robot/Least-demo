# 上传数据集：格式要求与示例

## 0. 支持的文件类型

| 类型 | 扩展名 | 内容 | 有 ground truth？ |
|---|---|---|---|
| 数据表 | `.csv` | 每行一个样本、每列一个变量的数值矩阵 | ❌ F1 显示 "—" |
| 贝叶斯网络定义 | `.bif` / `.dsc` / `.net` | bnlearn / Hugin 格式的网络结构 + CPT | ✅ 网络边表即真值，可算 F1 / P / R / SHD |

网络文件上传后，系统会按 CPT 前向采样生成观测数据（默认 n = 2000，界面可调），
并把网络边表作为 ground truth。**标准基准网络见 `bn/` 子目录**（ASIA / CHILD / ALARM）。

## 1. 你的数据要长什么样

LEAST 学的是**线性结构方程模型（LSEM）**，所以它期望的是一个**普通的二维数值表格**：

```
      var_1   var_2   var_3   ...   var_d      ← 第 1 行：列名（每个列名 = 因果图的一个节点）
      1.23   -0.44    2.10   ...    0.88      ← 第 2 行起：每条一行观测
     -0.51    1.02    0.33   ...   -1.20
       ...     ...     ...    ...    ...
```

| 要求 | 说明 |
|---|---|
| **文件** | `.csv`，逗号分隔，UTF-8 |
| **行 = 样本** | 每一行是一条独立观测（一个用户 / 一次请求 / 一个时刻…） |
| **列 = 变量** | 每一列是因果图里的一个节点，**列名即节点名** |
| **表头** | 必须有（第一行）。若没有表头，网页上取消勾选"首行为列名"，节点自动命名为 V1, V2… |
| **数值** | 全部为数字。非数值单元格会按**列均值**填补 |
| **缺失值** | 允许留空或写 `NA`/`NaN`，同上按列均值填补 |
| **样本量** | 建议 `n ≥ 10 × d`（论文 §V-A 用 n=10d）。样本太少学不出稳定结构 |
| **变量数** | 网页 Demo 上限 400 列；更多请先做特征筛选，或改用 `least_sp.py` |
| **标准化** | 建议勾选"标准化"（各列减均值除标准差）。量纲差异大时强烈建议开启 |
| **i.i.d.** | 样本需独立同分布。时间序列请先差分/去趋势，否则会学到虚假的自相关边 |

### 一个**反例**（不要这样）

```
用户ID,性别,城市,是否购买      ← 类别型变量、ID 列：请先做数值编码或剔除
1,男,北京,是
```
- ID 列一定要删掉（它不是变量）
- 类别变量需先编码（one-hot / 序数），但注意 one-hot 会引入确定性依赖
- 若要复现论文 §VI-C 的推荐场景，请先**减去用户均值**（示例脚本里有演示）

---

## 2. 两个真实示例（UCI，可直接上传测试）

| 文件 | 来源 | 规模 | 说明 |
|---|---|---|---|
| `auto_mpg.csv` | UCI Auto MPG | 398 × 8 | 汽车油耗数据。`horsepower` 原含 6 个缺失值，已按列均值填补。**有直观的物理因果**：`cylinders → displacement`、`weight → mpg` 等 |
| `wine_quality_red.csv` | UCI Wine Quality (red) | 1599 × 12 | 红酒理化指标 + 质量评分。11 个理化指标之间存在真实的化学因果链（如 `density ↔ alcohol`、`fixed_acidity → pH`） |

两个文件都是**真实世界数据、非合成**，适合现场演示"拿自己的数据跑一遍"。

```bash
# 直接看前几行
head -3 auto_mpg.csv
# mpg,cylinders,displacement,horsepower,weight,acceleration,model_year,origin
# 18.0,8,307.0,130.0,3504,12.0,70,1
# 15.0,8,350.0,165.0,3693,11.5,70,1
```

---

## 3. 用命令行先试跑（不开网页）

```python
import numpy as np, pandas as pd
from least_core import least, count_accuracy
from least_sp import least_sp

df = pd.read_csv("example_data/wine_quality_red.csv")
X = df.values.astype(float)
X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)   # 标准化
names = list(df.columns)
d = X.shape[1]

res = least(X, d, lam=0.5, eps=1e-3, warm_start=True)         # d 小 → 稠密版
# res = least_sp(X, d, zeta=0.1, theta=1e-3)                  # d 大 → 稀疏版
W = res["W"] if "W_dense" not in res else res["W_dense"]

tau = 0.2
for i, j in zip(*np.where(np.abs(W) > tau)):
    print(f"{names[i]:>22s} -> {names[j]:<22s} w={W[i,j]:+.3f}")
```

---

## 4. 怎么解读结果（重要）

- **学到的是"马尔可夫等价类"里的一个 DAG**，不是唯一的真因果图。观测数据 + 线性高斯假设下，边的方向有一部分不可识别（会出现反向边）。
- 边权 `w` 的**绝对值**表示条件依赖强度；阈值 τ 越大图越稀疏。
- 没有 ground truth 时，网页用 τ（默认 0.1）过滤出图；建议你调 τ 看结构如何变化。
- 想做**根因分析**：在网页上选中出错的那个指标，点"根因分析"，会枚举所有指向它的因果路径。

---

## 5. 数据从哪来

- 自己的业务监控指标（推荐：论文 §VI-A 的机票预订场景）
- 用户-物品评分矩阵去均值后（论文 §VI-C 的可解释推荐场景）
- UCI / Kaggle 上的数值型表格数据（本目录两个示例即来自 UCI）
