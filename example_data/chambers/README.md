# Causal Chambers — Light Tunnel（真实物理装置数据 + 由物理决定的真值图）

- **来源**：Gamella, Peters & Bühlmann (2025), *Nature Machine Intelligence*,
  "Causal chambers as a real-world physical testbed for AI methodology",
  DOI [10.1038/s42256-024-00964-x](https://doi.org/10.1038/s42256-024-00964-x)
- **装置**：ETH Zürich 的 "Light Tunnel"（可控光源 + 旋转偏振片 + 相机 + 波长强度传感器）
- **许可**：CC BY 4.0（数据文件）
- **获取**：<https://causalchamber.ai/> · `pip install causalchamber`
- **本目录数据取自**：数据集 `lt_walks_v1` 的 `actuators_white` 实验（执行器随机游走，
  属于**真实测量**，非仿真）

## 为什么它是"最硬"的真实数据

真值 DAG **由装置的物理结构直接确定**（哪个传感器读哪个物理量、偏振角如何影响读数），
而不是由专家共识或统计反推得到 —— 这是它区别于 Sachs（共识图）、bnlearn（专家画的网络）
的关键。文献中也把它列为"真值由物理指定"的基准。

## 文件

| 文件 | 内容 |
|---|---|
| `chambers_lt_actuators_white.csv` | **10,000 × 20**（从原始 20,000 行等间隔抽样，保持时间顺序） |
| `chambers_lt_ground_truth.csv` | 真值边表，**39 条边 / 20 节点** |

### 变量（20 个）

- 执行器（光源）：`red`、`green`、`blue`（LED 强度）；`pol_1`、`pol_2`（偏振片角度）；
  `l_11/l_12/l_21/l_22/l_31/l_32`（六个灯的亮度）
- 传感器：`current`（总电流）；`ir_1/vis_1/ir_2/vis_2/ir_3/vis_3`（三个探测器 × 红外/可见）；
  `angle_1`、`angle_2`（偏振片实际角度）

### 真值结构（物理关系）

```
red/green/blue  → current, ir_1, vis_1, ir_2, vis_2, ir_3, vis_3   (各 7 条)
pol_1           → angle_1, ir_3, vis_3
pol_2           → angle_2, ir_3, vis_3
l_11, l_12      → ir_1, vis_1
l_21, l_22      → ir_2, vis_2
l_31, l_32      → ir_3, vis_3
```

## 处理说明（供引用时说明）

1. 原始 standard 配置真值图有 **38 节点 / 57 边**；本实验（`actuators_white`）中有
   **18 个变量恒定未变**（`osr_c`、`v_c`、`diode_*`、`t_*`、`osr_angle_*`、`v_angle_*` 等，
   其中 `v_angle_*` 取哨兵值 −9999）。结构学习无法从常量列恢复边，故**只保留 20 个变动变量**，
   真值取其在原图上的**诱导子图（39 边）**。
2. 数据为真实测量，含传感器噪声；关系是单调但**非线性**（光学衰减、偏振余弦关系），
   因此线性方法（LEAST / NOTEARS-linear）的 F1 会低于其理论上限，属正常现象。
3. 执行器在该实验中被随机设定（random walk），这等价于对根节点做随机化，
   对结构学习是有利的；但**不是**打破边的硬干预。
