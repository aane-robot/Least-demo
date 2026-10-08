# Sachs 蛋白信号网络（真实连续数据 + 公认真值图）

- **来源**：Sachs, Perez, Pe'er, Lauffenburger & Nolan (2005), *Science* 308:523-529
  —— 用流式细胞术测量人原代 CD4+ T 细胞中 11 种磷酸化蛋白/磷脂的表达。
- **获取**：Zenodo 记录 [10.5281/zenodo.7681811](https://zenodo.org/records/7681811)（`sachs.zip`）
- **许可**：CC BY 4.0（由 G. Nolan 与 D. Lauffenburger 授权）

## 文件

| 文件 | 内容 |
|---|---|
| `sachs_cd3cd28.csv` | 观测子集（anti-CD3/CD4 刺激条件），**853 × 11**，列名 = 变量名，行 = 单个细胞 |
| `sachs_ground_truth.csv` | 原文献共识图的有向边表，**20 条边**（原始文件，含 `PKA→PIP3` 与 `PIP3→PKA` 这一对反向边） |
| `sachs_ground_truth_dag.csv` | 去环版本，**19 条边**（仅删去 `PKA→PIP3`，破除原图中唯一的 2-环，供需要 DAG 真值的算法使用） |

## 数据是真测的，真值图是"推导出来的"

- **数据为真测**：流式细胞仪逐个细胞实测 11 种磷酸化蛋白/磷脂（原文："a multiparameter flow
  cytometer simultaneously recorded levels of 11 phosphoproteins and phospholipids in
  individual cells"），共 9 种扰动条件。本目录用的是 `cd3cd28.csv`（CD3/CD28 刺激，853 个细胞）。
- **Zenodo 集合里确实混了模拟文件**：`cd3cd28icam2_aktinhib / _g0076 / _psit / _u0126 / _ly`
  这 5 个文件在官方说明中标注为 **"simulated dataset"**，本系统**未使用**它们。
- **真值图不是"大自然直接给的"**：20 条边来自 Sachs 等人对 9 组干预实验的**反推共识**，
  属实验+专家推导结果，因此存在争议（见下）。

## 使用注意（务必转述给观众）

1. **真值版本口径不统一**：原 Zenodo 文件给出 20 条边；而 NOTEARS / GOLEM / DAG-GNN 等论文
   普遍引用"11 节点、17 条边"的共识 DAG（不同整理方对部分边取舍不同）。本目录以原始
   20 条边为准，并额外提供 19 条边的去环版本，**引用数字时请说明是哪一版**。
2. **真值本身存在争议**：该"共识图"来自原作者的多组干预实验，后续研究指出它可能
   ① 不满足因果充分性（存在未测变量）；② 干预未必如所述那样特异；③ 细胞信号网络
   本身含反馈环，真 DAG 假设不一定成立。
3. **线性假设**：LEAST / NOTEARS-linear 假设线性 SEM，而蛋白信号关系是非线性的；
   因此这里的 F1 只能作为**真实数据上的定性/横向对比**，不能与合成数据 F1 直接比较。
4. 文献中常见做法是对数据取 **log 变换**（表达量呈长尾）；网页内置版本默认取
   `log(1+x)`，可在 `demo_datasets.sachs(log=True/False)` 中切换。
