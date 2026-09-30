# JumpNicheRewire

当前 `JumpnicheRewire` 文件夹的逐步执行说明见 [EXECUTE.md](EXECUTE.md)。该手册EXECUTE覆盖从 `raw_data` 重建四套模型输入、重建 P22 ATAC 支持标签、七方法 benchmark、新案例拟合及图件生成。

论文列出 scMultiSim 与 P22 两项边排序比较（各七种方法），以及小鼠肾脏、人乳腺癌两张空间案例图。

- `results/` 保存正文对应的逐边评分、支持标签、逐点绘图值和来源指标；
- `model_code/` 保存模型、基准方法、配置与评价代码；
- `model_inputs/` 保存拟合使用的处理后输入；
- `model_fits/` 保存 P22 和两张切片的缓存拟合。
- `raw_data/` 包含 scMultiSim 作者对象、P22 RNA/空间文件及两张案例的 10x 原始矩阵与空间文件。
- `reference_data/` 保存 P22 标签筛选所用的 motif 表。
- `MANIFEST_SHA256.json` 逐文件记录大小和 SHA-256，可校验解压后的文件。

P22 配对 ATAC 原始片段文件约 1.53 GB，未放入压缩包；其来源见第 3 节。包内已提供从该文件计算出的 `results/p22/support_edges.csv` 和完整评价候选集合，因此第 1 节的论文结果复算不需要重新下载 ATAC。若要从原始片段重新构建支持标签，则需要下载 ATAC 文件及第 3 节的参考资源。

## 1. 直接重算正文数值

在解压后的目录运行：

```powershell
python -m pip install numpy==2.2.6 pandas==2.3.3 scipy==1.15.2 scikit-learn==1.7.2 pyarrow==24.0.0
python reproduce_reported_results.py
python reproduce_case_from_fits.py
```

第一个脚本从 `results/scmultisim/scores/` 和 `results/p22/scores/` 的七个逐边 CSV 读取标签与分数，重新计算 AP、AUROC、早期精度和前 10% 候选的 F1、F0.1。并列边按随机排列的期望命中数处理。它逐项核对原始指标文件（AP、AUROC、早期精度；P22 另核对两项 F 分数），以及两张案例图使用的 spot 数、增量分位差和平均绝对增量。第二个脚本从保存的 LR 暴露和交互系数重新计算两张切片的全部逐点边增量，并核对论文所列前三条边及主图边的 LR 贡献。输出 `PASS` 表示保存的拟合与正文数值对应。这两步**不重新训练模型**。

scMultiSim 的两项 F 分数采用前 `ceil(624×10%)=63` 条边计算；原始实验指标文件中的 `F1_at_10` 指前 10 条边，不能用它代替正文的前 10% 指标。P22 相应截断为前 84 条边。scMultiSim 是同一公开模拟数据上的参数开发与评价；这些数值是该数据上的排序结果。

## 2. 重绘图与编译论文

解压后在复现包根目录运行：

```powershell
python -m venv .venv-plots
.\.venv-plots\Scripts\python.exe -m pip install numpy==1.26.4 pandas==2.1.4 matplotlib==3.8.4
.\.venv-plots\Scripts\python.exe paper/redraw_manuscript_figures.py
```

该脚本读取每个案例的 `selected_edge_spatial_values.csv` 和 `expression_umap.csv`，生成论文中的两张矢量 PDF，并保存源文件与图的 SHA-256。图内只有 A–D 子图标记，图注写在 LaTeX 主稿中。编译论文须安装 XeLaTeX、Biber、`ctex`、`biblatex`、TikZ 等宏包：

```powershell
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error main_dmaa_revision_zh.tex
```

## 3. 从公开原始数据重新准备输入与拟合

公开来源：

| 数据 | 原始地址 | 本文用途 |
|---|---|---|
| scMultiSim 作者数据 `main_tree1` | https://doi.org/10.5281/zenodo.13119261 | 模拟表达、坐标和生成网络真值 |
| P22 小鼠脑 RNA/空间 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6753043 | TF--gene 网络输入 |
| P22 配对 ATAC | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6758285 | 目标基因启动子可及性 |
| mm10 cisTarget 基因级排名 | https://resources.aertslab.org/cistarget/databases/mus_musculus/mm10/refseq_r80/mc_v10_clust/gene_based/ | TF motif 支持标签 |
| CellChatDB.mouse | https://github.com/jinworks/CellChat | 候选 L--R 关系 |
| 10x 成年小鼠肾脏 FFPE | https://www.10xgenomics.com/datasets/adult-mouse-kidney-ffpe-1-standard-1-3-0 | 肾脏空间案例 |
| 10x 人乳腺癌 FFPE | https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0 | 乳腺癌空间案例 |

`model_inputs/` 中包含本研究实际使用的处理后表达、坐标、候选分子及先验表。`model_code/experiment_parameters/` 中的参数快照记录原实验所用设置与源文件哈希；其中的 Windows 绝对路径是原运行环境记录，换机重新拟合时需改成当地路径。`model_code/configs/` 是可编辑配置源码。主入口 `model_code/main_jumpniche.py` 从 `configs/active.py` 读取 profile 和 action，需在重新拟合前明确选定对应数据及输出目录，再执行 `python main_jumpniche.py`。

本包保存了 P22 与两张空间切片的缓存拟合，便于逐层核对；scMultiSim 保存了七种已报告方法的固定预测分数和原输入。重新运行全部方法还需要原方法软件、对应运行环境与计算资源。原实验的基准方法配置分散在 `model_code/configs/`，快照记录了最终参数。**固定预测评价**可用第 1 节单独复现；**重新拟合**须完成上述环境和路径配置。论文中的空间增量是模型内条件关联，不表示通讯机制的因果效应。

