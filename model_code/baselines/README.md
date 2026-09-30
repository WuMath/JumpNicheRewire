# 独立基线与统一评价：执行教程

更新：2026-09-07。适用于 `C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct`。

指标补充：现在支持 F1、预定 top-k 下的 Precision/Recall，以及每个任务×指标独立 Excel/CSV。
公式、文献依据、配置方法和表格布局见 [METRICS_AND_TABLES.md](METRICS_AND_TABLES.md)。本次仅增加评价功能，不要求重跑/重新冻结模型。

本次只新增 `baselines/` 和 `src/independent_baselines/`，不修改 JumpNiche 模型、原来的两个入口、活动配置或已有结果。下面的命令由你按需执行；本次没有替你安装九个环境或运行九种正式实验。

## 1. 先决定比较什么

不要把所有方法强行塞进同一张“总分排行榜”。GRN 是 TF→靶基因，CCI 是细胞间通信，LR→gene 是一对配体–受体对下游基因的贡献。它们的输出单位、所需参考资源不同。

| 方法 / 入口 | 此接口导出的任务 | 优先使用的数据 | 重要限制 |
|---|---|---|---|
| GENIE3 / `run_genie3.py` | 全局 TF→target | SERGIO、核心正式模拟；Visium探索 | 官方 Arboreto 实现；没有细胞特异网络、没有 CCI |
| GRNBoost2 / `run_grnboost2.py` | 全局 TF→target | 同上 | 与 GENIE3 共用环境，但独立运行、独立结果 |
| CeSpGRN / `run_cespgrn.py` | 细胞特异基因网络的 TF 投影 | SERGIO、核心正式模拟 | 偏相关网络本质无向，不把 TF 投影当因果方向 |
| SpaGRN / `run_spagrn.py` | motif 剪枝后的 regulon TF→target | 小鼠 Visium | 需要真实基因名、匹配 cisTarget；不能给 G0/TF1 随便映射 motif |
| LIANA / `run_liana.py` | 细胞类型对×LR | sCCIgen；Visium探索 | 默认为相同输入 LR 资源；不是独立运行的原版 CellChat/CellPhoneDB |
| CellPhoneDB / `run_cellphonedb.py` | 细胞类型对×简单有方向 LR | 人类 sCCIgen | 标准数据库是人类；不能把鼠基因改大写来代替同源映射 |
| CellChat / `run_cellchat.R` | 细胞类型对×LR | sCCIgen、Visium探索 | 原版 R 包，使用原生人/鼠 CellChatDB；资源与 matched-LR 不同 |
| iMIRACLE-LRComponent / `run_imiracle.py` | LR 视图表达贡献的绝对值 | 核心模拟的 LR→gene 探索 | 是明确命名的适配变体，不冒充论文最终 regulatory score；不能据此声称复现全文 |
| CLARIFY / `run_clarify.py` | GAE 细胞网络和细胞内基因网络 | 满足资源与内存条件的小鼠数据 | 本版源码的人类分支不完整，不能直接跑当前 SERGIO/sCCIgen；全 Visium 可能内存不足 |

**建议实际顺序：** 先 SERGIO 的 JumpNiche + GENIE3 + GRNBoost2 + CeSpGRN；再核心正式模拟的同一 GRN 比较；再 sCCIgen 的 CCI 比较；最后 Visium 伪真值和案例。扰动协议未验收，暂不混入主结论。MERFISH 已排除。

## 2. 文件在哪里，各自做什么

```text
reconstruct/
  main_jumpniche.py                    原有 JumpNiche 主入口，不动
  preprocess_jumpniche.py              原有数据处理入口，不动
  baselines/
    README.md                          本教程
    baseline_settings/
      datasets.py                      所有 Python 基线共用的数据/输入/输出配置类
      evaluation.py                    选择要评价的方法、已有 JumpNiche 输出路径、统计设置
      genie3.py ... clarify.py          各方法的 MethodConfig 参数类
      cellchat.R                       CellChat 独立 R6 配置类
    environments/*.yml                 各方法独立 Conda 环境
    run_*.py / run_cellchat.R           每个方法一个运行文件
    install_cellchat.R                  仅手动执行时安装 R 依赖
    evaluate_comparison.py             只读结果与真值，统一评价，不训练模型
    tests/                             小数据接口/评价回归测试，不是正式 benchmark
  src/independent_baselines/
    io.py                              白名单输入读取、基因/细胞对齐、分批 Parquet
    runtime.py                         版本检查、参数冻结、逐数据集独立子进程、状态/内存
    adapters.py                        调用各官方库并转成统一输出
    comparison.py                      真值对齐、AP/排名、统计、Excel、绘图
```

基线只导入 `independent_baselines`，不导入 `jump_niche_rewire`。官方源码保存在 `reconstruct/vendor/external_methods/`，这次没有把第三方源码复制进你的核心模型。基线与 JumpNiche 之间只交换输入和结果文件。

所有实验参数都在配置类中手动改。不要给运行文件添加 `--dataset`、`--lambda` 等命令行参数。下面安装命令的 `-f`、版本号是安装工具自身的选项，不是模型实验参数。

## 3. 最快开始：在已有五个 SERGIO 上比较

### 3.1 进入目录，创建树模型环境

```powershell
cd C:\Users\Wuyan\Desktop\GRN\Futher\GRN2\reconstruct
conda env create -f .\baselines\environments\trees.yml
conda activate jnr-trees
python -c "import sys, numpy, scipy, arboreto; print(sys.executable); print(numpy.__version__)"
python -m pip check
```

不要在 `(base)` 下运行实验；也不要把所有老版本依赖混装进 `jumpniche`。环境文件已存在时，直接激活即可，不必重复创建。出现下载字节不足是网络/缓存问题，不是模型错误；先处理下载，不能跳过报错继续跑。

### 3.2 设置输入（目前默认就是 SERGIO）

打开 `baselines/baseline_settings/datasets.py`，保持：

```python
suite: str = "sergio"
dataset_ids: tuple[str, ...] | None = None
expression_scale: str = "raw_counts"
gene_panel_file: Path | None = None
```

`None` 不是不选数据，而是用类中 `sergio_ids` 的五个明确编号：

```text
sergio_grn_100g_seed_20260902
sergio_grn_100g_seed_20260903
sergio_grn_100g_seed_20260904
sergio_grn_100g_seed_20260905
sergio_grn_100g_seed_20260906
```

注册表只负责把编号映射到已准备好的数据文件夹。它在：
`C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/processed/evidence_benchmark/03_registries/evidence_registry.csv`。

无需再次下载或模拟这五份数据。每份 450 cells × 100 genes；它们共享一个网络拓扑，不是五种独立网络。SERGIO 没有真实空间通信真值，不能用它证明 CCI 优势。

### 3.3 每个方法均执行三次：check → freeze → run

以 GENIE3 为例，打开 `baselines/baseline_settings/genie3.py`。

1. `mode: str = "check"`，运行下面命令：核对环境、输入形状，不训练、不读取真值。
2. `mode: str = "freeze"`，再次运行：把当前参数、代码版本、依赖、资源哈希写入冻结文件。
3. `mode: str = "run"`，再次运行：逐个数据集训练并保存结果。

三次都是同一个命令：

```powershell
python .\baselines\run_genie3.py
```

然后按相同步骤修改 `grnboost2.py` 的类，运行：

```powershell
python .\baselines\run_grnboost2.py
```

两个方法不会互相调用。结果分别写入：

```text
reconstruct/data/result/independent_comparison_v1/sergio/
  00_protocols/GENIE3.json
  00_protocols/GRNBoost2.json
  02_methods/GENIE3/sergio_grn_100g_seed_20260902/...
  02_methods/GRNBoost2/sergio_grn_100g_seed_20260902/...
```

冻结并不等于“已经找到最优参数”。这里默认是预先固定的一套参数，不是自动超参数搜索器。若要调参，只能在 train/validation 比较候选，给候选设置不同 `output_root`，保存选择记录后把选中设置带入正式协议；不得看 test 后修改参数。

### 3.4 再加入 CeSpGRN

见第 4 节安装与下载。安装后也是 `check → freeze → run`，命令 `python .\baselines\run_cespgrn.py`。

SERGIO 请保持 `kernel_input="expression_pca"`：用表达的低维表示构造相似性，不把适配器产生的布局冒充真实组织坐标。核心正式模拟如果要利用真实模拟坐标，可以在该 suite 冻结前明确改为 `"spatial"`；不要跑完 test 再择优选空间/表达核。

### 3.5 单独执行评价（不再训练）

打开 `baselines/baseline_settings/evaluation.py`：

```python
methods: tuple[str, ...] = ("JumpNicheRewire", "GENIE3", "GRNBoost2", "CeSpGRN")
analysis_name: str = "comparison_v1"
```

`our_sergio_root` 已指向你在 reconstruct 中完成的 SERGIO 结果。你如果只完成了前两个基线，可临时只列这两个和 JumpNiche，但必须把结论称为“当前已完成方法比较”。

```powershell
conda activate jumpniche
python .\baselines\evaluate_comparison.py
```

输出：
`C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/independent_comparison_v1/sergio/03_evaluation/comparison_v1/`。

追加方法后，不用重跑已有模型；把 `analysis_name` 改为 `comparison_v2`，再次执行评价。程序拒绝覆盖旧评价目录。评价环境缺包时，可另建 `environments/evaluation.yml`，激活 `jnr-evaluation` 使用。

## 4. 各方法：下载、环境、调用和输出分数

所有以下命令默认工作目录都是 `reconstruct`。Git 下载方法源码，不下载你的 benchmark 数据。若网络不便，可以在另一台能联网的电脑克隆后连同 `.git` 文件夹整体拷贝；运行时要校验提交号，单纯 ZIP 解压没有 `.git` 会被拒绝。不要删除版本校验来绕过它。

### 4.1 GENIE3 / GRNBoost2

环境：`trees.yml`，Python 3.10；官方实现通过 `arboreto==0.1.6` 安装，无须单独克隆。代码调用点是 `adapters.tree()` 中 `arboreto.algo.genie3/grnboost2`。输入为 cells×genes 的 log-normalized 表达和表达中的 TF 清单。输出 feature importance 越大越强；两者不提供可信的激活/抑制符号。

入口与参数：`run_genie3.py` / `genie3.py`，`run_grnboost2.py` / `grnboost2.py`。主要改 `n_jobs`、`seed`；配置没有穷举库里每个内部常量。当前设置固定为官方算法默认树参数加显式随机种子。

官方：[Arboreto 使用说明](https://arboreto.readthedocs.io/en/latest/userguide.html)。

### 4.2 CeSpGRN

```powershell
git clone https://github.com/PeterZZQ/CeSpGRN.git .\vendor\external_methods\CeSpGRN
git -C .\vendor\external_methods\CeSpGRN checkout 2fb222f8a26edf1bdae14a95b1491543f0aaa4e8
conda env create -f .\baselines\environments\cespgrn.yml
conda activate jnr-cespgrn
python -c "import torch, numpy, scipy; print(torch.__version__); print(torch.eig(torch.eye(2)))"
python -m pip check
python .\baselines\run_cespgrn.py
```

最后一条按 `cespgrn.py` 中 mode 分别执行三次。环境采用 Python 3.9、PyTorch 1.11 的旧接口栈，避免直接给原始源码装新 Torch；仍需在你的目标机器完成上述导入测试，不能保证所有平台二进制依赖都已验证。

调用：`kernel.calc_kernel_neigh` → `g_admm.est_cov` → `G_admm_minibatch.train`。保存 `official_graphs.npy` 后按细胞分批导出 `parquet/grn_edges.parquet`。配置中的 `lamb` 是稀疏惩罚，`neighbors/bandwidth` 控制相似性平滑，`max_iters` 为优化上限。

该方法产生 cell×gene×gene 张量，有 O(cells×genes²) 内存成本；8 GiB 是保守的预计工作量保护阈值，不是实际内存硬限制。不能把 `memory_limit_gib` 盲目改得很大就认为解决了内存。官方：[CeSpGRN 仓库](https://github.com/PeterZZQ/CeSpGRN)。

### 4.3 SpaGRN（优先用于 Visium）

```powershell
git clone https://github.com/BGI-Qingdao/SpaGRN.git .\vendor\external_methods\SpaGRN
git -C .\vendor\external_methods\SpaGRN checkout 61690ca69a766f14c580de5dd47d4bc7b603813a
conda env create -f .\baselines\environments\spagrn.yml
conda activate jnr-spagrn
python -m pip install --no-deps -e .\vendor\external_methods\SpaGRN
python -c "from spagrn.regulatory_network import InferRegulatoryNetwork; import pyarrow; print(pyarrow.__version__)"
python -m pip check
python .\baselines\run_spagrn.py
```

必须做 `-e` 安装这一步：环境先装依赖，随后固定到核对过的源码 API。此版本自身限制 Python 3.8。老科学计算依赖较多，建议 WSL2/Linux；本次没有替你完成整个环境求解和真实拟合。

先把 `datasets.py` 改成 `suite="visium", dataset_ids=None`。`spagrn.py` 已填写你现有的小鼠资源，无须重下：

- `data/raw/ModelData/mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather`
- `data/raw/ModelData/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl.txt`

是父项目 `GRN2/data/raw/ModelData`，不是 reconstruct 下面另造一份。调用 `InferRegulatoryNetwork.infer()`，保存官方 AnnData，导出 motif 剪枝后 regulon 的基因权重。当前接口不是拿未剪枝相关系数冒充完整 SpaGRN。

如改成人类 sCCIgen，必须在类中换真实的人类 cisTarget、motif 注释与 `reference_species="human"`。但 sCCIgen 当前标签用于通信，不是 TF-target 真值，单有 SpaGRN GRN 不能做这项 CCI 评价。官方：[SpaGRN 源码和版本约束](https://github.com/BGI-Qingdao/SpaGRN)。

### 4.4 LIANA（sCCIgen 的 matched-resource 比较）

```powershell
conda env create -f .\baselines\environments\liana.yml
conda activate jnr-liana
python -c "import liana, anndata; print(liana.__version__)"
python -m pip check
python .\baselines\run_liana.py
```

数据类：`suite="sccigen", dataset_ids=None`。方法类 `liana.py`：保持 `resource_mode="matched"`、`n_perms=1000`，按三次 mode 运行。这里调用 `li.mt.rank_aggregate`，输出 `score = 1 - magnitude_rank`，越大排名越靠前。它调用 LIANA 自身的聚合实现，不等于独立运行了原版 CellChat 或 CellPhoneDB。

matched 模式使用现有输入 `lr_pairs.csv` 中表达的简单基因对，不根据评价真值临时添加正例。native 模式使用 `resource_name` 指定的内置资源；若使用鼠数据，要选择匹配物种的真实资源或保持 matched，不要照搬人类 consensus。

sCCIgen 植入的跨细胞基因对不一定全部是真实数据库中的 LR。原生数据库方法的低覆盖应报告为资源/任务限制，不能简单宣称算法差。官方：[LIANA API](https://liana-py.readthedocs.io/en/stable/api.html)。

### 4.5 CellPhoneDB（人类数据；需手动下载资源）

1. 浏览器打开 [固定版本 v5.0.0 数据库文件](https://github.com/ventolab/cellphonedb-data/blob/v5.0.0/cellphonedb.zip)。
2. 点击 **Download raw file** 下载，保存为 `cellphonedb.zip`，不要保存网页 HTML，也不要解压后改名伪装 ZIP。
3. 用资源管理器创建文件夹，并保存到：
   `C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/raw/baseline_resources/cellphonedb.zip`。
4. 打开 `baseline_settings/cellphonedb.py` 核对 `database` 路径。

```powershell
conda env create -f .\baselines\environments\cellphonedb.yml
conda activate jnr-cellphonedb
python -c "from cellphonedb.src.core.methods import cpdb_statistical_analysis_method; print('CellPhoneDB import OK')"
python -m pip check
python .\baselines\run_cellphonedb.py
```

`datasets.py` 使用 `sccigen`。程序自动把共用表达转换成 log-normalized H5AD，把 cell_type 写成 meta.tsv，再调用官方统计方法；不需要另外手工转矩阵。

保留完整官方结果，同时统一导出 `interaction_scores`。根据 receptor_a/receptor_b 确定 ligand→receptor 方向；不能确认方向、缺失单基因标识的复合物，不随意拆分成新的 LR。原始复合物仍在官方输出中，本接口只比较简单有方向 LR 子集。标准流程不支持直接输入小鼠。官方：[CellPhoneDB 说明](https://github.com/ventolab/CellphoneDB)、[输出解释](https://cellphonedb.readthedocs.io/en/stable/RESULTS-DOCUMENTATION.html)。

### 4.6 CellChat（独立原版 R 方法）

Windows 方案：安装/使用已有 R 4.3 或更新兼容版本，源码编译需要与 R 对应的 Rtools。你机器上检测到 R 4.4.1，可先检查 `Rscript --version`；本次只做语法解析，没有在该 R 库中安装 CellChat。

```powershell
Rscript .\baselines\install_cellchat.R
Rscript .\baselines\run_cellchat.R
```

第一条是安装文件，不是每次都执行；其中安装位置由你当前 R library 决定，先检查是否可写。固定 CellChat v2.1.2 提交，`remotes` 不自动升级已有全部依赖。Linux/WSL 可先 `conda env create -f baselines/environments/cellchat.yml`，激活 `jnr-cellchat` 后再执行安装 R 文件。不要在 Windows 上强行使用只提供 Linux 包的 R Conda 组合。

参数单独位于 `baseline_settings/cellchat.R` 的 `CellChatConfig` 类，**不会跟随 Python datasets.py 自动改变**。同样把 `mode` 依次改为 `check`、`freeze`、`run`。

默认 sCCIgen 配置：`suite="sccigen"`、`species="human"`、编号 `sccigen_human_breast_example1`。Visium 时同时改为：

```r
suite = "visium",
species = "mouse",
dataset_ids = c("real_pseudo__visium_ffpe_mouse_brain"),
```

调用 `createCellChat` → 表达/相互作用筛选 → `computeCommunProb` → `filterCommunication`。在转表前保存 `official_cellchat.rds`。全量统一结果是 `cci_edges.csv`，Excel 仅预览；统一评价器直接读 CSV，不受 Excel 行数限制。这里使用内置 CellChatDB，不是 matched-resource，论文必须注明资源区别。R 入口未可靠测量 peak RSS，状态中为 null，不能报成 0 MB。官方：[CellChat](https://github.com/jinworks/CellChat)。

### 4.7 iMIRACLE：先确认输出含义，再决定是否纳入主表

```powershell
git clone https://github.com/aicb-ZhangLabs/iMiracle.git .\vendor\external_methods\iMiracle
git -C .\vendor\external_methods\iMiracle checkout 6f8c554a746c23d9df227a20d97288f3bf49dee7
conda env create -f .\baselines\environments\imiracle.yml
conda activate jnr-imiracle
python -m pip install --only-binary=:all: --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
python -c "import torch, torch_geometric, torch_sparse; print(torch.__version__)"
python -m pip check
python .\baselines\run_imiracle.py
```

PyG wheel 必须与 Torch/CPU 对应；若当前平台没有匹配 wheel，命令会停止，不默默进行长时间源码编译。若你改用 CUDA，按 [PyG 对应版本安装文档](https://pytorch-geometric.readthedocs.io/en/2.2.0/notes/installation.html) 配齐同一版本后重新冻结，不能拿 CPU wheel 混用。

对应调用：`MultiViewGraph` → `iMiracle` → `iMiracleTrainer(mode="ranking")` → `train_model_for_ranking()`。输入 metadata cell_type 的 one-hot，表达、坐标、现有 LR 候选；不读取 truth。保存权重及每个 LR 的表达贡献数组，逐批导出 `receiver_id, lr_id, target, score`。

**本接口名称固定为 `iMIRACLE-LRComponent`**。评分是 `abs(saved_X_lr_GRN)`，属于可追踪的表达贡献适配；不是已确认复现论文的最终 regulatory ranking，更不是因果效应。当前 sCCIgen 只有 CCI 标签，没有对应 LR→gene 标签，会记录“不兼容任务”，不会把它转成伪造的 CCI 分数。

多 LR 视图的稠密矩阵可能非常大；内存保护会先停止。若需缩小面板，应在验证阶段预先定义共同数据子集，对所有方法和 JumpNiche 一起处理、重新注册和运行，不能只给一个方法随意截断或从 test 真值选择 LR。当前框架没有自动实现论文全部下游分析。官方：[iMiracle](https://github.com/aicb-ZhangLabs/iMiracle)。

### 4.8 CLARIFY：当前数据存在明确兼容障碍

提供独立调用文件，**不意味着当前所有数据都能直接运行**。核对固定提交的 `select_LRgenes` 后发现：实现了 `lr_database=0` 小鼠与 `2` 自带 scMultiSim；文档中的 `1` 人类分支未实现。`2` 会读仓库自带 `cci_gt.csv`，不能拿这个分支适配你的 SERGIO/sCCIgen、也不能让训练偷读 benchmark 真值。本接口明确拒绝这两条不合格路径。

```powershell
git clone https://github.com/MihirBafna/clarify.git .\vendor\external_methods\clarify
git -C .\vendor\external_methods\clarify checkout de5dccfdfc9c22d95c29d2a3e5ffdb05adc5de03
conda env create -f .\baselines\environments\clarify.yml
conda activate jnr-clarify
python -m pip install --only-binary=:all: --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
python -c "import torch, torch_geometric, wandb; print(torch.__version__)"
python -m pip check
python .\baselines\run_clarify.py
```

环境选择 Torch 1.11 与旧初始 GRN 依赖兼容，**与官方 installation.md 的 1.12.1 栈不同**；这是待实机验证的兼容方案，不宣称原环境已完整复现。安装 `wandb` 仅满足官方 import；调用代码没有登录、上传或启用远程实验记录。

小鼠 Visium 的 2,264 cells、45 genes/cell，gene-level 全矩阵约 `(2264×45)²` 元素；单个 float32 就约 39 GiB，还有梯度、mask、临时矩阵，不能在普通笔记本直接忽略内存保护。当前默认会在估计超限时停止，保留审计。要做它，应先确定所有方法共用的小型案例面板/细胞子集或申请足够内存超算，再增加独立注册表和匹配的 JumpNiche 结果；这不是简单把上限调大。

训练日志里的 `CLARIFY Test AP` 是其内部 KNN 图边重建验证，不是这里对生物学真值的正式 AP。统一比较仅用导出的预测表与独立 truth。官方：[CLARIFY](https://github.com/MihirBafna/clarify)。

## 5. 换用核心正式模拟，不重跑已有 JumpNiche test

在 `datasets.py`：

```python
suite: str = "core_final"
dataset_ids: tuple[str, ...] | None = None
split: str = "validation"
unlock_test: bool = False
```

当前核心注册表是 `formal_benchmark_core_final_v1/registries/final/dataset_registry.csv`。四个场景 `clean_500 / dropout_500 / low_prior_500 / boundary_500`；20 train、12 validation、20 test 是数据集数量，不是细胞数。程序按 split 选取，检查 `benchmark_profile=final`，不混 pilot。

先在 validation 确定方法参数。若采用固定默认参数，写明“预先固定默认参数”，不要声称做了公平等预算大搜索。确认后在各方法类设置 `mode="freeze"` 执行，保存本 suite 的协议。再把数据类改为：

```python
split: str = "test"
unlock_test: bool = True
```

方法类 `mode="run"`，分别执行三个 GRN 方法入口。推理阶段白名单没有 truth 文件；正式评价才读标签。程序禁止在 split=test 下新建冻结协议，且参数/代码/资源不匹配会停止。

评价类的 `our_formal_root` 指向旧工程已完成的：
`C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/result/benchmark/core_final_v1/final/02_runs`。
评价器读取其中 `frozen_parameters.json` 的 `parameter_id`，自动定位 `jumpniche/<parameter_id>/<dataset_id>`，不会重新调用 JumpNiche。

仍然只运行：

```powershell
conda activate jumpniche
python .\baselines\evaluate_comparison.py
```

新结果在 `independent_comparison_v1/core_final/03_evaluation/<analysis_name>/`。不要把 validation 和 test 的评价写成同名目录；建议分别用 `validation_v1` 和 `test_v1`。

## 6. sCCIgen、Visium、扰动分别怎么评价

### sCCIgen：评价细胞类型对×基因对

`datasets.py`：`suite="sccigen", dataset_ids=None`，输入是已处理的 `sccigen_human_breast_example1`（3998×4751），不是重新运行模拟器。逐个运行 LIANA、CellPhoneDB、CellChat；R 配置需要独立同步。

评价类方法名示例：

```python
methods = ("JumpNicheRewire", "LIANA", "CellPhoneDB", "CellChat")
```

真值 `truth_sccigen_cci.csv` 用 `sender_cell_type, receiver_cell_type, ligand, receptor` 匹配。JumpNiche 的细胞对先用 metadata 映射为类型对，再对同一键取最大 score；基线直接产生类型对。所有方法在同一个候选全集评分，缺失预测补 0，同时报告预测覆盖。

当前只有一份 sCCIgen example，不能对几百万条边做“几百万次独立重复”的显著性检验。若某方法的内置 LR 资源不含植入基因对，结果反映资源覆盖差异，另在报告注明。

### Visium：数据库伪真值 + 生物学案例，不是严格准确率

`datasets.py`：`suite="visium", dataset_ids=None`。已准备数据 `real_pseudo__visium_ffpe_mouse_brain`（2264×2000）。GRN 方法可比较已知 TF-target 检索；LIANA、CellChat 可比较已知 LR 检索。SpaGRN 需用同物种 cisTarget。

`evaluation.py` 中 `our_visium_root` 是预期路径，当前 reconstruct 目录未发现该 suite 已完成结果；必须核对你实际 JumpNiche 运行输出，必要时手动改成真正的 `02_runs/real_pseudotruth`。路径不存在会审计为 missing，不会伪造结果。

伪真值标签 0 是“未被数据库标注”，不等于实验证明不相互作用。因此指标带 `PU_` 前缀（positive-unlabeled，正例–未标注）；不能解释为真实 precision/recall 或因果准确率。`real_prior_independence_verified=False` 默认保留。若用于输入的 TF-target/LR 先验和评价标签重叠，需要先建立按来源独立的留出标签，不能只把开关改 True。

当前代码不自动修复已有先验与标签的重叠，也不把探索性结果升级成独立泛化证据。

### 扰动：暂时保留验收门禁

`suite="perturbation"` 默认被 `perturbation_protocol_validated=False` 拦截。既有处理尚需核验控制组限定的基因选择、guide→target、非靶向对照、生物学重复和差异表达标签。先读 `reconstruct/docs/KNOWN_LIMITATIONS.md`（如其实际保存于 reports 中，以项目现有同名记录为准）。

验收完成后，显式填写正确 `perturbation_registry` 和 `dataset_ids`，再设置标志、冻结方法。可复用 GENIE3/GRNBoost2 的 GRN 输入输出与 `truth_perturbation_grn.csv`；但本次**没有完成干预效应方向评价、guide 映射修复或独立 activity 升级**。不能为了凑完整 benchmark 直接放行。非空间 K562 不应补造坐标参加空间通信比较。

## 7. 输入是什么，如何确保公平

拟合白名单（每个 dataset_dir 下）：

| 文件 | 格式与示例 | 使用目的 |
|---|---|---|
| `baseline_input/expression_cells_by_genes.csv`，不存在则 `expression.csv` | 第一列 cell ID，后面每列一个基因；行 cells、列 genes | 共用表达；`raw_counts` 才做归一化+log1p |
| `metadata.csv` | 至少 `cell_id,cell_type`；可有 region/sample_id | 精确细胞 ID 对齐，通信类型分组；不能编造 cell_type |
| `coordinates.csv` | 第一列 cell ID，x/y 数值 | 空间方法；不会按行号假装匹配 |
| `tf_list.csv` | 第一列 TF 基因名 | TF 候选与表达基因求交；不根据 test 真值挑选 |
| `lr_pairs.csv` | `ligand,receptor`，可含 `lr_id` | LR 先验；复合物不通过拆字符串伪造基因对 |
| 可选共同 `gene_panel_file` | 一列 `gene` | train/validation 预定并冻结的面板，不自动取 test 中效果最好的基因 |

数字字符串 ID 要保留原样；程序验证唯一性与非有限数。默认输入不是 z-score，负值会拒绝。已经 log 的输入应明确设置 `expression_scale="log1p_normalized"`，否则会重复归一化。SpaGRN `danb` 模型需要 counts。

所有拟合程序不读取 `truth_*.csv`。评价程序才读取注册表指定的 `truth_dir`。不要把下载的 TF-target 数据库直接复制成模型输入先验与测试真值，然后称为独立测试。

同一 GRN 比较：候选边集合由共同真值表确定；预测不到的边补 0；重复 TF-target 取最大 score。全局比较时，细胞特异预测跨 cell 取最大。核心模拟另算 `GRN_cell_common_universe`：全局方法将同一 TF-target 分数广播到各 cell，细胞方法按 cell 对齐。广播不是让全局方法“获得了细胞特异能力”，只是给两类方法同一考卷。

原生数据库与 matched-resource 要分开解释；带 motif/通路先验的方法与无先验方法比较属于“端到端系统比较”，不能直接把所有差距归因于优化算法。若限制共同面板，JumpNiche 也必须用相同面板重新跑；不能把其旧全基因结果拿来宣称同输入对照。

## 8. 输出有哪些，如何看

每份数据一个目录，不在一个大进程中连续挤占所有内存：

```text
<output_root>/<suite>/02_methods/<method>/<dataset_id>/
  run_status.json            completed/failed/interrupted，耗时、峰值 RSS、报错
  effective_config.json      本次实际参数和输入位置（Python 方法）
  console.log                官方方法日志
  official_*                 官方原始网络/权重/表格等，各方法不同
  parquet/grn_edges.parquet  全量 GRN，若该方法支持
  parquet/cci_edges.parquet  全量 CCI，若该方法支持
  parquet/lr_gene.parquet    全量 LR→gene，若该方法支持
  <method>_results.xlsx      只供查看，默认最多 10000 行预览 + manifest
```

CellChat 全表为同目录 `cci_edges.csv`。评价器只读完整 Parquet/CSV，绝不拿 Excel 预览充当全部预测。

统一列（越大越强的非负有限 score）：

- GRN：`tf,target,score`；细胞网络再有 `cell_id`。当前无向/特征重要性方法 `signed_score` 为空，不能当调控方向。
- CCI：`sender_id,receiver_id,lr_id,score`，或 `sender_cell_type,receiver_cell_type,ligand,receptor,lr_id,score`。
- LR→gene：`receiver_id,lr_id,target,score`。

没有输出某个任务代表不支持，不生成一张 0 分表。官方方法没有可比的统一收敛定义时 `convergence_passed=null`，不是伪造 100%；JumpNiche 保留原收敛验收。方法返回成功也不等于所有数值收敛已严格证明。

Python 的 RSS 统计会抽样本进程和子进程内存，单位 bytes，转 GiB 除以 `1024**3`；它不是精确 CPU 时间或总 GPU 显存，原生崩溃前也可能来不及落盘。比较耗时需相同硬件、并发设置并排除环境安装时间。R 的 RSS 暂缺。

## 9. 评价指标和统计如何解释

统一评价输出：

| 文件 | 含义 |
|---|---|
| `run_audit.csv` | 每方法×数据集是否有完整、可评价结果；先看它 |
| `dataset_metrics.csv` | 每份数据的任务、指标、分数、候选数、预测覆盖率 |
| `summary.csv` | 跨数据集均值与 bootstrap CI；注意成功数据集数量 |
| `paired_tests.csv` | 与 JumpNiche 在共同完成数据集上的配对差、区间、Wilcoxon 与 Holm 校正 |
| `comparison.xlsx` | 上述四张表的方便阅读版；不是海量预测表 |
| `*_paired.png/pdf` | 一条线对应同一数据集，展示方法间配对分数 |
| `evaluation_provenance.json` | 配置、结果/真值文件哈希，供追溯 |
| `REPORT.md` | 本次完成情况及证据限制 |

主要指标：

- **AP (Average Precision，平均精确率)**：沿分数阈值排序，按召回增量加权 precision；越大越好。代码调用 sklearn 的 AP，不是把 PR 曲线用梯形积分得到的另一个数。所有相同分数整体进入同一阈值，不按输入行顺序决定谁先。
- **AUROC**：随机正边得分高于随机负边的概率，同分记半次；极稀疏网络不能只看它，需同时看 AP。
- **Precision@k / Recall@k**：看最高分 k 条里期望多少真边。若 k=2，第一条肯定为真，接下来三条同分中只有一条真，只剩一个位置，则期望真边数 `1+1×(1/3)`，precision 为 `2/3`，不挑对某方法有利的行。
- **Early precision**：k 取候选全集中正边数。无正例的 AP/recall 返回缺失而不是 0。
- **Positive prevalence / AP_over_prevalence**：正例占比及其归一化参考；是便于看任务稀疏度的辅助量，不把有限样本中每次随机排序的 AP 都说成严格等于占比。
- **NDCG_linear / Spearman / Kendall tau-b**：仅用于匹配的 LR→gene 连续真值排名。NDCG 用线性贡献作增益，同分组取期望折扣得分；秩相关处理并列，常量分数返回缺失。不是二分类 AP 的替代品。

AP/指标定义可核对 [scikit-learn average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)。

CI 默认 95%，重采样单位是**数据集/模拟 seed**，不是边。至少两份有效重复才给跨数据集区间；SERGIO 共用拓扑，区间只刻画这组模拟重复，不代表跨网络/跨病人的泛化。正式模拟多个场景混合均值也只代表预定场景混合，单场景应查看 dataset_metrics 分层结果。

默认预先指定主指标 `primary_metrics=("AP",)`；双侧 Wilcoxon 用共同成功数据集配对；Holm 对所有本轮预定检验做多重校正。5 对样本时显著性检验能力有限，不能反复换检验直到 p<0.05。`mean_delta > 0` 仅表示这组匹配数据上 JumpNiche 均值较高，不自动等于显著优于。

不能只展示成功子集而隐藏失败。若 A 成功 20/20、B 成功 5/20，二者成功子集均值不能当完整公平结论。也不把缺失、不可运行、未收敛、不支持的任务记为真实得分 0。

## 10. 失败与重复运行

- `check` 先验证安装包/源码版本与基础输入；不执行完整算法，因此检查通过仍可能遇到官方运行时问题。
- 一份数据失败后 Python 调度继续下一份，详细异常在该目录 `run_status.json`；原生崩溃的 queued/running 状态转为 interrupted。
- 仅补跑失败项：在 `datasets.py` 写 `dataset_ids=("准确编号",)`，方法保持 `mode="run"`。完成结果及输入/参数/输出哈希未变时跳过。
- 参数或文件改变而已有 completed 结果时默认拒绝覆盖。换 `output_root` 保留原实验；不要靠删除状态来掩盖参数变化。
- 官方数值数组/模型尽量在统一转表前保存，但本适配器**尚未实现各方法完整的“只重做导出、不再拟合”自动恢复**。已有 `official_*` 有助于单独修复导出，不保证再次 run 自动复用。
- 评价默认限制预测与真值各最多 500 万行，超限明确审计，不偷偷截断。需要更大规模时先规划流式评价或足够内存机器；调大保护值不会减少真实内存需求。
- WSL/Linux 要在配置类改绝对路径；注册表里的 Windows 路径可用 `path_prefix_from="C:/Users/Wuyan/Desktop/GRN/Futher/GRN2"`、`path_prefix_to="/你的实际项目路径"` 显式转换，源码/资源/输出类路径也要改。程序不靠隐藏环境变量寻找数据。

## 11. 本次交付验收边界

已提供：独立运行文件、按方法环境定义、配置类、完整结果契约、参数冻结、单数据子进程、缓存与失败审计、统一评价/配对统计/绘图、轻量回归测试。

未声称完成：九个官方环境的实机完整安装、所有官方模型拟合、CLARIFY 对当前全部数据兼容、iMIRACLE 论文全指标复现、扰动协议修复、真实标签与先验独立性验收，以及“证明 JumpNiche 一定更好”。结果必须由实际公平实验决定。

测试只使用小型临时数据，命令：

```powershell
conda activate jumpniche
python -m pytest .\baselines\tests -q -p no:cacheprovider
```

各方法环境不要求安装 pytest。详细交付和验证记录见 `DELIVERY_RECORD.md`。
