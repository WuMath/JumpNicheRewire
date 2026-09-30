# JumpNicheRewire

## 0. 执行顺序

执行分为四层：①从原始文件生成 `reproduction_runs/prepared/`；②重新拟合到 `reproduction_runs/fits/` 或运行七方法队列；③用固定候选集合评价；④重新生成案例表图。若仅要核对**论文已经展示的数值**，可直接运行第 4 节的保存结果核对脚本，无须重新训练。

## 1. Data preprocess：从 `raw_data` 生成模型输入

入口是 `reproduction/prepare_inputs.py`。它调用 `model_code/src/` 中的原数据转换函数，输入和产物如下。`--rebuilt-input` 选项会使后续新拟合读取本节产物。

| 数据 | 读取的原始文件和参考表 | 新模型输入 |
|---|---|---|
| scMultiSim | `raw_data/scmultisim/export/`，由作者 `author_result.rds`、`author_observed_counts.csv` 导出的表达、坐标、TF、LR 和网络真值 | `reproduction_runs/prepared/scmultisim/` |
| P22 | `raw_data/p22/` 中的 RNA 矩阵、空间文件、CellChatDB；`raw_data/references/tf_list.csv` | `reproduction_runs/prepared/p22/` |
| 小鼠肾脏 | `raw_data/case_mouse_kidney/` 中的 10x H5 和空间包；`raw_data/references/` | `reproduction_runs/prepared/case_mouse_kidney/` |
| 人乳腺癌 | `raw_data/case_human_breast_cancer/` 中的 10x H5 和空间包；`raw_data/references_human/`；`raw_data/publication_20260920/references/human_signaling.tsv` | `reproduction_runs/prepared/case_human_breast_cancer/` |

按下面顺序运行。已生成的产物会先与论文保存的输入核对；一致时显示 `REUSE` 并继续，内容不一致或不完整时才报错。脚本不会覆盖既有产物。

```powershell
& $Py .\reproduction\prepare_inputs.py scmultisim
& $Py .\reproduction\prepare_inputs.py p22
& $Py .\reproduction\prepare_inputs.py case_mouse_kidney
& $Py .\reproduction\prepare_inputs.py case_human_breast_cancer
```

逐表核对新输入与论文使用的 `model_inputs/`。`PASS` 表示七张主要模型输入表在值和行列顺序上相同。

```powershell
& $Py .\reproduction\prepare_inputs.py compare --dataset scmultisim
& $Py .\reproduction\prepare_inputs.py compare --dataset p22
& $Py .\reproduction\prepare_inputs.py compare --dataset case_mouse_kidney
& $Py .\reproduction\prepare_inputs.py compare --dataset case_human_breast_cancer
```

P22 评价标签还需独立生成。这一步读取 `raw_data/p22/GSM6758285_..._ATAC_fragments.tsv.gz`、`raw_data/mouse_annotations/Mus_musculus.GRCm38.86.gtf.gz` 和 `reference_data/motif_tf_gene.parquet`，逐片段统计目标基因启动子的可及性。它只用于**评价**，不参与模型拟合。

```powershell
& $Py .\reproduction\prepare_inputs.py p22-labels
& $Py .\reproduction\prepare_inputs.py compare --dataset p22-labels
```

评价前按 motif TF 覆盖和启动子注释筛选，得到 840 条候选、100 条支持边。核对脚本要求新标签表与 `results/p22/support_edges.csv` 逐项相同。全量 2,400 条未筛选标签在 `reproduction_runs/prepared/p22_aux/02_reference/gse205055_p22/support_edges.csv`。P22 九张模型输入 CSV 另存于 `reproduction_runs/prepared/p22_aux/01_model_input/gse205055_p22/`，供七方法队列读取。

`raw_data/publication_20260920/benchmark/` 另有 26 份历史开发/测试数据。**英文正文的 scMultiSim 表只使用其中的 `published_tree1_500_cells110_genes_sigma0.1_1`**；不要把那 26 份数据混入这张表。

## 2. Benchmark：重新拟合、评价与正文表对应

### 2.1 参数放在哪里

| 用途 | 文件 | 说明 |
|---|---|---|
| scMultiSim 论文参数快照 | `model_code/experiment_parameters/参数配置_scMultiSim.json` | 保存 JumpNicheRewire 和基线的实际选择、完整模型配置、输入与预测哈希。历史路径仅是来源记录。 |
| P22 论文参数快照 | `model_code/experiment_parameters/参数配置_GSE205055_P22.json` | 保存参考构建规则、方法参数和来源记录。 |
| P22 完整模型配置 | `model_fits/p22/model_config.json` | 当时拟合的模型参数。 |
| 七方法新队列配置 | `model_code/configs/benchmark_selected.py` | 已选中正文的 scMultiSim、P22 和七种方法，输入路径指向本目录的新预处理产物；`SELECTED.action` 决定 `check/run/status/evaluate`。 |
| 当前入口选择 | `model_code/configs/active.py` | 当前 `profile="benchmark_selected"`，`main_action=SELECTED.action`；通常无需修改。 |

scMultiSim 的论文 JumpNicheRewire 配置为 `l1_penalty=0.075`、`fusion_penalty=0.0005`、`group_penalty=0.01`、`ridge_penalty=0.0001`，全局边分数按各单元系数的平均绝对值汇总。P22 的完整配置以保存的 `model_config.json` 为准。不要把快照里的候选名当作论文中需要解释的生物学参数。

### 2.2 先核对论文现有七方法分数

```powershell
Set-Location -LiteralPath $Root
& $Py .\reproduce_reported_results.py
```

该脚本读取 `results/scmultisim/scores/*.csv` 和 `results/p22/scores/*.csv`，重新计算正文的 AP、AUROC、早期精度、前 10% 候选的 F1 与 F0.1，不训练方法。预期 scMultiSim 是 **624 条候选、130 条真值正边**，P22 是 **840 条候选、100 条支持边**；最后应出现 `PASS: fixed-score paper results reproduced.`。输出直接对应 `paper/EN/main.tex` 的 `tab:benchsim`、`tab:benchp22`。七方法逐边文件也在这两个 `scores/` 目录，可检查每条 TF–target 的标签和分数。

### 2.3 可选：单独重新训练 JumpNicheRewire

本节与第 2.4 节是两条独立的复现路径，**不需要顺序执行**。如果要重新运行七种方法并生成同一队列的比较表，直接执行第 2.4 节，跳过本节。本节仅用于单独核对 JumpNicheRewire 的已保存完整参数快照。它读取第 1 节生成的输入，写入独立的新拟合目录；原论文的 `model_fits/` 不会被覆盖。

```powershell
& $Py .\reproduction\run_jnr.py scmultisim --rebuilt-input --check
& $Py .\reproduction\run_jnr.py p22 --rebuilt-input --check
& $Py .\reproduction\run_jnr.py scmultisim --rebuilt-input
& $Py .\reproduction\run_jnr.py p22 --rebuilt-input
& $Py .\reproduction\evaluate_fresh.py scmultisim
& $Py .\reproduction\evaluate_fresh.py p22
```

拟合结果分别在 `reproduction_runs/fits/scmultisim/`、`reproduction_runs/fits/p22/`；新 AP 等指标和完整逐边对齐表在 `reproduction_runs/evaluation/`。先检查各 `run_status.json` 的 `state=completed` 和收敛诊断，再比较新值与第 2.2 节的固定论文值。**这是新运行**，有环境差异时不能用保存的论文表代替新运行的评价。

### 2.4 从同一输入重新运行七种方法

`model_code/configs/benchmark_selected.py` 目前固定两套数据和 `JumpNicheRewire、CeSpGRN、CLR、GENIE3、GRNBoost2、PPCOR、RegDiffusion`。**运行本节时不再执行第 2.3 节。**当前冻结队列包含 15 次拟合：scMultiSim 七次；P22 除七种方法外，另有一次 JumpNicheRewire 的基础系数核对拟合。该额外拟合是队列的审计记录，不是第八种方法，也不应成为正文的独立结果。P22 正文只展示最终通信网络。配置中模型参数、随机种子、数据路径、评价参考、输出目录和资源上限均可查；拟合过程不读真值。

当前机器的 Python/R 环境路径已写入配置。CeSpGRN 和 RegDiffusion 的源码放在 `model_code/vendor/external_methods/`；RegDiffusion 的解释器及固定 R 库目前仍指向原机器的安装位置。换机时需先重建这些软件环境，再改配置中的 `reg_python`、`r_library`、`r_software`、`model_python`、`tree_python`、`cesp_python`、`rscript`。**仅复制 `raw_data` 不等于已安装七种方法。**

从项目根目录进入 `model_code`，先运行只读检查：

```powershell
Set-Location -LiteralPath (Join-Path $Root 'model_code')
& $Py .\main_jumpniche.py
```

`benchmark_selected.py` 中只需修改 `action: str = 'check'` 这一行，无须分别运行七条命令。每次修改后，都在 `model_code` 目录执行同一个入口 `& $Py .\main_jumpniche.py`：

1. 保持 `action: str = 'check'`，执行入口，核对输出显示两套数据、七种方法和 15 次拟合，并确认资源预检通过。`check` 不训练模型。
2. 将该行改为 `action: str = 'run'`，再次执行入口，并等待全部拟合结束。它按固定队列运行 15 次拟合，写入 `reproduction_runs/selected_benchmark/03_runs/`，以 `01_plan.json` 固定输入、参数和代码哈希。
3. 将该行改为 `action: str = 'status'`，再次执行入口，查看全部任务是否完成。
4. 全部任务完成后，将该行改为 `action: str = 'evaluate'`，再次执行入口，生成 `reproduction_runs/selected_benchmark/04_evaluation/selected_metrics.csv` 和 `all_independent_fits.csv`。前者是新运行的正文候选表，后者保留 P22 两个独立拟合视图。

步骤 1 至 4 只改 `action`，不要同时改数据、参数或输出路径。任务中断后保留已完成结果和失败记录，先查 `reproduction_runs/selected_benchmark/study_process.json` 与相应任务的 `run_status.json`，不要删除记录后伪装成连续运行。对照 `paper/EN/main.tex` 的两张 benchmark 表时，以数据集、方法、候选集合和五项指标逐项比较。新运行若与论文保存值不同，保留差异并追查环境、版本、随机性及参数，不改动论文原始 `results/`。

**当前运行记录（2026-09-30）**：`reproduction_runs/selected_benchmark/study_process.json` 显示队列已完成，`latest_progress.json` 显示 15/15 次拟合完成。无需再次执行 `run`。要查看状态或生成评价表，只按上面的步骤把 `action` 改为 `status` 或 `evaluate` 并运行同一入口。先前一次 `check` 曾因系统可用内存不足而停止；随后启动的队列已完成，该旧错误不代表当前队列失败。

## 3. 案例分析：重新拟合、边增量和论文图

### 3.1 参数与输入

两张切片的完整参数在 `model_fits/case_mouse_kidney/model_config.json` 和 `model_fits/case_human_breast_cancer/model_config.json`。原始 10x 文件在 `raw_data/case_*/`；第 1 节预处理得到 `reproduction_runs/prepared/case_*/`，其中 `expression.csv`、`coordinates.csv`、`metadata.csv`、`tf_list.csv`、`lr_pairs.csv`、`prior_tf_gene.csv`、`prior_lr_tf.csv` 是模型直接读取的七张表。`model_code/configs/public_sim_data/publication_study.py` 记录案例预处理选择量（如 600 基因、12 TF、40 LR、种子 20260920）；`reproduction/run_jnr.py` 重新拟合时使用每个案例保存的**完整模型配置**，同时把输入、输出改为新目录。

### 3.2 保存拟合的论文结果核对与重绘

```powershell
Set-Location -LiteralPath $Root
& $Py .\reproduce_case_from_fits.py
& $Py .\paper\redraw_manuscript_figures.py
```

第一个脚本从 `model_fits/case_*/mechanism_coefficients.npz` 和 LR 暴露表重建逐 spot 的 `Δβ`，核对 `results/cases/case_*/` 的全部边、首位边及主 LR 分量；预期肾脏 **3,124 spots、441 条拟合边**，乳腺癌 **2,518 spots、296 条拟合边**，两例均输出 `PASS`。第二个脚本读取 `results/cases/case_*/selected_edge_spatial_values.csv` 和 `expression_umap.csv`，重绘到 `paper/figures/`。若要更新英文稿图件，再将两个 PDF 复制到 `paper/EN/figures/` 后编译英文稿。

英文稿表 3（`tab:caseedgevariation`）取两份 `complete_edge_delta_summary.csv` 各自按 `delta_q90_minus_q10` 降序排列的前三行。该值为同一条边全部 spot 的 `Δβ` 第 90 百分位减第 10 百分位；`mean_abs_delta` 为全部 spot 的 `|Δβ|` 均值。`reproduce_case_from_fits.py` 从保存的暴露和交互系数独立重算，并逐边核对这些数值。表中的四位小数仅为显示时四舍五入。

若要从拟合产物直接重新**计算**表 3，而非读取预制排序表，在项目根目录运行 `& $Py .\reproduction\calculate_table3.py`。脚本默认读取 `model_inputs/case_*/` 与 `model_fits/case_*/`，输出到 `reproduction_runs/table3_recomputed/`：`table3_top3.csv` 为六行表值，`table3_case_edges.tex` 可复制到英文稿，两个 `*_all_edges.csv` 保留完整排序，`calculation_audit.json` 记录输入文件哈希。若要用新拟合计算，传入 `--inputs .\reproduction_runs\prepared --fits .\reproduction_runs\fits --output .\reproduction_runs\table3_from_fresh_fits`。两种模式均已核对得到表 3 的六行数值。

图件版本需区分：早期完整案例图有 A--F 六个面板，其中 E 的主导 LR 空间贡献与 C 的总增量图形相近；当前中英文稿使用 A--D 四面板的 `*_edge_only.pdf`，删除了 E 和通路图 F。若用第 3.3 节的**新拟合**生成 `reproduction_runs/figures/`，B、D 的 UMAP 可能与论文保存图不同，因为该流程重新计算 `expression_umap.csv`。已核对新旧拟合的 `selected_edge_spatial_values.csv` 哈希相同，A、C 所用的 spot、坐标和增量不变；两份 UMAP CSV 哈希不同。要精确重绘论文现有图，使用本节读取论文保存 CSV 的脚本；不要直接以新 UMAP 图覆盖正文。

| 论文位置 | 数据或图文件 |
|---|---|
| 肾脏图 `fig:kidneycommunication` | `paper/EN/figures/mouse_kidney_communication_case_edge_only.pdf`；源值在 `results/cases/case_mouse_kidney/selected_edge_spatial_values.csv` |
| 乳腺癌图 `fig:breastcommunication` | `paper/EN/figures/human_breast_communication_case_edge_only.pdf`；源值在相应的 `selected_edge_spatial_values.csv` |
| 前三条边表 `tab:caseedgevariation` | 两个 `results/cases/case_*/complete_edge_delta_summary.csv` 的前三行 |
| 文中 spot 数、首位边、分位差、LR 来源 | 两个 `results/cases/case_*/analysis_summary.json` 和 `selected_edge_region_summary.csv` |

### 3.3 从新输入重新拟合案例并生成新表图

```powershell
& $Py .\reproduction\run_jnr.py case_mouse_kidney --rebuilt-input --check
& $Py .\reproduction\run_jnr.py case_human_breast_cancer --rebuilt-input --check
& $Py .\reproduction\run_jnr.py case_mouse_kidney --rebuilt-input
& $Py .\reproduction\run_jnr.py case_human_breast_cancer --rebuilt-input
& $Py .\reproduction\analyze_fresh_case.py case_mouse_kidney --rebuilt-input
& $Py .\reproduction\analyze_fresh_case.py case_human_breast_cancer --rebuilt-input
& $Py .\paper\redraw_manuscript_figures.py --source .\reproduction_runs\cases --output .\reproduction_runs\figures
```

新拟合在 `reproduction_runs/fits/case_*/`；新分析表在 `reproduction_runs/cases/case_*/`；新 PDF 在 `reproduction_runs/figures/`。`complete_edge_delta_summary.csv` 按各边的 `Δβ` 第 90 百分位减第 10 百分位排序，`mean_abs_delta` 是该边在全切片的 `|Δβ|` 均值。图中的 A、B 是同一批 spot 的空间位置与表达 UMAP，C、D 是首位边变化在相应两种坐标中的分布。检查新分析得到的首位边、spot 数和表值后，才考虑替换英文稿图片。

## 4. 已验证范围与需要保留的记录

在当前目录及当前机器上，四套输入的第 1 节逐表核对均通过；P22 从 1.53 GB fragments 重建的 840 条评价边与 100 条支持标签逐项吻合；`benchmark_selected` 的七方法只读 `check` 通过；论文固定分数与保存案例系数的核对脚本也通过。**本手册新增的完整七方法训练和四次新 JumpNicheRewire 拟合尚未执行**，因此不能声称新训练已经得到论文数值。新训练需按第 2、3 节运行，并保存各自 `run_status.json`、`effective_config.json`、`01_plan.json`、逐边预测和评价表。

根目录现有 `JumpNicheRewire_论文数据代码结果完整包.zip` 是旧归档，它不包含后来放入目录的 1.53 GB P22 fragments，也不包含本手册新增的脚本。对外共享前，应从当前文件夹重新制作归档，并核对它实际包含的文件；不能把旧 ZIP 当作当前目录的完整镜像。
