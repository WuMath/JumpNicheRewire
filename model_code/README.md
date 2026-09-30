# JumpNicheRewire 重构版

当前作者公开 scMultiSim **单数据集九方法评价**的只读本地复现入口：[文件索引与运行教程](docs/SCMULTISIM_LOCAL_REPRODUCTION.md)。中文论文源码为 [LaTeX 草稿](paper/scmultisim_development_zh.tex)，编译预览在 `output/pdf/scmultisim_development_zh.pdf`。该开发集评价与下面重新调用 R 模拟器的 `scmultisim_v1` 教程是两个不同实验；不要按下面教程重跑以为能得到九方法历史表。

新增独立空间 benchmark：[scMultiSim 完整教程](docs/SCMULTISIM_BENCHMARK.md)。
配置为 `configs/public_sim_data/scmultisim.py`，包含官方模拟、独立预处理、模型/基线运行和 GRN/通信分任务评价。

本目录是独立代码副本，不替换、不移动、不修改父目录的旧代码和旧实验结果。
核心数值算法保持原样；本次整理的是入口、配置位置、导入路径和运行组织。

## 两个入口

```powershell
conda activate jumpniche
Set-Location "C:\Users\Wuyan\Desktop\GRN\Futher\GRN2\reconstruct"

python .\preprocess_jumpniche.py
python .\main_jumpniche.py
```

没有命令行参数。先修改 `configs/active.py` 中的 **ActiveConfig 类字段**：

```python
@dataclass
class ActiveConfig:
    profile: str = "public_sim_data.sergio"
    main_action: str = "check"
    preprocess_action: str = "check"
```

上面示例是检查模式，不启动模型、不下载、不读评价真值、不写输出。实际操作取决于你保存的 ActiveConfig 字段；此次入口简化不修改你当前的选择。
两个程序读取同一个配置选择，但各自只执行自己的 action。

## 目录

```text
reconstruct/
├── main_jumpniche.py                 唯一模型/benchmark/评价入口
├── preprocess_jumpniche.py           唯一数据处理入口
├── configs/
│   ├── active.py                    选择数据配置及操作
│   ├── paths.py                     旧数据位置、新输出位置、写入白名单
│   ├── common/                      共享模型参数类型、默认值、评价接口
│   ├── public_sim_data/             SERGIO、sCCIgen、手动下载目录检查
│   ├── real_data/                   Visium、通用 H5AD、先验与伪真值
│   ├── perturbation/               Spatial Perturb-Seq 转换/准备参数
│   └── self_sim_data/               小示例、pilot、核心 final、模拟/调参配置
├── src/
│   ├── jumpniche_app/               统一分派、路径保护、配置快照、预处理缓存
│   ├── jump_niche_rewire/           核心模型
│   ├── preprocess/                  Visium/参考表/注释等处理实现
│   ├── data_acquisition/            离线 H5AD/矩阵读取实现
│   ├── evidence_benchmark/          公开模拟器/伪真值/扰动实验实现
│   ├── formal_benchmark/            自建模拟 benchmark 三阶段实现
│   ├── evaluation/                  统一指标、排名同分处理
│   ├── baselines/                   独立结果格式的公共适配函数
│   └── grn2_project/                新副本的路径兼容层
├── data/
│   ├── raw/                        新版本需要解压/转换的中间原始文件
│   ├── processed/                  新版本生成的标准模型输入和注册表
│   └── result/                     新模型结果、评价、日志、检查点
├── environments/                   SERGIO/Stereopy 专用环境文件
├── docs/                           数据说明、参数映射、迁移记录、验证报告
└── tests/                          回归测试，不是额外的实验入口
```

旧数据默认从父项目 `data/raw`、`data/processed` **只读复用**，不复制大型矩阵。
计算全部使用本目录 `src`，不会导入父项目的模型代码。

## 可以选择的配置

| profile | 数据/任务 | 默认输入 |
|---|---|---|
| `public_sim_data.sergio` | 5 个 SERGIO 种子，GRN 恢复 | 旧的已准备数据 |
| `public_sim_data.sccigen` | 官方 sCCIgen example1，跨细胞影响评价 | 旧的已准备数据 |
| `real_data.visium` | Visium 脑组织，可信数据库关系检索 | 旧的已准备伪真值数据 |
| `real_data.h5ad` | 用户自行指定的真实空间 H5AD | 新预处理输出，需先填真实文件名 |
| `perturbation.spatial_perturbseq` | GSE274447 | 新转换/预处理输出；正式评价有保护门 |
| `self_sim_data.demo` | 24 个细胞的小示例，仅验证工程流程 | 新生成 |
| `self_sim_data.pilot` | pilot 参数选择与流程验证 | 新注册表，可只读复用旧数据 |
| `self_sim_data.core_final` | 4 场景、52 份数据，其中 20 份 test | 新注册表，可只读复用旧数据 |

MERFISH 不在默认可选实验中，不会自动重跑。

## 在哪里改参数、输入和输出

先打开选择的配置文件，例如 `configs/public_sim_data/sergio.py`：

- `DataParameters`：模拟/预处理的参数及输出。
- `DatasetParameters`：模型读取哪份数据，哪些 dataset_id。
- `RunParameters`：结果目录、是否覆盖、是否隔离子进程、冻结参数文件。
- `ModelParameters`：共享模型默认值来自 `configs/common/presets.py`；完整参数类定义在 `configs/common/model.py`。允许在具体数据配置中定义其子类，覆盖嵌套配置。
- `self_sim_data` 的 `BenchmarkParameters`：调参网格、阶段、benchmark 大小限制等；`EvaluationParameters`：统计与评价配置。

例如为某一数据单独限制 TF/target，不改共享默认值：

```python
from dataclasses import dataclass, field
from configs.common.model import BaseGRNConfig
from configs.common.presets import ModelParameters as SharedModelParameters

@dataclass
class LocalGRNParameters(BaseGRNConfig):
    maximum_tfs: int = 20
    maximum_targets: int = 100
    l1_penalty: float = 0.02
    fusion_penalty: float = 0.15

@dataclass
class ModelParameters(SharedModelParameters):
    base_grn: BaseGRNConfig = field(default_factory=LocalGRNParameters)
```

该文件已有的 `build_config()` 会实例化新的 `ModelParameters`。
共享定义无需复制多份；针对具体数据的覆盖仍写在所属类别的配置文件中。

**冻结文件优先级**：SERGIO、sCCIgen、Visium、扰动默认读取父项目已经冻结的 `smooth` JSON。
文件中 10 个已选调参字段覆盖手动模型值，其余模型参数仍由配置类提供。
如要完全手动指定参数，设置 `RunParameters.frozen_parameters_file = None`。
程序会保存冻结来源、SHA256 和实际配置，不会假装自动沿用旧 `sparser`。

## 最短运行示例：用已有 SERGIO 数据

1. `ActiveConfig.profile = "public_sim_data.sergio"`。
2. `main_action = "check"`，执行 `python .\main_jumpniche.py`。
3. 检查列出的 5 个数据集及新输出位置。
4. `main_action = "run"`，再次执行同一个程序。
5. 完成后 `main_action = "status"` 查看状态；`main_action = "evaluate"` 才读取评价真值。

已有标准化数据可直接运行，不需要先重复模拟。
不要在旧 core test 占用内存时，同时启动 Visium 或新的大规模模拟运行。

新结果位置示例：

```text
data/result/public_sim_data/sergio/
├── 00_selection/                   选中数据清单、配置快照、冻结来源
├── 00_logs/                        每次入口运行的状态/错误记录
├── 00_preprocess/                  预处理完成标记（若执行了预处理）
├── 01_jobs/                        独立基线任务清单
├── 02_runs/
│   ├── public_simulator/sergio_grn_100g_seed_20260902/
│   │   ├── run_status.json
│   │   ├── effective_config.json
│   │   ├── JumpNicheRewire_results.xlsx
│   │   └── parquet/
│   └── baselines/                  导入的独立方法结果
└── 03_evaluation/                  指标、统计、评价工作簿
```

每个证据数据集自动使用独立子进程，子进程直接调用 `main_jumpniche.py` 的 `main()`，接收已经确定的配置、数据行和签名。
不再使用 `entry.py`、`run_worker_from_request()`、请求文件或请求路径环境变量。
启动时打印配置文件位置；开始计算每份数据时打印实际输入和输出目录。
旧输出中若还有 `_worker_requests`，它只是历史文件，新程序不读取，也没有自动删除它。

## 主程序现在如何调用（2026-09-06 简化）

```text
main_jumpniche.py 的 main()
  → configs/active.py：选择 profile 和 main_action
  → 对应数据配置文件的 build_config()：读取配置类
  → src/jumpniche_app/model_runner.py 的 run()：选择运行/检查/评价
  → src/evidence_benchmark/run_model.py 的 run_models()：逐份调度证据数据
  → 独立进程直接调用 main(配置, 数据行, 签名)
  → _run_one_dataset() → JumpNicheRewirePipeline(...).run()
```

你只需修改配置文件，不必填写 `main()` 括号内的三个内部参数。
例如使用 SERGIO，编辑 `configs/public_sim_data/sergio.py`：

```python
@dataclass
class DatasetParameters(InputSelection):
    source: str = "existing"
    # 可把注册表路径直接写在这里，不必查看继承的默认值。
    existing_registry: Path = Path(
        "C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/processed/"
        "evidence_benchmark/03_registries/evidence_registry.csv"
    )
    dataset_ids: tuple[str, ...] = ("sergio_grn_100g_seed_20260902",)

@dataclass
class RunParameters(RunOptions):
    output_root: Path = Path(
        "C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/"
        "data/result/public_sim_data/sergio_new_run"
    )
    frozen_parameters_file: Path | None = None  # 使用手动模型值；正式比较请保留已冻结参数。
```

注册表是数据清单；每行 `dataset_dir` 才是该份数据的模型输入目录，不是环境变量。
单份自己的数据可用 `real_data.h5ad` 或 `self_sim_data.demo` 的 `DatasetParameters.single_input_dir` 直接指定目录。
数据处理入口的读取与执行逻辑在 `preprocess_jumpniche.py`，不经过第三个入口。
主程序仅保留数值库线程用的环境变量，它们不选择数据或参数路径。

本次代码变化会影响已有的代码签名；如只是查看已完成 SERGIO，请用 `status` 或 `evaluate`。
不要为了查看结果改成 `run`，否则可能重新计算。确需重新计算时，可像上例一样使用新的输出目录，保留旧结果。
修改记录及测试说明见 `docs/ENTRY_SIMPLIFICATION_20260906.md`。

## 重新生成/预处理数据

| profile | preprocess_action | 实际工作 |
|---|---|---|
| SERGIO | `prepare` 或 `simulate` | 调用官方 SERGIO 生成 5 个种子，再标准化；用 `jumpniche-sergio` 环境 |
| sCCIgen | `prepare` | 处理已下载的 example1 count/meta/pattern；不是重新运行 R 模拟 |
| Visium | `convert` | 原始 10x H5 + spatial 坐标 → 新 full 模型输入 |
| Visium | `prepare` | 从 `DataParameters.datasets` 指定的输入构建伪真值 benchmark |
| H5AD | `prepare` | 按指定 X/layer/raw、spatial 和 obs 键转换 |
| Spatial Perturb-Seq | `convert` | GEF → H5AD，需人工填写真实样本映射 |
| Spatial Perturb-Seq | `prepare` | 旧版扰动协议；默认保护门禁止作为正式评价直接执行 |
| demo/pilot/core_final | `prepare` 或 `simulate` | 生成相应规模的自建模拟数据 |
| evidence/pilot/core_final | `reuse` | 只复制/合并选中数据的注册条目；原始大矩阵不复制 |
| real_data | `references` | 调用参考表、cisTarget、严格 LR、LR→TF 处理步骤 |

所有操作均调用 `python .\preprocess_jumpniche.py`。

`DatasetParameters.source` 决定**模型**读取哪里：

- `"existing"`：读取旧项目注册表，结果仍写入新目录。
- `"prepared"`：读取本目录的新注册表。

重新预处理成功后，如要用新数据运行，请显式切换到 `"prepared"`。
Visium 的 `convert` 和 `prepare` 是两步：完成 convert 后，把 `DataParameters.datasets` 中的输入目录改为 `VisiumParameters.output_dir`，再 prepare。
如果重新生成了先验，还需将 `reference_dir`/`lr_reference` 改到新 references 目录；不能误以为新先验已自动加入旧输入。

输入/参数/代码不变且输出存在时，预处理完成标记可避免重复处理。
如要明确重建，设置具体配置返回的 `ExperimentConfig.overwrite_preprocessing=True`，或选择新的输出目录。
预处理缓存使用输入大小/修改时间、参数与代码签名及输出存在性/大小；这不是原始数据的逐字节认证。需要原始下载校验时用 `catalog`。

## 手动下载与环境

默认复用已有 `jumpniche` 环境。不要在 `(base)` 中运行模型，也不要为了重构重装已正常工作的环境。
`requirements.txt` 是主模型依赖清单，包含 RSS 监测所需的 psutil。

仅在对应环境缺失时安装：

```powershell
conda env create -f .\environments\sergio.yml
conda env create -f .\environments\stereopy.yml
```

SERGIO 用 `conda activate jumpniche-sergio`，GEF 转换用 `conda activate jumpniche-stereopy`；完成后切回 `jumpniche` 运行模型。
不将 Stereopy 的旧依赖混入主环境。转换配置/路径保护已避免依赖 Python 3.9+ 的类型求值和 Path.is_relative_to；本机未安装 Stereopy 环境，因此未实际执行 GEF 转换。

`preprocess_action="catalog"` 只检查已下载文件、生成手动下载说明和校验记录；保持 `download_enabled=False`。
`preprocess_action="install"` 只将已下载的 SERGIO 源码解压到新 `vendor`，不安装全局 R 包。
解压后若使用新源码，修改 `DataParameters.official_repo` 指向新 `vendor/external_simulators/SERGIO`。

## 自建模拟 benchmark 的统一执行顺序

选择 `self_sim_data.pilot` 或 `self_sim_data.core_final`。

1. `preprocess_action="reuse"` 复用已有数据；或 `"prepare"` 生成新数据。两者不同时做。
2. 保持 `RunParameters.unlock_test=False`。
3. `main_action="tune"`，运行主程序，仅使用 train/validation。
4. `main_action="audit"` 检查收敛，再 `"freeze"` 生成新版本参数审计与冻结文件。
5. 查看**新版本**输出，手动设置 `RunParameters.unlock_test=True`。
6. `main_action="test"` 推理，不读取 test 真值。
7. `main_action="evaluate"` 运行 Stage 3 揭盲评价。

新版本不导入旧运行缓存或旧运行状态，不会接管你已经启动的旧 core test。
若只是想完成目前正在进行的旧 benchmark，请继续使用原代码；无需立即在这里重复所有正式计算。

## 独立基线和评价

对于 evidence 配置：

- `main_action="jobs"`：生成所选数据的官方方法任务清单。**不等于安装或运行官方方法**。
- 各官方方法在各自环境中独立运行，使用 `baseline_input_dir`，不读取 truth。
- 在 `configs/common/comparison.py` 的 `ExternalParameters.specifications` 填入外部结果。也可以在具体数据配置里设置 `ExperimentConfig.external_import`。
- `main_action="import_external"`：转换为统一 Parquet 与状态文件。
- `main_action="evaluate"`：汇总本方法及已导入方法。

可接受的基本接口：GRN 为 `tf,target,score`，可选 `signed_score`；CCI 为 `sender_id,receiver_id,ligand,receptor,score`。
细胞类型级 CCI 使用 `ExternalResultSpec.cci_unit_type="cell_type"` 并匹配列名。
这些参数和路径均写在类中，不传脚本参数。

真实 H5AD 没有真值时，单样本应用只支持 check/run/status，不伪造评价答案。
三种证据任务分开解释；不将模拟精度、伪真值检索和扰动响应合成一个总准确率。
目前 evidence 评价输出表格和报告；完整自动绘图只保留在 formal Stage 3，未新增论文绘图模块。

## 输出、重跑与局限

- Parquet 是程序实际保留结果的完整表；Excel 默认每表预览 10,000 行。
- `top_grn_rows`、`top_lr_gene_rows` 等候选数量限制仍然有效，不能把 Parquet 误解为“所有可能边均未截断”。
- 数值检查点、target 检查点、残差历史、收敛判定与内存优化沿用旧实现。
- 相同模型/数据签名且已完成的运行可跳过；配置或代码改变会产生不同签名。
- 某个证据数据集失败时，后续数据继续运行；入口最终返回失败并保留成功结果及 traceback。
- 默认只允许写新 `data`/`vendor`，旧项目代码与结果受保护。
- 扰动协议的已知泄漏与重复不足问题**没有在这次目录重构中偷偷改变**；正式路径默认阻止执行，见 `docs/KNOWN_LIMITATIONS.md`。

详情见 `docs/MIGRATION_RECORD.md`、`docs/DATA_CONTRACT.md`、`docs/MODEL_PARAMETER_MAP.md` 和 `docs/VERIFICATION.md`。
