# 本次新增文件与验证记录

日期：2026-09-07。目标目录：`C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct`。

## 改动范围

仅新增 `baselines/` 与 `src/independent_baselines/`。没有修改 `main_jumpniche.py`、`preprocess_jumpniche.py`、`configs/active.py`、`src/jump_niche_rewire`、旧 benchmark 管理代码、既有冻结参数或输出。没有运行原项目数据预处理、模拟或模型 test。

共用配置在 `baselines/baseline_settings/datasets.py`；各方法参数在同目录各自文件。新增八个 Python 方法入口、一个 R 方法入口、一个统一评价入口，九份独立环境 YAML。完整执行教程为同目录 `README.md`。

所有方法独立调用官方包/源码；适配器与统一评价不导入 JumpNiche 模型。读取原模型已有结果只发生在评价阶段。

## 已完成验证

- 43 项轻量 pytest 回归测试通过：输入白名单、ID 对齐、非有限数拒绝、pilot/final/test 门禁、扰动门禁、同分 AP/top-k、LR 排名、完整 Parquet 而非 Excel 预览、候选全集、缺失方法审计、冻结协议不静默覆盖。
- 模拟官方 API 验证 GENIE3、GRNBoost2、CeSpGRN、SpaGRN、LIANA 调用参数和输出转换；验证 CellPhoneDB 方向规则及物种限制。模拟 API 测试不等于第三方模型已经拟合成功。
- 实际启动两个轻量独立 Python 子进程，有意触发不存在的适配器；两份数据分别记录 failed、不同 PID，第一份失败未终止后续数据。没有执行真实第三方模型。
- 用本机 R 4.4.1 解析三个 R 文件，语法通过；有系统 locale 警告，没有运行安装或 CellChat 算法。
- 只读核对实际注册表：SERGIO 五份，每份 450×100，输入可对齐，10 个表达 TF；sCCIgen 一份、Visium 一份；核心正式 validation 12、test 20，不混 pilot。
- 只读核对五份 reconstruct SERGIO 的 JumpNiche 状态：completed 且 convergence_passed=true。
- sCCIgen 的 reconstruct JumpNiche 输出已存在，状态 completed、convergence_passed=true；无需为了基线比较重跑这一结果。
- 新配置预期的 reconstruct Visium 结果目录未发现 run_status.json；后续需指定实际已有结果位置，或按原模型流程独立完成案例。不能把 missing 当完成。
- 核心正式结果路径可按旧 `frozen_parameters.json` 定位 `jumpniche/smooth/...`；本次只检查路径与注册表，没有重新揭盲计算 test 指标。

## 未完成、不可据此宣称完成的工作

- 未在各独立环境完成依赖安装/实际训练；环境 YAML 是按核对过的源码与版本约束给出的安装方案，不是所有平台的已验证锁定容器。
- CeSpGRN/SpaGRN/GNN 方法依赖旧科学计算栈，建议在隔离环境完成导入测试；需要时使用 Linux/WSL/超算。
- CLARIFY 的当前公开源码不能直接适配本文已有 SERGIO/sCCIgen；全 Visium 的稠密 gene-level 解码矩阵有严重内存成本，默认内存保护会阻止超限计算。
- `iMIRACLE-LRComponent` 是明确标注的表达贡献适配，不是假装复现论文全部 regulatory ranking。
- 尚未完成真实伪真值与输入先验的来源独立性审计；扰动流程仍有门禁，不因新增评价接口而自动合格。
- 未生成任何方法优胜结论。缺失/失败/不支持任务不补 0 分；比较结果取决于后续实际实验。

## 保存与重复运行

默认新输出仅在 `reconstruct/data/result/independent_comparison_v1` 中创建。冻结、拟合、评价分别保存；已完成且哈希不变的 Python 运行跳过。新增方法后仅换评价 `analysis_name` 重新汇总即可。

该目录目前不包含人为编造的正式方法结果。测试结果仅来自临时小数据，不写入实际 benchmark 输出。
