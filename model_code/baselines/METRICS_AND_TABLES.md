# 指标、参考文献与逐指标表格

更新：2026-09-07。本文对应 `src/independent_baselines/comparison.py`、`metric_references.py`、`metric_tables.py`。

## 1. 能否与 JumpNicheRewire 统一评价？

能统一的是**同一个任务、同一批数据、同一候选全集上的预测**，不是所有方法的每一种输出都互相可比。

| 同一任务 | 可纳入的方法 | 必须明确的限制 |
|---|---|---|
| 全局 TF→target（GRN） | JumpNicheRewire、GENIE3、GRNBoost2、CeSpGRN；资源匹配时的 SpaGRN、兼容条件满足时的 CLARIFY | 全局方法与细胞方法跨 cell 聚合后比较；偏相关/图解码网络不能因 TF 投影就被解释为有向因果网络 |
| 细胞特异 GRN | JumpNicheRewire、CeSpGRN、适配成功的 CLARIFY；全局方法可作为广播参照 | 广播指同一个 TF-target 分数用于所有 cell，不是全局方法真的推断了细胞差异 |
| sCCIgen 细胞类型对×LR/跨细胞基因对 | JumpNicheRewire、LIANA、CellPhoneDB、CellChat | 按 metadata 和基因符号对齐。植入基因对未必全在原生 LR 数据库中，要报告资源覆盖 |
| 核心模拟的 cell→cell 总 CCI | JumpNicheRewire、能给出该粒度预测的方法 | 当前 LIANA/CellPhoneDB/CellChat 类型对不能自动当成精确 cell 对；CLARIFY 当前合成数据分支不合格，不直接列入正式结果 |
| LR→gene 连续贡献排名 | JumpNicheRewire、iMIRACLE-LRComponent 探索性适配 | 使用 NDCG/秩相关，不把表达贡献当成二分类 GRN、通信概率或因果效应 |

**当前不能声称九种方法都能在所有已有数据上完整运行。** SpaGRN 的 motif 资源不匹配 G0/TF1；CellPhoneDB 标准数据库要求人类基因；CLARIFY 公布的部分数据分支存在兼容问题且内存需求大；iMIRACLE 此接口的贡献分数也不等同论文最终 regulatory score。详见 README 的逐方法限制及官方源码链接。

不能比较的任务写 N/A 或留空并附原因，不能填 0。某方法运行失败，与“成功运行但预测真的很差”是两种不同情况。

## 2. Precision、Recall、F-score 到底是什么？

把每条候选网络边看成一道判断题。“真边”表示真值认为这条相互作用存在，“预测边”表示模型把它选进了推荐集合。

- TP（真阳性）：选出来而且确实为真的边。
- FP（假阳性）：选出来但标签为假的边。
- FN（假阴性）：确实为真却没有选出来的边。
- TN（真阴性）：标签为假，而且没有选出来的边。

公式：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
F-beta    = (1 + beta²) * Precision * Recall / (beta² * Precision + Recall)
```

直观例子：真值一共有 20 条真边，模型推荐 10 条，其中 8 条为真，则 TP=8、FP=2、FN=12。
Precision=8/10=0.8，Recall=8/20=0.4，F1≈0.5333。这是教学例子，不是你的实验结果。

Precision 高：生物学家按照推荐做实验时，少做无效验证。Recall 高：遗漏的真实调控关系少。F1 是二者的调和平均，避免只追求其中一个。F0.5 更偏重 precision，F2 更偏重 recall。经典定义与解释见 [Manning 等，2008，第 8 章](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html)。

“精准度/精确率”通常指 Precision，**不是 Accuracy（准确率）**。Accuracy=(TP+TN)/全部候选。在极稀疏网络中，把所有边都判为“不存在”也可能有很高 Accuracy，因此本项目不把它当主指标，也未新增 Accuracy。关于类别不平衡评价的依据见 [Saito 与 Rehmsmeier，2015](https://doi.org/10.1371/journal.pone.0118432)。

## 3. 先定义怎么“选边”，才有唯一的 F1

各方法通常输出连续 score，不是直接给出 True/False。GENIE3 的特征重要性、CeSpGRN 的偏相关绝对值、LIANA 的聚合排名不是同一个尺度。统一用 `score>0.5` 会引入不合理差异。

当前实现采用**预先固定 top-k**：各方法各取最高分 k 条，使用相同验证预算比较。参数在：

`C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/baselines/baseline_settings/evaluation.py`

```python
top_k: tuple[int, ...] = (10, 50, 100)
classification_top_k: int = 100
f_betas: tuple[float, ...] = (1.0,)
export_per_metric_tables: bool = True
```

- 无后缀 `Precision`、`Recall`、`F1` 对应 `classification_top_k=100`。
- 同时输出 `Precision_at_10/50/100`、`Recall_at_10/50/100`、`F1_at_10/50/100`。
- 如果想补充 F0.5 和 F2，把 `f_betas` 改成 `(0.5, 1.0, 2.0)`；F1 始终保留。
- 候选数小于 k 时，实际取 `min(k, candidate_count)`，表中保存候选总数。
- `classification_top_k` 是评价预算，不是模型训练参数；不能看 test 的 F1 后选择对自己最有利的 k。
- 无正标签时 Recall/F 返回 NA；AP、AUROC 在各自不满足标签条件时也返回 NA。不能把未定义指标写成 0。

**没有新增“test 最佳 F1”搜索，也没有自动选择各方法阈值。** 若以后确实需要 score 阈值版 Precision/Recall/F1，应在 validation 为每个方法确定阈值并冻结，再应用 test，另命名报告。当前不要把 top-k 版冒充阈值校准版。

如果在看到既有 test 结果之后才决定新增 F1，应在论文中如实标为补充/事后分析，不能回头声称原来就预注册了这个终点。默认显著性主指标仍是 AP，没有因为增加 F1 自动扩大主检验范围。

## 4. 同分如何处理，F1 是否正确？

例子：第一条高分边为真，后面三条边同分，其中一条真、两条假。只取 top-2，三条同分边只剩一个名额。

```text
期望 TP = 1 + 1 × (1/3) = 4/3
期望 FP = 2 - 4/3 = 2/3
```

不是“有 1.333 条实际边”，而是对同分的所有可能排列求平均。对于固定预测数量 k 和固定真边数 P：

```text
F1 = 2*TP / (k + P)
F-beta = (1+beta²)*TP / (k + beta²*P)
```

分母固定，所以用期望 TP 得到的 F-score，等于所有同分排列的平均 F-score；不是一般情况下随便把“平均的比值”和“比值的平均”混用。新增测试显式枚举小例子的全部排列核对了这一点。参考：[McSherry 与 Najork，2008](https://www.microsoft.com/en-us/research/publication/computing-information-retrieval-performance-measures-efficiently-in-the-presence-of-tied-scores/)。

**AP 的同分约定不同：** 本项目保留 sklearn 的“相同分数整体进入一个阈值”的非插值 AP；它不等于上述论文的随机排列平均 AP，也不等于 PR 曲线梯形积分面积。论文里应写清 `average_precision_score`，不要把几种算法得到的数统称为同一个 AUPRC。实现依据：[scikit-learn AP 文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)。

## 5. 每个指标的来源及其意义

这些引用包括经典定义、实现规范和 GRN 领域使用依据，不意味着某一篇文章“最早发明了全部指标”。项目派生量会特别标注，不虚构发明来源。

| 输出名称 | 回答的问题 | 文献或实现依据 | 解释边界 |
|---|---|---|---|
| Precision、Recall、F1/F-beta，及各自 @k | 推荐是否可靠、找回是否充分、二者如何权衡 | Manning 等，2008；McSherry 与 Najork，2008 | 必须指定 top-k/阈值；不能用 test 寻优 |
| AP | 真边是否在整体排序中靠前 | Manning 等，2008；sklearn AP 规范；Saito 与 Rehmsmeier，2015 | 非插值阈值分组 AP；随候选全集和正例占比变化 |
| AUROC | 正边是否比负边分数高 | Fawcett，2006 | 正负同分记 0.5；不能作为稀疏网络唯一指标 |
| Early_precision | 排名前部、约一个真实网络规模的候选质量 | Pratapa 等，BEELINE，2020 | 本项目输出原始 early precision，不是 EPR 比值；k 使用标签正例数仅用于评价 |
| NDCG_linear | 强 LR→gene 贡献是否优先排在前面 | Jarvelin 与 Kekalainen，2002；McSherry 与 Najork，2008 | 线性增益、log2(i+1) 折扣的明确实例；不冒充所有 NDCG 变体 |
| Spearman | 预测和真值贡献的总体顺序是否一致 | Spearman，1904 | 使用平均秩，常量无定义；不是因果方向 |
| Kendall_tau_b | 两两候选的顺序一致性 | Kendall，1945；SciPy tau-b 规范 | 带并列校正，常量无定义 |
| Positive_prevalence | 数据有多稀疏 | Saito 与 Rehmsmeier，2015 的类别不平衡框架 | 数据属性，不是模型优劣得分 |
| AP_over_prevalence | 相对正例占比的归一化参考 | 基于 PR 基准思想的项目派生量 | 不冒充 BEELINE 原文 AUPRC ratio；有限随机排列的期望 AP 不严格等于 prevalence |

GRN 领域采用 PR 和 early precision 的具体依据：[Pratapa 等，2020，Nature Methods](https://doi.org/10.1038/s41592-019-0690-6)。它支持这些指标在 GRN benchmark 中的用途，但不能用“别人使用过”替代本项目对候选边、真值和并列规则的详细说明。

LR 排名每个 receiver×target 分别计算，再对有效组宏平均。F1 只适用于有二分类标签的任务，不应先把连续 LR 贡献随意阈值化来凑一张 F1 表。

Visium 的 `PU_` 前缀表示正例–未标注检索。标签 0 不是实验证实的负例，因此 `PU_Precision/PU_Recall/PU_F1` 只描述检索已知标签的情况，不能声称真实生物学准确率。仍需做输入先验与标签来源独立性审计。

## 6. 每个指标都会单独输出表吗？

会。每个**任务×指标**单独一份 Excel 和对应 CSV。比如 GRN 的 AP 与 CCI 的 AP 是两个文件，不混合。

```text
<output_root>/<suite>/03_evaluation/<analysis_name>/
  comparison.xlsx
  metric_definitions.csv
  metric_references.csv
  metric_table_manifest.csv
  by_metric/
    GRN_global/
      AP.xlsx
      AP.csv
      AP_summary.csv
      AP_matched.csv
      AP_long.csv
      Precision.xlsx
      Recall.xlsx
      F1.xlsx
      F1_at_10.xlsx
      F1_at_50.xlsx
      F1_at_100.xlsx
      ...
    GRN_cell_common_universe/...
    sCCIgen_cross_cell_gene_pair/...
```

只为本轮实际有指标记录的任务生成文件；没有任何方法结果时，不造一堆空白文件冒充完成。如果部分方法缺失，已生成表中仍保留它们的列/行，为空并附审计。

每个 `AP.xlsx` 内有以下 sheet，其他指标结构一样：

| sheet | 行和列是什么 |
|---|---|
| `by_dataset` | 一行一个数据集；第一列 dataset_id，后面每个方法一列，全部单元格都是同一个指标 |
| `method_summary` | 一行一个方法；前两列 `method, AP`（或当前指标名），后面 CI、有效数据集数、缺失数 |
| `matched_summary` | 同样一行一个方法，但仅用“所有所选方法共同有效”的相同数据集计算均值 |
| `long` | dataset_id、method、AP；保留候选数、覆盖率和固定 top-k 等细节 |
| `audit` | 每方法×数据集的状态、该指标是否可用，解释空值 |
| `definition` | 当前指标公式、取值方向、判边规则、同分规则、适用限制 |
| `references` | 本指标及统计方法的引用、可复制的 DOI/URL |

`by_dataset` 示意（横线是未展示数值，不是你的实验结果）：

| dataset_id | JumpNicheRewire | GENIE3 | GRNBoost2 | CeSpGRN |
|---|---|---|---|---|
| seed_20260902 | — | — | — | — |
| seed_20260903 | — | — | — | — |

`method_summary` 示意：

| method | F1 | evaluated_datasets | missing_or_undefined |
|---|---|---|---|
| JumpNicheRewire | — | — | — |
| GENIE3 | — | — | — |

因此同时满足两种习惯：**方法各占一列的比较矩阵**，以及**“方法一列、同一个指标值一列”的两列主体表**。在汇总表中，指标值是有效数据集逐份算分后的宏平均，不是把所有数据的边混成一个巨型网络再计算。

## 7. 置信区间与统计检验也有来源

这些是对性能差异的不确定性分析，不是新的模型得分。

- Bootstrap CI：[Efron，1979](https://doi.org/10.1214/aos/1176344552)。本项目对数据集/seed 重采样，不对数百万条相关边当独立重复；少于两份有效数据时不输出跨数据集 CI。
- 配对 signed-rank 检验：[Wilcoxon，1945](https://doi.org/10.2307/3001968)。比较相同数据集上两方法的差。其假设及小样本/大量并列的局限仍需讨论；5 个 seed 不能期待很强的检验能力。
- 多重检验：[Holm，1979](https://www.jstor.org/stable/4615733)。对本轮预先指定主指标的多个比较校正，避免挑选偶然显著结果。

SERGIO 的五个 seed 共用拓扑，CI 只能解释为该拓扑下模拟重复的变化，不是跨真实网络泛化。sCCIgen/Visium 单份数据不能对每条边重采样来假装有大量生物学样本。

若 A 完成 20/20，B 完成 5/20，`method_summary` 的有效子集均值不直接公平。先看 `audit`，再看 `matched_summary` 或 `paired_tests`。如果选入不支持本任务的方法导致全方法共同集合为零，`matched_summary` 留空；应明确任务对应的方法清单，而不是把 N/A 换成 0。

## 8. 完整参考文献

1. Manning CD, Raghavan P, Schutze H. *Introduction to Information Retrieval*. Cambridge University Press, 2008. [第 8 章定义](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html)，[排序评价](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)。
2. Saito T, Rehmsmeier M. *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLoS ONE 10(3):e0118432, 2015. [DOI](https://doi.org/10.1371/journal.pone.0118432)。
3. Fawcett T. *An introduction to ROC analysis*. Pattern Recognition Letters 27(8):861–874, 2006. [DOI](https://doi.org/10.1016/j.patrec.2005.10.010)。
4. Pratapa A, Jalihal AP, Law JN, Bharadwaj A, Murali TM. *Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data*. Nature Methods 17:147–154, 2020. [DOI](https://doi.org/10.1038/s41592-019-0690-6)。
5. McSherry F, Najork M. *Computing Information Retrieval Performance Measures Efficiently in the Presence of Tied Scores*. ECIR 2008, LNCS 4956:414–421. [作者机构论文页](https://www.microsoft.com/en-us/research/publication/computing-information-retrieval-performance-measures-efficiently-in-the-presence-of-tied-scores/)。
6. Jarvelin K, Kekalainen J. *Cumulated gain-based evaluation of IR techniques*. ACM TOIS 20(4):422–446, 2002. [DOI](https://doi.org/10.1145/582415.582418)。
7. Spearman C. *The Proof and Measurement of Association between Two Things*. American Journal of Psychology 15(1):72–101, 1904. [DOI](https://doi.org/10.2307/1412159)。
8. Kendall MG. *The treatment of ties in ranking problems*. Biometrika 33(3):239–251, 1945. [DOI](https://doi.org/10.1093/biomet/33.3.239)。
9. Efron B. *Bootstrap Methods: Another Look at the Jackknife*. Annals of Statistics 7(1):1–26, 1979. [DOI](https://doi.org/10.1214/aos/1176344552)。
10. Wilcoxon F. *Individual Comparisons by Ranking Methods*. Biometrics Bulletin 1(6):80–83, 1945. [DOI](https://doi.org/10.2307/3001968)。
11. Holm S. *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics 6(2):65–70, 1979. [原文存档](https://www.jstor.org/stable/4615733)。

公式来源与实现细节分开记录；不以一个 Python 函数文档替代所有学术引用，也不为自定义辅助量编造文献。

## 9. 如何生成，不用重跑模型

在 `evaluation.py` 保持或修改上面的评价参数；在同文件 `methods` 选定当前任务对应的方法。`datasets.py` 仍决定数据 suite 和 train/validation/test。

给新评价换一个不覆盖旧结果的名字：

```python
analysis_name: str = "comparison_metrics_v2"
```

运行：

```powershell
cd C:\Users\Wuyan\Desktop\GRN\Futher\GRN2\reconstruct
conda activate jumpniche
python .\baselines\evaluate_comparison.py
```

SERGIO 的默认位置为：

`C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/independent_comparison_v1/sergio/03_evaluation/comparison_metrics_v2/by_metric/`

核心模拟则把目录中的 `sergio` 换成 `core_final`。评价只读完整 Parquet/CSV 和真值，不读 Excel 预览，不调用模型；本次更新没有改模型训练冻结协议。若某基线尚未运行，表中会显示缺失，不能称为完整多方法比较。

## 10. 本次修改和验证范围

只修改独立评价程序和评价配置，新增引用登记、逐指标导出模块及测试。没有修改模型算法、拟合入口、各方法参数、数据注册表或已有模型结果。现有 AP/AUROC/Precision@k/Recall@k 的定义没有替换，新增 F-score 和预定 top-k 下的无后缀分类指标。

60 项轻量测试通过，覆盖固定预算 F1、同分所有排列的期望、异常 beta/k、无正例、逐指标引用登记、宽表/汇总表方向、不同任务分开、缺失保留和共同数据集汇总。实际多方法分数需各方法完成后再评价，本次没有编造或替你运行正式结果。

被替换的三个文件（comparison.py、evaluation.py、README.md）的原件备份在
`C:/Users/Wuyan/Documents/ChatGPT/GRN2/metric_reporting_build_20260907/original_files/`。
模型本体、基线 runtime/adapters/io 均未改动，因此既有模型冻结协议不会因这次评价更新失效。
