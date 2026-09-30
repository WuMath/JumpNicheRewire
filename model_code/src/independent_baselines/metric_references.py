"""指标定义、文献来源与项目实现的区别。不是给自定义量虚构出处。"""
from __future__ import annotations
import re
import pandas as pd

REFERENCES={
    "IR2008": (
        "Manning CD, Raghavan P, Schutze H. Introduction to Information Retrieval. Cambridge University Press, 2008, Chapter 8.",
        "https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html",
        "Precision、Recall、F-beta 的定义与解释；不是声称本书最早发明全部指标。"),
    "IR_RANK2008": (
        "Manning CD, Raghavan P, Schutze H. Introduction to Information Retrieval. 2008, Section 8.4.",
        "https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html",
        "排序检索中的 AP 和 top-k 评价；本项目 AP 的同分实现另外明确记录。"),
    "SAITO2015": (
        "Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLoS ONE 10(3):e0118432, 2015.",
        "https://doi.org/10.1371/journal.pone.0118432",
        "稀疏正例下使用 precision-recall 的动机及正例占比影响。"),
    "FAWCETT2006": (
        "Fawcett T. An introduction to ROC analysis. Pattern Recognition Letters 27(8):861-874, 2006.",
        "https://doi.org/10.1016/j.patrec.2005.10.010",
        "ROC/AUC 的解释与使用条件。"),
    "BEELINE2020": (
        "Pratapa A, Jalihal AP, Law JN, Bharadwaj A, Murali TM. Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nature Methods 17:147-154, 2020.",
        "https://doi.org/10.1038/s41592-019-0690-6",
        "GRN benchmark 使用 PR 和 early precision 的领域依据；本项目原始 Early_precision 不冒充 EPR 比值。"),
    "TIES2008": (
        "McSherry F, Najork M. Computing Information Retrieval Performance Measures Efficiently in the Presence of Tied Scores. ECIR 2008, LNCS 4956:414-421.",
        "https://www.microsoft.com/en-us/research/publication/computing-information-retrieval-performance-measures-efficiently-in-the-presence-of-tied-scores/",
        "Precision/Recall/F/NDCG 的同分排列期望依据；本项目 AP 用阈值分组，不等同其随机排列 AP。"),
    "NDCG2002": (
        "Jarvelin K, Kekalainen J. Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems 20(4):422-446, 2002.",
        "https://doi.org/10.1145/582415.582418",
        "分级相关性和位置折扣；本项目明确采用线性增益与 log2(i+1) 折扣。"),
    "SPEARMAN1904": (
        "Spearman C. The Proof and Measurement of Association between Two Things. American Journal of Psychology 15(1):72-101, 1904.",
        "https://doi.org/10.2307/1412159",
        "Spearman 秩相关的经典文献。"),
    "KENDALL1945": (
        "Kendall MG. The treatment of ties in ranking problems. Biometrika 33(3):239-251, 1945.",
        "https://doi.org/10.1093/biomet/33.3.239",
        "Kendall 并列秩处理；本项目使用 tau-b。"),
    "SKLEARN_AP": (
        "scikit-learn documentation: average_precision_score.",
        "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html",
        "实现规范：非插值 AP 为召回增量加权 precision，不是 PR 梯形面积。"),
    "SCIPY_RANK": (
        "SciPy documentation: spearmanr and kendalltau.",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html",
        "带并列的 tau-b 实现；库版本由评价环境记录。"),
    "EFRON1979": (
        "Efron B. Bootstrap Methods: Another Look at the Jackknife. Annals of Statistics 7(1):1-26, 1979.",
        "https://doi.org/10.1214/aos/1176344552",
        "Bootstrap 重采样方法；本项目以数据集/种子为重复单位。"),
    "WILCOXON1945": (
        "Wilcoxon F. Individual Comparisons by Ranking Methods. Biometrics Bulletin 1(6):80-83, 1945.",
        "https://doi.org/10.2307/3001968",
        "配对 signed-rank 检验；不是性能指标，不能把 p 值当模型得分。"),
    "HOLM1979": (
        "Holm S. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics 6(2):65-70, 1979.",
        "https://www.jstor.org/stable/4615733",
        "多重比较的 Holm 校正。"),
}

def bibliography(ids=None):
    return pd.DataFrame([dict(reference_id=key,citation=value[0],url=value[1],role=value[2])
                         for key,value in REFERENCES.items() if ids is None or key in ids])

def definition(metric,ec):
    pu=metric.startswith("PU_")
    name=metric[3:] if pu else metric
    result=dict(metric=metric,kind="performance",direction="higher_is_better",range="0 to 1",
                definition="",biological_use="",decision_rule="threshold_free",
                tie_policy="",limitations="",reference_ids="",reference_urls="")
    refs=[]
    at_k=re.fullmatch(r"(Precision|Recall|F[0-9]+(?:\.[0-9]+)?)_at_([0-9]+)",name)
    unqualified=name in {"Precision","Recall"} or re.fullmatch(r"F[0-9]+(?:\.[0-9]+)?",name)
    if at_k or unqualified:
        family=at_k.group(1) if at_k else name
        k=int(at_k.group(2)) if at_k else ec.classification_top_k
        result["decision_rule"]=f"top-{k}, effective_k=min({k}, candidate_count); prespecified, no test optimization"
        result["tie_policy"]="边界同分组均匀随机排列的期望；预测数量固定，因此 F-beta 也可由期望 TP 精确计算"
        refs=["IR2008","IR_RANK2008","TIES2008"]
        if family=="Precision":
            result.update(definition="TP / (TP + FP)",biological_use="推荐给实验验证的候选中，真相互作用占多少，关联验证成本。")
        elif family=="Recall":
            result.update(definition="TP / (TP + FN)",biological_use="所有真实相互作用中找回多少，关联遗漏风险。")
        else:
            beta=float(family[1:])
            result.update(definition=f"(1+{beta:g}^2)*P*R/({beta:g}^2*P+R)",
                          biological_use="同时考察 precision 和 recall；beta=1 等权，beta<1 偏重 precision，beta>1 偏重 recall。")
        result["limitations"]="依赖预先给定判边规则；不把模型 score 当校准概率。不在 test 上选最佳 k/阈值。无正标签的 Recall/F 返回 NA。"
    elif name=="AP":
        refs=["IR_RANK2008","SKLEARN_AP","SAITO2015","BEELINE2020"]
        result.update(definition="sum_j (Recall_j - Recall_(j-1)) * Precision_j",
                      biological_use="考察真边在整个排名中是否靠前；适合正边稀少的网络筛选。",
                      tie_policy="同分整体进入阈值；非插值 sklearn AP，非随机排列期望 AP，非 PR 梯形面积",
                      limitations="随正例占比和候选全集改变，不能跨不同全集直接比较；无正例返回 NA。")
    elif name=="AUROC":
        refs=["FAWCETT2006","SAITO2015"]
        result.update(definition="Pr(score_positive > score_negative) + 0.5*Pr(tie)",
                      biological_use="真边相对负边的总体排序区分能力。",tie_policy="正负同分贡献 0.5",
                      limitations="只有一种标签时为 NA；稀疏网络应同时报告 AP/precision，不单看 AUROC。")
    elif name=="Early_precision":
        refs=["BEELINE2020","TIES2008"]
        result.update(definition="Precision@k where k=number_of_positive_labels",
                      biological_use="评估排名前部约一个真实网络规模的候选质量。",
                      decision_rule="k=真值中的正例数，仅定义评价预算，不提供给拟合程序",tie_policy="边界同分取期望",
                      limitations="这是原始 early precision，不是除以随机参考的 EPR。真实未知网络大小不能拿来当部署阈值。")
    elif name=="Positive_prevalence":
        refs=["SAITO2015","FAWCETT2006"]
        result.update(kind="dataset_diagnostic",direction="not_a_model_quality_score",definition="positive_label_count / candidate_count",
                      biological_use="解释考卷稀疏程度及 AP 的背景。",tie_policy="不适用",
                      limitations="数据属性，同一任务/数据对所有方法相同；不能用它评判模型优劣。")
    elif name=="AP_over_prevalence":
        refs=["SAITO2015","BEELINE2020","SKLEARN_AP"]
        result.update(kind="project_derived_diagnostic",range="nonnegative, not bounded by 1",
                      definition="AP / Positive_prevalence",
                      biological_use="相对于数据正例占比的描述性归一化参考。",tie_policy="继承本项目 AP",
                      limitations="项目派生量，不冒充文献原定义的 AUPRC ratio；有限随机排列的期望 AP 不严格等于 prevalence。")
    elif name=="NDCG_linear":
        refs=["NDCG2002","TIES2008"]
        result.update(definition="DCG/ideal_DCG; DCG=sum_i max(true_score_i,0)/log2(i+1)",
                      biological_use="强下游贡献是否优先排在前面，帮助选择重点 LR 机制。",
                      tie_policy="同分组的平均线性增益乘该组折扣和",
                      limitations="仅 LR→gene 连续真值；此线性增益/折扣实例不是所有 NDCG 变体。按 receiver×target 计算后宏平均；全零真值为 NA。")
    elif name=="Spearman":
        refs=["SPEARMAN1904"]
        result.update(range="-1 to 1",definition="Pearson correlation of average ranks",
                      biological_use="预测贡献与真贡献的整体单调顺序是否一致。",tie_policy="并列赋平均秩",
                      limitations="仅 LR→gene 连续排名；常量数组为 NA；不表达作用方向或因果识别。")
    elif name=="Kendall_tau_b":
        refs=["KENDALL1945","SCIPY_RANK"]
        result.update(range="-1 to 1",definition="(concordant_pairs-discordant_pairs)/sqrt((C+D+T)*(C+D+U))",
                      biological_use="两两候选顺序的一致性，对大量并列作明确校正。",tie_policy="tau-b 并列校正",
                      limitations="仅 LR→gene 连续排名；常量数组为 NA。")
    else:
        raise ValueError(f"指标 {metric} 尚无定义/文献登记，拒绝导出无来源指标。")
    result["reference_ids"]=", ".join(refs)
    result["reference_urls"]="\n".join(REFERENCES[key][1] for key in refs)
    if pu:
        result["kind"]="positive_unlabeled_exploratory_"+result["kind"]
        result["limitations"]="PU 标签 0 是未标注，不是证实负例；只能解释为已知证据检索，不是真实准确率。 "+result["limitations"]
    return result

def catalog(names,ec):
    return pd.DataFrame([definition(str(name),ec) for name in names])
