"""同一真值宇宙、同分无偏排名、数据集配对统计；不调用任何模型。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon, spearmanr, kendalltau
from sklearn.metrics import average_precision_score, roc_auc_score
from .io import selected, inputs, mapped_path, json_write, digest, sha256, safe_name

def f_score(precision, recall, beta=1.0):
    """固定预测边数量下，同分随机排序的期望 F-beta。"""
    if not np.isfinite(beta) or beta<=0:
        raise ValueError("F-beta 的 beta 必须为有限正数")
    if not np.isfinite(precision) or not np.isfinite(recall):
        return np.nan
    denominator=beta**2*precision+recall
    return float((1+beta**2)*precision*recall/denominator) if denominator else 0.0

def check_budget(k):
    if isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or k<1:
        raise ValueError("top-k 必须为正整数，并在 test 前预先指定")
    return int(k)

def precision_at_k(labels,scores,k):
    labels,scores=np.asarray(labels),np.asarray(scores,dtype=float)
    k=min(k,len(labels))
    if k < 1:
        return np.nan,np.nan
    boundary=np.sort(scores)[-k]
    above,tied=scores>boundary,scores==boundary
    expected=labels[above].sum()+(k-above.sum())*labels[tied].mean()
    return float(expected/k),float(expected/labels.sum()) if labels.sum() else np.nan

def metrics(labels,scores,ks,decision_k=None,f_betas=(1.0,)):
    labels,scores=np.asarray(labels),np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all() or not np.isin(labels,[0,1]).all():
        raise ValueError("评价标签只能为0/1，分数必须有限")
    labels=labels.astype(int)
    betas=tuple(dict.fromkeys((1.0,)+tuple(float(b) for b in f_betas)))
    for beta in betas:
        if not np.isfinite(beta) or beta<=0:
            raise ValueError("f_betas 必须为有限正数")
    p=int(labels.sum())
    prevalence=float(labels.mean()) if len(labels) else np.nan
    result=dict(AP=float(average_precision_score(labels,scores)) if p else np.nan,
                AUROC=float(roc_auc_score(labels,scores)) if 0<p<len(labels) else np.nan,
                Positive_prevalence=prevalence)
    result["AP_over_prevalence"]=result["AP"]/prevalence if prevalence>0 else np.nan
    result["Early_precision"]=precision_at_k(labels,scores,p)[0] if p else np.nan
    for k in ks:
        check_budget(k)
        precision,recall=precision_at_k(labels,scores,k)
        result[f"Precision_at_{k}"]=precision
        result[f"Recall_at_{k}"]=recall
        for beta in betas:
            result[f"F{beta:g}_at_{k}"]=f_score(precision,recall,beta)
    if decision_k is not None:
        precision,recall=precision_at_k(labels,scores,check_budget(decision_k))
        result.update(Precision=precision,Recall=recall)
        for beta in betas:
            result[f"F{beta:g}"]=f_score(precision,recall,beta)
    return result

def ranking(relevance,scores):
    relevance,scores=np.asarray(relevance,dtype=float),np.asarray(scores,dtype=float)
    if not np.isfinite(relevance).all() or not np.isfinite(scores).all():
        raise ValueError("LR 排名输入非有限")
    # 连续非负贡献作为线性增益，避免 2**大数 溢出；不与指数增益版本混称。
    gains=np.maximum(relevance,0)
    discounts=1/np.log2(np.arange(2,len(gains)+2))
    order=np.argsort(-scores,kind="stable")
    ordered=scores[order]
    bounds=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1,len(order)]
    dcg=sum(gains[order[a:b]].mean()*discounts[a:b].sum() for a,b in zip(bounds[:-1],bounds[1:]))
    ideal=(np.sort(gains)[::-1]*discounts).sum()
    variable=len(np.unique(scores))>1 and len(np.unique(gains))>1
    return dict(NDCG_linear=float(dcg/ideal) if ideal>0 else np.nan,
                Spearman=float(spearmanr(gains,scores).statistic) if variable else np.nan,
                Kendall_tau_b=float(kendalltau(gains,scores).statistic) if variable else np.nan)

def table(root,name,limit):
    parquet=Path(root)/"parquet"/f"{name}.parquet"
    csv=Path(root)/f"{name}.csv"
    if parquet.is_file():
        import pyarrow.parquet as pq
        count=pq.ParquetFile(parquet).metadata.num_rows
        if count>limit:
            raise MemoryError(f"{parquet}: {count} 行超过评价内存保护 {limit}；请使用更大内存环境并显式修改上限。")
        frame=pd.read_parquet(parquet)
    elif csv.is_file():
        chunks=[]
        count=0
        for chunk in pd.read_csv(csv,chunksize=100000):
            count+=len(chunk)
            if count>limit:
                raise MemoryError(f"{csv}: exceeds maximum_prediction_rows")
            chunks.append(chunk)
        frame=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()
    else:
        return None
    if not frame.empty:
        if ("score" not in frame or not np.isfinite(pd.to_numeric(frame.score,errors="coerce")).all()
                or (pd.to_numeric(frame.score,errors="coerce")<0).any()):
            raise ValueError(f"{name}: 分数无效，不能静默丢弃")
    return frame

def global_grn(prediction, aggregation="max"):
    """Explicit region-free projection; one directed TF-target score per dataset.

    Do not average scores across METHODS. NaN signed effects are not zero.
    Cell/region predictions may be reduced by the prespecified rule; native
    global predictions must have unique TF-target keys, never duplicated regions.
    """
    if aggregation not in {"max", "mean"}:
        raise ValueError("global_grn_aggregation must be max or mean, fixed before evaluation")
    pred = prediction.copy()
    if not {"tf", "target", "score"}.issubset(pred):
        raise ValueError("GRN predictions require tf, target, score")
    for key in ("tf", "target"):
        if pred[key].isna().any():
            raise ValueError(f"Missing {key} in GRN predictions")
        pred[key] = pred[key].astype(str)
    scores = pd.to_numeric(pred.score, errors="raise")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("GRN scores must be finite and nonnegative")
    pred["score"] = scores
    pred = pred.loc[pred.tf.ne(pred.target)]
    if "unit_type" in pred and not pred.empty:
        kinds = set(pred.unit_type.dropna())
        if "global" in kinds and kinds != {"global"}:
            raise ValueError("Cannot mix global and cell/region predictions")
        if kinds == {"global"} and pred.duplicated(["tf", "target"]).any():
            raise ValueError("Global GRN contains duplicate TF-target pairs")
    return pred.groupby(["tf", "target"], as_index=False, sort=True).score.agg(aggregation)


def align(truth,prediction,keys):
    truth=truth.copy()
    if truth.duplicated(keys).any():
        raise ValueError(f"真值存在重复主键 {keys}")
    prediction=prediction.copy()
    for key in keys:
        truth[key]=truth[key].astype(str)
        prediction[key]=prediction[key].astype(str)
    pred=prediction.groupby(keys,as_index=False).score.max()
    result=truth.merge(pred,on=keys,how="left",validate="one_to_one")
    coverage=float(result.score.notna().mean()) if len(result) else np.nan
    result["score"]=result.score.fillna(0.0)
    return result,coverage

def truth_table(path,ec):
    chunks=[]
    count=0
    for chunk in pd.read_csv(path,chunksize=100000):
        count+=len(chunk)
        if count>ec.maximum_truth_rows:
            raise MemoryError(f"真值 {path} 超过 maximum_truth_rows；停止评价，不截断候选全集")
        chunks.append(chunk)
    return pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()

def evaluate_dataset(row,root,dc,ec):
    truth_dir=Path(row["truth_dir"])
    tasks=[]
    suite=dc.suite
    grn=table(root,"grn_edges",ec.maximum_prediction_rows) if suite!="sccigen" else None
    cci=table(root,"cci_edges",ec.maximum_prediction_rows) if suite in {"sccigen","scmultisim","visium","core_final"} else None
    lr_gene=table(root,"lr_gene",ec.maximum_prediction_rows) if suite=="core_final" else None
    # 统一候选全集由真值文件定义；缺失预测计0，绝不只评价“方法输出的边”。
    if grn is not None:
        filename="truth_pseudo_grn.csv" if suite=="visium" else (
            "truth_perturbation_grn.csv" if suite=="perturbation" else "truth_grn.csv")
        if not (truth_dir/filename).is_file():
            raise FileNotFoundError(truth_dir/filename)
        if (truth_dir/filename).is_file():
            truth=truth_table(truth_dir/filename,ec)
            global_truth=truth.loc[truth.tf.astype(str).ne(truth.target.astype(str))].groupby(["tf","target"],as_index=False).label.max()
            aggregation = getattr(ec, "global_grn_aggregation", "max")
            global_prediction = global_grn(grn, aggregation)
            aligned,coverage=align(global_truth,global_prediction,["tf","target"])
            tasks.append(("GRN_global",aligned,coverage))
            if suite=="core_final" and "cell_id" in truth:
                explicitly_global = ("unit_type" in grn and grn.unit_type.eq("global").all())
                cellwise="cell_id" in grn and grn.cell_id.notna().all() and not explicitly_global
                if cellwise:
                    aligned,coverage=align(truth,grn,["cell_id","tf","target"])
                else:
                    pred=grn.groupby(["tf","target"],as_index=False).score.max()
                    aligned=truth.merge(pred,on=["tf","target"],how="left",validate="many_to_one")
                    coverage=float(aligned.score.notna().mean())
                    aligned["score"]=aligned.score.fillna(0)
                tasks.append(("GRN_cell_common_universe",aligned,coverage))
    if cci is not None and suite in {"sccigen", "scmultisim"}:
        keys=["sender_cell_type","receiver_cell_type","ligand","receptor"]
        # 只有总细胞邻接，没有 LR 身份的 CLARIFY 不支持该任务。
        if {"ligand","receptor"}.issubset(cci.columns):
            pred=cci[cci.get("lr_id",pd.Series("",index=cci.index)).ne("__all__")].copy()
            if not {"sender_cell_type","receiver_cell_type"}.issubset(pred.columns):
                meta=pd.read_csv(Path(row["dataset_dir"])/"metadata.csv",dtype=str).set_index("cell_id")
                pred["sender_cell_type"]=pred.sender_id.astype(str).map(meta[dc.cell_type_column])
                pred["receiver_cell_type"]=pred.receiver_id.astype(str).map(meta[dc.cell_type_column])
                if pred[keys[:2]].isna().any().any():
                    raise ValueError("CCI 细胞 ID 不能匹配 metadata")
            truth=truth_table(truth_dir/("truth_scmultisim_cci.csv" if suite=="scmultisim" else "truth_sccigen_cci.csv"),ec)
            aligned,coverage=align(truth,pred,keys)
            tasks.append(("scMultiSim_type_LR" if suite=="scmultisim" else "sCCIgen_cross_cell_gene_pair",aligned,coverage))
    if cci is not None and suite=="visium" and {"ligand","receptor"}.issubset(cci.columns):
        pred=cci.copy()
        pred["lr_id"]=pred.ligand.astype(str)+"__"+pred.receptor.astype(str)
        truth=truth_table(truth_dir/"truth_pseudo_lr.csv",ec)
        aligned,coverage=align(truth,pred,["lr_id"])
        tasks.append(("Curated_LR_retrieval",aligned,coverage))
    if cci is not None and suite=="core_final" and {"sender_id","receiver_id"}.issubset(cci.columns):
        pred=cci.copy()
        if "lr_id" in pred and pred.lr_id.eq("__all__").any():
            pred=pred[pred.lr_id.eq("__all__")]
        else:
            pred=pred.groupby(["sender_id","receiver_id"],as_index=False).score.sum()
        aligned,coverage=align(truth_table(truth_dir/"truth_cci.csv",ec),pred,["sender_id","receiver_id"])
        tasks.append(("CCI_cell",aligned,coverage))
    if lr_gene is not None and suite=="core_final":
        truth=truth_table(truth_dir/"truth_lr_gene.csv",ec)
        aligned,coverage=align(truth,lr_gene,["receiver_id","lr_id","target"])
        tasks.append(("LR_gene_ranking",aligned,coverage))
    records=[]
    for task,aligned,coverage in tasks:
        if task=="LR_gene_ranking":
            ranks=[ranking(group.true_score,group.score) for _,group in aligned.groupby(["receiver_id","target"])]
            values=pd.DataFrame(ranks).mean().to_dict()
        else:
            values=metrics(aligned.label,aligned.score,ec.top_k,ec.classification_top_k,ec.f_betas)
        prefix="PU_" if row.get("truth_kind")=="positive_unlabeled" else ""
        for metric,value in values.items():
            records.append(dict(task=task,metric=prefix+metric,value=value,
                                prediction_unit="global" if task=="GRN_global" else "task_specific",
                                prediction_aggregation=getattr(ec,"global_grn_aggregation","max") if task=="GRN_global" else "task_specific",
                                matching_keys="tf,target" if task=="GRN_global" else "task_specific",
                                candidate_count=len(aligned),prediction_coverage=coverage,
                                classification_rule="not_applicable" if task=="LR_gene_ranking" else "fixed_top_k",
                                classification_top_k=np.nan if task=="LR_gene_ranking" else ec.classification_top_k))
    return records

def mean_interval(values,ec,rng):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if len(values)<2:
        return np.nan,np.nan
    draws=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(ec.bootstrap_iterations)])
    alpha=(1-ec.confidence_level)/2
    return tuple(np.quantile(draws,[alpha,1-alpha]))

def summarize(frame,ec):
    columns=["method","task","metric","mean","ci_lower","ci_upper","datasets"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows=[]
    for keys,group in frame.groupby(["method","task","metric"],sort=True):
        # 相同 task 的方法使用相同 seed 流；数据集/模拟种子才是重复单位。
        rng=np.random.default_rng(ec.random_seed)
        values=group.value.dropna().to_numpy()
        low,high=mean_interval(values,ec,rng)
        rows.append(dict(zip(columns,[*keys,float(values.mean()) if len(values) else np.nan,low,high,len(values)])))
    return pd.DataFrame(rows,columns=columns)

def paired(frame,ec):
    rows=[]
    if frame.empty:
        return pd.DataFrame(columns=["method","task","metric","pairs","mean_delta","p_value","p_holm"])
    for (task,metric),group in frame.groupby(["task","metric"]):
        if metric not in ec.primary_metrics:  # PU 必须明确预先选择，不能自动宣称正式优越
            continue
        wide=group.pivot(index="dataset_id",columns="method",values="value")
        if "JumpNicheRewire" not in wide:
            continue
        for method in wide.columns:
            if method=="JumpNicheRewire":
                continue
            common=wide[["JumpNicheRewire",method]].dropna()
            diff=(common["JumpNicheRewire"]-common[method]).to_numpy()
            n=len(diff)
            p=(1.0 if n and np.all(diff==0) else float(wilcoxon(diff,alternative="two-sided").pvalue)) if n>=2 else np.nan
            low,high=mean_interval(diff,ec,np.random.default_rng(ec.random_seed))
            rows.append(dict(method=method,task=task,metric=metric,pairs=n,
                             mean_delta=float(diff.mean()) if n else np.nan,
                             delta_ci_lower=low,delta_ci_upper=high,p_value=p))
    out=pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"]=np.nan
    valid=out.p_value.dropna().sort_values()
    adjusted=np.maximum.accumulate(np.minimum(1,valid.to_numpy()*np.arange(len(valid),0,-1)))
    out.loc[valid.index,"p_holm"]=adjusted
    return out

def our_root(row,dc,ec):
    if dc.suite=="core_final":
        lock=json.loads((ec.our_formal_root/"frozen_parameters.json").read_text(encoding="utf-8"))
        return ec.our_formal_root/"jumpniche"/lock["parameter_id"]/row["dataset_id"]
    return getattr(ec,f"our_{dc.suite}_root")/row["dataset_id"]

def evaluate(dc,ec):
    from .metric_tables import export_metric_tables
    from .metric_references import catalog, bibliography
    check_budget(ec.classification_top_k)
    for k in ec.top_k:
        check_budget(k)
    rows=selected(dc,evaluation=True)
    output=dc.output_root/dc.suite/"03_evaluation"/safe_name(ec.analysis_name)
    if output.exists():
        raise FileExistsError(f"评价输出已存在：{output}；修改 analysis_name，不覆盖旧评价。")
    audit,metrics_rows,provenance=[],[],[]
    for row in rows:
        for method in ec.methods:
            root=our_root(row,dc,ec) if method=="JumpNicheRewire" else dc.output_root/dc.suite/"02_methods"/method/row["dataset_id"]
            status_file=root/"run_status.json"
            record=dict(dataset_id=row["dataset_id"],method=method,root=str(root),state="missing")
            try:
                status=json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {}
                record.update(state=status.get("state","missing"),runtime_seconds=status.get("runtime_seconds"),
                              peak_rss_bytes=status.get("peak_rss_bytes"),error_message=status.get("error_message"),
                              convergence_passed=status.get("convergence_passed"),
                              protocol_signature=status.get("protocol_signature"))
                if record["state"]!="completed":
                    audit.append(record)
                    continue
                if method=="JumpNicheRewire" and ec.require_our_convergence and status.get("convergence_passed") is not True:
                    record.update(state="convergence_not_passed")
                    audit.append(record)
                    continue
                scores=evaluate_dataset(row,root,dc,ec)
                if not scores:
                    record.update(state="no_compatible_task")
                predictions=list((root/"parquet").glob("*.parquet"))
                predictions += [root/f"{name}.csv" for name in ("grn_edges","cci_edges","lr_gene") if (root/f"{name}.csv").is_file()]
                provenance_record=dict(dataset_id=row["dataset_id"],method=method,status_sha256=sha256(status_file),
                                       table_files={str(p):sha256(p) for p in sorted(predictions)},
                                       truth_files={str(p):sha256(p) for p in sorted(Path(row["truth_dir"]).glob("truth_*.csv"))})
                for score in scores:
                    metrics_rows.append(dict(dataset_id=row["dataset_id"],scenario=row["scenario"],
                                             method=method,truth_kind=row["truth_kind"],**score))
                provenance.append(provenance_record)
            except Exception as error:
                record.update(state="evaluation_failed",error_message=str(error))
            audit.append(record)
    output.mkdir(parents=True)
    frame=pd.DataFrame(metrics_rows)
    summary=summarize(frame,ec)
    tests=paired(frame,ec)
    audit_frame=pd.DataFrame(audit)
    references=catalog(frame.metric.unique() if not frame.empty else [],ec)
    sources=bibliography()
    for name,table_frame in (("dataset_metrics",frame),("summary",summary),("paired_tests",tests),("run_audit",audit_frame)):
        table_frame.to_csv(output/f"{name}.csv",index=False)
    with pd.ExcelWriter(output/"comparison.xlsx",engine="openpyxl") as writer:
        for name,table_frame in (("dataset_metrics",frame),("summary",summary),("paired_tests",tests),("run_audit",audit_frame),
                                 ("metric_definitions",references),("references",sources)):
            table_frame.to_excel(writer,sheet_name=name,index=False)
    references.to_csv(output/"metric_definitions.csv",index=False,encoding="utf-8-sig")
    sources.to_csv(output/"metric_references.csv",index=False,encoding="utf-8-sig")
    if ec.export_per_metric_tables:
        export_metric_tables(frame,audit_frame,rows,ec,output)
    json_write(output/"evaluation_provenance.json",dict(data_config=dc.__dict__,evaluation_config=ec.__dict__,
                                                       runs=provenance,truth_only_read_in_evaluation=True,
                                                       evaluation_code_sha256={name:sha256(Path(__file__).parent/name)
                                                         for name in ("comparison.py","metric_references.py","metric_tables.py")}))
    plot(frame,output,ec)
    missing=audit_frame[~audit_frame.state.eq("completed")]
    notes=["# 独立方法比较报告", "",
           f"数据类别：{dc.suite}；所选数据集：{len(rows)}；所选方法：{len(ec.methods)}。",
           f"未完成或不兼容的运行条目：{len(missing)}。缺失/不支持任务没有记零分。",
           "完整指标在 comparison.xlsx；优先查看 run_audit，再看 AP 与 paired_tests。",
           "AP=按分数阈值整体纳入同分组；top-k 同分边界按期望真边数处理。",
           f"Precision/Recall/F1 的预定判边规则：最高分 {ec.classification_top_k} 条；同分取期望，未使用 test 寻优。",
           "by_metric/<task>/<metric>.xlsx 为逐指标独立表；by_dataset 是方法为列，method_summary 是方法与同一指标值两列。",
           "每份指标表附定义、参考文献、缺失审计；matched_summary 只使用所选全部方法共同有效的数据集。",
           "配对检验只使用相同数据集且双方成功的运行；同时报告缺失，不能以成功子集代表全部表现。",
           "95% CI 跨数据集/种子重采样，不把数百万条边当数百万个生物学重复。",
           "SERGIO 的五种子共用拓扑，不能代表五种独立网络；单个 sCCIgen/Visium 无跨样本置信区间。",
           "本表不含无法由该方法输出的因果等级、干预效应或受体机制链，不能据此声称完整 benchmark 已完成。"]
    if dc.suite=="visium":
        notes+=["", "真实数据结果为 PU 检索：未标注边不是已证实负例。",
                f"资源独立性验收标志：{ec.real_prior_independence_verified}。必须另外核对 motif/LR/TF-target 先验和标签的重叠；"
                "本程序不会把资源重叠结果升级为独立泛化证据。"]
    if dc.suite=="scmultisim":
        notes += ["", "scMultiSim：当前种子共用官方 GRN 模板，不代表独立网络拓扑重复。",
                  "GRN_global 评价 TF→gene；scMultiSim_type_LR 评价细胞类型对×合成 LR 支持关系。",
                  "CCI 标签是生成器允许的类型级相互作用，不是每对空间邻居实际发生通信的标签。",
                  "合成 SYN_L/SYN_R 不是真实基因符号；不能用真实 motif/原生 LR 资源强行映射。",
                  "不包含完整 L–R–TF–gene 机制链或因果证据等级的真值。GRN 与 CCI 请分别选支持该任务的方法评价。"]
    if not missing.empty:
        notes+=["","存在缺失方法/失败任务，当前比较不完整；不能只看平均分下结论。"]
    (output/"REPORT.md").write_text("\n\n".join(notes),encoding="utf-8")
    return output

def plot(frame,output,ec):
    if frame.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (task,metric),group in frame.groupby(["task","metric"]):
        if metric not in {"AP","PU_AP","NDCG_linear"}:
            continue
        wide=group.pivot(index="dataset_id",columns="method",values="value")
        fig,ax=plt.subplots(figsize=(max(6,len(wide.columns)*1.3),4.5))
        x=np.arange(len(wide.columns))
        for _,values in wide.iterrows():
            ax.plot(x,values.to_numpy(),marker="o",alpha=0.5)
        ax.set_xticks(x,wide.columns,rotation=25,ha="right")
        ax.set_title(task+" / "+metric+" (paired datasets)")
        ax.set_ylabel(metric)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(output/f"{task}_{metric}_paired.png",dpi=180)
        fig.savefig(output/f"{task}_{metric}_paired.pdf")
        plt.close(fig)
