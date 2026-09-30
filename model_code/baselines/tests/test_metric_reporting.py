from itertools import permutations
import numpy as np
import pandas as pd
import pytest
from independent_baselines.comparison import metrics,f_score
from independent_baselines.metric_references import definition,catalog,bibliography
from independent_baselines.metric_tables import export_metric_tables

def test_f1_perfect_and_empty_hits():
    m=metrics([1,0,0],[1,.2,.1],(1,2),decision_k=1)
    assert m["Precision"]==m["Recall"]==m["F1"]==1
    assert m["F1_at_2"]==pytest.approx(2/3)
    assert metrics([1,0],[0,1],(1,),decision_k=1)["F1"]==0

def test_fscore_tie_matches_all_permutations():
    # 第一条真边最高分，后三条同分中一条真，top-2 只剩一个位置。
    y=[1,1,0,0]
    scores=[.9,.5,.5,.5]
    m=metrics(y,scores,(2,),decision_k=2,f_betas=(.5,1,2))
    for beta in (.5,1,2):
        possible=[]
        for order in permutations([1,0,0]):
            tp=1+order[0]
            possible.append((1+beta**2)*tp/(beta**2*2+2))
        assert m[f"F{beta:g}"]==pytest.approx(np.mean(possible))
        assert m[f"F{beta:g}_at_2"]==pytest.approx(np.mean(possible))

def test_fscore_beta_changes_tradeoff():
    assert f_score(.8,.2,.5)>f_score(.8,.2,1)>f_score(.8,.2,2)

@pytest.mark.parametrize("beta",[0,-1,np.inf,np.nan])
def test_invalid_beta(beta):
    with pytest.raises(ValueError): metrics([1,0],[1,0],(1,),f_betas=(beta,))

@pytest.mark.parametrize("k",[0,-1,1.2,True])
def test_invalid_budget(k):
    with pytest.raises(ValueError): metrics([1,0],[1,0],(1,),decision_k=k)

def test_fscore_no_positive_policy():
    m=metrics([0,0],[1,0],(1,),decision_k=1)
    assert m["Precision"]==0
    assert np.isnan(m["Recall"]) and np.isnan(m["F1"])

def test_every_metric_has_defined_source(toy):
    dc,ec,root=toy
    all_names=list(metrics([1,0,1],[.9,.2,.2],ec.top_k,ec.classification_top_k,(.5,1,2)))
    all_names+= ["NDCG_linear","Spearman","Kendall_tau_b"]
    for name in all_names:
        item=definition(name,ec)
        assert item["definition"] and item["reference_ids"] and item["reference_urls"].startswith("http")
        assert "PU 标签" in definition("PU_"+name,ec)["limitations"]
    assert len(catalog(all_names,ec))==len(all_names)
    assert "EFRON1979" in set(bibliography().reference_id)
    with pytest.raises(ValueError): definition("Invented_metric",ec)

def test_derived_metric_not_claimed_original_paper(toy):
    dc,ec,root=toy
    d=definition("AP_over_prevalence",ec)
    assert d["kind"]=="project_derived_diagnostic"
    assert "项目派生量" in d["limitations"]

def test_per_metric_tables_both_orientations_and_missing(toy,tmp_path):
    dc,ec,root=toy
    ec.methods=("JumpNicheRewire","GENIE3","Missing")
    rows=[dict(dataset_id=d) for d in ("d1","d2")]
    records=[]
    audits=[]
    for d in ("d1","d2"):
        for method in ec.methods:
            audits.append(dict(dataset_id=d,method=method,state="missing" if method=="Missing" else "completed"))
            if method!="Missing":
                for task in ("GRN_global","CCI_cell"):
                    records.append(dict(dataset_id=d,method=method,task=task,metric="F1",value=.8 if method=="JumpNicheRewire" else .6))
    frame=pd.DataFrame(records)
    manifest=export_metric_tables(frame,pd.DataFrame(audits),rows,ec,tmp_path)
    assert len(manifest)==2 # 同名 F1 不能把 GRN 与 CCI 混在一起
    book=tmp_path/"by_metric/GRN_global/F1.xlsx"
    wide=pd.read_excel(book,sheet_name="by_dataset")
    assert list(wide)==["dataset_id",*ec.methods]
    assert len(wide)==2 and wide["Missing"].isna().all()
    summary=pd.read_excel(book,sheet_name="method_summary")
    assert list(summary)[:2]==["method","F1"]
    assert summary.loc[summary.method.eq("Missing"),"F1"].isna().all()
    common=pd.read_excel(book,sheet_name="matched_summary")
    assert common.common_datasets.eq(0).all() and common.F1.isna().all()
    refs=pd.read_excel(book,sheet_name="references")
    assert "IR2008" in set(refs.reference_id)
    assert (tmp_path/"by_metric/GRN_global/F1_summary.csv").is_file()

def test_matched_subset_is_identical_for_all_methods(toy,tmp_path):
    dc,ec,root=toy
    ec.methods=("A","B")
    rows=[dict(dataset_id=d) for d in ("d1","d2")]
    frame=pd.DataFrame([dict(dataset_id=d,method=m,task="GRN_global",metric="AP",value=v)
                        for d,m,v in (("d1","A",.8),("d1","B",.6),("d2","A",.9))])
    audit=pd.DataFrame([dict(dataset_id=d,method=m,state="completed") for d in ("d1","d2") for m in ec.methods])
    export_metric_tables(frame,audit,rows,ec,tmp_path)
    result=pd.read_csv(tmp_path/"by_metric/GRN_global/AP_matched.csv")
    assert result.common_datasets.tolist()==[1,1]
    assert result.AP.tolist()==[.8,.6]
    assert result.ci_lower.isna().all()

def test_ap_convention_is_not_random_permutation_ap(toy):
    dc,ec,root=toy
    assert "非随机排列期望 AP" in definition("AP",ec)["tie_policy"]
