import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from independent_baselines import io
from independent_baselines.comparison import metrics,precision_at_k,ranking,table,align,evaluate_dataset,paired,evaluate,truth_table
from independent_baselines.adapters import cpdb_table,_grn_chunks,_budget
from independent_baselines.runtime import protocol,launch
from baseline_settings.genie3 import MethodConfig

def test_whitelist_alignment(toy,monkeypatch):
    dc,ec,root=toy
    row=io.selected(dc)[0]
    assert "truth_dir" not in row and "signature" not in row
    original=pd.read_csv
    reads=[]
    def guarded(path,*a,**kw):
        reads.append(str(path))
        assert "truth" not in str(path)
        return original(path,*a,**kw)
    monkeypatch.setattr(pd,"read_csv",guarded)
    data=io.inputs(row,dc)
    assert data["metadata"].cell_type.tolist()==["A","B","B"]
    assert data["tfs"]==["TF1"]
    assert len(io.simple_lr(data))==1
    assert np.isfinite(data["log"]).all().all()
    assert len(reads)==5

@pytest.mark.parametrize("bad",[-1,np.nan,np.inf])
def test_bad_expression_rejected(toy,bad):
    dc,ec,root=toy
    frame=pd.read_csv(root/"expression.csv",index_col=0).astype(float)
    frame.iloc[0,0]=bad
    frame.to_csv(root/"expression.csv")
    with pytest.raises(ValueError): io.inputs(io.selected(dc)[0],dc)

def test_log_already_normalized(toy):
    dc,ec,root=toy
    dc.expression_scale="log1p_normalized"
    data=io.inputs(io.selected(dc)[0],dc)
    np.testing.assert_array_equal(data["log"],data["raw"])

def test_panel_changes_all_methods_same_input(toy):
    dc,ec,root=toy
    dc.gene_panel_file=root/"panel.csv"
    pd.DataFrame(dict(gene=["TF1","G3"])).to_csv(dc.gene_panel_file,index=False)
    data=io.inputs(io.selected(dc)[0],dc)
    assert list(data["raw"])==["TF1","G3"]
    assert io.simple_lr(data).empty

def test_test_gate_pilot_isolation(toy):
    dc,ec,root=toy
    dc.suite="core_final"
    dc.formal_registry=dc.evidence_registry
    pd.DataFrame([dict(dataset_id="d1",dataset_dir=str(root),split="test",scenario="clean_500",benchmark_profile="final")]).to_csv(dc.formal_registry,index=False)
    dc.split="test"
    with pytest.raises(PermissionError): io.selected(dc)
    dc.unlock_test=True
    assert len(io.selected(dc))==1
    frame=pd.read_csv(dc.formal_registry)
    frame["benchmark_profile"]="pilot"
    frame.to_csv(dc.formal_registry,index=False)
    with pytest.raises(ValueError,match="pilot/final"): io.selected(dc)

def test_merfish_excluded(toy):
    dc,ec,root=toy
    frame=pd.read_csv(dc.evidence_registry)
    frame["dataset_id"]="merfish_d1"
    frame.to_csv(dc.evidence_registry,index=False)
    dc.dataset_ids=("merfish_d1",)
    with pytest.raises(ValueError,match="MERFISH"): io.selected(dc)

def test_perturbation_guard(toy):
    dc,ec,root=toy
    dc.suite="perturbation"
    with pytest.raises(ValueError,match="扰动协议"): io.selected(dc)

def test_path_map_boundary(toy):
    dc,ec,root=toy
    dc.path_prefix_from="C:/Old"
    dc.path_prefix_to="/srv/new"
    assert str(io.mapped_path("C:/Old/x",dc)).replace("\\","/")=="/srv/new/x"
    assert str(io.mapped_path("C:/Older/x",dc)).replace("\\","/")=="C:/Older/x"

@pytest.mark.parametrize("name",["..",".","../escape","a/b","a\\b"])
def test_name_safety(name):
    with pytest.raises(ValueError): io.safe_name(name)

def test_all_ties_are_order_invariant():
    labels=np.array([1,0,1,0,0])
    scores=np.ones(5)
    first=metrics(labels,scores,(1,3))
    assert first["AP"]==pytest.approx(.4)
    assert first["AUROC"]==pytest.approx(.5)
    assert first["Precision_at_1"]==pytest.approx(.4)
    assert first==metrics(labels[::-1],scores,(1,3))

def test_partial_boundary_tie():
    p,r=precision_at_k([1,1,0,0],[.9,.5,.5,.5],2)
    assert p==pytest.approx((1+1/3)/2)
    assert r==pytest.approx((1+1/3)/2)

def test_no_positive_na_not_zero():
    result=metrics([0,0],[.1,.2],(1,))
    assert np.isnan(result["AP"]) and np.isnan(result["AUROC"])

def test_fractional_labels_not_silently_cast():
    with pytest.raises(ValueError): metrics([1,.5],[.2,.3],(1,))

def test_rank_tie_expected_discount():
    result=ranking([2,0],[.5,.5])
    assert result["NDCG_linear"]==pytest.approx((1+1/np.log2(3))/2)
    assert np.isnan(result["Spearman"])
    assert ranking([0,2],[.5,.5])["NDCG_linear"]==result["NDCG_linear"]

def test_streaming_full_table_not_excel_preview(tmp_path):
    writer=io.TableWriter(tmp_path,"TEST",preview_rows=2)
    def chunks():
        for i in range(4):
            yield pd.DataFrame(dict(tf=["A"]*3,target=[str(i*3+j) for j in range(3)],score=[.1,.2,.3]))
    writer.write("grn_edges",chunks())
    workbook=writer.finish()
    assert len(table(tmp_path,"grn_edges",20))==12
    assert len(pd.read_excel(workbook,sheet_name="grn_edges"))==2
    assert pd.read_excel(workbook,sheet_name="manifest").rows.iloc[0]==12
    with pytest.raises(MemoryError): table(tmp_path,"grn_edges",10)

def test_bad_score_not_committed(tmp_path):
    writer=io.TableWriter(tmp_path,"TEST")
    with pytest.raises(ValueError): writer.write("grn_edges",[pd.DataFrame(dict(score=[np.nan]))])
    assert not (tmp_path/"parquet/grn_edges.parquet").exists()

def test_universe_and_missing_predictions():
    truth=pd.DataFrame(dict(tf=["A","A"],target=["B","C"],label=[1,0]))
    pred=pd.DataFrame(dict(tf=["A","A","OTHER"],target=["B","B","G"],score=[.5,.8,5]))
    result,coverage=align(truth,pred,["tf","target"])
    assert result.score.tolist()==[.8,0]
    assert coverage==.5 and len(result)==2

def test_duplicate_truth_fails():
    truth=pd.DataFrame(dict(tf=["A","A"],target=["B","B"],label=[1,1]))
    with pytest.raises(ValueError): align(truth,truth.assign(score=1),["tf","target"])

def test_cpdb_orientation_and_complex_exclusion():
    frame=pd.DataFrame(dict(gene_a=["L","R","A",None],gene_b=["R","L","B","R"],
                            receptor_a=[False,True,True,False],receptor_b=[True,False,True,True],
                            **{"T1|T2":[1.,2.,3.,4.]}))
    out=cpdb_table(frame)
    assert len(out)==2
    assert out.sender_cell_type.tolist()==["T1","T2"]
    assert out.receiver_cell_type.tolist()==["T2","T1"]
    assert out.lr_id.tolist()==["L__R","L__R"]

def test_grn_chunk_no_sign_claim(toy):
    dc,ec,root=toy
    data=io.inputs(io.selected(dc)[0],dc)
    frames=list(_grn_chunks(np.full((3,3,3),-.7),data))
    assert len(frames)==3 and all(len(f)==2 for f in frames)
    assert frames[0].score.eq(.7).all()
    assert frames[0].signed_score.isna().all()

def test_memory_guard():
    with pytest.raises(MemoryError): _budget(9*1024**3,8,"test")

def test_truth_memory_guard(toy):
    dc,ec,root=toy
    ec.maximum_truth_rows=1
    with pytest.raises(MemoryError): truth_table(root/"truth/truth_grn.csv",ec)

def test_global_prediction_broadcast_cell_truth(toy,tmp_path):
    dc,ec,root=toy
    row=io.selected(dc,evaluation=True)[0]
    dc.suite="core_final"
    pd.DataFrame(dict(cell_id=["c1","c1","c2","c2"],tf=["TF1"]*4,target=["G2","G3"]*2,label=[1,0,0,1])).to_csv(root/"truth/truth_grn.csv",index=False)
    out=tmp_path/"pred"
    w=io.TableWriter(out,"M")
    w.write("grn_edges",[pd.DataFrame(dict(tf=["TF1","TF1"],target=["G2","G3"],score=[.8,.1]))])
    scores=evaluate_dataset(row,out,dc,ec)
    score=next(s for s in scores if s["task"]=="GRN_cell_common_universe" and s["metric"]=="AP")
    assert score["candidate_count"]==4
    assert score["value"]==.5

def test_sccigen_cell_to_type_mapping(toy,tmp_path):
    dc,ec,root=toy
    row=io.selected(dc,evaluation=True)[0]
    dc.suite="sccigen"
    pd.DataFrame(dict(sender_cell_type=["A","B"],receiver_cell_type=["B","A"],ligand=["G2"]*2,receptor=["G3"]*2,label=[1,0])).to_csv(root/"truth/truth_sccigen_cci.csv",index=False)
    out=tmp_path/"pred"
    w=io.TableWriter(out,"M")
    w.write("cci_edges",[pd.DataFrame(dict(sender_id=["c1"],receiver_id=["c2"],ligand=["G2"],receptor=["G3"],lr_id=["G2__G3"],score=[1.]))])
    scores=evaluate_dataset(row,out,dc,ec)
    assert next(s["value"] for s in scores if s["metric"]=="AP")==1

def test_paired_intersection_and_holm(toy):
    dc,ec,root=toy
    records=[]
    for dataset,ours,other in [("d1",.9,.4),("d2",.8,.3),("d3",.7,np.nan)]:
        for method,value in [("JumpNicheRewire",ours),("M",other)]:
            records.append(dict(dataset_id=dataset,method=method,value=value,task="GRN_global",metric="AP"))
    result=paired(pd.DataFrame(records),ec)
    assert result.pairs.iloc[0]==2
    assert result.mean_delta.iloc[0]==pytest.approx(.5)
    assert result.p_holm.iloc[0]>=result.p_value.iloc[0]

def test_evaluator_missing_not_zero_and_no_overwrite(toy):
    dc,ec,root=toy
    ec.methods=("JumpNicheRewire","Missing")
    out=ec.our_sergio_root/"d1"
    w=io.TableWriter(out,"JumpNicheRewire")
    w.write("grn_edges",[pd.DataFrame(dict(tf=["TF1","TF1"],target=["G2","G3"],score=[1.,0.]))])
    io.json_write(out/"run_status.json",dict(state="completed",convergence_passed=True))
    result=evaluate(dc,ec)
    frame=pd.read_csv(result/"dataset_metrics.csv")
    audit=pd.read_csv(result/"run_audit.csv")
    assert set(frame.method)=={"JumpNicheRewire"}
    assert audit.loc[audit.method.eq("Missing"),"state"].iloc[0]=="missing"
    assert (result/"GRN_global_AP_paired.png").is_file()
    with pytest.raises(FileExistsError): evaluate(dc,ec)

def test_protocol_freezes_parameters_not_output_mode(toy):
    dc,ec,root=toy
    cfg=MethodConfig()
    first=protocol("genie3",cfg,dc,{"fake":"1"})
    cfg.mode="run"
    assert io.digest(first)==io.digest(protocol("genie3",cfg,dc,{"fake":"1"}))
    cfg.n_jobs+=1
    assert io.digest(first)!=io.digest(protocol("genie3",cfg,dc,{"fake":"1"}))

def test_freeze_and_no_silent_refreeze(toy,monkeypatch):
    from independent_baselines import runtime
    dc,ec,root=toy
    monkeypatch.setattr(runtime,"environment",lambda *args:{"fake":"1"})
    cfg=MethodConfig(mode="freeze")
    assert launch("genie3",cfg,dc)==0
    cfg.n_jobs+=1
    with pytest.raises(FileExistsError): launch("genie3",cfg,dc)

def test_source_independence_and_syntax():
    root=Path(__file__).resolve().parents[2]
    for path in (root/"src/independent_baselines").glob("*.py"):
        source=path.read_text(encoding="utf-8")
        ast.parse(source)
        assert "from jump_niche_rewire" not in source
        assert "JNR_EVIDENCE_WORKER_REQUEST" not in source
    for path in (root/"baselines").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
