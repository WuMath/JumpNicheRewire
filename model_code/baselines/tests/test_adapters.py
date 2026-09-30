"""模拟官方 API 的契约测试，不宣称真的训练了第三方模型。"""
import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from independent_baselines import adapters,io,runtime

class Sink:
    def __init__(self): self.frames={}
    def write(self,name,chunks): self.frames[name]=pd.concat(list(chunks),ignore_index=True)

@pytest.mark.parametrize("name",["genie3","grnboost2"])
def test_tree_official_call(toy,monkeypatch,tmp_path,name):
    from baseline_settings.genie3 import MethodConfig
    dc,ec,root=toy
    row=io.selected(dc)[0]
    data=io.inputs(row,dc)
    seen=[]
    def algorithm(**kw):
        seen.append(kw)
        assert kw["tf_names"]==["TF1"]
        return pd.DataFrame(dict(TF=["TF1"],target=["G2"],importance=[2.5]))
    class Context:
        def __init__(self,*a,**kw): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
    monkeypatch.setitem(sys.modules,"arboreto",SimpleNamespace())
    monkeypatch.setitem(sys.modules,"arboreto.algo",SimpleNamespace(genie3=algorithm,grnboost2=algorithm))
    monkeypatch.setitem(sys.modules,"distributed",SimpleNamespace(Client=Context,LocalCluster=Context))
    writer=Sink()
    adapters.tree(name,MethodConfig(),dc,row,data,tmp_path,writer)
    assert len(seen)==1
    assert writer.frames["grn_edges"].score.tolist()==[2.5]

def test_cespgrn_shape_and_official_kernel(toy,monkeypatch,tmp_path):
    from baseline_settings.cespgrn import MethodConfig
    dc,ec,root=toy
    row=io.selected(dc)[0]
    data=io.inputs(row,dc)
    def kernel(latent,**kw):
        assert latent.shape[0]==3 and kw["truncate"]
        return np.eye(3),np.eye(3)
    class Model:
        def __init__(self,**kw): assert kw["X"].shape==(3,1,3)
        def train(self,**kw):
            assert kw["lamb"]==.1
            return np.full((3,3,3),-.2)
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(device=lambda value:value))
    monkeypatch.setitem(sys.modules,"kernel",SimpleNamespace(calc_kernel_neigh=kernel))
    monkeypatch.setitem(sys.modules,"g_admm",SimpleNamespace(est_cov=lambda **kw:np.eye(3),G_admm_minibatch=Model))
    writer=Sink()
    adapters.cespgrn(MethodConfig(),dc,row,data,tmp_path,writer)
    assert len(writer.frames["grn_edges"])==6
    assert (tmp_path/"official_graphs.npy").is_file()
    with pytest.raises(ValueError,match="布局"):
        adapters.cespgrn(MethodConfig(kernel_input="spatial"),dc,row,data,tmp_path,writer)

def test_liana_rank_inversion_and_matched_resource(toy,monkeypatch,tmp_path):
    from baseline_settings.liana import MethodConfig
    dc,ec,root=toy
    row=io.selected(dc)[0]
    data=io.inputs(row,dc)
    obj=SimpleNamespace(uns={})
    def infer(obj,**kw):
        assert kw["resource_name"] is None
        assert kw["resource"].ligand.tolist()==["G2"]
        obj.uns["liana_res"]=pd.DataFrame(dict(source=["A"],target=["B"],ligand_complex=["G2"],receptor_complex=["G3"],magnitude_rank=[.02]))
    monkeypatch.setitem(sys.modules,"liana",SimpleNamespace(mt=SimpleNamespace(rank_aggregate=infer)))
    monkeypatch.setattr(adapters,"ann_data",lambda *a,**kw:obj)
    writer=Sink()
    adapters.liana(MethodConfig(),dc,row,data,tmp_path,writer)
    assert writer.frames["cci_edges"].score.iloc[0]==pytest.approx(.98)

def test_spagrn_current_api_and_regulons(toy,monkeypatch,tmp_path):
    from baseline_settings.spagrn import MethodConfig
    dc,ec,root=toy
    row=io.selected(dc)[0]
    data=io.inputs(row,dc)
    with pytest.raises(ValueError,match="cisTarget"):
        adapters.spagrn(MethodConfig(),dc,row,data,tmp_path,Sink())
    row["species"]="mouse"
    obj=SimpleNamespace(X=np.zeros((3,3)),layers={},write_h5ad=lambda path:None)
    class Model:
        def __init__(self,obj,project_name):
            self.regulons=[SimpleNamespace(transcription_factor="TF1",gene2weight={"G2":.8})]
        def infer(self,**kw):
            assert kw["layers"]=="raw_counts" and kw["model"]=="danb"
            assert not {"method","prefix","c_threshold"}.intersection(kw)
    monkeypatch.setattr(adapters,"ann_data",lambda *a,**kw:obj)
    monkeypatch.setitem(sys.modules,"spagrn",SimpleNamespace())
    monkeypatch.setitem(sys.modules,"spagrn.regulatory_network",SimpleNamespace(InferRegulatoryNetwork=Model))
    writer=Sink()
    adapters.spagrn(MethodConfig(),dc,row,data,tmp_path,writer)
    assert writer.frames["grn_edges"].score.tolist()==[.8]

def test_cpdb_mouse_blocked_before_import(toy,tmp_path):
    from baseline_settings.cellphonedb import MethodConfig
    dc,ec,root=toy
    row=io.selected(dc)[0]
    row["species"]="mouse"
    with pytest.raises(ValueError,match="人类"):
        adapters.cellphonedb(MethodConfig(),dc,row,io.inputs(row,dc),tmp_path,Sink())

def test_clarify_synthetic_blocked(toy,monkeypatch,tmp_path):
    from baseline_settings.clarify import MethodConfig
    dc,ec,root=toy
    row=io.selected(dc)[0]
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace())
    with pytest.raises(ValueError,match="分支未实现"):
        adapters.clarify(MethodConfig(),dc,row,io.inputs(row,dc),tmp_path,Sink())

def test_real_spawn_failure_isolated(toy,monkeypatch):
    from baseline_settings.genie3 import MethodConfig
    dc,ec,root=toy
    frame=pd.read_csv(dc.evidence_registry)
    second=frame.copy()
    second["dataset_id"]="d2"
    pd.concat([frame,second]).to_csv(dc.evidence_registry,index=False)
    dc.dataset_ids=("d1","d2")
    monkeypatch.setattr(runtime,"environment",lambda *a:{"test":"1"})
    cfg=MethodConfig(method="TEST",mode="freeze")
    assert runtime.launch("intentional_missing_adapter",cfg,dc)==0
    cfg.mode="run"
    # 子进程中有意调用不存在的 adapter；无任何第三方拟合。
    assert runtime.launch("intentional_missing_adapter",cfg,dc)==1
    statuses=[json.loads((dc.output_root/"sergio/02_methods/TEST"/d/"run_status.json").read_text(encoding="utf-8")) for d in dc.dataset_ids]
    assert all(s["state"]=="failed" for s in statuses)
    assert statuses[0]["worker_pid"]!=statuses[1]["worker_pid"]
    assert all(s["truth_read_during_inference"] is False for s in statuses)
