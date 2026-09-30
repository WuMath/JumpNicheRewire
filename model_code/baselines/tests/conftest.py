from pathlib import Path
import sys
import pytest
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"baselines"))

@pytest.fixture
def toy(tmp_path):
    from baseline_settings.datasets import DatasetConfig
    from baseline_settings.evaluation import EvaluationConfig
    data=tmp_path/"dataset"
    data.mkdir()
    pd.DataFrame([[1,2,0],[3,0,1],[0,4,2]],index=["c1","c2","c3"],columns=["TF1","G2","G3"]).to_csv(data/"expression.csv")
    pd.DataFrame(dict(cell_id=["c3","c1","c2"],cell_type=["B","A","B"])).to_csv(data/"metadata.csv",index=False)
    pd.DataFrame(dict(x=[1,2,3],y=[0,0,1]),index=["c1","c2","c3"]).to_csv(data/"coordinates.csv")
    pd.DataFrame(dict(tf=["TF1","unexpressed"])).to_csv(data/"tf_list.csv",index=False)
    pd.DataFrame(dict(ligand=["G2","G2_G3"],receptor=["G3","TF1"])).to_csv(data/"lr_pairs.csv",index=False)
    truth=data/"truth"
    truth.mkdir()
    pd.DataFrame(dict(tf=["TF1","TF1"],target=["G2","G3"],label=[1,0])).to_csv(truth/"truth_grn.csv",index=False)
    registry=tmp_path/"registry.csv"
    pd.DataFrame([dict(dataset_id="d1",dataset_dir=str(data),truth_dir=str(truth),truth_kind="exact",species="synthetic")]).to_csv(registry,index=False)
    dc=DatasetConfig(evidence_registry=registry,dataset_ids=("d1",),output_root=tmp_path/"output")
    ec=EvaluationConfig(bootstrap_iterations=50,our_sergio_root=tmp_path/"ours")
    return dc,ec,data
