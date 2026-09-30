"""Read-only 624-edge single-development-dataset comparison.

The data used to select configurations are also used to score them; this is
deliberately not a confirmation experiment.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from independent_baselines.comparison import metrics
from .cespgrn_one import SingleCeSpGRN, verify
from .regdiffusion_one import SingleRegDiffusion, verify as verify_regdiffusion


ROOT = Path(__file__).resolve().parents[2]
ROUND02 = ROOT/'data/result/optimization_20260921/round_02'
PERFORMANCE = ('AP','AUROC','Early_precision','Precision_at_10','Recall_at_10',
    'F1_at_10','Precision_at_50','Recall_at_50','F1_at_50',
    'Precision_at_100','Recall_at_100','F1_at_100')


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def compare(include_cespgrn=True, include_regdiffusion=True):
    from benchmark_round02.workflow import verified_plan
    from configs.public_sim_data.benchmark_round02 import Round02Config
    cfg=Round02Config()
    verified_plan(cfg)
    ext=SingleCeSpGRN()
    verify(ext)
    original=pd.read_csv(ROUND02/'03_development/all_candidate_and_ablation_metrics.csv')
    data=original.loc[original.dataset_id.eq(ext.dataset_id)&original.role.eq('candidate')]
    candidates=data[['method','candidate']].drop_duplicates()
    if len(candidates)!=43 or data.groupby(['method','candidate']).metric.nunique().ne(12).any():
        raise ValueError('Prior candidate set is incomplete or includes fixed ablations')
    ap=data.loc[data.metric.eq('AP')].sort_values(
        ['method','value','candidate'],ascending=[True,False,True])
    chosen=ap.groupby('method',sort=True).head(1)
    if len(chosen)!=6: raise ValueError('All six existing methods must be selected')
    selected=dict(zip(chosen.method,chosen.candidate))
    values={}
    for method,candidate in selected.items():
        subset=data.loc[data.method.eq(method)&data.candidate.eq(candidate)]
        values[method]=dict(zip(subset.metric,subset.value.astype(float)))
        if set(values[method])!=set(PERFORMANCE):
            raise ValueError(f'Incomplete metric set: {method}')
    report=dict(dataset_id=ext.dataset_id,common_tf_target_edges=624,
        same_data_used_for_tuning_and_evaluation=True,independent_confirmation=False,
        selected_candidates=selected,methods=values,
        historical_fit_count=276,new_fits_this_evaluation=0,
        fixed_ablations_excluded=True)
    if include_cespgrn:
        status=json.loads((ext.output_dir/'run_status.json').read_text(encoding='utf8'))
        if status.get('state')!='completed' or status.get('dataset_id')!=ext.dataset_id:
            raise RuntimeError('CeSpGRN full-dataset fit not completed and accepted')
        prediction_path=ext.output_dir/'global_grn.parquet'
        graph_path=ext.output_dir/'cell_graphs.npz'
        if _sha(prediction_path)!=status['global_sha256'] or _sha(graph_path)!=status['graph_sha256']:
            raise ValueError('CeSpGRN output hash mismatch')
        manifest=json.loads((ext.dataset_dir/'evidence_manifest.json').read_text(encoding='utf8'))
        truth_path=ext.dataset_dir/'truth/truth_grn.csv'
        if _sha(truth_path)!=manifest['files']['truth/truth_grn.csv']:
            raise ValueError('Development truth hash mismatch')
        truth=pd.read_csv(truth_path,dtype={'tf':str,'target':str})
        prediction=pd.read_parquet(prediction_path)
        if (len(truth)!=624 or len(prediction)!=624 or
                truth.duplicated(['tf','target']).any() or
                prediction.duplicated(['tf','target']).any()):
            raise ValueError('Incomplete or duplicated common TF-target universe')
        joined=truth.merge(prediction,on=['tf','target'],how='left',validate='one_to_one')
        if joined.score.isna().any() or not np.isfinite(joined.score).all():
            raise ValueError('CeSpGRN missed common edges or has nonfinite values')
        result=metrics(joined.label,joined.score,(10,50,100))
        report['methods']['CeSpGRN']={key:float(result[key]) for key in PERFORMANCE}
        report['selected_candidates']['CeSpGRN']='spatial_default'
        report['cespgrn_is_undirected_association_not_causal']=True
        report['cespgrn_graph_sha256']=status['graph_sha256']
        report['cespgrn_global_sha256']=status['global_sha256']
    if include_regdiffusion:
        rd=SingleRegDiffusion()
        verify_regdiffusion(rd)
        status=json.loads((rd.output_dir/'run_status.json').read_text(encoding='utf8'))
        if status.get('state')!='completed' or status.get('dataset_id')!=rd.dataset_id:
            raise RuntimeError('RegDiffusion full-dataset fit not completed and accepted')
        prediction_path=rd.output_dir/'global_grn.parquet'
        adjacency_path=rd.output_dir/'adjacency.npz'
        if (_sha(prediction_path)!=status['global_sha256'] or
                _sha(adjacency_path)!=status['adjacency_sha256']):
            raise ValueError('RegDiffusion output hash mismatch')
        manifest=json.loads((rd.dataset_dir/'evidence_manifest.json').read_text(encoding='utf8'))
        truth_path=rd.dataset_dir/'truth/truth_grn.csv'
        if _sha(truth_path)!=manifest['files']['truth/truth_grn.csv']:
            raise ValueError('Development truth hash mismatch')
        truth=pd.read_csv(truth_path,dtype={'tf':str,'target':str})
        prediction=pd.read_parquet(prediction_path)
        if (len(truth)!=624 or len(prediction)!=624 or
                truth.duplicated(['tf','target']).any() or
                prediction.duplicated(['tf','target']).any()):
            raise ValueError('Incomplete or duplicated common TF-target universe')
        joined=truth.merge(prediction,on=['tf','target'],how='left',validate='one_to_one')
        if joined.score.isna().any() or not np.isfinite(joined.score).all():
            raise ValueError('RegDiffusion missed common edges or has nonfinite values')
        result=metrics(joined.label,joined.score,(10,50,100))
        report['methods']['RegDiffusion']={key:float(result[key]) for key in PERFORMANCE}
        report['selected_candidates']['RegDiffusion']='official_default'
        report['regdiffusion_removed_zero_variance']=status['removed_zero_variance']
        report['regdiffusion_adjacency_sha256']=status['adjacency_sha256']
        report['regdiffusion_global_sha256']=status['global_sha256']
    return report
