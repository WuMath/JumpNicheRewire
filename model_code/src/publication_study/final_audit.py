"""Read-only numerical cross-checks; writes a separate audit, never changes fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, hypergeom
from configs.public_sim_data.publication_study import StudyConfig
from .resources import file_hash, verify_model_inputs, write_json, storage_guard
from .workflow import implementation_hash


def independent_metrics(y, score):
    y=np.asarray(y);score=np.asarray(score);positive=y.sum();n=len(y)
    # Independent threshold summation, not the production metric helper.
    ap=0.; previous=0
    for threshold in np.unique(score)[::-1]:
        selected=score>=threshold;tp=y[selected].sum()
        ap+=(tp-previous)/positive*tp/selected.sum();previous=tp
    auc=(rankdata(score)[y==1].sum()-positive*(positive+1)/2)/(positive*(n-positive))
    values={'AP':ap,'AUROC':auc}
    for k in (10,50,100,int(positive)):
        k=min(k,n);boundary=np.sort(score)[-k];above=score>boundary;equal=score==boundary
        tp=y[above].sum()+(k-above.sum())*y[equal].mean()
        if k==int(positive):values['Early_precision']=tp/k
        if k in (10,50,100):
            values.update({f'Precision_at_{k}':tp/k,f'Recall_at_{k}':tp/positive,f'F1_at_{k}':2*tp/(k+positive)})
    return values


def run():
    cfg=StudyConfig();out=cfg.result_root/'06_audit';out.mkdir(exist_ok=True)
    registry=pd.concat([pd.read_csv(cfg.processed_root/f'{kind}_registry.csv') for kind in ('benchmark','case')])
    saved=pd.read_csv(cfg.result_root/'03_evaluation/dataset_metrics.csv')
    frozen=json.loads((cfg.result_root/'02_runs/frozen_parameters.json').read_text(encoding='utf-8'))
    signature=implementation_hash();states=[];checks=0
    for row in registry.to_dict('records'):
        verify_model_inputs(row)
        methods=('JumpNicheRewire',) if row['split']=='case' else cfg.methods
        for method in methods:
            folder=cfg.result_root/'02_runs'/row['dataset_id']/method
            state=json.loads((folder/'run_status.json').read_text(encoding='utf-8'))
            assert state['state']=='completed' and state['implementation_hash']==signature
            assert state['input_signature']==row['signature']
            assert file_hash(folder/'global_grn.parquet')==state['prediction_sha256']
            if method=='JumpNicheRewire':assert state['convergence_passed'] is True
            states.append(state)
            if row['split']=='case':continue
            root=Path(row['dataset_dir']);manifest=json.loads((root/'evidence_manifest.json').read_text(encoding='utf-8'))
            assert file_hash(root/'truth/truth_grn.csv')==manifest['files']['truth/truth_grn.csv']
            truth=pd.read_csv(root/'truth/truth_grn.csv',dtype={'tf':str,'target':str})
            predictions=pd.read_parquet(folder/'global_grn.parquet')
            predictions[['tf','target']]=predictions[['tf','target']].astype(str)
            assert not predictions.duplicated(['tf','target']).any()
            joined=truth.merge(predictions,on=['tf','target'],how='left',validate='one_to_one')
            values=independent_metrics(joined.label,joined.score.fillna(0))
            for metric,value in values.items():
                actual=saved.loc[saved.dataset_id.eq(row['dataset_id'])&saved.method.eq(method)&saved.metric.eq(metric),'value']
                assert len(actual)==1 and np.isclose(actual.iloc[0],value,atol=1e-12), (row['dataset_id'],method,metric)
                checks+=1
    pathway_tests=0;case_summary=[]
    for folder in (cfg.result_root/'04_cases').glob('case_*'):
        if not folder.is_dir():continue
        frame=pd.read_csv(folder/'pathway_enrichment_all_tests.csv')
        p=hypergeom.sf(frame.overlap-1,frame.background_size,frame.pathway_in_background,frame.query_size)
        assert np.allclose(frame.p_value,p,atol=1e-14)
        # BH recomputed by sorted ranks, independently of analysis.bh.
        order=np.argsort(p);q=np.empty(len(p));best=1.
        for i in range(len(p)-1,-1,-1):
            best=min(best,p[order[i]]*len(p)/(i+1));q[order[i]]=best
        assert np.allclose(frame.q_value,q,atol=1e-12)
        pathway_tests+=len(frame)
        case_summary.append(dict(dataset_id=folder.name,tests=len(frame),significant_tests=int((q<=.05).sum()),
                                 unique_significant_pathways=frame.loc[q<=.05,'pathway'].nunique()))
    assert len(states)==81
    result=dict(status='passed',formal_runs=81,independent_metric_checks=checks,pathway_tests_recomputed=pathway_tests,
                implementation_hash=signature,cases=case_summary,
                missing_full_peak_rss=[s['dataset_id']+'/'+s['method'] for s in states if s.get('peak_rss_gib') is None],
                new_gib=storage_guard(cfg)/1024**3,
                scope='Engineering and numerical cross-check, not external peer review or biological validation')
    write_json(out/'independent_numerical_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':run()
