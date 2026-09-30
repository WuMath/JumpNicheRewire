"""Development-only comparisons. Does not unlock confirmation or application cases."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from publication_study.analysis import accepted_prediction
from publication_study.resources import file_hash,write_json
from independent_baselines.comparison import metrics

PERFORMANCE=('AP','AUROC','Early_precision','Precision_at_10','Recall_at_10','F1_at_10',
             'Precision_at_50','Recall_at_50','F1_at_50','Precision_at_100','Recall_at_100','F1_at_100')

def select_candidates(data, audit, expected_datasets):
    selected={}; records=[]
    for (method,candidate), group in audit.groupby(['method','candidate']):
        valid=set(group.loc[group.state.eq('accepted'),'dataset_id'])==set(expected_datasets)
        ap=data.loc[data.method.eq(method)&data.candidate.eq(candidate)&data.metric.eq('AP')]
        score=float(ap.groupby('independence_group').value.mean().mean()) if valid and len(ap)==len(expected_datasets) else np.nan
        records.append(dict(method=method,candidate=candidate,complete=bool(valid),selection_AP=score))
    table=pd.DataFrame(records)
    for method, group in table.loc[table.complete & table.selection_AP.notna()].groupby('method'):
        selected[method]=group.sort_values(['selection_AP','candidate'],ascending=[False,True]).iloc[0].candidate
    return selected,table

def evaluate(cfg):
    from .workflow import rows_for_development,lock_plan
    rows=rows_for_development(cfg);plan=lock_plan(cfg,rows)
    out=cfg.result_root/'03_development';out.mkdir(parents=True,exist_ok=True)
    audits=[];values=[]
    for row in rows.to_dict('records'):
        manifest=json.loads((Path(row['dataset_dir'])/'evidence_manifest.json').read_text(encoding='utf-8'))
        truth_path=Path(row['truth_dir'])/'truth_grn.csv'
        if file_hash(truth_path)!=manifest['files']['truth/truth_grn.csv']: raise ValueError('Development truth checksum changed')
        truth=pd.read_csv(truth_path,dtype={'tf':str,'target':str})
        for method in cfg.methods:
            for candidate in cfg.candidates()[method]:
                root=cfg.result_root/'02_runs'/method/candidate/row['dataset_id']
                frame,state=accepted_prediction(root,method)
                if frame is not None:
                    record=json.loads((root/'run_status.json').read_text(encoding='utf-8'))
                    if record.get('implementation_hash')!=plan['implementation_hash'] or record.get('input_signature')!=row['signature']:
                        frame=None;state='provenance_mismatch'
                audit=dict(dataset_id=row['dataset_id'],method=method,candidate=candidate,state=state)
                if frame is not None:
                    joined=truth.merge(frame[['tf','target','score']],on=['tf','target'],how='left',validate='one_to_one')
                    audit.update(n_candidates=len(truth),n_predicted_candidates=int(joined.score.notna().sum()))
                    for metric,value in metrics(joined.label,joined.score.fillna(0),cfg.classification_top_k).items():
                        values.append(dict(dataset_id=row['dataset_id'],independence_group=row['independence_group'],
                            method=method,candidate=candidate,metric=metric,value=value))
                audits.append(audit)
    audit=pd.DataFrame(audits);data=pd.DataFrame(values,columns=['dataset_id','independence_group','method','candidate','metric','value'])
    audit.to_csv(out/'run_audit.csv',index=False);data.to_csv(out/'all_candidate_metrics.csv',index=False)
    selected,selection=select_candidates(data,audit,rows.dataset_id.tolist())
    selection.to_csv(out/'parameter_selection_audit.csv',index=False)
    keep=[selected.get(r.method)==r.candidate for r in data.itertuples()]
    chosen=data.loc[keep]
    summary=chosen.groupby(['metric','method']).value.mean().unstack('method') if len(chosen) else pd.DataFrame()
    summary.to_csv(out/'selected_method_means.csv')
    tables=out/'per_metric';tables.mkdir(exist_ok=True)
    for metric, group in chosen.groupby('metric'):
        group.pivot(index='dataset_id',columns='method',values='value').to_csv(tables/f'{metric}.csv')
    wins=[]
    if set(cfg.methods).issubset(summary.columns):
        for other in cfg.methods[1:]:
            for metric in PERFORMANCE:
                a=float(summary.loc[metric,'JumpNicheRewire']);b=float(summary.loc[metric,other])
                wins.append(dict(metric=metric,comparator=other,jnr=a,other=b,difference=a-b,strictly_better=a>b))
    pd.DataFrame(wins).to_csv(out/'development_comparisons.csv',index=False)
    result=dict(development_fits_accepted=int(audit.state.eq('accepted').sum()),expected_fits=len(audit),
                selected_candidates=selected,all_methods_selected=len(selected)==len(cfg.methods),
                development_all_metrics_better_than_all_five=bool(wins and all(x['strictly_better'] for x in wins)),
                formal_superiority_established=False,confirmation_complete=False,cases_enabled=False,
                warning='Scores used for selection are optimistic development estimates, not independent test evidence.')
    write_json(out/'development_audit.json',result)
    write_json(cfg.result_root/'case_gate.json',dict(allowed=False,reason='Independent confirmation protocol and results not completed'))
    return result
