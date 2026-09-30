"""One isolated development fit, using official baselines. Never reads truth."""
from dataclasses import asdict, replace
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
import psutil
from publication_study.resources import write_json,file_hash,verify_model_inputs

def run_worker(dataset_id,method,candidate):
    from configs.public_sim_data.benchmark_optimization import OptimizationConfig
    cfg=OptimizationConfig()
    rows=pd.read_csv(cfg.processed_root/'development_registry.csv')
    match=rows.loc[rows.dataset_id.eq(dataset_id)]
    if len(match)!=1 or match.iloc[0]['split']!='development': raise ValueError('Development data only')
    row=match.iloc[0].to_dict();verify_model_inputs(row)
    params=cfg.candidates()[method][candidate]
    out=cfg.result_root/'02_runs'/method/candidate/dataset_id
    out.mkdir(parents=True,exist_ok=True)
    state=dict(dataset_id=dataset_id,method=method,candidate=candidate,state='running',pid=os.getpid(),
               process_started=psutil.Process().create_time(),
               started_epoch=time.time(),input_signature=row['signature'],parameters=params)
    write_json(out/'run_status.json',state)
    try:
        if method=='JumpNicheRewire':
            from publication_study.inference import build_model
            from jump_niche_rewire import JumpNicheRewirePipeline
            from jump_niche_rewire.convergence import summarize_convergence
            updated=replace(cfg,**{k:v for k,v in params.items() if k!='aggregation'})
            model=build_model(updated,row,out)
            write_json(out/'model_config.json',asdict(model))
            result=JumpNicheRewirePipeline(model).run()
            beta=result.rewiring.beta_ccc
            score=aggregate(beta,params['aggregation'])
            frame=pd.DataFrame(dict(tf=np.repeat(result.base.tf_names,len(result.base.target_names)),
                  target=np.tile(result.base.target_names,len(result.base.tf_names)),score=score.ravel()))
            state.update(summarize_convergence(result.base,result.rewiring,model.runtime.minimum_target_convergence_rate,
                  communication_rewiring_disabled=model.rewiring.disable_communication_rewiring))
            if state.get('convergence_passed') is not True: raise RuntimeError('Convergence not accepted')
        else:
            root=Path(row['dataset_dir'])
            expression=pd.read_csv(root/'expression.csv',index_col=0)
            tfs=pd.read_csv(root/'tf_list.csv',dtype=str).tf.tolist()
            logged=np.log1p(expression.div(expression.sum(axis=1).replace(0,1),axis=0)*10000)
            if method in ('GENIE3','GRNBoost2'):
                from arboreto.algo import diy
                from arboreto.core import RF_KWARGS,SGBM_KWARGS
                from distributed import Client,LocalCluster
                kwargs=dict(RF_KWARGS if method=='GENIE3' else SGBM_KWARGS);kwargs.update(params)
                with LocalCluster(n_workers=1,threads_per_worker=1,processes=False,dashboard_address=None) as cluster, Client(cluster) as client:
                    frame=diy(expression_data=logged,regressor_type='RF' if method=='GENIE3' else 'GBM',
                              regressor_kwargs=kwargs,tf_names=tfs,client_or_address=client,seed=cfg.seed)
                frame.to_csv(out/'official_grn.csv',index=False)
                frame=frame.rename(columns={'TF':'tf','importance':'score'})
                state['arboreto_version']=importlib.metadata.version('arboreto')
            else:
                logged.to_csv(out/'log_expression.csv')
                write_json(out/'job.json',dict(method=method,params=params,seed=cfg.seed,
                    expression=str(out/'log_expression.csv'),tfs=str(root/'tf_list.csv'),
                    library=str(cfg.raw_root/'R_library'),software=str(cfg.raw_root/'software')))
                subprocess.run([str(cfg.rscript),'--vanilla',str(Path(__file__).with_name('official_r.R'))],
                               cwd=out,check=True)
                frame=pd.read_csv(out/'official_grn.csv',dtype={'tf':str,'target':str})
            state.update(convergence_passed=None,convergence_policy='not_applicable_to_this_algorithm')
        if frame.duplicated(['tf','target']).any() or not np.isfinite(frame.score).all() or (frame.score<0).any():
            raise ValueError('Invalid full GRN prediction')
        frame.to_parquet(out/'global_grn.parquet',index=False)
        state.update(state='completed',prediction_sha256=file_hash(out/'global_grn.parquet'))
    except Exception as error:
        state.update(state='failed',error=str(error),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        state.update(finished_epoch=time.time(),runtime_seconds=time.time()-state['started_epoch'])
        state['versions']={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy')}
        write_json(out/'run_status.json',state)
    return state

def aggregate(beta,mode):
    if mode=='max_abs': return np.max(np.abs(beta),axis=0)
    if mode=='mean_abs': return np.mean(np.abs(beta),axis=0)
    raise ValueError('Aggregation must be prespecified, not chosen per metric/dataset')
