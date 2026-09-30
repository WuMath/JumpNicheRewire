"""One isolated model fit. This module never reads a truth file."""
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from configs.public_sim_data.publication_study import StudyConfig
from .resources import file_hash, write_json

def build_model(cfg,row,out):
    from configs.common.presets import ModelParameters
    model=ModelParameters()
    root=Path(row['dataset_dir'])
    genes=pd.read_csv(root/'expression.csv',nrows=0,index_col=0).columns.tolist()
    tfs=pd.read_csv(root/'tf_list.csv',dtype=str).tf.tolist()
    lr=pd.read_csv(root/'lr_pairs.csv')
    spatial=bool(row['is_spatial'])
    model.data.input_dir=root
    model.data.species=row['species']
    model.data.is_spatial=spatial
    model.data.minimum_detection_rate=0.0
    model.data.minimum_total_count=0.0
    model.rewiring.disable_communication_rewiring=not spatial or lr.empty
    model.base_grn.maximum_tfs=len(tfs)
    model.base_grn.maximum_targets=len(genes)
    model.base_grn.l1_penalty=cfg.l1_penalty
    model.base_grn.fusion_penalty=cfg.fusion_penalty
    model.rewiring.group_penalty=cfg.group_penalty
    model.rewiring.ridge_penalty=cfg.ridge_penalty
    model.rewiring.maximum_interaction_tfs=len(tfs)
    model.rewiring.maximum_active_lr=len(lr)
    model.spatial.spatial_permutations=cfg.spatial_permutations
    model.spatial.minimum_spatial_genes=len(genes)
    model.spatial.maximum_spatial_genes=len(genes)
    if row['split']=='case':
        model.spatial.minimum_spatial_genes=cfg.case_maximum_targets
        model.spatial.maximum_spatial_genes=cfg.case_maximum_targets
        model.base_grn.maximum_targets=cfg.case_maximum_targets
    model.runtime.random_seed=cfg.seed
    # Retain resource-limited previews; evaluation below uses ALL numeric coefficients.
    model.output.output_dir=out
    model.output.maximum_excel_rows_per_table=200
    model.output.maximum_grn_rows=20000
    model.rewiring.top_grn_rows=20000
    model.rewiring.top_lr_gene_rows=20000
    model.rewiring.top_path_rows=10000
    model.output.maximum_spatial_edge_rows=30000
    model.runtime.checkpoint_data_signature=row['signature']
    return model

def run_inference(dataset_id,method):
    cfg=StudyConfig()
    registries=[pd.read_csv(cfg.processed_root/'benchmark_registry.csv')]
    cases=cfg.processed_root/'case_registry.csv'
    if cases.exists(): registries.append(pd.read_csv(cases))
    rows=pd.concat(registries,ignore_index=True)
    selected=rows.loc[rows.dataset_id.eq(dataset_id)]
    if len(selected)!=1: raise ValueError('Dataset ID must identify exactly one prepared input')
    row=selected.iloc[0].to_dict()
    out=cfg.result_root/'02_runs'/dataset_id/method
    out.mkdir(parents=True,exist_ok=True)
    state=dict(dataset_id=dataset_id,method=method,state='running',pid=os.getpid(),
               input_signature=row['signature'],started_epoch=time.time(),split=row['split'])
    write_json(out/'run_status.json',state)
    start=time.perf_counter()
    try:
        if method=='JumpNicheRewire':
            from jump_niche_rewire import JumpNicheRewirePipeline
            from jump_niche_rewire.convergence import ConvergenceError,summarize_convergence
            model=build_model(cfg,row,out)
            write_json(out/'model_config.json',asdict(model))
            failed=None
            try: result=JumpNicheRewirePipeline(model).run()
            except ConvergenceError as error:
                result=error.result;failed=error
            # Exact global aggregation before any output top-k; no Python dictionary per cell-edge.
            beta=result.rewiring.beta_ccc
            score=np.max(np.abs(beta),axis=0)
            signed=np.take_along_axis(beta,np.argmax(np.abs(beta),axis=0)[None,:,:],axis=0)[0]
            frame=pd.DataFrame(dict(tf=np.repeat(result.base.tf_names,len(result.base.target_names)),
                target=np.tile(result.base.target_names,len(result.base.tf_names)),score=score.ravel(),
                signed_score=signed.ravel()))
            frame.to_parquet(out/'global_grn.parquet',index=False)
            summary=summarize_convergence(result.base,result.rewiring,model.runtime.minimum_target_convergence_rate,
                communication_rewiring_disabled=model.rewiring.disable_communication_rewiring)
            state.update(summary)
            # All-target activity maps and complete regional gamma are small scientific checkpoints.
            np.savez_compressed(out/'mechanism_coefficients.npz',gamma=result.rewiring.gamma,
                active_lr_indices=result.rewiring.active_lr_indices,
                tf_names=np.asarray(result.base.tf_names,dtype=str),target_names=np.asarray(result.base.target_names,dtype=str),
                regions=np.asarray(result.rewiring.regions,dtype=str))
            if failed is not None: raise failed
        elif method in ('GENIE3','GRNBoost2'):
            from arboreto.algo import genie3,grnboost2
            from distributed import Client,LocalCluster
            root=Path(row['dataset_dir'])
            expression=pd.read_csv(root/'expression.csv',index_col=0)
            tfs=pd.read_csv(root/'tf_list.csv',dtype=str).tf.tolist()
            logged=np.log1p(expression.div(expression.sum(axis=1).replace(0,1),axis=0)*10000)
            algorithm=genie3 if method=='GENIE3' else grnboost2
            with LocalCluster(n_workers=1,threads_per_worker=1,processes=False,dashboard_address=None) as cluster, Client(cluster) as client:
                frame=algorithm(expression_data=logged,tf_names=tfs,client_or_address=client,seed=cfg.seed,verbose=False)
            frame.to_csv(out/'official_grn.csv',index=False)
            frame=frame.rename(columns={'TF':'tf','importance':'score'})
            frame.to_parquet(out/'global_grn.parquet',index=False)
            state.update(convergence_passed=None,convergence_policy='not_applicable_to_tree_ensemble',
                         arboreto_version=importlib.metadata.version('arboreto'))
        else: raise ValueError(f'Unsupported model {method}')
        state['prediction_sha256']=file_hash(out/'global_grn.parquet')
        state['state']='completed'
    except Exception as error:
        state.update(state='failed',error=str(error),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        state.update(runtime_seconds=time.perf_counter()-start,finished_epoch=time.time())
        state['versions']={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy')}
        write_json(out/'run_status.json',state)
    return state
