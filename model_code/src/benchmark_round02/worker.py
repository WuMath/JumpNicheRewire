"""Official estimators and independent v2 fits; never opens truth files."""
from dataclasses import replace
import importlib.metadata
from pathlib import Path
import subprocess
import os
import sys
import numpy as np
import pandas as pd
from jump_niche_rewire_v2.adapter import load_development, sha
from jump_niche_rewire_v2.execution import fit_prepared, atomic_json
from jump_niche_rewire_v2.persistence import ResourceGuard, digest


def complete_candidates(frame, tfs, targets):
    frame = frame[['tf', 'target', 'score']].copy()
    if frame.duplicated(['tf', 'target']).any() or not np.isfinite(frame.score).all() or (frame.score < 0).any():
        raise ValueError('Invalid estimator output')
    candidates = pd.MultiIndex.from_product([tfs, targets], names=['tf', 'target']).to_frame(index=False)
    result = candidates.merge(frame, on=['tf', 'target'], how='left', validate='one_to_one')
    result['score'] = result.score.fillna(0.)
    return result


def run_worker(cfg, dataset_id, method, candidate, role, plan_hash):
    from .workflow import verified_plan, task_root
    plan = verified_plan(cfg)
    if digest(plan) != plan_hash: raise ValueError('Worker plan differs from controller')
    out = task_root(cfg, dataset_id, method, candidate, role)
    numerical, adapter = cfg.numerical, cfg.adapter
    if role == 'ablation':
        if method != 'JumpNicheRewire': raise ValueError('Unknown ablation method')
        adapter = replace(adapter, **cfg.ablations()[candidate])
    else:
        params = cfg.candidates()[method][candidate]
        if method == 'JumpNicheRewire': numerical = replace(numerical, **params)
    settings = replace(cfg, adapter=adapter, numerical=numerical)
    prepared = load_development(settings, dataset_id)
    identity = dict(dataset_id=dataset_id, method=method, candidate=candidate, role=role, plan_sha256=plan_hash)
    guard = ResourceGuard(cfg); guard()
    if method == 'JumpNicheRewire':
        result = fit_prepared(prepared, numerical, adapter, out, identity, guard)
    else:
        row = pd.read_csv(cfg.development_registry).set_index('dataset_id').loc[dataset_id]
        root = Path(row.dataset_dir)
        expression = pd.read_csv(root/'expression.csv', index_col=0)
        logged = np.log1p(expression.div(expression.sum(axis=1).replace(0, 1), axis=0)*10000)
        tfs, targets = prepared.tfs, prepared.targets
        if method in ('GENIE3', 'GRNBoost2'):
            from arboreto.algo import diy
            from arboreto.core import RF_KWARGS, SGBM_KWARGS
            from distributed import Client, LocalCluster
            kwargs = dict(RF_KWARGS if method == 'GENIE3' else SGBM_KWARGS); kwargs.update(params)
            with LocalCluster(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None) as cluster, Client(cluster) as client:
                frame = diy(expression_data=logged, regressor_type='RF' if method == 'GENIE3' else 'GBM',
                    regressor_kwargs=kwargs, tf_names=tfs, client_or_address=client, seed=cfg.seed)
            frame.to_csv(out/'official_grn.csv', index=False)
            frame = frame.rename(columns={'TF': 'tf', 'importance': 'score'})
        else:
            logged.to_csv(out/'log_expression.csv')
            atomic_json(out/'job.json', dict(method=method, params=params, seed=cfg.seed,
                expression=str(out/'log_expression.csv'), tfs=str(root/'tf_list.csv'),
                library=str(cfg.raw_root/'R_library'), software=str(cfg.raw_root/'software')))
            wrapper = Path(__file__).parents[1]/'benchmark_optimization/official_r.R'
            # R emits directly to a file so the Python supervisor does not buffer its output.
            with (out/'official_r.log').open('a', encoding='utf8') as log:
                subprocess.run([str(cfg.rscript), '--vanilla', str(wrapper)], cwd=out, check=True,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            frame = pd.read_csv(out/'official_grn.csv', dtype={'tf': str, 'target': str})
        frame = complete_candidates(frame, tfs, targets)
        temporary = out/'global_grn.parquet.writing'
        frame.to_parquet(temporary, index=False)
        if (out/'global_grn.parquet').exists():
            if sha(temporary) != sha(out/'global_grn.parquet'): raise ValueError('Previous prediction differs; preserve and audit')
            temporary.unlink()
        else: temporary.replace(out/'global_grn.parquet')
        result = dict(state='completed', rows=len(frame), prediction_sha256=sha(out/'global_grn.parquet'),
                      convergence_rate=None, convergence_policy='not_applicable_to_this_algorithm')
    result.update(**identity, input_hashes=prepared.audit['input_hashes'],
        python_executable=sys.executable, truth_read=False,
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'pyarrow')})
    return result
