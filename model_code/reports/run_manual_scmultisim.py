"""Re-fit ONE fixed method, or evaluate all nine, under an isolated output root.

No CLI arguments. Select action in configs/public_sim_data/manual_scmultisim.py.
This is a replication harness, not a replacement production model entry point.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
for folder in (ROOT, ROOT/'src'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

from configs.public_sim_data.manual_scmultisim import ManualScMultiSim, DATASET

METHODS = ('JumpNicheRewire', 'TIGRESS', 'CLR', 'PPCOR', 'GRNBoost2',
           'GENIE3', 'CeSpGRN', 'RegDiffusion', 'CellOracle_Net_core')
CANDIDATES = {
    'JumpNicheRewire': 'l1_0075', 'TIGRESS': 'steps3', 'CLR': 'official',
    'PPCOR': 'kendall', 'GRNBoost2': 'half_depth1', 'GENIE3': 'leaf5_half',
    'CeSpGRN': 'spatial_default', 'RegDiffusion': 'official_default',
    'CellOracle_Net_core': 'net_core_all_candidates',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_input(cfg):
    from publication_study.resources import verify_model_inputs
    if cfg.dataset_id != DATASET:
        raise ValueError('This harness is fixed to one author-published development dataset')
    manifest = json.loads((cfg.dataset_dir/'evidence_manifest.json').read_text(encoding='utf8'))
    if manifest['dataset_id'] != DATASET:
        raise ValueError('Dataset manifest identity differs')
    verify_model_inputs({'dataset_dir': str(cfg.dataset_dir)})
    if len(__import__('pandas').read_csv(cfg.dataset_dir/'tf_list.csv')) != 6:
        raise ValueError('Expected six TFs')
    return manifest


def preflight(cfg):
    from jump_niche_rewire_v2.persistence import ResourceGuard
    import psutil
    verify_input(cfg)
    if psutil.virtual_memory().available < cfg.minimum_available_ram_gib*1024**3:
        raise MemoryError('Available RAM is below 1.5 GiB')
    if shutil.disk_usage(cfg.output_root.anchor).free < cfg.minimum_free_disk_gib*1024**3:
        raise MemoryError('Free disk is below 5 GiB')
    guard = ResourceGuard(cfg)
    guard(reserve_bytes=cfg.storage_reserve_bytes)
    return guard


def output_dir(cfg, method):
    if method == 'JumpNicheRewire':
        return cfg.output_root/method/'02_runs'/CANDIDATES[method]
    return cfg.output_root/method/CANDIDATES[method]/cfg.dataset_id


def baseline_worker(cfg, method):
    """Fixed official estimator settings; never opens the truth table."""
    import numpy as np
    import pandas as pd
    from benchmark_round02.worker import complete_candidates
    from configs.public_sim_data.benchmark_optimization import OptimizationConfig
    from configs.public_sim_data.benchmark_round02 import Round02Config
    from publication_study.resources import verify_model_inputs
    from jump_niche_rewire_v2.execution import atomic_json

    verify_model_inputs({'dataset_dir': str(cfg.dataset_dir)})
    old = OptimizationConfig()
    candidate = CANDIDATES[method]
    params = (Round02Config().candidates()[method][candidate]
              if method in ('GRNBoost2', 'GENIE3') else old.candidates()[method][candidate])
    expression = pd.read_csv(cfg.dataset_dir/'expression.csv', index_col=0)
    logged = np.log1p(expression.div(expression.sum(axis=1).replace(0, 1), axis=0)*10000)
    tfs = pd.read_csv(cfg.dataset_dir/'tf_list.csv', dtype=str).tf.tolist()
    targets = [gene for gene in expression.columns if gene not in set(tfs)]
    if (expression.shape, len(tfs), len(targets)) != ((500, 110), 6, 104):
        raise ValueError('Common input panel changed')
    out = output_dir(cfg, method)
    if method in ('GRNBoost2', 'GENIE3'):
        from arboreto.algo import diy
        from arboreto.core import RF_KWARGS, SGBM_KWARGS
        from distributed import Client, LocalCluster
        kwargs = dict(RF_KWARGS if method == 'GENIE3' else SGBM_KWARGS)
        kwargs.update(params)
        with LocalCluster(n_workers=1, threads_per_worker=1, processes=False,
                          dashboard_address=None) as cluster, Client(cluster) as client:
            frame = diy(expression_data=logged,
                        regressor_type='RF' if method == 'GENIE3' else 'GBM',
                        regressor_kwargs=kwargs, tf_names=tfs,
                        client_or_address=client, seed=20260921)
        frame.to_csv(out/'official_grn.csv', index=False)
        frame = frame.rename(columns={'TF': 'tf', 'importance': 'score'})
    else:
        import subprocess
        logged.to_csv(out/'log_expression.csv')
        atomic_json(out/'job.json', dict(method=method, params=params, seed=20260921,
            expression=str(out/'log_expression.csv'), tfs=str(cfg.dataset_dir/'tf_list.csv'),
            library=str(cfg.r_library), software=str(cfg.r_software)))
        wrapper = ROOT/'src/benchmark_optimization/official_r.R'
        with (out/'official_r.log').open('a', encoding='utf8') as log:
            subprocess.run([str(cfg.rscript), '--vanilla', str(wrapper)], cwd=out,
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        frame = pd.read_csv(out/'official_grn.csv', dtype={'tf': str, 'target': str})
    frame = complete_candidates(frame, tfs, targets)
    if len(frame) != 624:
        raise ValueError('Incomplete common candidate universe')
    prediction = out/'global_grn.parquet'
    frame.to_parquet(prediction, index=False)
    return dict(state='completed', method=method, candidate=candidate,
                dataset_id=DATASET, rows=len(frame), prediction_sha256=sha(prediction),
                input_sha256=sha(cfg.dataset_dir/'expression.csv'), truth_read=False,
                parameters=params)


def run_one(cfg):
    method = cfg.action
    out = output_dir(cfg, method)
    if (out/'run_status.json').exists() or (method == 'JumpNicheRewire' and (out/'global_grn.parquet').exists()):
        raise FileExistsError(f'Existing attempt; inspect or evaluate it, do not overwrite: {out}')
    if method in ('CeSpGRN', 'RegDiffusion'):
        import importlib.util
        required = ('torch',) if method == 'CeSpGRN' else ('torch', 'lightgraph')
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            message = (f'{method} needs its isolated Python environment; missing {missing}; '
                       f'current Python: {sys.executable}; no output created')
            if method == 'RegDiffusion':
                interpreter = cfg.regdiffusion_python
                # Print a copyable PowerShell command; do not install packages,
                # change the active environment, or start an unrequested fit.
                quote = lambda path: "'" + str(path).replace("'", "''") + "'"
                if interpreter.is_file():
                    message += ('\nKeep action=\'RegDiffusion\' and run in PowerShell:\n'
                                f'& {quote(interpreter)} -B {quote(Path(__file__).resolve())}')
                else:
                    message += f'\nConfigured RegDiffusion Python not found: {interpreter}'
                message += ('\nUse the pinned Torch 2.5.1 CPU / lightgraph 1.2.0 environment; '
                            'do not upgrade or install these into jnr-cespgrn.')
            raise ImportError(message)
    preflight(cfg)
    if method == 'JumpNicheRewire':
        import time
        from configs.public_sim_data.single_dataset_jnr_tuning import OneHourJNR
        from communication_benchmark.jnr_one_hour import _verify, _supervise
        # The original worker writes output_root/02_runs/<candidate>.
        jnr = replace(OneHourJNR(), output_root=cfg.output_root/method,
                      registry=cfg.dataset_dir.parent/'development_registry.csv',
                      python=cfg.model_python,
                      budget_roots=cfg.budget_roots,
                      cumulative_storage_limit=int(cfg.maximum_new_gib*1024**3))
        _verify(jnr)
        params = dict(l1_penalty=.075, fusion_penalty=.0005,
                      group_penalty=.01, ridge_penalty=.0001, aggregation='mean_abs')
        status = _supervise(jnr, CANDIDATES[method], params, time.monotonic()+cfg.maximum_task_hours*3600)
    elif method == 'CeSpGRN':
        from communication_benchmark.cespgrn_one import SingleCeSpGRN, run
        status = run(replace(SingleCeSpGRN(), output_dir=out, budget_roots=cfg.budget_roots,
                             dataset_dir=cfg.dataset_dir,
                             source_dir=cfg.cesp_source,
                             cumulative_storage_limit=int(cfg.maximum_new_gib*1024**3)))
    elif method == 'RegDiffusion':
        from communication_benchmark.regdiffusion_one import SingleRegDiffusion, run
        status = run(replace(SingleRegDiffusion(), output_dir=out, budget_roots=cfg.budget_roots,
                             dataset_dir=cfg.dataset_dir,
                             source_dir=cfg.reg_source,
                             cumulative_storage_limit=int(cfg.maximum_new_gib*1024**3)))
    elif method == 'CellOracle_Net_core':
        from communication_benchmark.celloracle_core_one import CellOracleCoreOne, run
        status = run(replace(CellOracleCoreOne(), output_dir=out, budget_roots=cfg.budget_roots,
                             package_dir=cfg.cell_package,
                             cumulative_storage_limit=int(cfg.maximum_new_gib*1024**3)))
    else:
        # The independent adapters supervise their own workers. In particular,
        # CeSpGRN's Python 3.9 must not import this Python 3.10-only supervisor.
        from jump_niche_rewire_v2.supervisor import run_isolated, SupervisorConfig
        python = cfg.tree_python if method in ('GENIE3', 'GRNBoost2') else cfg.model_python
        status = run_isolated(baseline_worker, (cfg, method), out, cfg,
                              SupervisorConfig(python_executable=str(python)))
    if status['state'] != 'completed':
        raise RuntimeError(f'{method}: {status["state"]}; inspect {out}')
    print(f'COMPLETED {method}: {out}')


def evaluate(cfg):
    import numpy as np
    import pandas as pd
    from independent_baselines.comparison import metrics
    manifest = verify_input(cfg)
    truth_path = cfg.dataset_dir/'truth/truth_grn.csv'
    if sha(truth_path) != manifest['files']['truth/truth_grn.csv']:
        raise ValueError('Truth checksum changed')
    truth = pd.read_csv(truth_path, dtype={'tf': str, 'target': str})
    if len(truth) != 624 or int(truth.label.sum()) != 130 or truth.duplicated(['tf','target']).any():
        raise ValueError('Truth candidate universe changed')
    records = []
    for method in METHODS:
        out = output_dir(cfg, method)
        status = json.loads((out/'run_status.json').read_text(encoding='utf8'))
        if status['state'] != 'completed' or status.get('worker_exit_code', status.get('exit_code', 0)) != 0:
            raise ValueError(f'Incomplete worker: {method}')
        result = status.get('result', status)
        prediction = out/'global_grn.parquet'
        if result.get('prediction_sha256', result.get('global_sha256')) != sha(prediction):
            raise ValueError(f'Prediction hash differs: {method}')
        frame = pd.read_parquet(prediction)
        if frame.duplicated(['tf','target']).any() or not np.isfinite(frame.score).all():
            raise ValueError(f'Invalid predictions: {method}')
        joined = truth.merge(frame[['tf','target','score']], on=['tf','target'],
                             how='left', validate='one_to_one')
        values = metrics(joined.label, joined.score.fillna(0), (10, 50, 100))
        records.append(dict(method=method, candidate=CANDIDATES[method],
                            **{key: value for key, value in values.items()
                               if key in ('AP','AUROC','Early_precision') or key.startswith(('Precision_at_', 'Recall_at_', 'F1_at_'))}))
    result = pd.DataFrame(records).sort_values('AP', ascending=False)
    destination = cfg.output_root/'final_comparison/nine_method_metrics.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Existing final table; do not overwrite: {destination}')
    result.to_csv(destination, index=False)
    print(result[['method','AP','AUROC','Early_precision']].to_string(index=False))
    print(f'COMPLETED evaluation: {destination}')


def main():
    if len(sys.argv) != 1:
        raise ValueError('No experiment CLI arguments; edit the config class action')
    cfg = ManualScMultiSim()
    if cfg.action == 'check':
        guard = preflight(cfg)
        print(f'INPUT OK: {cfg.dataset_dir}')
        print(f'BUDGET BYTES: {guard.cumulative_bytes}; reserve={cfg.storage_reserve_bytes}')
        print(f'NEW OUTPUT: {cfg.output_root}')
        for method in METHODS:
            out = output_dir(cfg, method)
            print(method, 'completed' if (out/'run_status.json').exists() and
                  json.loads((out/'run_status.json').read_text(encoding='utf8')).get('state') == 'completed'
                  else 'not completed')
    elif cfg.action == 'evaluate':
        evaluate(cfg)
    elif cfg.action in METHODS:
        run_one(cfg)
    else:
        raise ValueError(f'Unknown action: {cfg.action}')


if __name__ == '__main__':
    main()
