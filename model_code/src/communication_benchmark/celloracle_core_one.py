"""One-dataset CellOracle Net-core comparison, with no truth-derived prior.

The official CellOracle 0.18.0 Net/regression source is loaded unchanged.  Its
package-level import depends on pysam, which does not build on this Windows
host; this adapter loads only the official GRN core modules needed here.
This is explicitly not the full Oracle/KNN-imputation workflow.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import contextlib
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import types

from .cespgrn_one import ROOT, DATASET, _atomic_json, _budget_bytes, _sha


SOURCE_SHA256 = {
    'network/net_core.py': 'f56e198f27878590b68ea52de8729924a935ba33ec7ee018a531cf440b1f8b72',
    'network/regression_models.py': '27eea89eda4fd601ce15e691b99338e56b3c9aab3271a9468116f4a590715bdf',
    'utility/utility.py': '743f09d578e1e15a0df731339ce7443e81dea5c6742c46cf623d31ff4c92fb8d',
    'utility/hdf5_processing.py': 'f74ad269f3cdd19a7c9cb436aaa30976f83bec217c4cf5618bb168fe8d09c2d4',
}


@dataclass(frozen=True)
class CellOracleCoreOne:
    dataset_id: str = DATASET
    version: str = '0.18.0'
    alpha: int = 10
    bagging_number: int = 20
    n_jobs: int = 1
    python_hash_seed: str = '0'
    worker_rss_limit: int = int(3.5 * 1024**3)
    minimum_available_ram: int = int(1.5 * 1024**3)
    minimum_disk_free: int = 5 * 1024**3
    cumulative_storage_limit: int = 9 * 1024**3
    storage_reserve: int = 256 * 1024**2
    task_seconds: int = 4 * 3600
    package_dir: Path = ROOT/'data/result/optimization_20260921/single_dataset_extension/CellOracle/venv/Lib/site-packages/celloracle'
    dataset_dir: Path = ROOT/'data/processed/publication_20260920'/DATASET
    output_dir: Path = ROOT/'data/result/optimization_20260921/single_dataset_extension/CellOracle/net_core_all_candidates'/DATASET
    budget_roots: tuple[Path, ...] = (
        ROOT/'data/raw/publication_20260920',
        ROOT/'data/processed/publication_20260920',
        ROOT/'data/result/publication_20260920',
        ROOT/'data/raw/optimization_20260921',
        ROOT/'data/processed/optimization_20260921',
        ROOT/'data/result/optimization_20260921',
        Path('C:/Anaconda/envs/jnr-cespgrn'),
        ROOT/'vendor/external_methods/RegDiffusion',
    )


def verify(cfg):
    if (cfg.dataset_id != DATASET or cfg.version != '0.18.0' or cfg.alpha != 10 or
            cfg.bagging_number != 20 or cfg.n_jobs != 1 or cfg.python_hash_seed != '0'):
        raise ValueError('CellOracle core plan changed')
    for relative, expected in SOURCE_SHA256.items():
        if _sha(cfg.package_dir/relative) != expected:
            raise ValueError(f'Official CellOracle source changed: {relative}')
    manifest = json.loads((cfg.dataset_dir/'evidence_manifest.json').read_text(encoding='utf8'))
    if manifest['dataset_id'] != DATASET:
        raise ValueError('Development dataset changed')
    inputs = {}
    for name in ('expression.csv', 'tf_list.csv'):
        digest = _sha(cfg.dataset_dir/name)
        if digest != manifest['files'][name]:
            raise ValueError(f'Input changed: {name}')
        inputs[name] = digest
    return dict(dataset_id=DATASET, celloracle_version=cfg.version,
                source_sha256=SOURCE_SHA256, input_sha256=inputs,
                inference_truth_files_read=False,
                all_tf_to_non_tf_pairs_as_uninformative_candidate_prior=True,
                official_algorithm='Net.fit_genes bagging_ridge',
                full_oracle_workflow=False, knn_imputation=False,
                preprocessing='log1p(10000 * counts / cell_library); input-only zero-variance filter',
                alpha=cfg.alpha, bagging_number=cfg.bagging_number,
                n_jobs=cfg.n_jobs, celloracle_internal_random_state=123,
                score='coef_abs, mean across official bagged ridge coefficients; absent edges zero')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_official_net(package_dir):
    """Load byte-verified official core without unrelated top-level imports."""
    base = 'celloracle_core_official'
    for name, path in ((base, package_dir),
                       (base+'.utility', package_dir/'utility'),
                       (base+'.network', package_dir/'network')):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    utility = _load_module(base+'.utility.utility', package_dir/'utility/utility.py')
    parent = sys.modules[base+'.utility']
    for name in ('standard', 'intersect', 'save_as_pickled_object'):
        setattr(parent, name, getattr(utility, name))
    _load_module(base+'.utility.hdf5_processing', package_dir/'utility/hdf5_processing.py')
    _load_module(base+'.network.regression_models', package_dir/'network/regression_models.py')
    core = _load_module(base+'.network.net_core', package_dir/'network/net_core.py')
    return core.Net


def _worker(cfg):
    import numpy as np
    import pandas as pd

    status_path = cfg.output_dir/'run_status.json'
    with (cfg.output_dir/'worker.log').open('w', encoding='utf8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                verify(cfg)
                Net = load_official_net(cfg.package_dir)
                expression = pd.read_csv(cfg.dataset_dir/'expression.csv', index_col='cell_id')
                tf = pd.read_csv(cfg.dataset_dir/'tf_list.csv')['tf'].astype(str).tolist()
                if len(expression) != 500 or len(expression.columns) != 110 or len(tf) != 6:
                    raise ValueError('Unexpected one-dataset dimensions')
                target = [gene for gene in expression if gene not in set(tf)]
                if len(target) != 104 or len(set(tf)) != 6:
                    raise ValueError('Unexpected common candidate universe')
                values = expression.to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Invalid expression counts')
                library = values.sum(axis=1)
                if (library <= 0).any():
                    raise ValueError('Zero cell library')
                normalized = pd.DataFrame(np.log1p(values / library[:, None] * 10000),
                                          index=expression.index, columns=expression.columns)
                removed = normalized.columns[normalized.var(axis=0).eq(0)].tolist()
                fitted_input = normalized.drop(columns=removed)
                eligible_targets = [gene for gene in target if gene not in removed]
                eligible_tf = [gene for gene in tf if gene not in removed]
                if not eligible_tf:
                    raise ValueError('All TFs have zero variance')
                prior = {gene: eligible_tf[:] for gene in eligible_targets}
                model = Net(gene_expression_matrix=fitted_input,
                            TFinfo_dic=prior, verbose=False)
                model.fit_genes(target_genes=eligible_targets,
                                bagging_number=cfg.bagging_number,
                                scaling=True, model_method='bagging_ridge',
                                alpha=cfg.alpha, n_jobs=cfg.n_jobs,
                                verbose=False)
                if model.failed_genes or set(model.fitted_genes) != set(eligible_targets):
                    raise RuntimeError(f'Incomplete CellOracle fit: {model.failed_genes}')
                raw = []
                for gene in eligible_targets:
                    scores = model.stats_dict[gene]
                    for regulator, row in scores.iterrows():
                        raw.append(dict(tf=str(regulator), target=gene,
                                        coef_mean=float(row.coef_mean),
                                        coef_abs=float(row.coef_abs),
                                        p=float(row.p)))
                raw_frame = pd.DataFrame(raw)
                if raw_frame.duplicated(['tf', 'target']).any():
                    raise ValueError('Duplicate raw network edges')
                universe = pd.MultiIndex.from_product([tf, target], names=['tf', 'target'])
                scores = raw_frame.set_index(['tf', 'target']).coef_abs.reindex(universe, fill_value=0)
                if len(scores) != 624 or not np.isfinite(scores.to_numpy()).all():
                    raise ValueError('Invalid complete score network')
                prediction = scores.rename('score').reset_index()
                raw_path = cfg.output_dir/'raw_grn.parquet'
                prediction_path = cfg.output_dir/'global_grn.parquet'
                raw_frame.to_parquet(raw_path, index=False)
                prediction.to_parquet(prediction_path, index=False)
                _atomic_json(status_path, dict(state='completed', dataset_id=DATASET,
                    method='CellOracle_Net_core', candidate='net_core_all_candidates',
                    fitted_targets=len(model.fitted_genes), removed_zero_variance=removed,
                    complete_edges=len(prediction), raw_edges=len(raw_frame),
                    raw_sha256=_sha(raw_path), global_sha256=_sha(prediction_path),
                    ended_utc=datetime.now(timezone.utc).isoformat()))
            except BaseException as error:
                _atomic_json(status_path, dict(state='failed', dataset_id=DATASET,
                    method='CellOracle_Net_core', reason=str(error),
                    traceback=traceback.format_exc(),
                    ended_utc=datetime.now(timezone.utc).isoformat()))
                raise


def run(cfg=None):
    import psutil
    cfg = cfg or CellOracleCoreOne()
    plan = verify(cfg)
    status_path = cfg.output_dir/'run_status.json'
    if status_path.exists():
        prior = json.loads(status_path.read_text(encoding='utf8'))
        if prior.get('state') == 'completed':
            raise FileExistsError('CellOracle fit already completed; reuse it')
        raise RuntimeError('Prior CellOracle attempt exists; audit before explicit resume')
    if psutil.virtual_memory().available < cfg.minimum_available_ram:
        raise MemoryError('Available RAM below 1.5 GiB')
    if shutil.disk_usage(cfg.output_dir.anchor).free < cfg.minimum_disk_free:
        raise MemoryError('Disk free below 5 GiB')
    initial = _budget_bytes(cfg)
    if initial + cfg.storage_reserve > cfg.cumulative_storage_limit:
        raise MemoryError('Cumulative storage plus reserve exceeds 9 GiB')
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(cfg.output_dir/'preflight.json', dict(**plan,
        cumulative_storage_bytes_before=initial,
        cumulative_storage_limit_bytes=cfg.cumulative_storage_limit,
        storage_reserve_bytes=cfg.storage_reserve,
        budget_roots=[str(root) for root in cfg.budget_roots],
        sampled_resource_limits_are_not_os_hard_caps=True))
    _atomic_json(status_path, dict(state='starting', dataset_id=DATASET,
        method='CellOracle_Net_core', started_utc=datetime.now(timezone.utc).isoformat()))
    previous_seed = os.environ.get('PYTHONHASHSEED')
    os.environ['PYTHONHASHSEED'] = cfg.python_hash_seed
    try:
        process = mp.get_context('spawn').Process(target=_worker, args=(cfg,))
        process.start()
    finally:
        if previous_seed is None:
            os.environ.pop('PYTHONHASHSEED', None)
        else:
            os.environ['PYTHONHASHSEED'] = previous_seed
    started = time.monotonic()
    peak = 0
    storage = initial
    reason = None
    last_storage = started
    while process.is_alive():
        try:
            member = psutil.Process(process.pid)
            members = [member] + member.children(recursive=True)
            rss = sum(p.memory_info().rss for p in members if p.is_running())
            peak = max(peak, rss)
            if rss > cfg.worker_rss_limit:
                reason = 'worker_rss_limit'
            elif psutil.virtual_memory().available < cfg.minimum_available_ram:
                reason = 'minimum_available_ram'
            elif shutil.disk_usage(cfg.output_dir.anchor).free < cfg.minimum_disk_free:
                reason = 'minimum_disk_free'
            elif time.monotonic() - started > cfg.task_seconds:
                reason = 'task_seconds'
            if reason is None and time.monotonic() - last_storage > 60:
                storage = _budget_bytes(cfg)
                last_storage = time.monotonic()
                if storage > cfg.cumulative_storage_limit:
                    reason = 'cumulative_storage_limit'
            if reason:
                for item in reversed(members):
                    try:
                        item.terminate()
                    except psutil.Error:
                        pass
                break
        except psutil.Error:
            pass
        process.join(timeout=.25)
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    status = json.loads(status_path.read_text(encoding='utf8'))
    if reason:
        status.update(state='resource_stopped', reason=reason,
                      peak_sampled_rss_bytes=peak, sampled_cumulative_storage_bytes=storage)
    else:
        status.update(peak_sampled_rss_bytes=peak, worker_exit_code=process.exitcode,
                      sampled_cumulative_storage_bytes=storage)
        if process.exitcode != 0 and status['state'] != 'failed':
            status.update(state='failed', reason='worker_exited_without_result')
    _atomic_json(status_path, status)
    return status
