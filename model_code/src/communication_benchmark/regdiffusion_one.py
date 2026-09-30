"""One locked, resource-supervised RegDiffusion development fit."""
from dataclasses import dataclass
from datetime import datetime, timezone
import contextlib
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import traceback

from .cespgrn_one import ROOT, DATASET, _sha, _blob, _atomic_json, _budget_bytes

COMMIT = '0af2d95a8751df791f7efb07825087a6c3ac3fe6'
SOURCE_MANIFEST_SHA256 = '78941f7c2acc820b61fcce4d6c32cac8ca6b16b1b804c41d694b3b684e745e3b'


@dataclass(frozen=True)
class SingleRegDiffusion:
    dataset_id: str = DATASET
    seed: int = 20260923
    n_steps: int = 1000
    device: str = 'cpu'
    worker_rss_limit: int = int(3.5 * 1024**3)
    minimum_available_ram: int = int(1.5 * 1024**3)
    minimum_disk_free: int = 5 * 1024**3
    cumulative_storage_limit: int = 9 * 1024**3
    storage_reserve: int = 256 * 1024**2
    task_seconds: int = 4 * 3600
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
    source_dir: Path = ROOT/'vendor/external_methods/RegDiffusion'
    dataset_dir: Path = ROOT/'data/processed/publication_20260920'/DATASET
    output_dir: Path = ROOT/'data/result/optimization_20260921/single_dataset_extension/RegDiffusion/official_default'/DATASET


def verify(cfg):
    if cfg.dataset_id != DATASET or cfg.seed != 20260923 or cfg.n_steps != 1000 or cfg.device != 'cpu':
        raise ValueError('Locked RegDiffusion plan changed')
    manifest = json.loads((cfg.dataset_dir/'evidence_manifest.json').read_text(encoding='utf8'))
    if manifest['dataset_id'] != DATASET:
        raise ValueError('Dataset changed')
    inputs = {}
    for name in ('expression.csv', 'coordinates.csv', 'metadata.csv', 'tf_list.csv', 'lr_pairs.csv'):
        actual = _sha(cfg.dataset_dir/name)
        if actual != manifest['files'][name]:
            raise ValueError(f'Input changed: {name}')
        inputs[name] = actual
    source = {}
    for path in sorted((cfg.source_dir/'regdiffusion').rglob('*.py')):
        source[str(path.relative_to(cfg.source_dir)).replace('\\','/')] = _blob(path)
    if len(source) != 16:
        raise ValueError(f'Expected 16 official Python files; got {len(source)}')
    digest = hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest != SOURCE_MANIFEST_SHA256:
        raise ValueError('Pinned official source manifest changed')
    return dict(dataset=DATASET,input_sha256=inputs,author_commit=COMMIT,
                official_git_blobs=source,source_manifest_sha256=digest,
                parameters=dict(n_steps=cfg.n_steps,
                seed=cfg.seed,device=cfg.device,other_hyperparameters='official defaults',
                normalization='log1p(10000*counts/library)',
                zero_variance_filter='input-only; zero score on removed genes'),
                inference_truth_files_read=False)


def _worker(cfg):
    import sys
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from independent_baselines.io import inputs
    sys.path.insert(0, str(cfg.source_dir))
    from regdiffusion import RegDiffusionTrainer
    from regdiffusion.logger import LightLogger
    status = cfg.output_dir/'run_status.json'
    with (cfg.output_dir/'worker.log').open('w',encoding='utf8',buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                provenance = verify(cfg)
                data = inputs({'dataset_dir':str(cfg.dataset_dir),'dataset_id':DATASET},
                    SimpleNamespace(expression_scale='raw_counts',target_sum=10000.,gene_panel_file=None))
                matrix = data['log'].to_numpy(dtype=np.float32)
                genes = list(data['log'].columns)
                tfs = list(data['tfs'])
                targets = [g for g in genes if g not in set(tfs)]
                if (matrix.shape,len(tfs),len(targets)) != ((500,110),6,104):
                    raise ValueError('Common input universe changed')
                keep = np.var(matrix,axis=0) > 0
                removed = [g for g,k in zip(genes,keep) if not k]
                if removed != ['gene84']:
                    raise ValueError(f'Unexpected zero-variance genes: {removed}')
                fit_genes = [g for g,k in zip(genes,keep) if k]
                logger = LightLogger(result_dir=str(cfg.output_dir/'train_logs'))
                trainer = RegDiffusionTrainer(matrix[:,keep],device=cfg.device,
                    seed=cfg.seed,n_steps=cfg.n_steps,logger=logger)
                trainer.train()
                adj = np.asarray(trainer.get_adj(),dtype=np.float32)
                if adj.shape != (109,109) or not np.isfinite(adj).all():
                    raise ValueError('Official adjacency missing or nonfinite')
                ix = {g:i for i,g in enumerate(fit_genes)}
                records = pd.DataFrame([(tf,target,float(abs(adj[ix[tf],ix[target]]))
                    if tf in ix and target in ix else 0.)
                    for tf in tfs for target in targets],columns=['tf','target','score'])
                if len(records) != 624 or records.duplicated(['tf','target']).any():
                    raise ValueError('Incomplete common edge universe')
                tmp = cfg.output_dir/'adjacency.writing.npz'
                np.savez_compressed(tmp,adjacency=adj,genes=np.asarray(fit_genes),
                    removed_zero_variance=np.asarray(removed))
                tmp.replace(cfg.output_dir/'adjacency.npz')
                table_tmp = cfg.output_dir/'global_grn.writing.parquet'
                records.to_parquet(table_tmp,index=False)
                table_tmp.replace(cfg.output_dir/'global_grn.parquet')
                _atomic_json(status,dict(state='completed',dataset_id=DATASET,
                    method='RegDiffusion',candidate='official_default',source=provenance,
                    fit_genes=len(fit_genes),removed_zero_variance=removed,
                    tf_target_edges=len(records),
                    adjacency_sha256=_sha(cfg.output_dir/'adjacency.npz'),
                    global_sha256=_sha(cfg.output_dir/'global_grn.parquet'),
                    full_method_native_lr_decomposition=False))
            except Exception as error:
                _atomic_json(status,dict(state='failed',error_type=type(error).__name__,
                    error=str(error),traceback=traceback.format_exc()))
                raise


def run(cfg=None):
    import psutil
    cfg = cfg or SingleRegDiffusion()
    plan = verify(cfg)
    result = cfg.output_dir/'run_status.json'
    if result.exists():
        prior = json.loads(result.read_text(encoding='utf8'))
        if prior.get('state') == 'completed':
            raise FileExistsError('Completed fit exists; reuse, do not refit')
        raise RuntimeError('Prior attempt exists; audit before explicit resume')
    if psutil.virtual_memory().available < cfg.minimum_available_ram:
        raise MemoryError('Available RAM below limit before start')
    if shutil.disk_usage(cfg.output_dir.anchor).free < cfg.minimum_disk_free:
        raise MemoryError('Disk free below limit before start')
    initial = _budget_bytes(cfg)
    if initial + cfg.storage_reserve > cfg.cumulative_storage_limit:
        raise MemoryError('Cumulative storage plus reserve exceeds 9 GiB')
    cfg.output_dir.mkdir(parents=True,exist_ok=True)
    _atomic_json(cfg.output_dir/'preflight.json',dict(**plan,
        cumulative_storage_bytes_before=initial,
        cumulative_storage_roots=[str(p) for p in cfg.budget_roots],
        cumulative_storage_limit_bytes=cfg.cumulative_storage_limit,
        storage_reserve_bytes=cfg.storage_reserve,
        sampled_resource_limits_are_not_os_hard_caps=True))
    _atomic_json(result,dict(state='starting',dataset_id=DATASET,method='RegDiffusion',
        candidate='official_default',started_utc=datetime.now(timezone.utc).isoformat()))
    process = mp.get_context('spawn').Process(target=_worker,args=(cfg,))
    process.start()
    started = time.monotonic()
    peak = 0
    reason = None
    last_storage = 0.
    storage = initial
    while process.is_alive():
        try:
            child = psutil.Process(process.pid)
            members = [child] + child.children(recursive=True)
            rss = sum(p.memory_info().rss for p in members if p.is_running())
            peak = max(peak,rss)
            if rss > cfg.worker_rss_limit: reason = 'worker_rss_limit'
            elif psutil.virtual_memory().available < cfg.minimum_available_ram: reason = 'minimum_available_ram'
            elif shutil.disk_usage(cfg.output_dir.anchor).free < cfg.minimum_disk_free: reason = 'minimum_disk_free'
            elif time.monotonic()-started > cfg.task_seconds: reason = 'task_seconds'
            if not reason and time.monotonic()-last_storage > 60:
                storage = _budget_bytes(cfg)
                last_storage = time.monotonic()
                if storage > cfg.cumulative_storage_limit: reason = 'cumulative_storage_limit'
            if reason:
                for member in reversed(members):
                    try: member.terminate()
                    except psutil.Error: pass
                break
        except psutil.Error:
            pass
        process.join(timeout=.25)
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    state = json.loads(result.read_text(encoding='utf8'))
    if reason:
        state.update(state='resource_stopped',reason=reason,peak_sampled_rss_bytes=peak,
                     sampled_cumulative_storage_bytes=storage)
    else:
        state.update(peak_sampled_rss_bytes=peak,worker_exit_code=process.exitcode,
                     sampled_cumulative_storage_bytes=storage)
        if process.exitcode != 0 and state['state'] != 'failed':
            state.update(state='failed',reason='worker_exited_without_result')
    _atomic_json(result,state)
    return state
