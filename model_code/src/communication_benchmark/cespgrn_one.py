"""Audited single-dataset CeSpGRN fit with a parent resource supervisor.

Internal worker callable, not a new command-line experiment interface.
The fixed plan uses one author-published development dataset and no labels.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
DATASET = 'published_tree1_500_cells110_genes_sigma0.1_1'
COMMIT = '2fb222f8a26edf1bdae14a95b1491543f0aaa4e8'
BLOBS = {'g_admm.py':'448b49fb79acc798f1c6c7d54e5bdd28945ead7b',
         'kernel.py':'9179ff9c9f30f4aa6c4b863a4349383c0cfe57a2',
         'torch_sqrtm.py':'4de49f9c74662b341184b0b339c95c38dab5efc3'}


@dataclass(frozen=True)
class SingleCeSpGRN:
    dataset_id: str = DATASET
    kernel_input: str = 'spatial'
    kernel_k: int = 5
    bandwidth: float = 1.
    neighbors: int = 30
    lamb: float = .1
    max_iters: int = 1000
    batch_size: int = 32
    seed: int = 20260923
    worker_rss_limit: int = int(3.5*1024**3)
    minimum_available_ram: int = int(1.5*1024**3)
    minimum_disk_free: int = 5*1024**3
    cumulative_storage_limit: int = 9*1024**3
    storage_reserve: int = 256*1024**2
    task_seconds: int = 4*3600
    budget_roots: tuple[Path, ...] = (
        ROOT/'data/raw/publication_20260920',
        ROOT/'data/processed/publication_20260920',
        ROOT/'data/result/publication_20260920',
        ROOT/'data/raw/optimization_20260921',
        ROOT/'data/processed/optimization_20260921',
        ROOT/'data/result/optimization_20260921',
        Path('C:/Anaconda/envs/jnr-cespgrn'),
    )
    source_dir: Path = ROOT/'vendor/external_methods/CeSpGRN/src'
    dataset_dir: Path = ROOT/'data/processed/publication_20260920'/DATASET
    output_dir: Path = ROOT/'data/result/optimization_20260921/single_dataset_extension/CeSpGRN/spatial_default'/DATASET


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _blob(path):
    content=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()


def _atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    temporary.replace(path)


def _tree_bytes(root):
    total=0
    for directory,_,files in os.walk(root):
        for name in files:
            try: total+=Path(directory,name).stat().st_size
            except FileNotFoundError: pass
    return total


def _budget_bytes(cfg):
    roots=sorted({path.resolve() for path in cfg.budget_roots},key=lambda path:len(path.parts))
    unique=[]
    for root in roots:
        if not any(root==parent or parent in root.parents for parent in unique):
            unique.append(root)
    return sum(_tree_bytes(root) for root in unique)


def verify(cfg):
    if cfg.dataset_id!=DATASET or cfg.kernel_input!='spatial':
        raise ValueError('This registered one-dataset candidate is fixed')
    manifest=json.loads((cfg.dataset_dir/'evidence_manifest.json').read_text(encoding='utf8'))
    if manifest['dataset_id']!=DATASET:
        raise ValueError('Wrong author-published dataset')
    inputs={}
    for name in ('expression.csv','coordinates.csv','metadata.csv','tf_list.csv','lr_pairs.csv'):
        actual=_sha(cfg.dataset_dir/name)
        if actual!=manifest['files'][name]: raise ValueError(f'Input checksum changed: {name}')
        inputs[name]=actual
    for name,expected in BLOBS.items():
        if _blob(cfg.source_dir/name)!=expected:
            raise ValueError(f'Official pinned source blob changed: {name}')
    return dict(dataset=DATASET,input_sha256=inputs,author_commit=COMMIT,
                official_git_blobs=BLOBS,parameters={
                    'kernel_input':cfg.kernel_input,'kernel_k':cfg.kernel_k,
                    'bandwidth':cfg.bandwidth,'neighbors':cfg.neighbors,'lamb':cfg.lamb,
                    'max_iters':cfg.max_iters,'batch_size':cfg.batch_size,'seed':cfg.seed,
                    'covariance_weighted_kt':True,'device':'cpu'},
                inference_truth_files_read=False,
                model_task='undirected cell-specific association; TF projection to common GRN edges')


def _worker(cfg):
    import sys
    import numpy as np
    import pandas as pd
    import torch
    from types import SimpleNamespace
    from independent_baselines.io import inputs
    from sklearn.decomposition import PCA

    sys.path.insert(0,str(cfg.source_dir))
    import g_admm, kernel
    status=cfg.output_dir/'run_status.json'
    with (cfg.output_dir/'worker.log').open('w',encoding='utf8',buffering=1) as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            try:
                provenance=verify(cfg)
                row={'dataset_dir':str(cfg.dataset_dir),'dataset_id':DATASET}
                data=inputs(row,SimpleNamespace(expression_scale='raw_counts',
                    target_sum=10000.,gene_panel_file=None))
                values=data['log'].to_numpy(dtype=np.float64)
                genes=list(data['log'].columns)
                tfs=list(data['tfs'])
                targets=[gene for gene in genes if gene not in set(tfs)]
                coordinates=data['coordinates'].to_numpy(dtype=np.float64)
                n,g=values.shape
                if (n,g,len(tfs),len(targets))!=(500,110,6,104):
                    raise ValueError('Common input universe changed')
                torch.manual_seed(cfg.seed)
                np.random.seed(cfg.seed)
                K,K_trun=kernel.calc_kernel_neigh(coordinates,k=cfg.kernel_k,
                    bandwidth=cfg.bandwidth,truncate=True,truncate_param=cfg.neighbors)
                covariance=g_admm.est_cov(X=values,K_trun=K_trun,weighted_kt=True)
                model=g_admm.G_admm_minibatch(X=values[:,None,:],K=K,pre_cov=covariance,
                    batchsize=cfg.batch_size,seed=cfg.seed,device=torch.device('cpu'))
                graphs=np.asarray(model.train(max_iters=cfg.max_iters,n_intervals=100,
                                              lamb=cfg.lamb,njobs=1))
                if graphs.shape!=(n,g,g) or not np.isfinite(graphs).all():
                    raise ValueError('Official network is missing or nonfinite')
                tf_ix=[genes.index(tf) for tf in tfs]
                target_ix=[genes.index(target) for target in targets]
                scores=np.mean(np.abs(graphs[:,tf_ix,:][:,:,target_ix]),axis=0)
                records=pd.DataFrame([(tf,target,float(scores[i,j]))
                    for i,tf in enumerate(tfs) for j,target in enumerate(targets)],
                    columns=['tf','target','score'])
                if len(records)!=624 or records.duplicated(['tf','target']).any():
                    raise ValueError('Incomplete TF-target projection')
                graph_tmp=cfg.output_dir/'cell_graphs.writing.npz'
                np.savez_compressed(graph_tmp,graphs=graphs,cells=np.asarray(data['log'].index),
                                    genes=np.asarray(genes))
                graph_tmp.replace(cfg.output_dir/'cell_graphs.npz')
                table_tmp=cfg.output_dir/'global_grn.writing.parquet'
                records.to_parquet(table_tmp,index=False)
                table_tmp.replace(cfg.output_dir/'global_grn.parquet')
                _atomic_json(status,dict(state='completed',dataset_id=DATASET,
                    method='CeSpGRN',candidate='spatial_default',source=provenance,
                    cells=n,genes=g,tf_target_edges=len(records),
                    graph_sha256=_sha(cfg.output_dir/'cell_graphs.npz'),
                    global_sha256=_sha(cfg.output_dir/'global_grn.parquet'),
                    full_method_native_lr_decomposition=False))
            except Exception as error:
                _atomic_json(status,dict(state='failed',error_type=type(error).__name__,
                    error=str(error),traceback=traceback.format_exc()))
                raise


def run(cfg=None):
    import psutil
    cfg=cfg or SingleCeSpGRN()
    plan=verify(cfg)
    result=cfg.output_dir/'run_status.json'
    if result.exists():
        prior=json.loads(result.read_text(encoding='utf8'))
        if prior.get('state')=='completed':
            raise FileExistsError('Completed fit exists; reuse and evaluate, do not refit')
        raise RuntimeError('Prior attempted fit exists; audit before explicit resume')
    if psutil.virtual_memory().available<cfg.minimum_available_ram:
        raise MemoryError('Available RAM below limit before start')
    if shutil.disk_usage(cfg.output_dir.anchor).free<cfg.minimum_disk_free:
        raise MemoryError('Disk free below limit before start')
    initial_bytes=_budget_bytes(cfg)
    if initial_bytes+cfg.storage_reserve>cfg.cumulative_storage_limit:
        raise MemoryError('Cumulative storage plus output reserve exceeds 9 GiB')
    cfg.output_dir.mkdir(parents=True,exist_ok=True)
    _atomic_json(cfg.output_dir/'preflight.json',dict(**plan,
        cumulative_storage_bytes_before=initial_bytes,
        cumulative_storage_roots=[str(p) for p in cfg.budget_roots],
        cumulative_storage_limit_bytes=cfg.cumulative_storage_limit,
        storage_reserve_bytes=cfg.storage_reserve,
        sampled_resource_limits_are_not_os_hard_caps=True))
    _atomic_json(result,dict(state='starting',dataset_id=DATASET,method='CeSpGRN',
                             candidate='spatial_default',started_utc=datetime.now(timezone.utc).isoformat()))
    process=mp.get_context('spawn').Process(target=_worker,args=(cfg,))
    process.start()
    started=time.monotonic()
    peak=0
    reason=None
    last_storage_check=0.
    storage_bytes=initial_bytes
    while process.is_alive():
        try:
            child=psutil.Process(process.pid)
            members=[child]+child.children(recursive=True)
            rss=sum(p.memory_info().rss for p in members if p.is_running())
            peak=max(peak,rss)
            if rss>cfg.worker_rss_limit: reason='worker_rss_limit'
            elif psutil.virtual_memory().available<cfg.minimum_available_ram: reason='minimum_available_ram'
            elif shutil.disk_usage(cfg.output_dir.anchor).free<cfg.minimum_disk_free: reason='minimum_disk_free'
            elif time.monotonic()-started>cfg.task_seconds: reason='task_seconds'
            if not reason and time.monotonic()-last_storage_check>60:
                storage_bytes=_budget_bytes(cfg)
                last_storage_check=time.monotonic()
                if storage_bytes>cfg.cumulative_storage_limit: reason='cumulative_storage_limit'
            if reason:
                for p in reversed(members):
                    try: p.terminate()
                    except psutil.Error: pass
                break
        except psutil.Error:
            pass
        process.join(timeout=.25)
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    state=json.loads(result.read_text(encoding='utf8'))
    if reason:
        state.update(state='resource_stopped',reason=reason,peak_sampled_rss_bytes=peak,
                     sampled_cumulative_storage_bytes=storage_bytes)
    else:
        state.update(peak_sampled_rss_bytes=peak,worker_exit_code=process.exitcode,
                     sampled_cumulative_storage_bytes=storage_bytes)
        if process.exitcode!=0 and state['state']!='failed':
            state.update(state='failed',reason='worker_exited_without_result')
    _atomic_json(result,state)
    return state
