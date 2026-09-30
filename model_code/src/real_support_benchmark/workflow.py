"""Serial, bounded 14-fit real-data queue with frozen inputs and isolated evaluation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from jump_niche_rewire_v2.execution import atomic_json
from jump_niche_rewire_v2.supervisor import (SupervisorConfig, process_record,
                                              run_isolated, verified_live)

from .prepare import check_resources, sha
from .workers import BlindFitSettings, METHODS, fit


DATASETS = ('gse165141_liver', 'gse205055_p22')


def _plan(cfg) -> dict:
    files = {}
    for dataset in DATASETS:
        root = cfg.output_root / '01_model_input' / dataset
        reference = cfg.output_root / '02_reference' / dataset
        for name in ('expression.csv', 'coordinates.csv', 'metadata.csv', 'tf_list.csv',
                     'lr_pairs.csv', 'prior_tf_gene.csv', 'prior_lr_tf.csv',
                     'evaluation_tfs.csv', 'evaluation_targets.csv'):
            path = root / name
            files[str(path.resolve())] = sha(path)
        for name in ('support_edges.csv', 'reference_audit.json'):
            path = reference / name
            files[str(path.resolve())] = sha(path)
    sources = [Path(__file__), Path(__file__).with_name('workers.py'),
               Path(__file__).with_name('tigress_two_tf_compat.R'),
               Path(__file__).with_name('prepare.py'), Path(__file__).with_name('references.py'),
               Path(__file__).parents[2] / 'configs/real_data/support_benchmark.py',
               Path(__file__).parents[1] / 'benchmark_optimization/official_r.R']
    files.update({str(path.resolve()): sha(path) for path in sources})
    from benchmark_optimization.workflow import fingerprint
    from configs.public_sim_data.benchmark_optimization import OptimizationConfig
    implementation = fingerprint(OptimizationConfig())
    if implementation != 'ab208a3cd1b16c6f0668c780d6dc4450d676556fd3e8c1540c453fc6526c0c53':
        raise ValueError('JNR implementation differs from accepted scMultiSim fit')
    return dict(schema='real_support_benchmark_1', datasets=list(DATASETS), methods=list(METHODS),
                files=files, seed=cfg.seed, jnr_implementation_sha256=implementation,
                jnr_parameters=dict(l1_penalty=cfg.l1_penalty, fusion_penalty=cfg.fusion_penalty,
                    group_penalty=cfg.group_penalty, ridge_penalty=cfg.ridge_penalty,
                    aggregation='mean_abs'),
                compare_views='refit base vs refit communication; separately same-fit zero-LR reference vs total',
                baseline_parameters='frozen round_02 cumulative development AP selections',
                truth_role='evaluation only; unobserved edges are not verified negatives',
                limits=dict(storage_bytes=cfg.cumulative_storage_limit,
                    worker_rss_bytes=cfg.worker_rss_limit, minimum_ram_bytes=cfg.minimum_available_ram,
                    task_seconds=cfg.task_limit_seconds, minimum_disk_free_bytes=cfg.minimum_disk_free))


def _settings(cfg) -> BlindFitSettings:
    return BlindFitSettings(cfg.output_root, cfg.rscript, cfg.r_library, cfg.r_software,
                            cfg.seed, cfg.l1_penalty, cfg.fusion_penalty,
                            cfg.group_penalty, cfg.ridge_penalty, cfg.spatial_permutations)


def _accepted(path: Path, dataset: str, method: str) -> bool:
    status_path = path / 'run_status.json'
    prediction = path / 'global_grn.parquet'
    if not status_path.exists():
        return False
    state = json.loads(status_path.read_text(encoding='utf-8'))
    result = state.get('result') or {}
    return (state.get('state') == 'completed' and result.get('dataset') == dataset
            and result.get('method') == method and prediction.is_file()
            and result.get('prediction_sha256') == sha(prediction))


def run(cfg, action: str):
    if action == 'check':
        result = check_resources(cfg)
        result.update(plan=_plan(cfg), fits_expected=len(DATASETS) * len(METHODS))
        return result
    if action == 'status':
        rows = []
        for dataset in DATASETS:
            for method in METHODS:
                path = cfg.output_root / '03_runs' / dataset / method / 'run_status.json'
                if path.is_file():
                    state = json.loads(path.read_text(encoding='utf-8'))
                    rows.append(dict(dataset=dataset, method=method, state=state.get('state'),
                                     peak_sampled_worker_tree_rss_bytes=state.get('peak_sampled_worker_tree_rss_bytes')))
        return rows
    if action == 'evaluate':
        from .evaluate import evaluate
        return evaluate(cfg)
    if action not in ('run', 'resume'):
        raise ValueError('Real-support main supports check/status/run/resume/evaluate')

    check_resources(cfg)
    expected = _plan(cfg)
    plan_path = cfg.output_root / '01_plan.json'
    if action == 'resume':
        archived = cfg.output_root / 'attempt_history' / 'engineering_01'
        archived_controller = archived / 'study_process.json'
        archived_failure = archived / 'gse165141_liver_TIGRESS' / 'run_status.json'
        if not (plan_path.is_file() and archived_controller.is_file() and archived_failure.is_file()):
            raise RuntimeError('Explicit resume requires preserved stopped controller and TIGRESS failure')
        former = json.loads(plan_path.read_text(encoding='utf-8'))
        stopped = json.loads(archived_controller.read_text(encoding='utf-8'))
        failed = json.loads(archived_failure.read_text(encoding='utf-8'))
        if stopped.get('state') != 'stopped' or failed.get('state') != 'failed':
            raise RuntimeError('Archived attempt does not document the expected stopped failure')
        if any(former[key] != expected[key] for key in former if key != 'files'):
            raise ValueError('Scientific plan changed; cannot resume as an engineering repair')
        allowed = {'workflow.py', 'workers.py', 'tigress_two_tf_compat.R'}
        for filename, digest in former['files'].items():
            if Path(filename).name not in allowed and expected['files'].get(filename) != digest:
                raise ValueError(f'Frozen input/reference/source changed: {filename}')
        if set(expected['files']) - set(former['files']) != {
                str(Path(__file__).with_name('tigress_two_tf_compat.R').resolve())}:
            raise ValueError('Unexpected file added to engineering revision')
        revision = cfg.output_root / '01_plan_revision_01.json'
        if revision.exists():
            if json.loads(revision.read_text(encoding='utf-8')) != expected:
                raise ValueError('Engineering revision changed after registration')
        else:
            atomic_json(revision, expected)
    elif plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Frozen input/reference/code plan changed; refusing run')
    else:
        atomic_json(plan_path, expected)
    controller = cfg.output_root / 'study_process.json'
    if controller.exists():
        old = json.loads(controller.read_text(encoding='utf-8'))
        if verified_live(old):
            raise RuntimeError('Verified real-support controller is already running')
        if old.get('state') not in ('completed',):
            raise RuntimeError('Previous controller did not complete; audit before any explicit resume')
        return old
    state = dict(state='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 **process_record(os.getpid()))
    with controller.open('x', encoding='utf-8') as stream:
        json.dump(state, stream)
    blind = _settings(cfg)
    try:
        for dataset in DATASETS:
            for method in METHODS:
                if _plan(cfg) != expected:
                    raise ValueError('Frozen inputs/reference/source changed during queue')
                root = cfg.output_root / '03_runs' / dataset / method
                if _accepted(root, dataset, method):
                    print('[cache]', dataset, method, flush=True)
                    continue
                if (root / 'run_status.json').exists():
                    raise RuntimeError(f'Prior failed/incomplete attempt requires audit, not automatic retry: {root}')
                print('[start]', dataset, method, flush=True)
                python = cfg.tree_python if method in ('GENIE3', 'GRNBoost2') else cfg.model_python
                result = run_isolated(fit, (blind, dataset, method), root, cfg,
                                      SupervisorConfig(python_executable=str(python)))
                if result.get('state') != 'completed' or not _accepted(root, dataset, method):
                    raise RuntimeError(f'Fit failed or was resource-stopped: {dataset} {method}: {result.get("state")}')
                atomic_json(cfg.output_root / 'latest_progress.json', dict(dataset=dataset,
                    method=method, state='completed', as_of_utc=datetime.now(timezone.utc).isoformat(),
                    completed=sum(_accepted(cfg.output_root / '03_runs' / d / m, d, m)
                                  for d in DATASETS for m in METHODS), expected=14))
                print('[finished]', dataset, method, flush=True)
        from .evaluate import evaluate
        table = evaluate(cfg)
        state.update(state='completed', finished_utc=datetime.now(timezone.utc).isoformat(),
                     table=str(table))
        atomic_json(controller, state)
        return table
    except BaseException as error:
        state.update(state='stopped', finished_utc=datetime.now(timezone.utc).isoformat(),
                     error_type=type(error).__name__, error=str(error))
        atomic_json(controller, state)
        raise
