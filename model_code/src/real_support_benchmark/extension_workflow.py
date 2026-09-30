"""Serial six-fit extension; no prior 14-fit result is overwritten or refitted."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import psutil

from jump_niche_rewire_v2.execution import atomic_json
from jump_niche_rewire_v2.supervisor import process_record, verified_live

from .prepare import check_resources, sha
from .extension_workers import METHODS


DATASETS = ('gse165141_liver', 'gse205055_p22')


def _plan(cfg) -> dict:
    from communication_benchmark.cespgrn_one import BLOBS, _blob
    from communication_benchmark.regdiffusion_one import SOURCE_MANIFEST_SHA256
    from communication_benchmark.celloracle_core_one import SOURCE_SHA256

    files: dict[str, str] = {}
    for dataset in DATASETS:
        for folder, names in (
            (cfg.base_root / '01_model_input' / dataset,
             ('expression.csv', 'coordinates.csv', 'evaluation_tfs.csv',
              'evaluation_targets.csv')),
            (cfg.base_root / '02_reference' / dataset,
             ('support_edges.csv', 'reference_audit.json')),
        ):
            for name in names:
                path = folder / name
                files[str(path.resolve())] = sha(path)
        for method in ('JumpNicheRewire_base', 'JumpNicheRewire_communication',
                       'PPCOR', 'CLR', 'TIGRESS', 'GRNBoost2', 'GENIE3'):
            root = cfg.base_root / '03_runs' / dataset / method
            for name in ('run_status.json', 'global_grn.parquet'):
                path = root / name
                files[str(path.resolve())] = sha(path)
    for path in (cfg.base_root / '01_plan.json',
                 cfg.base_root / '01_plan_revision_01.json'):
        files[str(path.resolve())] = sha(path)
    for path in (Path(__file__), Path(__file__).with_name('extension_workers.py'),
                 Path(__file__).with_name('cespgrn_worker39.py'),
                 Path(__file__).with_name('extension_worker310.py'),
                 Path(__file__).parents[2] / 'configs/real_data/support_benchmark_extension.py'):
        files[str(path.resolve())] = sha(path)
    for name in ('g_admm.py', 'kernel.py', 'torch_sqrtm.py'):
        path = cfg.cesp_source / name
        if _blob(path) != BLOBS[name]:
            raise ValueError('Pinned official CeSpGRN source changed: ' + name)
        files[str(path.resolve())] = sha(path)
    reg_blobs = {}
    for path in sorted(cfg.reg_source.joinpath('regdiffusion').rglob('*.py')):
        reg_blobs[str(path.relative_to(cfg.reg_source)).replace('\\', '/')] = _blob(path)
        files[str(path.resolve())] = sha(path)
    reg_digest = hashlib.sha256(json.dumps(reg_blobs, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    if len(reg_blobs) != 16 or reg_digest != SOURCE_MANIFEST_SHA256:
        raise ValueError('Pinned official RegDiffusion source changed')
    for name in ('network/net_core.py', 'network/regression_models.py',
                 'utility/utility.py', 'utility/hdf5_processing.py'):
        path = cfg.cell_package / name
        if sha(path) != SOURCE_SHA256[name]:
            raise ValueError('Pinned official CellOracle Net core source changed: ' + name)
        files[str(path.resolve())] = sha(path)
    return dict(schema='real_support_extension_1', datasets=list(DATASETS),
                methods=list(METHODS), files=files,
                fit_parameters=dict(CeSpGRN='spatial_default, seed=20260923',
                    RegDiffusion='official_default, CPU, 1000 steps, seed=20260923',
                    CellOracle_Net_core='all-TF uninformative prior, bagging-ridge, alpha=10, bags=20'),
                truths_read_by_fit=False,
                inherited_scientific_plan='existing 14 fits and reference labels hashed, not refit',
                limits=dict(storage_bytes=cfg.cumulative_storage_limit,
                            worker_rss_bytes=cfg.worker_rss_limit,
                            minimum_ram_bytes=cfg.minimum_available_ram,
                            task_seconds=cfg.task_limit_seconds,
                            minimum_disk_free_bytes=cfg.minimum_disk_free))


def _accepted(root: Path, dataset: str, method: str) -> bool:
    status_path = root / 'run_status.json'
    prediction = root / 'global_grn.parquet'
    if not status_path.is_file() or not prediction.is_file():
        return False
    status = json.loads(status_path.read_text(encoding='utf-8'))
    result = status.get('result') or {}
    return (status.get('state') == 'completed' and result.get('state') == 'completed'
            and result.get('dataset') == dataset and result.get('method') == method
            and result.get('prediction_sha256') == sha(prediction))


def _owned(process: psutil.Process, started: float) -> list[psutil.Process]:
    if abs(process.create_time() - started) >= .001:
        raise RuntimeError('Owned extension worker process identity changed')
    return [process] + process.children(recursive=True)


def _stop_owned(members: list[psutil.Process]) -> None:
    for item in reversed(members):
        try:
            item.terminate()
        except psutil.Error:
            pass
    for item in reversed(members):
        try:
            item.wait(timeout=5)
        except psutil.Error:
            pass
        except psutil.TimeoutExpired:
            try:
                item.kill()
            except psutil.Error:
                pass


def _external_fit(cfg, dataset: str, method: str, out: Path) -> dict:
    from communication_benchmark.cespgrn_one import BLOBS, _blob

    if method == 'CeSpGRN':
        for name, expected in BLOBS.items():
            if _blob(cfg.cesp_source / name) != expected:
                raise ValueError(f'Pinned CeSpGRN official source differs: {name}')
        python = cfg.cesp_python
        script = Path(__file__).with_name('cespgrn_worker39.py')
    elif method == 'RegDiffusion':
        python = cfg.reg_python
        script = Path(__file__).with_name('extension_worker310.py')
    elif method == 'CellOracle_Net_core':
        python = cfg.cell_python
        script = Path(__file__).with_name('extension_worker310.py')
    else:
        raise ValueError('Unregistered external method')
    root = cfg.base_root / '01_model_input' / dataset
    out.mkdir(parents=True, exist_ok=False)
    inputs = {name: sha(root / name) for name in
              ('expression.csv', 'coordinates.csv', 'evaluation_tfs.csv',
               'evaluation_targets.csv')}
    atomic_json(out / 'job.json', dict(dataset=dataset, method=method,
        input_root=str(root.resolve()), source_dir=str(cfg.cesp_source.resolve()),
        input_hashes=inputs))
    with (out / 'run.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([str(python), '-B', str(script)],
            cwd=out, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        process = psutil.Process(child.pid)
        started = process.create_time()
        state = dict(state='running', **process_record(child.pid),
                     started_epoch=time.time())
        atomic_json(out / 'run_status.json', state)
        start_mono = time.monotonic()
        peak = 0
        stop_reason = None
        last_storage = start_mono
        try:
            while child.poll() is None:
                try:
                    members = _owned(process, started)
                    peak = max(peak, sum(item.memory_info().rss for item in members
                                         if item.is_running()))
                except psutil.NoSuchProcess:
                    members = []
                if peak > cfg.worker_rss_limit:
                    stop_reason = 'worker-tree RSS exceeded 3.5 GiB'
                elif psutil.virtual_memory().available < cfg.minimum_available_ram:
                    stop_reason = 'available RAM below 1.5 GiB'
                elif shutil.disk_usage(cfg.output_root.anchor).free < cfg.minimum_disk_free:
                    stop_reason = 'free disk below 5 GiB'
                elif time.monotonic() - start_mono > cfg.task_limit_seconds:
                    stop_reason = 'task exceeded four hours'
                if time.monotonic() - last_storage >= 60 and not stop_reason:
                    try:
                        check_resources(cfg)
                    except (MemoryError, OSError) as error:
                        stop_reason = str(error)
                    last_storage = time.monotonic()
                if stop_reason:
                    if members:
                        _stop_owned(members)
                    break
                time.sleep(.25)
            child.wait(timeout=5)
        except BaseException:
            try:
                _stop_owned(_owned(process, started))
            except psutil.Error:
                pass
            raise
    result_path = out / 'fit_result.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
    state.update(state='resource_limit' if stop_reason else
                 'completed' if child.returncode == 0 and result and result.get('state') == 'completed'
                 else 'failed', result=result, exit_code=child.returncode,
                 error=stop_reason, peak_sampled_worker_tree_rss_bytes=peak,
                 runtime_seconds=time.monotonic()-start_mono,
                 memory_measurement='sampled RSS, not an OS hard cap',
                 finished_epoch=time.time())
    atomic_json(out / 'run_status.json', state)
    return state


def run(cfg, action: str):
    if action == 'check':
        return dict(resources=check_resources(cfg), plan=_plan(cfg))
    if action == 'status':
        return [(dataset, method,
                 json.loads((cfg.output_root / '03_runs' / dataset / method /
                             'run_status.json').read_text(encoding='utf-8')).get('state'))
                for dataset in DATASETS for method in METHODS
                if (cfg.output_root / '03_runs' / dataset / method / 'run_status.json').exists()]
    if action == 'evaluate':
        from .extension_evaluate import evaluate
        return evaluate(cfg)
    if action not in ('run', 'resume'):
        raise ValueError('Extension supports check/status/run/resume/evaluate')

    check_resources(cfg)
    expected = _plan(cfg)
    plan_path = cfg.output_root / '01_extension_plan.json'
    if action == 'run':
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding='utf-8')) != expected:
                raise ValueError('Frozen extension inputs, source or earlier results changed')
        else:
            cfg.output_root.mkdir(parents=True, exist_ok=True)
            atomic_json(plan_path, expected)
    else:
        if not plan_path.is_file():
            raise RuntimeError('Original extension plan missing; cannot resume')
        old_plan = json.loads(plan_path.read_text(encoding='utf-8'))
        old_files = old_plan.get('files', {})
        new_files = expected['files']
        workflow_path = str(Path(__file__).resolve())
        worker_path = str(Path(__file__).with_name('extension_worker310.py').resolve())
        if set(new_files) - set(old_files) != {worker_path} or set(old_files) - set(new_files):
            raise ValueError('Engineering revision has unexpected file scope')
        if any(old_files[path] != new_files[path]
               for path in old_files if path != workflow_path):
            raise ValueError('Scientific inputs, source, or earlier results changed')
        for key in old_plan.keys() | expected.keys():
            if key != 'files' and old_plan.get(key) != expected.get(key):
                raise ValueError('Scientific extension plan changed')
        archive = cfg.output_root / 'attempt_history' / 'engineering_01'
        required = (archive / 'study_process.json',
                    archive / 'gse165141_liver_RegDiffusion' / 'run_status.json',
                    archive / 'full_extension.log', archive / 'full_extension.err.log')
        if any(not item.is_file() for item in required):
            raise RuntimeError('Stopped controller/failed fit audit archive incomplete')
        stopped = json.loads(required[0].read_text(encoding='utf-8'))
        failed = json.loads(required[1].read_text(encoding='utf-8'))
        if stopped.get('state') != 'stopped' or failed.get('state') == 'completed':
            raise RuntimeError('Prior stopped attempt not verified')
        revision_path = cfg.output_root / '01_extension_plan_revision_01.json'
        revision = dict(original_plan_sha256=sha(plan_path),
                        allowed_changed_file=workflow_path,
                        added_worker_file=worker_path,
                        reason='Correct cross-interpreter launch; no fit parameters changed',
                        expected_plan=expected)
        if revision_path.is_file():
            if json.loads(revision_path.read_text(encoding='utf-8')) != revision:
                raise ValueError('Engineering revision changed')
        else:
            atomic_json(revision_path, revision)
    controller = cfg.output_root / 'study_process.json'
    if controller.exists():
        old = json.loads(controller.read_text(encoding='utf-8'))
        if verified_live(old):
            raise RuntimeError('Verified extension controller already running')
        if old.get('state') != 'completed':
            raise RuntimeError('Prior extension did not complete; audit before resume')
        return old
    state = dict(state='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 **process_record(os.getpid()))
    with controller.open('x', encoding='utf-8') as stream:
        json.dump(state, stream)
    try:
        for dataset in DATASETS:
            for method in METHODS:
                if _plan(cfg) != expected:
                    raise ValueError('Frozen extension plan changed during queue')
                out = cfg.output_root / '03_runs' / dataset / method
                if _accepted(out, dataset, method):
                    print('[cache]', dataset, method, flush=True)
                    continue
                if out.exists():
                    raise RuntimeError(f'Unaccepted prior task requires audit: {out}')
                print('[start]', dataset, method, flush=True)
                result = _external_fit(cfg, dataset, method, out)
                if result.get('state') != 'completed' or not _accepted(out, dataset, method):
                    raise RuntimeError(f'Extension fit failed or resource-stopped: {dataset} {method}')
                atomic_json(cfg.output_root / 'latest_progress.json', dict(
                    dataset=dataset, method=method, completed=sum(
                        _accepted(cfg.output_root / '03_runs' / d / m, d, m)
                        for d in DATASETS for m in METHODS), expected=6,
                    as_of_utc=datetime.now(timezone.utc).isoformat()))
                print('[finished]', dataset, method, flush=True)
        state.update(state='completed', finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(controller, state)
        return controller
    except BaseException as error:
        state.update(state='stopped', finished_utc=datetime.now(timezone.utc).isoformat(),
                     error_type=type(error).__name__, error=str(error),
                     traceback=traceback.format_exc())
        atomic_json(controller, state)
        raise
