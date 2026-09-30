"""One-entry, one-config benchmark on fixed scMultiSim and real-data inputs.

Fit workers never read reference labels. Existing historical results stay read-only.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from configs.benchmark_selected import LIVER, METHODS, P22, SCMULTISIM
from jump_niche_rewire_v2.execution import atomic_json
from jump_niche_rewire_v2.persistence import ResourceGuard
from jump_niche_rewire_v2.supervisor import (SupervisorConfig, process_record,
                                              run_isolated, verified_live)
from real_support_benchmark.prepare import sha


REAL_INPUT_FILES = ('expression.csv', 'coordinates.csv', 'metadata.csv', 'tf_list.csv',
                    'lr_pairs.csv', 'prior_tf_gene.csv', 'prior_lr_tf.csv',
                    'evaluation_tfs.csv', 'evaluation_targets.csv')
SC_INPUT_FILES = ('expression.csv', 'coordinates.csv', 'metadata.csv', 'tf_list.csv',
                  'lr_pairs.csv', 'prior_tf_gene.csv', 'prior_lr_tf.csv',
                  'evaluation_targets.csv')
REAL_BASE = ('JumpNicheRewire_base', 'JumpNicheRewire_communication',
             'PPCOR', 'CLR', 'TIGRESS', 'GRNBoost2', 'GENIE3')
REAL_EXTENSION = ('CeSpGRN', 'RegDiffusion', 'CellOracle_Net_core')


def _tasks(cfg):
    if not cfg.datasets or len(set(cfg.datasets)) != len(cfg.datasets):
        raise ValueError('Select at least one distinct dataset')
    if not cfg.methods or len(set(cfg.methods)) != len(cfg.methods):
        raise ValueError('Select at least one distinct method')
    unknown = set(cfg.datasets) - {SCMULTISIM, LIVER, P22}
    unknown_methods = set(cfg.methods) - set(METHODS)
    if unknown or unknown_methods:
        raise ValueError(f'Unknown datasets={sorted(unknown)}, methods={sorted(unknown_methods)}')
    if cfg.action not in ('check', 'run', 'status', 'evaluate'):
        raise ValueError('SELECTED.action must be check/run/status/evaluate')
    for dataset in cfg.datasets:
        for method in cfg.methods:
            if dataset != SCMULTISIM and method == 'JumpNicheRewire':
                yield dataset, 'JumpNicheRewire_base'
                yield dataset, 'JumpNicheRewire_communication'
            else:
                yield dataset, method


def _python(cfg, method):
    if method in ('GENIE3', 'GRNBoost2'):
        return cfg.tree_python
    if method == 'CeSpGRN':
        return cfg.cesp_python
    if method == 'RegDiffusion':
        return cfg.reg_python
    if method == 'CellOracle_Net_core':
        return cfg.cell_python
    return cfg.model_python


def _sc_settings(cfg, method):
    from configs.public_sim_data.manual_scmultisim import ManualScMultiSim
    return replace(ManualScMultiSim(), action=method, dataset_dir=cfg.sc_input,
                   output_root=cfg.output_root/'03_runs'/SCMULTISIM,
                   model_python=cfg.model_python, tree_python=cfg.tree_python,
                   regdiffusion_python=cfg.reg_python, rscript=cfg.rscript,
                   r_library=cfg.r_library, r_software=cfg.r_software,
                   cesp_source=cfg.cesp_source, reg_source=cfg.reg_source,
                   cell_package=cfg.cell_package,
                   maximum_new_gib=cfg.maximum_new_gib,
                   maximum_worker_rss_gib=cfg.maximum_worker_rss_gib,
                   minimum_available_ram_gib=cfg.minimum_available_ram_gib,
                   minimum_free_disk_gib=cfg.minimum_free_disk_gib,
                   maximum_task_hours=cfg.maximum_task_hours,
                   storage_reserve_bytes=cfg.storage_reserve_bytes,
                   budget_roots=cfg.budget_roots)


def _sc_worker(settings):
    from reports.run_manual_scmultisim import output_dir, run_one
    run_one(settings)
    out = output_dir(settings, settings.action)
    status = json.loads((out/'run_status.json').read_text(encoding='utf-8'))
    return {'state': status['state'], 'dataset': SCMULTISIM,
            'method': settings.action, 'prediction_sha256': sha(out/'global_grn.parquet')}


def _real_base_settings(cfg):
    from real_support_benchmark.workers import BlindFitSettings
    return BlindFitSettings(cfg.output_root, cfg.rscript, cfg.r_library,
                            cfg.r_software, cfg.seed, cfg.l1_penalty,
                            cfg.fusion_penalty, cfg.group_penalty,
                            cfg.ridge_penalty, cfg.spatial_permutations)


def _output(cfg, dataset, method):
    if dataset == SCMULTISIM:
        from reports.run_manual_scmultisim import output_dir
        return output_dir(_sc_settings(cfg, method), method)
    return cfg.output_root/'03_runs'/dataset/method


def _accepted(cfg, dataset, method):
    out = _output(cfg, dataset, method)
    status_path, prediction = out/'run_status.json', out/'global_grn.parquet'
    if not status_path.is_file() or not prediction.is_file():
        return False
    status = json.loads(status_path.read_text(encoding='utf-8'))
    result = status.get('result') or status
    if status.get('state') != 'completed':
        return False
    if dataset == SCMULTISIM:
        digest = result.get('prediction_sha256') or result.get('global_sha256')
        return digest == sha(prediction)
    return (result.get('state') == 'completed' and result.get('dataset') == dataset
            and result.get('method') == method and
            result.get('prediction_sha256') == sha(prediction))


def _plan(cfg):
    tasks = list(_tasks(cfg))
    files = {}
    for dataset in cfg.datasets:
        if dataset == SCMULTISIM:
            for name in SC_INPUT_FILES + ('evidence_manifest.json',):
                path = cfg.sc_input/name
                files[str(path.resolve())] = sha(path)
        else:
            for name in REAL_INPUT_FILES:
                path = cfg.real_input_root/dataset/name
                files[str(path.resolve())] = sha(path)
    code = (Path(__file__), Path(__file__).parents[2]/'reports/run_manual_scmultisim.py',
            Path(__file__).with_name('sc_worker.py'),
            Path(__file__).with_name('real_worker310.py'),
            Path(__file__).parents[1]/'real_support_benchmark/workers.py',
            Path(__file__).parents[1]/'real_support_benchmark/extension_workers.py',
            Path(__file__).parents[1]/'real_support_benchmark/cespgrn_worker39.py',
            Path(__file__).parents[1]/'real_support_benchmark/extension_worker310.py',
            Path(__file__).parents[1]/'real_support_benchmark/tigress_two_tf_compat.R',
            Path(__file__).parents[1]/'benchmark_optimization/official_r.R',
            Path(__file__).parents[1]/'communication_benchmark/jnr_one_hour.py',
            Path(__file__).parents[1]/'communication_benchmark/cespgrn_one.py',
            Path(__file__).parents[1]/'communication_benchmark/regdiffusion_one.py',
            Path(__file__).parents[1]/'communication_benchmark/celloracle_core_one.py')
    files.update({str(path.resolve()): sha(path) for path in code})
    if 'CeSpGRN' in cfg.methods:
        for name in ('g_admm.py', 'kernel.py', 'torch_sqrtm.py'):
            path = cfg.cesp_source/name
            files[str(path.resolve())] = sha(path)
    if 'RegDiffusion' in cfg.methods:
        for path in sorted((cfg.reg_source/'regdiffusion').rglob('*.py')):
            files[str(path.resolve())] = sha(path)
    if 'CellOracle_Net_core' in cfg.methods:
        for name in ('network/net_core.py', 'network/regression_models.py',
                     'utility/utility.py', 'utility/hdf5_processing.py'):
            path = cfg.cell_package/name
            files[str(path.resolve())] = sha(path)
    selection = asdict(cfg)
    selection.pop('action')
    return json.loads(json.dumps({'schema': 'selected_benchmark_1', 'selection': selection,
            'expanded_tasks': tasks, 'files': files,
            'reference_read_by_fit': False,
            'reference_role': 'evaluation only; unsupported edges are not verified negatives'},
            default=str))


def _preflight(cfg, tasks):
    if cfg.maximum_new_gib != 15 or cfg.maximum_worker_rss_gib != 3.5:
        raise ValueError('Approved cumulative/worker limits are 15 GiB / 3.5 GiB')
    for dataset, method in tasks:
        executable = _python(cfg, method)
        if not executable.is_file():
            raise FileNotFoundError(f'{method} interpreter missing: {executable}')
        if method in ('PPCOR', 'CLR', 'TIGRESS') and not cfg.rscript.is_file():
            raise FileNotFoundError(f'Rscript missing: {cfg.rscript}')
        if method in ('PPCOR', 'CLR', 'TIGRESS') and not (cfg.r_library.is_dir() and cfg.r_software.is_dir()):
            raise FileNotFoundError('Pinned R library/software directory missing')
    guard = ResourceGuard(cfg)
    guard(reserve_bytes=cfg.storage_reserve_bytes)
    return guard.cumulative_bytes


def _copy_real_inputs(cfg, dataset):
    destination = cfg.output_root/'01_model_input'/dataset
    for name in REAL_INPUT_FILES:
        source, target = cfg.real_input_root/dataset/name, destination/name
        if target.exists():
            if sha(target) != sha(source):
                raise ValueError(f'Existing prepared input differs: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name('.'+target.name+'.writing')
        if temporary.exists():
            raise RuntimeError(f'Unfinished copy requires audit: {temporary}')
        shutil.copyfile(source, temporary)
        if sha(temporary) != sha(source):
            raise IOError(f'Input copy checksum differs: {source}')
        temporary.replace(target)


def _fit_one(cfg, dataset, method):
    out = _output(cfg, dataset, method)
    if _accepted(cfg, dataset, method):
        print('[cache]', dataset, method, flush=True)
        return
    if out.exists():
        raise RuntimeError(f'Existing incomplete attempt requires audit, not automatic retry: {out}')
    print('[start]', dataset, method, 'environment=', _python(cfg, method), flush=True)
    if dataset == SCMULTISIM:
        settings = _sc_settings(cfg, method)
        if method in ('CeSpGRN', 'RegDiffusion', 'CellOracle_Net_core'):
            state = _external_sc(cfg, settings, method)
        else:
            monitor = cfg.output_root/'00_monitor'/dataset/method
            if monitor.exists():
                raise RuntimeError(f'Prior monitor attempt requires audit: {monitor}')
            state = run_isolated(_sc_worker, (settings,), monitor, cfg,
                                 SupervisorConfig(python_executable=str(_python(cfg, method))))
    elif method in REAL_BASE:
        from real_support_benchmark.workers import fit
        state = run_isolated(fit, (_real_base_settings(cfg), dataset, method),
                             out, cfg, SupervisorConfig(python_executable=str(_python(cfg, method))))
    else:
        state = _external_real(cfg, dataset, method, out)
    if state.get('state') != 'completed' or not _accepted(cfg, dataset, method):
        raise RuntimeError(f'Fit failed/resource-stopped: {dataset} {method}; state={state.get("state")}')
    print('[finished]', dataset, method, flush=True)


def _external_sc(cfg, settings, method):
    """Direct launch for pinned Python 3.9/other isolated method runtimes."""
    monitor = cfg.output_root/'00_monitor'/SCMULTISIM/method
    job = json.loads(json.dumps(asdict(settings), default=str))
    return _supervise_script(cfg, method, Path(__file__).with_name('sc_worker.py'),
                             job, monitor, lambda: _accepted(cfg, SCMULTISIM, method))


def _external_real(cfg, dataset, method, out):
    root = cfg.output_root/'01_model_input'/dataset
    job = {'dataset': dataset, 'method': method,
           'input_root': str(root.resolve()),
           'source_dir': str(cfg.cesp_source.resolve()),
           'input_hashes': {name: sha(root/name) for name in
                            ('expression.csv', 'coordinates.csv',
                             'evaluation_tfs.csv', 'evaluation_targets.csv')},
           'base_root': str(cfg.output_root.resolve()),
           'output_root': str(cfg.output_root.resolve()),
           'reg_source': str(cfg.reg_source.resolve()),
           'cell_package': str(cfg.cell_package.resolve()),
           'reg_python': str(cfg.reg_python.resolve()),
           'cell_python': str(cfg.cell_python.resolve())}
    script = (Path(__file__).parents[1]/'real_support_benchmark/cespgrn_worker39.py'
              if method == 'CeSpGRN' else Path(__file__).with_name('real_worker310.py'))
    return _supervise_script(cfg, method, script, job, out,
                             lambda: _accepted(cfg, dataset, method))


def _supervise_script(cfg, method, script, job, monitor, accepted):
    from real_support_benchmark.extension_workflow import _owned, _stop_owned
    if monitor.exists():
        raise RuntimeError(f'Prior external attempt requires audit: {monitor}')
    monitor.mkdir(parents=True)
    atomic_json(monitor/'job.json', job)
    with (monitor/'run.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([str(_python(cfg, method)), '-B', str(script)],
            cwd=monitor, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        process = psutil.Process(child.pid)
        started = process.create_time()
        state = {'state': 'running', **process_record(child.pid), 'started_epoch': time.time()}
        atomic_json(monitor/'run_status.json', state)
        guard = ResourceGuard(cfg)
        peak = 0
        reason = None
        try:
            while child.poll() is None:
                try:
                    members = _owned(process, started)
                    total = sum(item.memory_info().rss for item in members if item.is_running())
                    peak = max(peak, total)
                    if total > cfg.worker_rss_limit:
                        reason = 'resource_limit: worker-tree RSS'
                except psutil.NoSuchProcess:
                    members = []
                if not reason:
                    try:
                        guard()
                    except RuntimeError as error:
                        reason = str(error)
                if reason:
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
    result_path = monitor/'fit_result.json'
    fit_result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.is_file() else None
    if 'dataset' in job:
        prediction = monitor/'global_grn.parquet'
        is_accepted = (fit_result is not None and fit_result.get('state') == 'completed'
                       and fit_result.get('dataset') == job['dataset']
                       and fit_result.get('method') == method and prediction.is_file()
                       and fit_result.get('prediction_sha256') == sha(prediction))
    else:
        is_accepted = accepted()
    state.update(state='resource_limit' if reason else
                 'completed' if child.returncode == 0 and is_accepted else 'failed',
                 result=fit_result, exit_code=child.returncode, error=reason,
                 peak_sampled_worker_tree_rss_bytes=peak,
                 finished_epoch=time.time(),
                 memory_measurement='sampled RSS, not an OS hard cap')
    atomic_json(monitor/'run_status.json', state)
    if state['state'] == 'completed' and not accepted():
        state.update(state='failed', error='Post-write acceptance check failed')
        atomic_json(monitor/'run_status.json', state)
    return state


def _evaluate(cfg, tasks):
    import numpy as np
    import pandas as pd
    from independent_baselines.comparison import metrics
    from real_support_benchmark.evaluate import _prediction, _reference
    from configs.real_data.support_benchmark import SupportBenchmarkSettings

    if not (cfg.output_root/'01_plan.json').is_file():
        raise RuntimeError('Run and freeze the selected fit plan before evaluation')
    rows = []
    same_fit_rows = []
    sc_truth = None
    for dataset, method in tasks:
        if not _accepted(cfg, dataset, method):
            raise RuntimeError(f'No accepted prediction: {dataset} {method}')
        prediction = _output(cfg, dataset, method)/'global_grn.parquet'
        if dataset == SCMULTISIM:
            if sc_truth is None:
                manifest = json.loads((cfg.sc_input/'evidence_manifest.json').read_text(encoding='utf-8'))
                truth_path = cfg.sc_input/'truth/truth_grn.csv'
                if sha(truth_path) != manifest['files']['truth/truth_grn.csv']:
                    raise ValueError('scMultiSim truth checksum changed')
                sc_truth = pd.read_csv(truth_path, dtype={'tf': str, 'target': str})
                if len(sc_truth) != 624 or int(sc_truth.label.sum()) != 130 or sc_truth.duplicated(['tf','target']).any():
                    raise ValueError('scMultiSim 624-edge reference changed')
            pred = pd.read_parquet(prediction)
            if pred.duplicated(['tf','target']).any() or len(pred) != 624 or not np.isfinite(pred.score).all():
                raise ValueError(f'Invalid scMultiSim prediction: {prediction}')
            aligned = sc_truth.merge(pred[['tf','target','score']], on=['tf','target'],
                                     how='left', validate='one_to_one')
            if aligned.score.isna().any():
                raise ValueError('Prediction lacks a scMultiSim candidate edge')
            labels, scores = aligned.label.to_numpy(int), aligned.score.to_numpy(float)
            layer = 'known_grn'
        else:
            tfs = pd.read_csv(cfg.output_root/'01_model_input'/dataset/'evaluation_tfs.csv', dtype=str).tf.tolist()
            targets = pd.read_csv(cfg.output_root/'01_model_input'/dataset/'evaluation_targets.csv', dtype=str).target.tolist()
            full = pd.MultiIndex.from_product([tfs, targets], names=['tf','target']).to_frame(index=False)
            source_settings = replace(SupportBenchmarkSettings(),
                                      output_root=cfg.real_reference_root.parent,
                                      motif_catalog=cfg.motif_catalog)
            reference, _, _ = _reference(source_settings, dataset)
            aligned = _prediction(prediction, full, reference)
            labels, scores = reference.label.to_numpy(int), aligned.score.to_numpy(float)
            layer = 'support_union' if dataset == LIVER else 'atac_motif_weak_support'
        k = max(1, math.ceil(len(labels)*.1))
        result = metrics(labels, scores, (), decision_k=k, f_betas=(.1,))
        rows.append({'dataset': dataset, 'evidence_layer': layer, 'method': method,
                     'candidates': len(labels), 'positives': int(labels.sum()),
                     'decision_k': k, 'AP': result['AP'], 'AUROC': result['AUROC'],
                     'Early_precision': result['Early_precision'], 'F1': result['F1'],
                     'F0.1': result['F0.1'], 'precision': result['Precision'],
                     'recall': result['Recall'], 'prediction_sha256': sha(prediction)})
        if dataset != SCMULTISIM and method == 'JumpNicheRewire_communication':
            status = json.loads((_output(cfg, dataset, method)/'run_status.json').read_text(encoding='utf-8'))
            fitted = status['result']
            for view in ('reference_zero_exposure', 'communication_delta'):
                view_path = _output(cfg, dataset, method)/(view+'.parquet')
                if sha(view_path) != fitted.get(view+'_sha256'):
                    raise ValueError(f'Same-fit view changed: {view_path}')
                view_aligned = _prediction(view_path, full, reference)
                view_result = metrics(labels, view_aligned.score.to_numpy(float), (),
                                      decision_k=k, f_betas=(.1,))
                same_fit_rows.append({
                    'dataset': dataset, 'evidence_layer': layer, 'view': view,
                    'candidates': len(labels), 'positives': int(labels.sum()),
                    'decision_k': k, 'AP': view_result['AP'],
                    'AUROC': view_result['AUROC'],
                    'Early_precision': view_result['Early_precision'],
                    'F1': view_result['F1'], 'F0.1': view_result['F0.1'],
                    'precision': view_result['Precision'],
                    'recall': view_result['Recall'],
                    'prediction_sha256': sha(view_path)})
    table = pd.DataFrame(rows)
    full_table = table.copy()
    # A single public JNR row is defensible only if both independent real fits agree.
    real = table.loc[table.dataset.ne(SCMULTISIM)]
    for dataset in real.dataset.unique():
        jnr = table.loc[table.dataset.eq(dataset) & table.method.str.startswith('JumpNicheRewire_')]
        if len(jnr) == 2 and jnr.prediction_sha256.nunique() == 1:
            table.loc[jnr.index[0], 'method'] = 'JumpNicheRewire'
            table = table.drop(jnr.index[1])
    out = cfg.output_root/'04_evaluation'
    out.mkdir(parents=True, exist_ok=True)
    path = out/'selected_metrics.csv'
    if path.exists():
        raise FileExistsError(f'Existing evaluation table retained: {path}')
    full_table.to_csv(out/'all_independent_fits.csv', index=False)
    table.to_csv(path, index=False)
    if same_fit_rows:
        pd.DataFrame(same_fit_rows).to_csv(out/'same_fit_views.csv', index=False)
    atomic_json(out/'evaluation_audit.json', {
        'decision': 'top ceil(10% of evaluable candidates); tie-expected precision/recall',
        'AP_AUROC': 'full evaluable candidate universe, not top-k',
        'real_data_label_zero': 'not experimentally supported, not verified negative',
        'scmultisim_role': 'development dataset; tuned on same data',
        'real_reference_root': str(cfg.real_reference_root.resolve()),
        'single_public_jnr_row_only_if_both_prediction_hashes_equal': True})
    return path


def run(cfg, action):
    tasks = list(_tasks(cfg))
    if action != cfg.action:
        raise ValueError('Active action and SELECTED.action differ; edit only benchmark_selected.py')
    if action == 'status':
        return [{'dataset': dataset, 'method': method,
                 'accepted': _accepted(cfg, dataset, method),
                 'path': str(_output(cfg, dataset, method))}
                for dataset, method in tasks]
    plan = _plan(cfg)
    if action == 'check':
        used = _preflight(cfg, tasks)
        return {'datasets': cfg.datasets, 'methods': cfg.methods, 'expanded_fits': len(tasks),
                'storage_used_bytes': used, 'storage_limit_bytes': cfg.cumulative_storage_limit,
                'environments': {method: str(_python(cfg, method)) for method in cfg.methods},
                'plan_files_hashed': len(plan['files']),
                'message': 'Read-only check; no fit or output file written'}
    plan_path = cfg.output_root/'01_plan.json'
    if action == 'evaluate':
        if not plan_path.is_file() or json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Frozen selected plan missing or changed; cannot evaluate')
        return _evaluate(cfg, tasks)
    _preflight(cfg, tasks)
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Frozen plan changed; choose a new output_root')
    else:
        cfg.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(plan_path, plan)
    controller = cfg.output_root/'study_process.json'
    if controller.is_file():
        previous = json.loads(controller.read_text(encoding='utf-8'))
        if verified_live(previous):
            raise RuntimeError('Verified selected benchmark controller already running')
        if previous.get('state') != 'completed':
            raise RuntimeError('Previous controller stopped; audit before any explicit resume')
        return previous
    for dataset in cfg.datasets:
        if dataset != SCMULTISIM:
            _copy_real_inputs(cfg, dataset)
    state = {'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
             **process_record(os.getpid())}
    with controller.open('x', encoding='utf-8') as stream:
        json.dump(state, stream)
    try:
        for dataset, method in tasks:
            if _plan(cfg) != plan:
                raise ValueError('Frozen source input or code changed during queue')
            _preflight(cfg, [(dataset, method)])
            _fit_one(cfg, dataset, method)
            atomic_json(cfg.output_root/'latest_progress.json', {
                'dataset': dataset, 'method': method,
                'completed': sum(_accepted(cfg, d, m) for d, m in tasks),
                'expected': len(tasks),
                'as_of_utc': datetime.now(timezone.utc).isoformat()})
        state.update(state='completed', finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(controller, state)
        return controller
    except BaseException as error:
        state.update(state='stopped', finished_utc=datetime.now(timezone.utc).isoformat(),
                     error_type=type(error).__name__, error=str(error))
        atomic_json(controller, state)
        raise
