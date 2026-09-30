"""Immutable second-round admission, cache provenance and isolated serial queue."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback
import pandas as pd
from configs.public_sim_data.benchmark_optimization import OptimizationConfig
from benchmark_optimization.workflow import fingerprint
from jump_niche_rewire_v2.adapter import load_development, sha
from jump_niche_rewire_v2.execution import atomic_json
from jump_niche_rewire_v2.persistence import digest, ResourceGuard
from jump_niche_rewire_v2.supervisor import run_isolated, SupervisorConfig, process_record, verified_live

ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    files = []
    for folder in ('src/benchmark_round02', 'src/jump_niche_rewire_v2', 'src/jumpniche_app'):
        files.extend((ROOT/folder).glob('*.py'))
    files.extend(ROOT/name for name in ('configs/public_sim_data/benchmark_round02.py',
        'configs/public_sim_data/decomposition_v2.py', 'main_jumpniche.py', 'preprocess_jumpniche.py'))
    return {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in sorted(files)}


def task_root(cfg, dataset, method, candidate, role='candidate'):
    return cfg.output_root/('02_ablations' if role == 'ablation' else '02_runs')/method/candidate/dataset


def tasks(cfg):
    result = [(method, candidate, dataset, 'candidate') for method in cfg.execution_order
              for candidate in cfg.candidates()[method] for dataset in cfg.dataset_ids]
    result.extend(('JumpNicheRewire', candidate, dataset, 'ablation')
                  for candidate in cfg.ablations() for dataset in cfg.dataset_ids)
    return result


def old_cache(cfg):
    old_plan = json.loads(cfg.round01_plan.read_text(encoding='utf8'))
    if fingerprint(OptimizationConfig()) != old_plan['implementation_hash']:
        raise ValueError('Locked round_01 source hash changed')
    rows = pd.read_csv(cfg.development_registry).set_index('dataset_id')
    records = []
    for method, candidates in old_plan['candidates'].items():
        for candidate in candidates:
            for dataset in cfg.dataset_ids:
                root = cfg.old_root/'02_runs'/method/candidate/dataset
                status = json.loads((root/'run_status.json').read_text(encoding='utf8'))
                if (status['state'] != 'completed' or status.get('exit_code') != 0 or
                    status['implementation_hash'] != old_plan['implementation_hash'] or
                    status['input_signature'] != rows.loc[dataset, 'signature'] or
                    status['prediction_sha256'] != sha(root/'global_grn.parquet')):
                    raise ValueError(f'Old successful cache changed: {root}')
                if method == 'JumpNicheRewire' and status.get('convergence_passed') is not True:
                    raise ValueError('Old JNR cache not converged')
                records.append(dict(method=method, candidate=candidate, dataset_id=dataset,
                    root=str(root), prediction_sha256=status['prediction_sha256'], status_sha256=sha(root/'run_status.json')))
    if len(records) != 138: raise ValueError('Expected all 138 old fits, not selected favorable results')
    return records


def proposed_plan(cfg):
    if sha(cfg.development_registry) != json.loads(cfg.round01_plan.read_text(encoding='utf8'))['registry_sha256']:
        raise ValueError('Development registry changed')
    rows = pd.read_csv(cfg.development_registry)
    if len(rows) != 6 or set(rows.dataset_id) != set(cfg.dataset_ids) or not rows.split.eq('development').all():
        raise ValueError('Only the original six development datasets are allowed')
    inputs = {}
    for dataset in cfg.dataset_ids:
        prepared = load_development(cfg, dataset)  # input only, no model fit or truth access
        inputs[dataset] = prepared.audit
        dataset_root = Path(rows.set_index('dataset_id').loc[dataset, 'dataset_dir'])
        inputs[dataset]['manifest_sha256'] = sha(dataset_root/'evidence_manifest.json')
    settings = asdict(cfg); settings.pop('resume_engineering_failures', None)
    value = dict(schema='round02_development_1', settings=settings, sources=source_hashes(),
        old_plan_sha256=sha(cfg.round01_plan), old_cache=old_cache(cfg), inputs=inputs,
        registry_sha256=sha(cfg.development_registry), new_candidates=cfg.candidates(), ablations=cfg.ablations(),
        ordered_tasks=tasks(cfg), new_candidate_tasks=120, ablation_tasks=18, new_tasks=138,
        reused_tasks=138, cumulative_unique_fits=276, cumulative_selectable_candidates=43,
        selection='one candidate per method by equal-independence-group mean development AP; lexical tie-break',
        ablations_selectable=False, global_readout='absolute_mean_backbone for v2, prespecified readout for old candidates',
        metrics=['AP','AUROC','Early_precision','Precision_at_10','Recall_at_10','F1_at_10',
                 'Precision_at_50','Recall_at_50','F1_at_50','Precision_at_100','Recall_at_100','F1_at_100'],
        test_enabled=False, cases_enabled=False, independent_confirmation_registered=False)
    if len(tasks(cfg)) != value['new_tasks']: raise ValueError('Declared finite budget differs from grid')
    return json.loads(json.dumps(value, default=str))


def register(cfg):
    ResourceGuard(cfg)()
    plan = proposed_plan(cfg)
    path = cfg.output_root/'01_plan/search_plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != plan: raise ValueError('Immutable plan differs; preserve this round')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf8') as stream: json.dump(plan, stream, ensure_ascii=False, indent=2)
    return plan


def verified_plan(cfg):
    path = cfg.output_root/'01_plan/search_plan.json'
    if not path.exists(): raise PermissionError('Run preprocess prepare to register BEFORE any scientific fitting')
    plan = json.loads(path.read_text(encoding='utf8'))
    settings = asdict(cfg); settings.pop('resume_engineering_failures', None)
    settings = json.loads(json.dumps(settings, default=str))
    if (plan['sources'] != source_hashes() or plan['settings'] != settings or
        plan['registry_sha256'] != sha(cfg.development_registry) or
        plan['old_plan_sha256'] != sha(cfg.round01_plan) or plan['old_cache'] != old_cache(cfg)):
        raise ValueError('Registered code/config/registry/old cache changed; stop for audit')
    for row in pd.read_csv(cfg.development_registry).to_dict('records'):
        if sha(Path(row['dataset_dir'])/'evidence_manifest.json') != plan['inputs'][row['dataset_id']]['manifest_sha256']:
            raise ValueError('Locked input/truth provenance manifest changed')
    return plan


def accepted_new(root, plan_hash, method, candidate, dataset, role):
    path = root/'run_status.json'
    if not path.exists(): return None, 'missing'
    status = json.loads(path.read_text(encoding='utf8'))
    result = status.get('result', {})
    if status['state'] != 'completed' or status.get('exit_code') != 0: return None, status['state']
    required = dict(plan_sha256=plan_hash, method=method, candidate=candidate, dataset_id=dataset, role=role)
    if any(result.get(key) != value for key, value in required.items()): raise ValueError('New cache identity mismatch')
    if result.get('prediction_sha256') != sha(root/'global_grn.parquet'): raise ValueError('New cache prediction changed')
    if method == 'JumpNicheRewire' and result.get('convergence_rate') != 1.: raise ValueError('Unconverged cache marked completed')
    return status, 'accepted'


def run(cfg, action):
    if action in ('freeze', 'test', 'cases'):
        raise PermissionError('New independent confirmation is absent. Development cannot unlock test or cases.')
    if action == 'check':
        plan = proposed_plan(cfg)
        return dict(new_tasks=plan['new_tasks'], reused_tasks=138, registered=(cfg.output_root/'01_plan/search_plan.json').exists())
    if action in ('prepare', 'reuse'): return dict(plan_sha256=digest(register(cfg)), registered=True, new_fits=0)
    if action == 'status':
        from collections import Counter
        return dict(Counter(json.loads(p.read_text(encoding='utf8'))['state'] for folder in ('02_runs', '02_ablations')
            for p in (cfg.output_root/folder).rglob('run_status.json') if 'attempt_history' not in p.parts))
    if action == 'evaluate':
        from .evaluate import evaluate
        return evaluate(cfg)
    if action not in ('run', 'resume'): raise ValueError('check / prepare / reuse / run / resume / status / evaluate only')
    plan = verified_plan(cfg); plan_hash = digest(plan)
    # Check actual workers before creating a controller; old PID numbers alone are never used.
    for folder in ('02_runs', '02_ablations'):
        for path in (cfg.output_root/folder).rglob('run_status.json'):
            if 'attempt_history' in path.parts: continue
            record = json.loads(path.read_text(encoding='utf8'))
            if record.get('state') == 'running' and verified_live(record): raise RuntimeError('Live/orphan worker; refusing duplicate')
    lock = cfg.output_root/'study_process.json'
    if lock.exists():
        prior = json.loads(lock.read_text(encoding='utf8'))
        if verified_live(prior): raise RuntimeError('Study controller is still alive')
        archive = cfg.output_root/'scheduler_history'/f'{time.time_ns()}.json'
        archive.parent.mkdir(exist_ok=True); lock.replace(archive)
    state = dict(state='running', **process_record(os.getpid()), plan_sha256=plan_hash, started_epoch=time.time())
    with lock.open('x', encoding='utf8') as stream: json.dump(state, stream)
    if os.name == 'nt': __import__('ctypes').windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        from .worker import run_worker
        counts = {}
        for index, (method, candidate, dataset, role) in enumerate(tasks(cfg)):
            verified_plan(cfg)
            out = task_root(cfg, dataset, method, candidate, role)
            cached, reason = accepted_new(out, plan_hash, method, candidate, dataset, role)
            if cached: result = cached
            elif reason != 'missing' and (reason == 'nonconverged' or action != 'resume'):
                result = json.loads((out/'run_status.json').read_text(encoding='utf8'))
            else:
                python = cfg.tree_python if method in ('GENIE3', 'GRNBoost2') else cfg.model_python
                print('[start]', index+1, '/', len(tasks(cfg)), role, method, candidate, dataset, flush=True)
                result = run_isolated(run_worker, (cfg, dataset, method, candidate, role, plan_hash), out, cfg,
                    SupervisorConfig(python_executable=str(python)))
            counts[result['state']] = counts.get(result['state'], 0)+1
            progress = dict(task_index=index+1, total_tasks=len(tasks(cfg)), method=method, candidate=candidate,
                dataset_id=dataset, role=role, state=result['state'], counts=counts, updated_epoch=time.time(),
                prior_reused_tasks=138, confirmation_complete=False, cases_enabled=False)
            atomic_json(cfg.output_root/'latest_progress.json', progress)
            print('[finished]', role, method, candidate, result['state'], counts, flush=True)
            if result['state'] == 'resource_limit': raise RuntimeError('Resource limit; stop queue and audit before resuming')
        from .evaluate import evaluate
        evaluation = evaluate(cfg)
        state.update(state='development_complete' if counts == {'completed': 138} else 'development_incomplete',
            counts=counts, evaluation=evaluation, finished_epoch=time.time())
        atomic_json(lock, state)
        return evaluation
    except BaseException as error:
        state.update(state='stopped', error=str(error), traceback=traceback.format_exc())
        atomic_json(lock, state); raise
    finally:
        if os.name == 'nt': __import__('ctypes').windll.kernel32.SetThreadExecutionState(0x80000000)
