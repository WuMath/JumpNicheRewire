"""One-hour parameter-only JNR development search on one fixed scMultiSim dataset."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import contextlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import traceback

from configs.public_sim_data.single_dataset_jnr_tuning import OneHourJNR
from .cespgrn_one import ROOT, _sha, _atomic_json, _budget_bytes

BASELINE_AP = 0.2499448546580186
FROZEN_IMPL = 'ab208a3cd1b16c6f0668c780d6dc4450d676556fd3e8c1540c453fc6526c0c53'
DEADLINE_UTC = datetime(2026,9,23,11,37,10,tzinfo=timezone.utc)


def _row(cfg):
    import pandas as pd
    from publication_study.resources import verify_model_inputs
    registry = pd.read_csv(cfg.registry)
    selected = registry.loc[registry.dataset_id.eq(cfg.dataset_id)]
    if len(selected) != 1 or selected.iloc[0]['split'] != 'development':
        raise ValueError('Exactly one fixed development dataset required')
    row = selected.iloc[0].to_dict()
    verify_model_inputs(row)
    return row


def _verify(cfg):
    from configs.public_sim_data.benchmark_optimization import OptimizationConfig
    from benchmark_optimization.workflow import fingerprint
    if cfg.dataset_id != 'published_tree1_500_cells110_genes_sigma0.1_1':
        raise ValueError('Dataset changed')
    if fingerprint(OptimizationConfig()) != FROZEN_IMPL:
        raise ValueError('JNR v1 implementation differs from accepted round_01')
    if len(cfg.candidates()) != 22 or any(p['aggregation'] != 'mean_abs' for p in cfg.candidates().values()):
        raise ValueError('Fixed finite parameter-only grid changed')
    row = _row(cfg)
    manifest = json.loads((Path(row['dataset_dir'])/'evidence_manifest.json').read_text(encoding='utf8'))
    for name in ('expression.csv','coordinates.csv','metadata.csv','tf_list.csv','lr_pairs.csv'):
        if _sha(Path(row['dataset_dir'])/name) != manifest['files'][name]:
            raise ValueError(f'Input changed: {name}')
    return row


def _worker(cfg,name,params):
    import os
    import numpy as np
    import pandas as pd
    import psutil
    from configs.public_sim_data.benchmark_optimization import OptimizationConfig
    from publication_study.inference import build_model
    from publication_study.resources import write_json
    from jump_niche_rewire import JumpNicheRewirePipeline
    from jump_niche_rewire.convergence import summarize_convergence
    from benchmark_optimization.worker import aggregate

    out = cfg.output_root/'02_runs'/name
    status_path = out/'run_status.json'
    with (out/'worker.log').open('w',encoding='utf8',buffering=1) as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            started = time.time()
            try:
                row = _verify(cfg)
                numerical = {k:v for k,v in params.items() if k != 'aggregation'}
                old_cfg = replace(OptimizationConfig(),**numerical)
                model = build_model(old_cfg,row,out)
                write_json(out/'model_config.json',asdict(model))
                result = JumpNicheRewirePipeline(model).run()
                summary = summarize_convergence(result.base,result.rewiring,
                    model.runtime.minimum_target_convergence_rate,
                    communication_rewiring_disabled=model.rewiring.disable_communication_rewiring)
                if summary.get('convergence_passed') is not True:
                    raise RuntimeError('Convergence not accepted')
                beta = result.rewiring.beta_ccc
                score = aggregate(beta,params['aggregation'])
                frame = pd.DataFrame(dict(tf=np.repeat(result.base.tf_names,len(result.base.target_names)),
                    target=np.tile(result.base.target_names,len(result.base.tf_names)),score=score.ravel()))
                if len(frame) != 624 or frame.duplicated(['tf','target']).any() or not np.isfinite(frame.score).all():
                    raise ValueError('Incomplete or nonfinite common TF-target prediction')
                temp = out/'global_grn.writing.parquet'
                frame.to_parquet(temp,index=False)
                temp.replace(out/'global_grn.parquet')
                _atomic_json(status_path,dict(state='completed',dataset_id=cfg.dataset_id,
                    method='JumpNicheRewire',candidate=name,parameters=params,
                    input_signature=row['signature'],implementation_hash=FROZEN_IMPL,
                    pid=os.getpid(),process_started=psutil.Process().create_time(),
                    runtime_seconds=time.time()-started,convergence=summary,
                    prediction_sha256=_sha(out/'global_grn.parquet')))
            except Exception as exc:
                _atomic_json(status_path,dict(state='failed',dataset_id=cfg.dataset_id,
                    candidate=name,parameters=params,error_type=type(exc).__name__,
                    error=str(exc),traceback=traceback.format_exc(),
                    runtime_seconds=time.time()-started))
                raise


def _score(cfg,name):
    import numpy as np
    import pandas as pd
    from independent_baselines.comparison import metrics
    out = cfg.output_root/'02_runs'/name
    state = json.loads((out/'run_status.json').read_text(encoding='utf8'))
    if state['state'] != 'completed' or state['implementation_hash'] != FROZEN_IMPL:
        raise ValueError('Fit incomplete or implementation changed')
    if _sha(out/'global_grn.parquet') != state['prediction_sha256']:
        raise ValueError('Prediction changed')
    row = _row(cfg)
    truth_path = Path(row['truth_dir'])/'truth_grn.csv'
    manifest = json.loads((Path(row['dataset_dir'])/'evidence_manifest.json').read_text(encoding='utf8'))
    if _sha(truth_path) != manifest['files']['truth/truth_grn.csv']:
        raise ValueError('Development truth changed')
    truth = pd.read_csv(truth_path,dtype={'tf':str,'target':str})
    frame = pd.read_parquet(out/'global_grn.parquet')
    joined = truth.merge(frame,on=['tf','target'],how='left',validate='one_to_one')
    if len(truth) != 624 or len(joined) != 624 or joined.score.isna().any() or not np.isfinite(joined.score).all():
        raise ValueError('Common 624-edge evaluation incomplete')
    return {k:float(v) for k,v in metrics(joined.label,joined.score,(10,50,100)).items()}


def _supervise(cfg,name,params,deadline_monotonic):
    import psutil
    out = cfg.output_root/'02_runs'/name
    out.mkdir(parents=True,exist_ok=True)
    status_path = out/'run_status.json'
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding='utf8'))
        if state.get('state') == 'completed':
            return state
        raise RuntimeError('Prior failed/interrupted attempt exists; audit before retry')
    _atomic_json(status_path,dict(state='starting',candidate=name,parameters=params,
                                  started_utc=datetime.now(timezone.utc).isoformat()))
    process = mp.get_context('spawn').Process(target=_worker,args=(cfg,name,params))
    process.start()
    peak = 0
    reason = None
    last_storage = 0.
    storage = _budget_bytes(cfg)
    while process.is_alive():
        try:
            root = psutil.Process(process.pid)
            members = [root]+root.children(recursive=True)
            rss = sum(p.memory_info().rss for p in members if p.is_running())
            peak = max(peak,rss)
            if rss > cfg.worker_rss_limit: reason = 'worker_rss_limit'
            elif psutil.virtual_memory().available < cfg.minimum_available_ram: reason = 'minimum_available_ram'
            elif shutil.disk_usage(cfg.output_root.anchor).free < cfg.minimum_disk_free: reason = 'minimum_disk_free'
            elif time.monotonic() >= deadline_monotonic: reason = 'one_hour_deadline'
            if not reason and time.monotonic()-last_storage > 30:
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
    state = json.loads(status_path.read_text(encoding='utf8'))
    if reason:
        state.update(state='resource_stopped' if reason != 'one_hour_deadline' else 'time_stopped',
            reason=reason,peak_sampled_rss_bytes=peak,sampled_storage_bytes=storage)
    else:
        state.update(worker_exit_code=process.exitcode,peak_sampled_rss_bytes=peak,
                     sampled_storage_bytes=storage)
        if process.exitcode != 0 and state['state'] != 'failed':
            state.update(state='failed',reason='worker_exited_without_result')
    _atomic_json(status_path,state)
    return state


def run(cfg=None):
    import psutil
    cfg = cfg or OneHourJNR()
    row = _verify(cfg)
    deadline = DEADLINE_UTC.timestamp()
    if time.time() >= deadline:
        raise TimeoutError('One-hour search window already elapsed')
    if psutil.virtual_memory().available < cfg.minimum_available_ram:
        raise MemoryError('Available RAM below limit')
    if shutil.disk_usage(cfg.output_root.anchor).free < cfg.minimum_disk_free:
        raise MemoryError('Disk free below limit')
    initial = _budget_bytes(cfg)
    if initial + cfg.storage_reserve > cfg.cumulative_storage_limit:
        raise MemoryError('Cumulative storage plus reserve exceeds 9 GiB')
    cfg.output_root.mkdir(parents=True,exist_ok=True)
    plan = dict(dataset_id=cfg.dataset_id,input_signature=row['signature'],
        implementation_hash=FROZEN_IMPL,parameters=cfg.candidates(),
        baseline_AP=BASELINE_AP,aggregation='mean_abs',
        deadline_utc=DEADLINE_UTC.isoformat(),
        other_methods_rerun=False,confirmation=False,
        initial_storage_bytes=initial,cumulative_storage_limit=cfg.cumulative_storage_limit,
        resource_limits_sampled_not_os_hard_caps=True)
    plan_path = cfg.output_root/'01_plan.json'
    if plan_path.exists():
        prior_plan = json.loads(plan_path.read_text(encoding='utf8'))
        # The original preflight byte sample is historical, not a plan field
        # that should drift as this same search writes audit files.
        plan['initial_storage_bytes'] = prior_plan.get('initial_storage_bytes')
        if prior_plan != plan:
            raise ValueError('One-hour plan changed')
    else:
        _atomic_json(plan_path,plan)
    report_path = cfg.output_root/'latest_report.json'
    if report_path.exists():
        prior_report = json.loads(report_path.read_text(encoding='utf8'))
        if prior_report['dataset_id'] != cfg.dataset_id or prior_report['deadline_utc'] != DEADLINE_UTC.isoformat():
            raise ValueError('Previous search report does not match locked plan')
        results = prior_report['results']
        best_name,best_ap = prior_report['best_candidate'],prior_report['best_AP']
    else:
        results = []
        best_name,best_ap = 'r01__more_sparse_mean',BASELINE_AP
    already_attempted = {r['candidate'] for r in results}
    for index,(name,params) in enumerate(cfg.candidates().items(),1):
        if name in already_attempted:
            continue
        if deadline-time.time() <= 5:
            break
        if _budget_bytes(cfg)+cfg.storage_reserve > cfg.cumulative_storage_limit:
            results.append(dict(candidate=name,state='resource_stopped',reason='cumulative_storage_reserve'))
            break
        print('[start]',index,'/',len(cfg.candidates()),name,flush=True)
        state = _supervise(cfg,name,params,time.monotonic()+max(0,deadline-time.time()))
        record = dict(candidate=name,state=state['state'],parameters=params,
                      peak_sampled_rss_bytes=state.get('peak_sampled_rss_bytes'))
        if state['state'] == 'completed':
            record['metrics'] = _score(cfg,name)
            ap = record['metrics']['AP']
            if ap > best_ap:
                best_ap,best_name = ap,name
        else:
            record['reason'] = state.get('reason',state.get('error'))
        results.append(record)
        report = dict(dataset_id=cfg.dataset_id,as_of_utc=datetime.now(timezone.utc).isoformat(),
            deadline_utc=DEADLINE_UTC.isoformat(),baseline_AP=BASELINE_AP,
            best_candidate=best_name,best_AP=best_ap,
            completed_candidates=sum(r['state']=='completed' for r in results),
            attempted_candidates=len(results),total_candidates=len(cfg.candidates()),
            results=results,other_methods_rerun=False,
            same_data_tuning_and_evaluation=True,independent_confirmation=False)
        _atomic_json(cfg.output_root/'latest_report.json',report)
        print('[finished]',name,state['state'],'AP',record.get('metrics',{}).get('AP'),
              'best',best_name,best_ap,flush=True)
        if state['state'] != 'completed' and state.get('error_type') != 'ConvergenceError':
            break
    return json.loads((cfg.output_root/'latest_report.json').read_text(encoding='utf8'))
