"""Second fixed one-hour JNR-only parameter search; no other method refits."""
from datetime import datetime, timezone
import json
import shutil
import time

from configs.public_sim_data.single_dataset_jnr_second_hour import SecondHourJNR
from configs.public_sim_data.single_dataset_jnr_tuning import ROOT
from .cespgrn_one import _sha, _atomic_json, _budget_bytes
from .jnr_one_hour import _verify, _supervise, _score, FROZEN_IMPL

DEADLINE_UTC = datetime(2026,9,23,13,23,11,tzinfo=timezone.utc)
FIRST_ROOT = ROOT/'data/result/optimization_20260921/single_dataset_jnr_tuning_20260923'
FIRST_NAME = 'fusion_002'
FIRST_AP = 0.273008661355451


def _verify_first_hour():
    report = json.loads((FIRST_ROOT/'latest_report.json').read_text(encoding='utf8'))
    if (report['best_candidate'] != FIRST_NAME or report['best_AP'] != FIRST_AP or
            report['completed_candidates'] != 16):
        raise ValueError('First-hour selected baseline changed')
    status = json.loads((FIRST_ROOT/'02_runs'/FIRST_NAME/'run_status.json').read_text(encoding='utf8'))
    path = FIRST_ROOT/'02_runs'/FIRST_NAME/'global_grn.parquet'
    if (status['state'] != 'completed' or not status['convergence']['convergence_passed'] or
            status['implementation_hash'] != FROZEN_IMPL or _sha(path) != status['prediction_sha256']):
        raise ValueError('First-hour baseline cache invalid')
    return next(x['metrics'] for x in report['results'] if x['candidate'] == FIRST_NAME)


def run(cfg=None):
    import psutil
    cfg = cfg or SecondHourJNR()
    row = _verify(cfg)
    baseline_metrics = _verify_first_hour()
    deadline = DEADLINE_UTC.timestamp()
    if time.time() >= deadline:
        raise TimeoutError('Second-hour window already elapsed')
    if psutil.virtual_memory().available < cfg.minimum_available_ram:
        raise MemoryError('Available RAM below 1.5 GiB')
    if shutil.disk_usage(cfg.output_root.anchor).free < cfg.minimum_disk_free:
        raise MemoryError('Disk free below 5 GiB')
    initial = _budget_bytes(cfg)
    if initial+cfg.storage_reserve > cfg.cumulative_storage_limit:
        raise MemoryError('Cumulative storage plus reserve exceeds 9 GiB')
    cfg.output_root.mkdir(parents=True,exist_ok=True)
    plan = dict(dataset_id=cfg.dataset_id,input_signature=row['signature'],
        implementation_hash=FROZEN_IMPL,parameters=cfg.candidates(),
        first_hour_baseline=FIRST_NAME,first_hour_baseline_AP=FIRST_AP,
        first_hour_prediction_sha256=_sha(FIRST_ROOT/'02_runs'/FIRST_NAME/'global_grn.parquet'),
        aggregation='mean_abs',deadline_utc=DEADLINE_UTC.isoformat(),
        other_methods_rerun=False,independent_confirmation=False,
        initial_storage_bytes=initial,cumulative_storage_limit=cfg.cumulative_storage_limit,
        resource_limits_sampled_not_os_hard_caps=True)
    plan_path = cfg.output_root/'01_plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding='utf8'))
        plan['initial_storage_bytes'] = previous.get('initial_storage_bytes')
        if previous != plan:
            raise ValueError('Second-hour plan changed')
    else:
        _atomic_json(plan_path,plan)
    report_path = cfg.output_root/'latest_report.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf8'))
        if previous['dataset_id'] != cfg.dataset_id or previous['deadline_utc'] != DEADLINE_UTC.isoformat():
            raise ValueError('Second-hour report changed')
        results = previous['results']
        best_name,best_ap,best_metrics = previous['best_candidate'],previous['best_AP'],previous['best_metrics']
    else:
        results = []
        best_name,best_ap,best_metrics = 'h1__'+FIRST_NAME,FIRST_AP,baseline_metrics
    seen = {r['candidate'] for r in results}
    for index,(name,params) in enumerate(cfg.candidates().items(),1):
        if name in seen:
            continue
        if deadline-time.time() <= 5:
            break
        if _budget_bytes(cfg)+cfg.storage_reserve > cfg.cumulative_storage_limit:
            results.append(dict(candidate=name,state='resource_stopped',reason='cumulative_storage_reserve'))
            break
        print('[start]',index,'/',len(cfg.candidates()),name,flush=True)
        status = _supervise(cfg,name,params,time.monotonic()+max(0,deadline-time.time()))
        record = dict(candidate=name,state=status['state'],parameters=params,
                      peak_sampled_rss_bytes=status.get('peak_sampled_rss_bytes'))
        if status['state'] == 'completed':
            record['metrics'] = _score(cfg,name)
            if record['metrics']['AP'] > best_ap:
                best_name,best_ap,best_metrics = name,record['metrics']['AP'],record['metrics']
        else:
            record['reason'] = status.get('reason',status.get('error'))
        results.append(record)
        summary = dict(dataset_id=cfg.dataset_id,as_of_utc=datetime.now(timezone.utc).isoformat(),
            deadline_utc=DEADLINE_UTC.isoformat(),first_hour_baseline_AP=FIRST_AP,
            best_candidate=best_name,best_AP=best_ap,best_metrics=best_metrics,
            completed_candidates=sum(r['state']=='completed' for r in results),
            attempted_candidates=len(results),total_candidates=len(cfg.candidates()),
            results=results,other_methods_rerun=False,
            same_data_tuning_and_evaluation=True,independent_confirmation=False)
        _atomic_json(report_path,summary)
        print('[finished]',name,status['state'],'AP',record.get('metrics',{}).get('AP'),
              'best',best_name,best_ap,flush=True)
        if status['state'] != 'completed' and status.get('error_type') != 'ConvergenceError':
            break
    return json.loads(report_path.read_text(encoding='utf8'))
