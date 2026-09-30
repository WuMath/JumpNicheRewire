"""Second one-hour positive-fusion JNR search; comparator results remain cached."""
from datetime import datetime, timezone
import json
import shutil
import time

from configs.public_sim_data.single_dataset_jnr_positive_fusion_followup import PositiveFusionFollowupJNR
from configs.public_sim_data.single_dataset_jnr_tuning import ROOT
from .cespgrn_one import _sha, _atomic_json, _budget_bytes
from .jnr_one_hour import _verify, _supervise, _score, FROZEN_IMPL


BASE_ROOT = ROOT/'data/result/optimization_20260921/single_dataset_jnr_positive_fusion_hour_20260924'
BASE_NAME = 'l1_008'
BASE_AP = 0.30522711386982515
MIN_FUSION = .0005


def _verify_baseline():
    report = json.loads((BASE_ROOT/'latest_report.json').read_text(encoding='utf8'))
    if (report['best_candidate'] != BASE_NAME or report['best_AP'] != BASE_AP or
            report['completed_candidates'] != 17):
        raise ValueError('Previous positive-fusion winner changed')
    item = next(x for x in report['results'] if x['candidate'] == BASE_NAME)
    if item['parameters']['fusion_penalty'] < MIN_FUSION:
        raise ValueError('Baseline violates positive-fusion minimum')
    status = json.loads((BASE_ROOT/'02_runs'/BASE_NAME/'run_status.json').read_text(encoding='utf8'))
    prediction = BASE_ROOT/'02_runs'/BASE_NAME/'global_grn.parquet'
    if (status['state'] != 'completed' or not status['convergence']['convergence_passed'] or
            status['implementation_hash'] != FROZEN_IMPL or
            _sha(prediction) != status['prediction_sha256']):
        raise ValueError('Positive-fusion baseline cache invalid')
    return item['metrics']


def run(cfg=None):
    import psutil
    cfg = cfg or PositiveFusionFollowupJNR()
    row = _verify(cfg)
    baseline_metrics = _verify_baseline()
    if psutil.virtual_memory().available < cfg.minimum_available_ram:
        raise MemoryError('Available RAM below 1.5 GiB before start')
    if shutil.disk_usage(cfg.output_root.anchor).free < cfg.minimum_disk_free:
        raise MemoryError('Disk free below 5 GiB before start')
    initial = _budget_bytes(cfg)
    if initial + cfg.storage_reserve > cfg.cumulative_storage_limit:
        raise MemoryError('Cumulative storage plus reserve exceeds 9 GiB before start')
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = cfg.output_root/'01_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf8'))
        if (plan['parameters'] != cfg.candidates() or plan['dataset_id'] != cfg.dataset_id or
                plan['implementation_hash'] != FROZEN_IMPL or
                plan['baseline_prediction_sha256'] != _sha(BASE_ROOT/'02_runs'/BASE_NAME/'global_grn.parquet') or
                plan['minimum_eligible_fusion_penalty'] != MIN_FUSION):
            raise ValueError('Follow-up locked plan changed')
    else:
        now = time.time()
        plan = dict(dataset_id=cfg.dataset_id, input_signature=row['signature'],
            implementation_hash=FROZEN_IMPL, parameters=cfg.candidates(),
            baseline_candidate=BASE_NAME, baseline_AP=BASE_AP,
            baseline_prediction_sha256=_sha(BASE_ROOT/'02_runs'/BASE_NAME/'global_grn.parquet'),
            minimum_eligible_fusion_penalty=MIN_FUSION,
            started_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
            deadline_utc=datetime.fromtimestamp(now + cfg.hour_limit_seconds, timezone.utc).isoformat(),
            aggregation='mean_abs', other_methods_rerun=False,
            independent_confirmation=False, initial_storage_bytes=initial,
            cumulative_storage_limit=cfg.cumulative_storage_limit,
            worker_rss_limit=cfg.worker_rss_limit,
            minimum_available_ram=cfg.minimum_available_ram,
            minimum_disk_free=cfg.minimum_disk_free,
            resource_limits_sampled_not_os_hard_caps=True)
        _atomic_json(plan_path, plan)
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    if time.time() >= deadline:
        raise TimeoutError('Follow-up hour elapsed; do not extend')
    report_path = cfg.output_root/'latest_report.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf8'))
        if previous['dataset_id'] != cfg.dataset_id or previous['deadline_utc'] != plan['deadline_utc']:
            raise ValueError('Follow-up report differs from locked plan')
        results = previous['results']
        best_name, best_ap, best_metrics = (previous['best_candidate'],
            previous['best_AP'], previous['best_metrics'])
    else:
        results = []
        best_name, best_ap, best_metrics = 'prior__'+BASE_NAME, BASE_AP, baseline_metrics
    seen = {item['candidate'] for item in results}
    for index, (name, params) in enumerate(cfg.candidates().items(), 1):
        if name in seen:
            continue
        if deadline - time.time() <= 5:
            break
        if _budget_bytes(cfg) + cfg.storage_reserve > cfg.cumulative_storage_limit:
            results.append(dict(candidate=name, state='resource_stopped',
                                reason='cumulative_storage_reserve'))
            break
        print('[start]', index, '/', len(cfg.candidates()), name, flush=True)
        status = _supervise(cfg, name, params, time.monotonic()+max(0, deadline-time.time()))
        record = dict(candidate=name, state=status['state'], parameters=params,
                      peak_sampled_rss_bytes=status.get('peak_sampled_rss_bytes'))
        if status['state'] == 'completed':
            record['metrics'] = _score(cfg, name)
            if record['metrics']['AP'] > best_ap:
                best_name, best_ap, best_metrics = name, record['metrics']['AP'], record['metrics']
        else:
            record['reason'] = status.get('reason', status.get('error'))
        results.append(record)
        summary = dict(dataset_id=cfg.dataset_id,
            as_of_utc=datetime.now(timezone.utc).isoformat(),
            deadline_utc=plan['deadline_utc'], baseline_AP=BASE_AP,
            best_candidate=best_name, best_AP=best_ap, best_metrics=best_metrics,
            completed_candidates=sum(item['state']=='completed' for item in results),
            attempted_candidates=len(results), total_candidates=len(cfg.candidates()),
            results=results, other_methods_rerun=False,
            same_data_tuning_and_evaluation=True, independent_confirmation=False)
        _atomic_json(report_path, summary)
        print('[finished]', name, status['state'], 'AP', record.get('metrics', {}).get('AP'),
              'best', best_name, best_ap, flush=True)
        if status['state'] != 'completed' and status.get('error_type') != 'ConvergenceError':
            break
    return json.loads(report_path.read_text(encoding='utf8'))
