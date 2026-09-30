"""Cumulative development evaluation with separately reported fixed ablations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_optimization.evaluate import PERFORMANCE, select_candidates
from independent_baselines.comparison import metrics
from jump_niche_rewire_v2.adapter import sha
from jump_niche_rewire_v2.persistence import digest
from jump_niche_rewire_v2.execution import atomic_json
from .workflow import verified_plan, tasks, task_root, accepted_new


def choose_main(data, audit, expected):
    return select_candidates(data.loc[data.role.eq('candidate')], audit.loc[audit.role.eq('candidate')], expected)


def evaluate(cfg):
    plan = verified_plan(cfg); plan_hash = digest(plan)
    out = cfg.output_root/'03_development'; out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(cfg.development_registry)
    records = [dict(**item, role='candidate', candidate_id='r01__'+item['candidate'], origin='round_01') for item in plan['old_cache']]
    records.extend(dict(method=method, candidate=candidate, candidate_id='r02__'+candidate, dataset_id=dataset,
        root=str(task_root(cfg, dataset, method, candidate, role)), role=role, origin='round_02')
        for method, candidate, dataset, role in tasks(cfg))
    values, audits = [], []
    for row in rows.to_dict('records'):
        manifest = json.loads((Path(row['dataset_dir'])/'evidence_manifest.json').read_text(encoding='utf8'))
        truth_path = Path(row['truth_dir'])/'truth_grn.csv'
        if sha(truth_path) != manifest['files']['truth/truth_grn.csv']: raise ValueError('Development truth checksum changed')
        truth = pd.read_csv(truth_path, dtype={'tf': str, 'target': str})
        if truth.duplicated(['tf', 'target']).any(): raise ValueError('Duplicate truth candidates')
        for item in [r for r in records if r['dataset_id'] == row['dataset_id']]:
            root = Path(item['root'])
            status, state = (True, 'accepted') if item['origin'] == 'round_01' else accepted_new(
                root, plan_hash, item['method'], item['candidate'], row['dataset_id'], item['role'])
            audit = dict(dataset_id=row['dataset_id'], method=item['method'], candidate=item['candidate_id'],
                role=item['role'], origin=item['origin'], state=state, source=str(root))
            if status:
                frame = pd.read_parquet(root/'global_grn.parquet')
                if frame.duplicated(['tf','target']).any() or not np.isfinite(frame.score).all() or (frame.score < 0).any():
                    raise ValueError('Invalid prediction in completed cache')
                joined = truth.merge(frame[['tf','target','score']], on=['tf','target'], how='left', validate='one_to_one')
                audit.update(n_candidates=len(truth), n_predicted_candidates=int(joined.score.notna().sum()))
                if item['origin'] == 'round_02' and (len(frame) != len(truth) or joined.score.isna().any()):
                    raise ValueError('New output lacks full TF-target universe')
                calculated = metrics(joined.label, joined.score.fillna(0), cfg.classification_top_k)
                for metric in PERFORMANCE:
                    values.append(dict(dataset_id=row['dataset_id'], independence_group=row['independence_group'],
                        method=item['method'], candidate=item['candidate_id'], role=item['role'], origin=item['origin'],
                        metric=metric, value=calculated[metric]))
            audits.append(audit)
    audit = pd.DataFrame(audits)
    data = pd.DataFrame(values)
    audit.to_csv(out/'run_audit.csv', index=False)
    data.to_csv(out/'all_candidate_and_ablation_metrics.csv', index=False)
    selected, selection = choose_main(data, audit, rows.dataset_id.tolist())
    selection.to_csv(out/'parameter_selection_audit.csv', index=False)
    chosen = data.loc[[r.role == 'candidate' and selected.get(r.method) == r.candidate for r in data.itertuples()]]
    summary = chosen.groupby(['metric','method','independence_group']).value.mean().groupby(['metric','method']).mean().unstack('method')
    summary.to_csv(out/'selected_method_means.csv')
    tables = out/'per_metric'; tables.mkdir(exist_ok=True)
    for metric, group in chosen.groupby('metric'):
        group.pivot(index='dataset_id', columns='method', values='value').to_csv(tables/f'{metric}.csv')
    data.loc[data.role.eq('ablation')].to_csv(out/'fixed_anchor_ablations.csv', index=False)
    wins = []
    if set(cfg.methods).issubset(summary.columns):
        for other in cfg.methods[1:]:
            for metric in PERFORMANCE:
                a, b = float(summary.loc[metric,'JumpNicheRewire']), float(summary.loc[metric,other])
                wins.append(dict(comparator=other, metric=metric, jnr=a, other=b, difference=a-b, strictly_better=a>b))
    pd.DataFrame(wins).to_csv(out/'development_comparisons.csv', index=False)
    dataset_wins = []
    for dataset, group in chosen.groupby('dataset_id'):
        table = group.pivot(index='metric', columns='method', values='value')
        if not set(cfg.methods).issubset(table.columns): continue
        details = plan['inputs'][dataset]
        for other in cfg.methods[1:]:
            for metric in PERFORMANCE:
                a, b = float(table.loc[metric,'JumpNicheRewire']), float(table.loc[metric,other])
                dataset_wins.append(dict(dataset_id=dataset, cells=details['cells'], genes=details['genes'],
                    tfs=details['tfs'], targets=details['targets'], candidates=details['candidates'], comparator=other,
                    metric=metric, jnr=a, other=b, difference=a-b, strictly_better=a>b))
    pd.DataFrame(dataset_wins).to_csv(out/'per_dataset_comparisons.csv', index=False)
    result = dict(plan_sha256=plan_hash, selected_candidates=selected,
        new_accepted=int((audit.origin.eq('round_02') & audit.state.eq('accepted')).sum()), new_expected=138,
        cumulative_accepted=int(audit.state.eq('accepted').sum()), cumulative_expected=276,
        fixed_ablations_excluded_from_selection=True,
        development_all_metrics_better_than_all_five=bool(wins and all(r['strictly_better'] for r in wins)),
        confirmation_complete=False, formal_superiority_established=False, cases_enabled=False,
        warning='Cumulative tuned development estimates; not independent evidence. Shared topology is not an independent replicate.')
    atomic_json(out/'development_audit.json', result)
    return result
