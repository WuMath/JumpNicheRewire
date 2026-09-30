"""Post-request descriptive classification metrics on prespecified support layers."""

from __future__ import annotations

import json
import math

import pandas as pd

from independent_baselines.comparison import metrics
from .evaluate import _prediction, _reference
from .extension_workflow import DATASETS, METHODS as NEW_METHODS, _accepted
from .prepare import sha
from .workers import METHODS as EXISTING_METHODS


# This is a descriptive decision budget, not a trained threshold. It is fixed
# across methods before any additional F-score comparison is computed.
DECISION_FRACTION = .10


def evaluate(cfg):
    from configs.real_data.support_benchmark import SupportBenchmarkSettings
    base_cfg = SupportBenchmarkSettings()
    if base_cfg.output_root.resolve() != cfg.base_root.resolve():
        raise ValueError('Extension base result directory differs from frozen original')
    plan_path = cfg.output_root / '01_extension_plan.json'
    controller = cfg.output_root / 'study_process.json'
    if not plan_path.is_file() or not controller.is_file():
        raise RuntimeError('Extension fit plan/controller missing')
    if json.loads(controller.read_text(encoding='utf-8')).get('state') != 'completed':
        raise RuntimeError('All six additional fits must complete before evaluation')
    from .extension_workflow import _plan
    revision_path = cfg.output_root / '01_extension_plan_revision_01.json'
    if revision_path.is_file():
        revision = json.loads(revision_path.read_text(encoding='utf-8'))
        if revision.get('original_plan_sha256') != sha(plan_path):
            raise ValueError('Original extension plan changed')
        if revision.get('expected_plan') != _plan(cfg):
            raise ValueError('Frozen engineering revision changed')
    elif json.loads(plan_path.read_text(encoding='utf-8')) != _plan(cfg):
        raise ValueError('Frozen extension inputs/results/source changed')
    rows = []
    same_fit_rows = []
    for dataset in DATASETS:
        input_root = cfg.base_root / '01_model_input' / dataset
        tfs = pd.read_csv(input_root / 'evaluation_tfs.csv', dtype=str).tf.tolist()
        targets = pd.read_csv(input_root / 'evaluation_targets.csv', dtype=str).target.tolist()
        full_keys = pd.MultiIndex.from_product([tfs, targets],
            names=['tf', 'target']).to_frame(index=False)
        reference, layers, _ = _reference(base_cfg, dataset)
        decision_k = math.ceil(len(reference) * DECISION_FRACTION)
        for method in EXISTING_METHODS + NEW_METHODS:
            old = method in EXISTING_METHODS
            out = (cfg.base_root if old else cfg.output_root) / '03_runs' / dataset / method
            status = json.loads((out / 'run_status.json').read_text(encoding='utf-8'))
            result = status.get('result') or {}
            if status.get('state') != 'completed' or result.get('state') != 'completed':
                raise RuntimeError(f'Incomplete fit: {dataset} {method}')
            if not old and not _accepted(out, dataset, method):
                raise ValueError(f'Unaccepted extension fit: {dataset} {method}')
            prediction = out / 'global_grn.parquet'
            if sha(prediction) != result.get('prediction_sha256'):
                raise ValueError(f'Prediction hash mismatch: {prediction}')
            aligned = _prediction(prediction, full_keys, reference)
            for layer in layers:
                y = reference[layer].to_numpy(dtype=int)
                if not 0 < int(y.sum()) < len(y):
                    raise ValueError(f'{dataset}/{layer} has only one label class')
                values = metrics(y, aligned.score.to_numpy(dtype=float), (),
                                 decision_k=decision_k, f_betas=(.1, 1.))
                rows.append(dict(dataset=dataset, evidence_layer=layer,
                    method=method, candidates=len(y), positives=int(y.sum()),
                    prevalence=float(y.mean()), decision_k=decision_k,
                    decision_fraction=DECISION_FRACTION,
                    AP=values['AP'], AUROC=values['AUROC'],
                    Early_precision=values['Early_precision'],
                    Precision=values['Precision'], Recall=values['Recall'],
                    F1=values['F1'], F0_1=values['F0.1'],
                    method_scope=('CeSpGRN undirected association TF projection'
                                  if method == 'CeSpGRN' else
                                  'CellOracle official Net core, not full Oracle'
                                  if method == 'CellOracle_Net_core' else
                                  'directed/global TF-target ranking or association baseline'),
                    prediction_coverage=result.get('candidate_coverage'),
                    peak_sampled_worker_tree_rss_bytes=status.get(
                        'peak_sampled_worker_tree_rss_bytes')))
            if method == 'JumpNicheRewire_communication':
                for view in ('reference_zero_exposure', 'communication_delta'):
                    view_path = out / (view + '.parquet')
                    if sha(view_path) != result.get(view + '_sha256'):
                        raise ValueError(f'Same-fit view hash mismatch: {view_path}')
                    view_scores = _prediction(view_path, full_keys, reference).score.to_numpy(dtype=float)
                    for layer in layers:
                        y = reference[layer].to_numpy(dtype=int)
                        values = metrics(y, view_scores, (), decision_k=decision_k,
                                         f_betas=(.1, 1.))
                        same_fit_rows.append(dict(dataset=dataset, evidence_layer=layer,
                            same_fit_view=view, candidates=len(y), positives=int(y.sum()),
                            decision_k=decision_k, AP=values['AP'], AUROC=values['AUROC'],
                            Early_precision=values['Early_precision'],
                            Precision=values['Precision'], Recall=values['Recall'],
                            F1=values['F1'], F0_1=values['F0.1']))
    output = cfg.output_root / '04_evaluation'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'benchmark_metrics_10_methods.csv'
    if path.exists():
        raise FileExistsError('Completed extension evaluation exists; do not overwrite')
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(same_fit_rows).to_csv(output / 'same_fit_views_top10.csv', index=False)
    audit = dict(methods=list(EXISTING_METHODS + NEW_METHODS), datasets=list(DATASETS),
                 decision_rule='top ceil(10% of evaluable TF-target candidates) within each evidence layer',
                 tie_rule='expected precision/recall under random ordering of equal scores',
                 decision_k={'gse165141_liver': math.ceil(234 * DECISION_FRACTION),
                             'gse205055_p22': math.ceil(840 * DECISION_FRACTION)},
                 decision_fraction=DECISION_FRACTION,
                 additional_metrics_are_post_request_descriptive=True,
                 threshold_not_optimized_on_reference=True,
                 F_beta='(1+beta^2)*P*R/(beta^2*P+R); beta=0.1 or 1',
                 unobserved_edges_are_not_verified_negatives=True,
                 input_source_plan_sha256=sha(plan_path),
                 engineering_revision_sha256=sha(revision_path) if revision_path.is_file() else None,
                 evaluator_source_sha256=sha(__import__('pathlib').Path(__file__)))
    (output / 'evaluation_audit.json').write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    return path
