"""Model orchestration only; numerical algorithms live in jump_niche_rewire/."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
from .configuration import effective_model
from .registry import MODEL_FILES, selected_rows, snapshot, validate_inputs
from .safety import writable, write_json, file_hash, digest, implementation_signature


def evidence_settings(config, registry):
    from configs.common.evidence_settings import EvidenceModelConfig
    model = effective_model(config)
    return EvidenceModelConfig(
        registry_path=registry, output_root=writable(config.run.output_root / '02_runs'),
        dataset_ids=config.inputs.dataset_ids or (config.name,),
        evidence_tracks=('public_simulator', 'real_pseudotruth', 'perturbation', 'application', 'self_sim_data'),
        isolate_datasets=config.run.isolate_datasets, overwrite=config.run.overwrite,
        require_convergence=model.runtime.require_convergence,
        minimum_convergence_rate=model.runtime.minimum_target_convergence_rate,
        maximum_tfs=model.base_grn.maximum_tfs, maximum_targets=model.base_grn.maximum_targets,
        maximum_active_lr=model.rewiring.maximum_active_lr,
        spatial_permutations=model.spatial.spatial_permutations,
        l1_penalty=model.base_grn.l1_penalty, fusion_penalty=model.base_grn.fusion_penalty,
        group_penalty=model.rewiring.group_penalty, ridge_penalty=model.rewiring.ridge_penalty,
        admm_max_iterations=model.base_grn.admm_max_iterations,
        fista_max_iterations=model.rewiring.fista_max_iterations,
        joint_iterations=model.rewiring.joint_iterations,
        iteration_tolerance=model.rewiring.iteration_tolerance,
        project_model=model, implementation_signature=implementation_signature(),
    )


def single_registry(config):
    root = Path(config.inputs.single_input_dir)
    validate_inputs(root)
    from .registry import write_rows
    row = dict(dataset_id=config.name, evidence_track='application', source=str(root),
               evaluation_scope='application', truth_kind='not_evaluated', species=config.model.data.species,
               tissue='configured', dataset_dir=str(root.resolve()), truth_dir=str(root / 'truth'),
               signature=digest({name: file_hash(root / name) for name in MODEL_FILES}))
    return write_rows(config.run.output_root / '00_selection/evidence_registry.csv', [row])


def check(config):
    print(f'[profile] {config.category}.{config.name} / workflow={config.workflow}')
    print(f'[input mode] {config.inputs.source}')
    print(f'[output] {writable(config.run.output_root)}')
    if config.workflow == 'single':
        validate_inputs(config.inputs.single_input_dir)
        print(f'[input] {config.inputs.single_input_dir}')
    else:
        rows = selected_rows(config)
        for row in rows:
            validate_inputs(row['dataset_dir'])
        print(f'[datasets] {len(rows)}: ' + ', '.join(r['dataset_id'] for r in rows))
    if config.workflow != 'formal':
        effective_model(config)
    print('[check passed] No fitting, truth reading, or output writing was performed.')


def status(config):
    root = Path(config.run.output_root)
    paths = sorted(root.rglob('run_status.json')) if root.exists() else []
    if not paths:
        print(f'[status] No model runs under {root}')
    for path in paths:
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
            print(item.get('dataset_id', path.parent.name), item.get('state'),
                  'convergence=', item.get('convergence_passed'), str(path))
        except (ValueError, OSError) as error:
            print(f'[unreadable] {path}: {error}')


def run(config, action):
    if config.processor == 'selected_benchmark':
        from selected_benchmark.workflow import run as run_selected
        return run_selected(config.preprocessing, action)
    if config.processor == 'real_support_extension':
        from real_support_benchmark.extension_workflow import run as run_extension
        return run_extension(config.preprocessing, action)
    if config.processor == 'real_support':
        from real_support_benchmark.workflow import run as run_real_support
        return run_real_support(config.preprocessing, action)
    if config.processor == 'benchmark_round02':
        from benchmark_round02.workflow import run
        return run(config.preprocessing, action)
    if config.processor == 'decomposition_v2':
        from jump_niche_rewire_v2.interface import run
        return run(config.preprocessing, action)
    if config.processor == 'optimization':
        from benchmark_optimization.workflow import run
        return run(config.preprocessing, action)
    if config.processor == 'publication':
        from publication_study.workflow import run
        return run(config.preprocessing, action)
    if config.processor == 'scmultisim':
        from evidence_benchmark.scmultisim_protocol import freeze, require_frozen
        if action == 'freeze':
            return freeze(config)
        if action == 'evaluate':
            raise ValueError('scMultiSim uses baselines/evaluate_comparison.py for joint GRN/CCI evaluation; see docs/SCMULTISIM_BENCHMARK.md')
        if config.preprocessing.split == 'test' and action not in {'check', 'status'}:
            require_frozen(config)
    if action == 'check':
        return check(config)
    if action == 'status':
        return status(config)
    writable(config.run.output_root)
    if config.category == 'perturbation' and not config.preprocessing.allow_exploratory_legacy_protocol:
        raise RuntimeError('Perturbation formal protocol is not validated. See docs/KNOWN_LIMITATIONS.md; conversion is available separately.')
    if config.workflow == 'formal':
        return _formal(config, action)
    if config.workflow == 'single' and action != 'run':
        raise ValueError('Single application/demo supports check/run/status. Formal/evidence profiles provide benchmark evaluation.')
    allowed = {'run', 'evaluate', 'jobs', 'import_external'}
    if action not in allowed:
        raise ValueError(f'Main action must be one of check/status/{sorted(allowed)}')
    registry = single_registry(config) if config.workflow == 'single' else snapshot(config)
    write_json(config.run.output_root / '00_selection/experiment_config.json', config)
    if config.run.frozen_parameters_file is not None:
        write_json(config.run.output_root / '00_selection/frozen_parameters_source.json', {
            'path': config.run.frozen_parameters_file,
            'sha256': file_hash(config.run.frozen_parameters_file),
            'content': json.loads(config.run.frozen_parameters_file.read_text(encoding='utf-8-sig')),
        })
    if action == 'run':
        from evidence_benchmark.run_model import run_models
        result = run_models(evidence_settings(config, registry))
        rows = __import__('pandas').read_csv(result)
        if not rows.state.eq('completed').all():
            raise RuntimeError(f'Some dataset workers failed; successful datasets are retained. Inspect {result}')
        return result
    from configs.common.comparison import EvaluationParameters, JobParameters, ExternalParameters
    if action == 'evaluate':
        from evidence_benchmark.evaluate import run_evaluation
        evaluation = replace(config.evaluation or EvaluationParameters(), registry_path=registry,
                             run_root=config.run.output_root / '02_runs',
                             output_root=writable(config.run.output_root / '03_evaluation'),
                             external_result_root=config.run.output_root / '02_runs/baselines')
        return run_evaluation(evaluation)
    if action == 'jobs':
        from evidence_benchmark.jobs import run_jobs
        jobs = replace(config.jobs or JobParameters(), registry_path=registry,
                       output_root=writable(config.run.output_root / '01_jobs'))
        return run_jobs(jobs)
    from evidence_benchmark.external_import import run_external_import
    settings = replace(config.external_import or ExternalParameters(),
                       output_root=writable(config.run.output_root / '02_runs/baselines'))
    if not set(spec.dataset_id for spec in settings.specifications).issubset(set(config.inputs.dataset_ids)):
        raise ValueError('External results must belong to the selected profile dataset_ids')
    return run_external_import(settings)


def _formal(config, action):
    from formal_benchmark.stage2_model import run_stage2
    from formal_benchmark.convergence_audit import audit_convergence, ConvergenceAuditConfig
    if action == 'run':
        action = config.formal_model.phase
    if action not in {'tune', 'audit', 'freeze', 'test', 'evaluate'}:
        raise ValueError('Formal profile: check/status/tune/audit/freeze/test/evaluate')
    if action in {'test', 'evaluate'} and not config.run.unlock_test:
        raise PermissionError('Set RunParameters.unlock_test=True only after inspecting the NEW frozen parameters and audit.')
    snapshot(config)
    model = replace(config.formal_model, phase='freeze' if action == 'audit' else action,
                    benchmark_root=writable(config.run.output_root),
                    unlock_test=config.run.unlock_test, overwrite=config.run.overwrite,
                    project_model=deepcopy(config.model))
    write_json(config.run.output_root / 'experiment_config.json', config)
    if action in {'audit', 'freeze'}:
        audit_convergence(ConvergenceAuditConfig(output_dir=config.run.output_root / 'convergence',
                          allowed_splits=('train', 'validation')), model_config=model)
        if action == 'audit':
            return config.run.output_root / 'convergence'
    if action == 'evaluate':
        lock = model.profile_benchmark_root / '02_runs/frozen_parameters.json'
        if not lock.exists():
            raise FileNotFoundError(f'Freeze before test evaluation: {lock}')
        from formal_benchmark.stage3_analysis import run_stage3
        return run_stage3(replace(config.evaluation, benchmark_root=config.run.output_root))
    return run_stage2(model)
