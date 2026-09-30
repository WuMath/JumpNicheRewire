
from __future__ import annotations
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import json

PROFILES = (
    'benchmark_selected',
    'real_data.support_benchmark_extension',
    'real_data.support_benchmark',
    'public_sim_data.benchmark_round02',
    'public_sim_data.decomposition_v2',
    'public_sim_data.benchmark_optimization',
    'public_sim_data.publication_study',
    'public_sim_data.sergio', 'public_sim_data.sccigen', 'public_sim_data.scmultisim', 'real_data.visium',
    'real_data.h5ad', 'perturbation.spatial_perturbseq', 'self_sim_data.demo',
    'self_sim_data.pilot', 'self_sim_data.core_final',
)


def load_profile(name):
    if name not in PROFILES:
        raise ValueError(f'Unknown profile {name!r}; choose from {PROFILES}')
    return import_module('configs.' + name).build_config()


def effective_model(config):
    model = deepcopy(config.model)
    path = config.run.frozen_parameters_file
    if path is None:
        return model
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    parameters = payload.get('parameters')
    if not isinstance(parameters, dict) or not payload.get('parameter_id'):
        raise ValueError(f'Invalid frozen parameter file: {path}')
    mapping = {
        'l1_penalty': (model.base_grn, 'l1_penalty'),
        'fusion_penalty': (model.base_grn, 'fusion_penalty'),
        'admm_rho': (model.base_grn, 'admm_rho'),
        'admm_max_iterations': (model.base_grn, 'admm_max_iterations'),
        'spatial_permutations': (model.spatial, 'spatial_permutations'),
        'group_penalty': (model.rewiring, 'group_penalty'),
        'ridge_penalty': (model.rewiring, 'ridge_penalty'),
        'fista_max_iterations': (model.rewiring, 'fista_max_iterations'),
        'joint_iterations': (model.rewiring, 'joint_iterations'),
        'iteration_tolerance': (model.rewiring, 'iteration_tolerance'),
    }
    missing = set(mapping) - set(parameters)
    unknown = set(parameters) - set(mapping) - {'parameter_id'}
    if missing or unknown:
        raise ValueError(f'Frozen schema mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}')
    for key, (section, attribute) in mapping.items():
        setattr(section, attribute, parameters[key])
    print(f"[frozen parameters] {payload['parameter_id']} <- {path}")
    return model
