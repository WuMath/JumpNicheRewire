"""One predeclared development dataset, cached fits only; no new training.

This is analysis, not an experiment entry or a replacement for the frozen
round-02 scoring policy. Inference cache and truth provenance are verified.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_round02.workflow import verified_plan
from independent_baselines.comparison import metrics
from jump_niche_rewire_v2.adapter import load_development, sha
from jump_niche_rewire_v2.execution import checkpoint_identity, target_path, _validate_checkpoint
from jump_niche_rewire_v2.persistence import load_target_checkpoint, ResourceGuard
from .views import coefficient_views, reference_in_centered_coordinates


DATASET = 'published_tree1_500_cells110_genes_sigma0.1_1'


def analyze(cfg):
    plan = verified_plan(cfg)
    if DATASET not in cfg.dataset_ids:
        raise ValueError('Single dataset is not in the frozen development registry')
    row = pd.read_csv(cfg.development_registry).set_index('dataset_id').loc[DATASET]
    manifest = json.loads((Path(row.dataset_dir)/'evidence_manifest.json').read_text(encoding='utf8'))
    truth_path = Path(row.truth_dir)/'truth_grn.csv'
    if sha(truth_path) != manifest['files']['truth/truth_grn.csv']:
        raise ValueError('Development labels differ from frozen manifest')
    truth = pd.read_csv(truth_path, dtype={'tf':str,'target':str})
    if len(truth) != 624 or truth.duplicated(['tf','target']).any():
        raise ValueError('Expected unique 624-edge common universe')
    full = load_development(cfg, DATASET)
    off = load_development(replace(cfg, adapter=replace(cfg.adapter,use_lr=False)), DATASET)
    guard = ResourceGuard(cfg)
    score_maps = {name:{} for name in ('full_refit','no_lr_refit','same_fit_mean_reference',
                                      'same_fit_raw_zero_reference')}
    delta_abs = []
    reconstruction_error = 0.0
    for candidate, role, data in [('v2_anchor','02_runs',full),('anchor_no_lr','02_ablations',off)]:
        root = Path(cfg.output_root)/role/'JumpNicheRewire'/candidate/DATASET
        contract = json.loads((root/'execution_contract.json').read_text(encoding='utf8'))
        if data.cells != contract['cells'] or data.tfs != contract['tfs'] or data.targets != contract['targets']:
            raise ValueError('Cache IDs differ from prepared data')
        for index, target in enumerate(data.targets):
            guard()
            record, arrays = load_target_checkpoint(target_path(root,index,target),
                checkpoint_identity(contract,target),cfg.numerical)
            if not _validate_checkpoint(record,arrays,target,data.tfs):
                raise ValueError('Incomplete or nonconverged target checkpoint')
            allowed = [k for k,members in enumerate(data.lr_members) if target not in members]
            if record['design_audit']['lr_ids'] != [data.lr_ids[k] for k in allowed]:
                raise ValueError('Target LR mapping differs')
            raw = data.exposure[:,allowed]
            zero,audit = reference_in_centered_coordinates(raw,np.zeros(len(allowed)),arrays['z'])
            if any(audit['inactive_reference_changed']):
                raise ValueError('Raw zero cannot be represented for an inactive LR column')
            keys = (('full_refit', 'same_fit_mean_reference','same_fit_raw_zero_reference')
                    if candidate == 'v2_anchor' else ('no_lr_refit',))
            for name in keys:
                ref = zero if name == 'same_fit_raw_zero_reference' else np.zeros(arrays['z'].shape[1])
                view = coefficient_views(arrays['backbone'][:,0],arrays['spatial'][:,:,0],
                    arrays['lr'][:,:,0],arrays['h'],arrays['z'],ref,reference_name=name)
                reconstruction_error = max(reconstruction_error,float(np.max(np.abs(
                    np.sum(arrays['x']*view.total,axis=1) -
                    (arrays['base_prediction']+arrays['spatial_prediction']+arrays['lr_prediction'])[:,0]))))
                part = 'reference' if name.startswith('same_fit') else 'total'
                values = view.global_scores()[part]
                for tf,value in zip(data.tfs,values):
                    score_maps[name][(tf,target)] = float(value)
                if name == 'same_fit_mean_reference':
                    delta_abs.extend(np.abs(view.communication_delta).ravel().tolist())
    results = {}
    for name, scores in score_maps.items():
        if set(scores) != set(zip(truth.tf,truth.target)):
            raise ValueError(f'{name}: incomplete common edge universe')
        values = [scores[(tf,target)] for tf,target in zip(truth.tf,truth.target)]
        result = metrics(truth.label,values,cfg.classification_top_k)
        results[name] = {k:float(result[k]) for k in ('AP','AUROC','Early_precision')}
        results[name]['nonzero_edges'] = int(np.count_nonzero(values))
        results[name]['unique_scores'] = int(len(np.unique(values)))
    return dict(dataset_id=DATASET,task='global TF-target recovery on one tuned development dataset',
        full_model='v2_anchor',refit_without_communication='anchor_no_lr',
        same_fit_reference_definition='mean LR exposure / centered z=0',
        raw_zero_reference_definition='raw LR exposure=0 transformed into fitted z coordinates',
        reference_and_total_scored_against_same_global_truth=True,
        communication_increment_truth_available=False,causal_interpretation=False,
        score_readout='mean over cells of absolute TF-target coefficient',
        results=results,mean_absolute_communication_coefficient_change=float(np.mean(delta_abs)),
        maximum_prediction_reconstruction_error=reconstruction_error,
        target_checkpoints_verified=208,new_fits=0,formal_confirmation=False,
        historical_round02_plan_sha256=plan.get('plan_sha256','see frozen plan'),
        peak_sampled_rss_bytes=guard.peak_rss_bytes)
