"""Read-only validation of existing v2 views, not a fitting/evaluation entry.

No benchmark labels are read and no files are written. Callers explicitly pass
the frozen configuration and candidate; public scheduling remains unregistered.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from jump_niche_rewire_v2.adapter import load_development
from jump_niche_rewire_v2.execution import checkpoint_identity, target_path, _validate_checkpoint
from jump_niche_rewire_v2.persistence import load_target_checkpoint, ResourceGuard
from .views import coefficient_views, reference_in_centered_coordinates


def validate_existing_anchor(cfg):
    from benchmark_round02.workflow import verified_plan
    verified_plan(cfg)  # Reject any changed frozen input/source/cache.
    guard = ResourceGuard(cfg)
    results = []
    for dataset in cfg.dataset_ids:
        prepared = load_development(cfg, dataset)
        roots = [('v2_anchor', '02_runs', prepared),
                 ('anchor_no_lr', '02_ablations',
                  load_development(replace(cfg, adapter=replace(cfg.adapter, use_lr=False)), dataset))]
        for candidate, role_dir, data in roots:
            root = Path(cfg.output_root)/role_dir/'JumpNicheRewire'/candidate/dataset
            contract = json.loads((root/'execution_contract.json').read_text(encoding='utf8'))
            if data.cells != contract['cells'] or data.tfs != contract['tfs'] or data.targets != contract['targets']:
                raise ValueError('Cached ID mapping changed')
            maximum = 0.
            outside = 0
            unsupported_constant = 0
            for index, target in enumerate(data.targets):
                guard()
                record, arrays = load_target_checkpoint(target_path(root,index,target),
                    checkpoint_identity(contract,target),cfg.numerical)
                if not _validate_checkpoint(record,arrays,target,data.tfs):
                    raise ValueError('Nonconverged checkpoint cannot be validated as complete')
                lr_ids=record['design_audit']['lr_ids']
                allowed=[k for k,m in enumerate(data.lr_members) if target not in m]
                if [data.lr_ids[k] for k in allowed] != lr_ids:
                    raise ValueError('Target-specific LR exclusion mapping differs')
                raw=data.exposure[:,allowed]
                zero_ref,audit=reference_in_centered_coordinates(raw,np.zeros(len(allowed)),arrays['z'])
                outside+=sum(audit['outside_marginal_support'])
                unsupported_constant+=sum(audit['inactive_reference_changed'])
                for name,ref in [('mean_exposure',np.zeros(len(allowed))),('raw_zero_exposure',zero_ref)]:
                    view=coefficient_views(arrays['backbone'][:,0],arrays['spatial'][:,:,0],
                        arrays['lr'][:,:,0],arrays['h'],arrays['z'],ref,reference_name=name)
                    reconstructed=np.sum(arrays['x']*view.total,axis=1)
                    stored=(arrays['base_prediction']+arrays['spatial_prediction']+arrays['lr_prediction'])[:,0]
                    error=float(np.max(np.abs(reconstructed-stored)))
                    maximum=max(maximum,error)
                    if not np.allclose(reconstructed,stored,atol=1e-10,rtol=1e-10):
                        raise ValueError('Coefficient network fails checkpoint prediction reconstruction')
            results.append(dict(dataset=dataset,candidate=candidate,targets=len(data.targets),
                maximum_prediction_reconstruction_error=maximum,
                raw_zero_outside_marginal_support_target_lr_count=outside,
                inactive_reference_changed_target_lr_count=unsupported_constant))
    return dict(state='cache_view_validation_passed',read_only=True,new_fits=0,
                benchmark_metrics_computed=False,formal_new_comparison_complete=False,
                full_reference='mean_exposure is NOT no communication',results=results,
                peak_sampled_rss_bytes=guard.peak_rss_bytes,
                historical_budget_roots_bytes=guard.cumulative_bytes)
