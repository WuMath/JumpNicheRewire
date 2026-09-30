"""One blind fit per registered method and real spatial subset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .prepare import sha


METHODS = ('JumpNicheRewire_base', 'JumpNicheRewire_communication',
           'PPCOR', 'CLR', 'TIGRESS', 'GRNBoost2', 'GENIE3')
BASELINE_PARAMETERS = {
    'PPCOR': {'method': 'kendall'},
    'CLR': {'estimator': 'pearson', 'disc': 'none', 'nbins': 10},
    'TIGRESS': {'alpha': .4, 'nstepsLARS': 5, 'nsplit': 200},
    'GRNBoost2': {'max_features': .5, 'max_depth': 1},
    'GENIE3': {'min_samples_leaf': 5, 'max_features': .5},
}


@dataclass(frozen=True)
class BlindFitSettings:
    output_root: Path
    rscript: Path
    r_library: Path
    r_software: Path
    seed: int
    l1_penalty: float
    fusion_penalty: float
    group_penalty: float
    ridge_penalty: float
    spatial_permutations: int


def _candidate_frame(root: Path) -> pd.DataFrame:
    tfs = pd.read_csv(root / 'evaluation_tfs.csv', dtype=str).tf.tolist()
    targets = pd.read_csv(root / 'evaluation_targets.csv', dtype=str).target.tolist()
    return pd.MultiIndex.from_product([tfs, targets], names=['tf', 'target']).to_frame(index=False)


def _complete(frame: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = frame[['tf', 'target', 'score']].copy()
    frame[['tf', 'target']] = frame[['tf', 'target']].astype(str)
    if frame.duplicated(['tf', 'target']).any() or not np.isfinite(frame.score).all() or (frame.score < 0).any():
        raise ValueError('Invalid or duplicated estimator output')
    aligned = candidates.merge(frame, on=['tf', 'target'], how='left', validate='one_to_one')
    coverage = int(aligned.score.notna().sum())
    aligned['score'] = aligned.score.fillna(0.)
    return aligned, coverage


def _save_prediction(path: Path, frame: pd.DataFrame) -> str:
    if path.exists():
        raise FileExistsError(f'Prediction already exists: {path}')
    temporary = path.with_name('.' + path.name + '.writing')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return sha(path)


def _jnr(cfg, dataset: str, method: str, root: Path, out: Path, candidates: pd.DataFrame) -> dict:
    from jump_niche_rewire import JumpNicheRewirePipeline
    from jump_niche_rewire.convergence import summarize_convergence
    from publication_study.inference import build_model

    row = dict(dataset_dir=str(root), species='mouse', is_spatial=True,
               split='development', signature=sha(root / 'expression.csv'))
    model = build_model(cfg, row, out)
    model.base_grn.l1_penalty = .075
    model.base_grn.fusion_penalty = .0005
    model.rewiring.group_penalty = .01
    model.rewiring.ridge_penalty = .0001
    model.base_grn.maximum_targets = pd.read_csv(root / 'expression.csv', nrows=0, index_col=0).shape[1]
    model.rewiring.disable_communication_rewiring = method.endswith('_base')
    model.rewiring.joint_iterations = 1 if method.endswith('_base') else model.rewiring.joint_iterations
    model.output.maximum_grn_rows = 20000
    model.rewiring.top_grn_rows = 20000
    model.runtime.verbose = True
    model.runtime.checkpoint_data_signature = row['signature'] + ':' + method
    (out / 'model_config.json').write_text(json.dumps(asdict(model), default=str, indent=2), encoding='utf-8')
    result = JumpNicheRewirePipeline(model).run()
    convergence = summarize_convergence(result.base, result.rewiring,
        model.runtime.minimum_target_convergence_rate,
        communication_rewiring_disabled=model.rewiring.disable_communication_rewiring)
    if convergence.get('convergence_passed') is not True:
        raise RuntimeError(f'JNR did not satisfy fixed convergence policy: {convergence}')
    tf_count, target_count = len(result.base.tf_names), len(result.base.target_names)
    keys = pd.DataFrame({'tf': np.repeat(result.base.tf_names, target_count),
                         'target': np.tile(result.base.target_names, tf_count)})
    total = np.mean(np.abs(result.rewiring.beta_ccc), axis=0).ravel()
    prediction, coverage = _complete(keys.assign(score=total), candidates)
    digest = _save_prediction(out / 'global_grn.parquet', prediction)
    details = dict(convergence=convergence, fitted_tfs=tf_count, fitted_targets=target_count,
                   candidate_coverage=coverage, prediction_sha256=digest)
    if method.endswith('_communication'):
        reference = np.mean(np.abs(result.base.beta), axis=0).ravel()
        delta = np.mean(np.abs(result.rewiring.beta_ccc - result.base.beta), axis=0).ravel()
        for name, values in [('reference_zero_exposure', reference), ('communication_delta', delta)]:
            internal, _ = _complete(keys.assign(score=values), candidates)
            details[name + '_sha256'] = _save_prediction(out / (name + '.parquet'), internal)
        details['reference_definition'] = 'same fit, LR exposure set to zero in fitted coefficient scale'
        details['reference_is_causal_counterfactual'] = False
    return details


def _baseline(cfg, method: str, root: Path, out: Path, candidates: pd.DataFrame) -> dict:
    expression = pd.read_csv(root / 'expression.csv', index_col=0)
    logged = np.log1p(expression.div(expression.sum(axis=1).replace(0, 1), axis=0) * 10000)
    tfs = pd.read_csv(root / 'tf_list.csv', dtype=str).tf.tolist()
    params = BASELINE_PARAMETERS[method]
    if method in ('GENIE3', 'GRNBoost2'):
        from arboreto.algo import diy
        from arboreto.core import RF_KWARGS, SGBM_KWARGS
        from distributed import Client, LocalCluster
        kwargs = dict(RF_KWARGS if method == 'GENIE3' else SGBM_KWARGS)
        kwargs.update(params)
        with LocalCluster(n_workers=1, threads_per_worker=1, processes=False,
                          dashboard_address=None) as cluster, Client(cluster) as client:
            result = diy(expression_data=logged, regressor_type='RF' if method == 'GENIE3' else 'GBM',
                         regressor_kwargs=kwargs, tf_names=tfs,
                         client_or_address=client, seed=cfg.seed)
        frame = result.rename(columns={'TF': 'tf', 'importance': 'score'})
    else:
        logged.to_csv(out / 'log_expression.csv')
        job = dict(method=method, params=params, seed=cfg.seed,
                   expression=str((out / 'log_expression.csv').resolve()),
                   tfs=str((root / 'tf_list.csv').resolve()),
                   library=str(cfg.r_library.resolve()), software=str(cfg.r_software.resolve()))
        (out / 'job.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
        wrapper = Path(__file__).resolve().parents[1] / 'benchmark_optimization/official_r.R'
        if method == 'TIGRESS' and len(tfs) == 2:
            wrapper = Path(__file__).with_name('tigress_two_tf_compat.R')
        with (out / 'official_r.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(cfg.rscript), '--vanilla', str(wrapper)], cwd=out, check=True,
                           stdout=log, stderr=subprocess.STDOUT,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        frame = pd.read_csv(out / 'official_grn.csv', dtype={'tf': str, 'target': str})
    prediction, coverage = _complete(frame, candidates)
    digest = _save_prediction(out / 'global_grn.parquet', prediction)
    return dict(candidate_coverage=coverage, prediction_sha256=digest,
                parameters=params, convergence_policy='not_applicable_to_baseline')


def fit(cfg, dataset: str, method: str) -> dict:
    if dataset not in ('gse165141_liver', 'gse205055_p22') or method not in METHODS:
        raise ValueError('Unregistered dataset/method')
    root = cfg.output_root / '01_model_input' / dataset
    out = cfg.output_root / '03_runs' / dataset / method
    if not root.is_dir() or (out / 'global_grn.parquet').exists():
        raise FileExistsError('Missing input or existing output; no silent overwrite')
    candidates = _candidate_frame(root)
    signature = {name: sha(root / name) for name in ('expression.csv', 'coordinates.csv',
        'metadata.csv', 'tf_list.csv', 'lr_pairs.csv', 'prior_tf_gene.csv',
        'prior_lr_tf.csv', 'evaluation_tfs.csv', 'evaluation_targets.csv')}
    details = (_jnr(cfg, dataset, method, root, out, candidates)
               if method.startswith('JumpNicheRewire') else
               _baseline(cfg, method, root, out, candidates))
    return dict(state='completed', dataset=dataset, method=method, candidates=len(candidates),
                input_hashes=signature, truth_read=False, **details)
