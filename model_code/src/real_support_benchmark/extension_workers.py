"""Blind RegDiffusion and CellOracle Net-core fits on frozen real-data subsets."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .prepare import sha
from .workers import _candidate_frame, _complete, _save_prediction


METHODS = ('CeSpGRN', 'RegDiffusion', 'CellOracle_Net_core')


def _logged(root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    expression = pd.read_csv(root / 'expression.csv', index_col=0)
    values = expression.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expression must contain finite nonnegative counts')
    library = values.sum(axis=1)
    if (library <= 0).any():
        raise ValueError('Zero expression library')
    logged = pd.DataFrame(np.log1p(values / library[:, None] * 10000),
                          index=expression.index, columns=expression.columns)
    tfs = pd.read_csv(root / 'evaluation_tfs.csv', dtype=str).tf.tolist()
    targets = pd.read_csv(root / 'evaluation_targets.csv', dtype=str).target.tolist()
    if not set(tfs + targets).issubset(logged.columns):
        raise ValueError('Evaluation genes absent from expression')
    return logged, tfs, targets


def _regdiffusion(cfg, root: Path, out: Path, logged: pd.DataFrame,
                  tfs: list[str], targets: list[str]) -> tuple[pd.DataFrame, dict]:
    sys.path.insert(0, str(cfg.reg_source))
    from regdiffusion import RegDiffusionTrainer
    from regdiffusion.logger import LightLogger

    matrix = logged.to_numpy(dtype=np.float32)
    keep = np.var(matrix, axis=0) > 0
    fit_genes = logged.columns[keep].tolist()
    removed = logged.columns[~keep].tolist()
    if not fit_genes:
        raise ValueError('All expression genes are constant')
    logger = LightLogger(result_dir=str(out / 'train_logs'))
    trainer = RegDiffusionTrainer(matrix[:, keep], device='cpu', seed=20260923,
                                  n_steps=1000, logger=logger)
    trainer.train()
    adjacency = np.asarray(trainer.get_adj(), dtype=np.float32)
    if adjacency.shape != (len(fit_genes), len(fit_genes)) or not np.isfinite(adjacency).all():
        raise ValueError('Official RegDiffusion adjacency invalid')
    ix = {gene: i for i, gene in enumerate(fit_genes)}
    frame = pd.DataFrame([(tf, target,
                           float(abs(adjacency[ix[tf], ix[target]]))
                           if tf in ix and target in ix else 0.)
                          for tf in tfs for target in targets],
                         columns=['tf', 'target', 'score'])
    np.savez_compressed(out / 'adjacency.npz', adjacency=adjacency,
                        genes=np.asarray(fit_genes), removed=np.asarray(removed))
    return frame, dict(removed_zero_variance=removed,
                       adjacency_sha256=sha(out / 'adjacency.npz'),
                       parameters={'seed': 20260923, 'n_steps': 1000,
                                   'device': 'cpu', 'other': 'official defaults'})


def _celloracle(cfg, root: Path, out: Path, logged: pd.DataFrame,
                tfs: list[str], targets: list[str]) -> tuple[pd.DataFrame, dict]:
    from communication_benchmark.celloracle_core_one import load_official_net

    Net = load_official_net(cfg.cell_package)
    removed = logged.columns[logged.var(axis=0).eq(0)].tolist()
    fitted_input = logged.drop(columns=removed)
    eligible_tfs = [gene for gene in tfs if gene not in removed]
    eligible_targets = [gene for gene in targets if gene not in removed]
    if not eligible_tfs or not eligible_targets:
        raise ValueError('No variable TF/target genes for CellOracle Net core')
    # Every TF is an uninformative candidate for every target: no ChIP/ATAC/motif labels.
    prior = {gene: eligible_tfs[:] for gene in eligible_targets}
    model = Net(gene_expression_matrix=fitted_input, TFinfo_dic=prior, verbose=False)
    model.fit_genes(target_genes=eligible_targets, bagging_number=20,
                    scaling=True, model_method='bagging_ridge', alpha=10,
                    n_jobs=1, verbose=False)
    if model.failed_genes or set(model.fitted_genes) != set(eligible_targets):
        raise RuntimeError(f'Incomplete CellOracle Net-core fit: {model.failed_genes}')
    raw = []
    for target in eligible_targets:
        for tf, row in model.stats_dict[target].iterrows():
            raw.append(dict(tf=str(tf), target=target, score=float(row.coef_abs)))
    raw_frame = pd.DataFrame(raw)
    if raw_frame.duplicated(['tf', 'target']).any():
        raise ValueError('Duplicate CellOracle Net-core edge')
    frame = _complete(raw_frame, _candidate_frame(root))[0]
    raw_frame.to_parquet(out / 'raw_grn.parquet', index=False)
    return frame, dict(removed_zero_variance=removed,
                       raw_sha256=sha(out / 'raw_grn.parquet'),
                       parameters={'alpha': 10, 'bagging_number': 20,
                                   'n_jobs': 1, 'method': 'bagging_ridge',
                                   'uninformative_all_TF_candidate_prior': True,
                                   'full_Oracle_workflow': False,
                                   'KNN_imputation': False})


def fit(cfg, dataset: str, method: str) -> dict:
    if dataset not in ('gse165141_liver', 'gse205055_p22') or method not in ('RegDiffusion', 'CellOracle_Net_core'):
        raise ValueError('Unregistered Python extension fit')
    root = cfg.base_root / '01_model_input' / dataset
    out = cfg.output_root / '03_runs' / dataset / method
    if not root.is_dir() or (out / 'global_grn.parquet').exists():
        raise FileExistsError('Missing input or existing prediction')
    logged, tfs, targets = _logged(root)
    frame, details = (_regdiffusion(cfg, root, out, logged, tfs, targets)
                      if method == 'RegDiffusion' else
                      _celloracle(cfg, root, out, logged, tfs, targets))
    prediction, coverage = _complete(frame, _candidate_frame(root))
    digest = _save_prediction(out / 'global_grn.parquet', prediction)
    return dict(state='completed', dataset=dataset, method=method,
                candidates=len(prediction), candidate_coverage=coverage,
                prediction_sha256=digest,
                input_hashes={name: sha(root / name) for name in
                              ('expression.csv', 'coordinates.csv', 'evaluation_tfs.csv',
                               'evaluation_targets.csv')},
                truth_read=False, **details)
