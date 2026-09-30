"""Support-retrieval AP/AUROC/Early precision on prespecified observable universes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from independent_baselines.comparison import metrics
from .prepare import sha
from .workers import METHODS


def _reference(cfg, dataset: str) -> tuple[pd.DataFrame, list[str], dict]:
    root = cfg.output_root / '02_reference' / dataset
    frame = pd.read_csv(root / 'support_edges.csv', dtype={'tf': str, 'target': str})
    audit = json.loads((root / 'reference_audit.json').read_text(encoding='utf-8'))
    if frame.duplicated(['tf', 'target']).any():
        raise ValueError('Duplicate reference candidate')
    if dataset == 'gse165141_liver':
        frame = frame.loc[frame.has_mm9_promoter.eq(1)].copy()
        return frame, ['proximal', 'distal', 'label'], audit
    motif_tfs = set(pd.read_parquet(cfg.motif_catalog, columns=['tf']).tf.astype(str))
    frame = frame.loc[frame.tf.isin(motif_tfs) & frame.has_mm10_promoter.eq(1)].copy()
    return frame, ['label'], audit


def _prediction(path: Path, full_keys: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not {'tf', 'target', 'score'}.issubset(frame) or frame.duplicated(['tf', 'target']).any():
        raise ValueError(f'Invalid or duplicate prediction keys: {path}')
    if not np.isfinite(frame.score).all() or (frame.score < 0).any():
        raise ValueError(f'Nonfinite or negative predictions: {path}')
    if len(frame) != len(full_keys) or not frame[['tf', 'target']].sort_values(['tf', 'target']).reset_index(drop=True).equals(
            full_keys.sort_values(['tf', 'target']).reset_index(drop=True)):
        raise ValueError(f'Prediction does not cover entire prespecified candidate universe: {path}')
    aligned = reference[['tf', 'target']].merge(frame[['tf', 'target', 'score']],
        on=['tf', 'target'], how='left', validate='one_to_one')
    if aligned.score.isna().any():
        raise ValueError(f'Eligible reference candidate missing from full prediction: {path}')
    return aligned


def evaluate(cfg) -> Path:
    rows = []
    internal = []
    audit = {}
    for dataset in ('gse165141_liver', 'gse205055_p22'):
        input_root = cfg.output_root / '01_model_input' / dataset
        tfs = pd.read_csv(input_root / 'evaluation_tfs.csv', dtype=str).tf.tolist()
        targets = pd.read_csv(input_root / 'evaluation_targets.csv', dtype=str).target.tolist()
        full_keys = pd.MultiIndex.from_product([tfs, targets], names=['tf', 'target']).to_frame(index=False)
        reference, layers, source_audit = _reference(cfg, dataset)
        if reference.empty or not set(zip(reference.tf, reference.target)).issubset(
                set(zip(full_keys.tf, full_keys.target))):
            raise ValueError(f'{dataset}: reference outside frozen model candidate universe')
        audit[dataset] = dict(full_candidates=len(full_keys), evaluable_candidates=len(reference),
                              source_audit=source_audit, layers={})
        for layer in layers:
            labels = reference[layer].to_numpy(dtype=int)
            positives = int(labels.sum())
            audit[dataset]['layers'][layer] = dict(positives=positives,
                unobserved=int(len(labels)-positives), prevalence=positives/len(labels),
                unobserved_are_verified_negatives=False)
        for method in METHODS:
            out = cfg.output_root / '03_runs' / dataset / method
            status_path = out / 'run_status.json'
            if not status_path.exists():
                raise FileNotFoundError(f'Missing method fit: {status_path}')
            status = json.loads(status_path.read_text(encoding='utf-8'))
            result = status.get('result') or {}
            if status.get('state') != 'completed' or result.get('state') != 'completed':
                raise RuntimeError(f'Cannot score incomplete fit: {dataset} {method}')
            path = out / 'global_grn.parquet'
            if sha(path) != result.get('prediction_sha256'):
                raise ValueError(f'Prediction hash changed: {path}')
            aligned = _prediction(path, full_keys, reference)
            for layer in layers:
                labels = reference[layer].to_numpy(dtype=int)
                available = 0 < int(labels.sum()) < len(labels)
                values = metrics(labels, aligned.score.to_numpy(dtype=float), ()) if available else {}
                rows.append(dict(dataset=dataset, evidence_layer=layer, method=method,
                    candidates=len(labels), positives=int(labels.sum()),
                    prevalence=float(labels.mean()), AP=values.get('AP'),
                    AUROC=values.get('AUROC'), Early_precision=values.get('Early_precision'),
                    status='evaluable' if available else 'not_estimable_single_class',
                    prediction_coverage=result.get('candidate_coverage'),
                    peak_sampled_worker_tree_rss_bytes=status.get('peak_sampled_worker_tree_rss_bytes')))
            if method == 'JumpNicheRewire_communication':
                for view in ('reference_zero_exposure', 'communication_delta'):
                    view_path = out / (view + '.parquet')
                    if sha(view_path) != result.get(view + '_sha256'):
                        raise ValueError(f'Same-fit view hash changed: {view_path}')
                    view_scores = _prediction(view_path, full_keys, reference).score.to_numpy(dtype=float)
                    for layer in layers:
                        labels = reference[layer].to_numpy(dtype=int)
                        available = 0 < int(labels.sum()) < len(labels)
                        values = metrics(labels, view_scores, ()) if available else {}
                        internal.append(dict(dataset=dataset, evidence_layer=layer,
                            same_fit_view=view, candidates=len(labels), positives=int(labels.sum()),
                            AP=values.get('AP'), AUROC=values.get('AUROC'),
                            Early_precision=values.get('Early_precision'),
                            status='evaluable' if available else 'not_estimable_single_class'))
    out = cfg.output_root / '04_evaluation'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'benchmark_metrics.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(internal).to_csv(out / 'same_fit_views.csv', index=False)
    (out / 'evaluation_audit.json').write_text(json.dumps(dict(
        datasets=audit, metric_definition='average_precision_score; roc_auc_score; tie-expected precision at number of positives',
        score_universe='observable subset of fixed TF x target product, not top-k model output',
        label_zero='unobserved, not experimentally disproven',
        truth_kind='experimental-support retrieval, not complete causal GRN'),
        indent=2, ensure_ascii=False), encoding='utf-8')
    return path
