"""Independent read-only verification of the 10-method real-support table."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/result/real_support_benchmark_20260925'
EXT = BASE / '05_extension'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_at_k(y: np.ndarray, score: np.ndarray, k: int) -> tuple[float, float]:
    cutoff = np.partition(score, len(score) - k)[len(score) - k]
    above = score > cutoff
    tied = score == cutoff
    hits = y[above].sum() + (k - above.sum()) * y[tied].mean()
    return float(hits / k), float(hits / y.sum())


def f_beta(precision: float, recall: float, beta: float) -> float:
    denominator = beta * beta * precision + recall
    return float((1 + beta * beta) * precision * recall / denominator) if denominator else 0.0


def main() -> None:
    table = pd.read_csv(EXT / '04_evaluation/benchmark_metrics_10_methods.csv')
    audit = json.loads((EXT / '04_evaluation/evaluation_audit.json').read_text())
    assert len(table) == 40 and len(audit['methods']) == 10
    assert json.loads((EXT / 'study_process.json').read_text())['state'] == 'completed'
    assert digest(EXT / '01_extension_plan.json') == audit['input_source_plan_sha256']
    assert digest(EXT / '01_extension_plan_revision_01.json') == audit['engineering_revision_sha256']
    checks = 0
    max_error = 0.0
    for row in table.itertuples(index=False):
        ref = pd.read_csv(BASE / '02_reference' / row.dataset / 'support_edges.csv',
                          dtype={'tf': str, 'target': str})
        if row.dataset == 'gse165141_liver':
            ref = ref.loc[ref.has_mm9_promoter.eq(1)]
        else:
            catalog = pd.read_parquet(ROOT.parent / 'data/processed/references/motif_tf_gene.parquet',
                                      columns=['tf'])
            ref = ref.loc[ref.tf.isin(set(catalog.tf.astype(str))) &
                          ref.has_mm10_promoter.eq(1)]
        y = ref[row.evidence_layer].to_numpy(dtype=int)
        assert len(y) == row.candidates and int(y.sum()) == row.positives
        k = math.ceil(.1 * len(y))
        assert k == row.decision_k
        output = (BASE if row.method not in audit['methods'][-3:] else EXT) / '03_runs' / row.dataset / row.method
        status = json.loads((output / 'run_status.json').read_text())
        assert status['state'] == status['result']['state'] == 'completed'
        prediction = output / 'global_grn.parquet'
        assert digest(prediction) == status['result']['prediction_sha256']
        frame = pd.read_parquet(prediction)
        assert not frame.duplicated(['tf', 'target']).any()
        merged = ref[['tf', 'target']].merge(frame[['tf', 'target', 'score']],
                                           on=['tf', 'target'], how='left', validate='one_to_one')
        score = merged.score.to_numpy(dtype=float)
        assert np.isfinite(score).all()
        precision, recall = expected_at_k(y, score, k)
        ep, _ = expected_at_k(y, score, int(y.sum()))
        wanted = dict(AP=average_precision_score(y, score), AUROC=roc_auc_score(y, score),
                      Early_precision=ep, Precision=precision, Recall=recall,
                      F1=f_beta(precision, recall, 1), F0_1=f_beta(precision, recall, .1))
        for name, value in wanted.items():
            error = abs(float(getattr(row, name)) - float(value))
            max_error = max(max_error, error)
            assert error < 1e-12, (row.dataset, row.method, row.evidence_layer, name, error)
            checks += 1
    print(json.dumps(dict(rows=len(table), methods=len(audit['methods']),
                          metric_checks=checks, max_absolute_error=max_error,
                          decision_k=audit['decision_k']), indent=2))


if __name__ == '__main__':
    main()
