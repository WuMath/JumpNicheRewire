"""Internal Python 3.9 worker for the frozen CeSpGRN spatial default fit.

Invoked only by the registered benchmark controller, without experiment CLI
arguments. Reads only the controller's blind job.json in its working folder.
"""

import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch


OFFICIAL_BLOBS = {
    'g_admm.py': '448b49fb79acc798f1c6c7d54e5bdd28945ead7b',
    'kernel.py': '9179ff9c9f30f4aa6c4b863a4349383c0cfe57a2',
    'torch_sqrtm.py': '4de49f9c74662b341184b0b339c95c38dab5efc3',
}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    out = Path.cwd()
    job = json.loads((out / 'job.json').read_text(encoding='utf-8'))
    try:
        root = Path(job['input_root'])
        source = Path(job['source_dir'])
        for name, digest in job['input_hashes'].items():
            if _sha(root / name) != digest:
                raise ValueError('Blind input changed: ' + name)
        for name, digest in OFFICIAL_BLOBS.items():
            if _blob(source / name) != digest:
                raise ValueError('Official CeSpGRN source changed: ' + name)
        sys.path.insert(0, str(source))
        import g_admm
        import kernel

        expression = pd.read_csv(root / 'expression.csv', index_col=0)
        coords = pd.read_csv(root / 'coordinates.csv', index_col=0)
        if not expression.index.equals(coords.index):
            raise ValueError('Expression and coordinates are not row-aligned')
        tfs = pd.read_csv(root / 'evaluation_tfs.csv', dtype=str).tf.tolist()
        targets = pd.read_csv(root / 'evaluation_targets.csv', dtype=str).target.tolist()
        values = expression.to_numpy(dtype=np.float64)
        library = values.sum(axis=1)
        if not np.isfinite(values).all() or (values < 0).any() or (library <= 0).any():
            raise ValueError('Invalid expression counts')
        logged = np.log1p(values / library[:, None] * 10000)
        genes = expression.columns.tolist()
        n, g = logged.shape
        torch.manual_seed(20260923)
        np.random.seed(20260923)
        K, K_trun = kernel.calc_kernel_neigh(coords[['x', 'y']].to_numpy(dtype=np.float64),
                                               k=5, bandwidth=1., truncate=True,
                                               truncate_param=30)
        covariance = g_admm.est_cov(X=logged, K_trun=K_trun, weighted_kt=True)
        model = g_admm.G_admm_minibatch(X=logged[:, None, :], K=K, pre_cov=covariance,
                                        batchsize=32, seed=20260923,
                                        device=torch.device('cpu'))
        graphs = np.asarray(model.train(max_iters=1000, n_intervals=100,
                                        lamb=.1, njobs=1))
        if graphs.shape != (n, g, g) or not np.isfinite(graphs).all():
            raise ValueError('Official CeSpGRN graph invalid or incomplete')
        tf_ix = [genes.index(tf) for tf in tfs]
        target_ix = [genes.index(target) for target in targets]
        projected = graphs[:, tf_ix, :][:, :, target_ix]
        scores = np.mean(np.abs(projected), axis=0)
        frame = pd.DataFrame([(tf, target, float(scores[i, j]))
                              for i, tf in enumerate(tfs)
                              for j, target in enumerate(targets)],
                             columns=['tf', 'target', 'score'])
        if len(frame) != len(tfs) * len(targets) or frame.duplicated(['tf', 'target']).any():
            raise ValueError('CeSpGRN TF-target projection incomplete')
        projection = out / 'cell_tf_target_graphs.npz'
        np.savez_compressed(projection, graphs=projected.astype(np.float32),
                            cells=np.asarray(expression.index), tfs=np.asarray(tfs),
                            targets=np.asarray(targets))
        prediction = out / 'global_grn.parquet'
        frame.to_parquet(prediction, index=False)
        result = dict(state='completed', dataset=job['dataset'], method='CeSpGRN',
                      candidates=len(frame), candidate_coverage=len(frame),
                      prediction_sha256=_sha(prediction),
                      projection_sha256=_sha(projection),
                      input_hashes=job['input_hashes'], truth_read=False,
                      parameters={'kernel_input': 'spatial', 'kernel_k': 5,
                                  'bandwidth': 1., 'neighbors': 30, 'lamb': .1,
                                  'max_iters': 1000, 'batch_size': 32,
                                  'seed': 20260923, 'device': 'cpu',
                                  'covariance_weighted_kt': True},
                      native_network='undirected cell-specific association')
        (out / 'fit_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return 0
    except BaseException as error:
        (out / 'fit_result.json').write_text(json.dumps(dict(
            state='failed', error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc()), indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
