"""Fixed input-only spatial/LR construction; no benchmark labels are opened."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse
from .core import build_design


@dataclass(frozen=True)
class AdapterConfig:
    is_spatial: bool = True
    library_scale: float = 10000.
    neighbors: int = 8
    spatial_basis_per_sample: int = 8
    saturation_quantile: float = .9
    complex_separator: str = '&'
    sample_column: str = 'sample_id'
    covariates: tuple[str, ...] = ('cell_type',)
    adjust_library: bool = True
    use_spatial: bool = True
    use_lr: bool = True
    maximum_input_file_bytes: int = 256*1024**2


@dataclass
class Prepared:
    cells: list
    genes: list
    tfs: list
    targets: list
    standardized: np.ndarray
    covariates: np.ndarray
    covariate_names: list
    spatial_basis: np.ndarray
    basis_names: list
    exposure: np.ndarray
    lr_ids: list
    lr_members: list
    graph: sparse.csr_matrix
    audit: dict


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _zscore(values):
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, 1.)
    return (values-mean)/scale


def spatial_features(coordinates, samples, cells, cfg):
    """Receiver-row kNN, deterministic distance ties, per-sample RBF columns.

    No expression is used to choose a graph or spatial basis. RBF is a fixed
    geometric representation, not claimed to be a newly invented spatial model.
    """
    n = len(cells)
    if not cfg.is_spatial:
        return sparse.csr_matrix((n, n)), np.empty((n, 0)), [], []
    if coordinates.shape != (n, 2) or not np.isfinite(coordinates).all():
        raise ValueError('Spatial coordinates must be finite n by 2')
    rows, columns, values, basis, names, sample_audit = [], [], [], [], [], []
    ids = np.asarray(cells, dtype=str)
    for sample in sorted(set(samples)):
        indices = np.flatnonzero(samples == sample)
        indices = indices[np.argsort(ids[indices], kind='stable')]
        points = coordinates[indices]
        m = len(indices)
        distances = []
        for position, receiver in enumerate(indices):
            squared = np.sum((points-points[position])**2, axis=1)
            eligible = np.flatnonzero(np.arange(m) != position)
            ordered = eligible[np.lexsort((ids[indices[eligible]], squared[eligible]))]
            chosen = ordered[:min(cfg.neighbors, m-1)]
            if not len(chosen):
                continue
            positive = squared[chosen][squared[chosen] > 0]
            bandwidth2 = float(positive[-1]) if len(positive) else 1.
            weight = np.exp(-squared[chosen]/(2*bandwidth2))
            weight /= weight.sum()
            rows.extend([int(receiver)]*len(chosen)); columns.extend(indices[chosen].tolist()); values.extend(weight.tolist())
            distances.extend(positive.tolist())
        bandwidth2 = float(np.median(distances)) if distances else 1.
        centers = []
        nearest = np.full(m, np.inf)
        for k in range(min(cfg.spatial_basis_per_sample, m)) if cfg.use_spatial else ():
            chosen = 0 if k == 0 else int(np.argmax(nearest))
            if k and nearest[chosen] <= 1e-14:
                break
            centers.append(str(ids[indices[chosen]]))
            squared = np.sum((points-points[chosen])**2, axis=1)
            nearest = np.minimum(nearest, squared)
            column = np.exp(-squared/(2*bandwidth2))
            column -= column.mean()
            full = np.zeros(n); full[indices] = column
            basis.append(full); names.append(f'{sample}:rbf{k}')
        sample_audit.append(dict(sample=str(sample), cells=m, rbf_bandwidth_squared=bandwidth2, anchors=centers))
    graph = sparse.csr_matrix((values, (rows, columns)), shape=(n, n))
    h = np.column_stack(basis) if basis else np.empty((n, 0))
    return graph, h, names, sample_audit


def prepare_frames(expression, metadata, coordinates, tfs, lr_pairs, cfg=None):
    cfg = cfg or AdapterConfig()
    if not (isinstance(cfg.neighbors, int) and cfg.neighbors >= 1 and
            isinstance(cfg.spatial_basis_per_sample, int) and cfg.spatial_basis_per_sample >= 0):
        raise ValueError('Invalid neighbors/basis count')
    if not 0 < cfg.saturation_quantile <= 1 or not np.isfinite(cfg.library_scale) or cfg.library_scale <= 0:
        raise ValueError('Invalid normalization settings')
    if not cfg.complex_separator:
        raise ValueError('Complex separator must be explicit')
    if expression.index.has_duplicates or expression.columns.has_duplicates or metadata.index.has_duplicates:
        raise ValueError('Duplicate cells or genes')
    if not expression.index.equals(metadata.index):
        raise ValueError('Metadata row order must equal expression cells')
    if cfg.is_spatial and (coordinates is None or not expression.index.equals(coordinates.index)):
        raise ValueError('Coordinate row order must equal expression cells')
    counts = expression.to_numpy(dtype=float)
    if len(counts) < 2 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Need finite nonnegative counts and at least two cells')
    genes, cells = expression.columns.astype(str).tolist(), expression.index.astype(str).tolist()
    if len(set(genes)) != len(genes) or len(set(cells)) != len(cells):
        raise ValueError('Identifiers collide after string conversion')
    tfs = list(map(str, tfs))
    if not tfs or len(set(tfs)) != len(tfs) or not set(tfs).issubset(genes):
        raise ValueError('Invalid or unexpressed TF identifiers')
    targets = [gene for gene in genes if gene not in set(tfs)]
    if not targets:
        raise ValueError('No non-TF targets')
    if cfg.sample_column not in metadata or metadata[cfg.sample_column].isna().any():
        raise ValueError('Explicit nonmissing sample IDs are required')
    samples = metadata[cfg.sample_column].astype(str).to_numpy()
    library = counts.sum(axis=1)
    if not np.isfinite(library).all(): raise ValueError('Library size overflow')
    logged = np.log1p(counts/np.maximum(library[:, None], 1.)*cfg.library_scale)
    standardized = _zscore(logged)
    cblocks, cnames = [], []
    if cfg.adjust_library:
        cblocks.append(_zscore(np.log1p(library[:, None]))); cnames.append('log_library_size')
    # Sample dummy effects are mandatory when samples differ.
    for name in dict.fromkeys((cfg.sample_column, *cfg.covariates)):
        if name not in metadata or metadata[name].isna().any():
            raise ValueError(f'Missing configured covariate: {name}')
        series = metadata[name]
        if pd.api.types.is_numeric_dtype(series) and name != cfg.sample_column:
            array = series.to_numpy(dtype=float)[:, None]
            if not np.isfinite(array).all(): raise ValueError('Non-finite covariate')
            if series.nunique() > 1: cblocks.append(_zscore(array)); cnames.append(name)
        else:
            dummy = pd.get_dummies(series.astype(str), drop_first=True, dtype=float)
            if dummy.shape[1]: cblocks.append(dummy.to_numpy()); cnames.extend([f'{name}:{v}' for v in dummy.columns])
    covariates = np.column_stack(cblocks) if cblocks else np.empty((len(cells), 0))
    graph, h, basis_names, graph_audit = spatial_features(
        coordinates[['x', 'y']].to_numpy(float) if cfg.is_spatial else None, samples, cells, cfg)
    required = {'lr_id', 'ligand', 'receptor'}
    if not required.issubset(lr_pairs.columns) or lr_pairs[list(required)].isna().any().any():
        raise ValueError('LR table must explicitly contain lr_id, ligand, receptor')
    if lr_pairs.lr_id.astype(str).duplicated().any(): raise ValueError('Duplicate LR identifiers')
    saturated = np.zeros_like(logged)
    for j in range(len(genes)):
        positive = logged[:, j][logged[:, j] > 0]
        if len(positive):
            scale = np.quantile(positive, cfg.saturation_quantile)
            saturated[:, j] = logged[:, j]/(logged[:, j]+scale)
    exposures, lr_ids, members, lr_audit = [], [], [], []
    for row in lr_pairs.sort_values('lr_id').itertuples(index=False):
        ligand = tuple(str(row.ligand).split(cfg.complex_separator))
        receptor = tuple(str(row.receptor).split(cfg.complex_separator))
        if any(not g or g.strip() != g for g in ligand+receptor): raise ValueError('Ambiguous complex member')
        missing = sorted(set(ligand+receptor)-set(genes))
        reason = 'missing_complex_members' if missing else ('nonspatial_or_disabled' if not cfg.is_spatial or not cfg.use_lr else '')
        lr_audit.append(dict(lr_id=str(row.lr_id), included=not bool(reason), reason=reason, missing_members=missing))
        if reason: continue
        ligand_level = np.min(saturated[:, [genes.index(g) for g in ligand]], axis=1)
        receptor_level = np.min(saturated[:, [genes.index(g) for g in receptor]], axis=1)
        exposures.append((graph@ligand_level)*receptor_level)
        lr_ids.append(str(row.lr_id)); members.append(sorted(set(ligand+receptor)))
    exposure = np.column_stack(exposures) if exposures else np.empty((len(cells), 0))
    audit = dict(cells=len(cells), genes=len(genes), tfs=len(tfs), targets=len(targets),
                 candidates=len(tfs)*len(targets), graph_edges=graph.nnz,
                 covariates=cnames, spatial_basis_columns=h.shape[1],
                 graph=graph_audit, lr_pairs=lr_audit, zero_library_cells=int((library == 0).sum()),
                 normalization='log1p(counts/library*configured_scale) then gene ddof=1 zscore',
                 library_scale=cfg.library_scale, truth_read=False,
                 limitations=['common library-size normalization retains compositional dependence',
                              'LR exposure is an expression proxy, not independent activity'])
    return Prepared(cells, genes, tfs, targets, standardized, covariates, cnames, h, basis_names,
                    exposure, lr_ids, members, graph, audit)


def target_design(prepared, target, numerical_config=None):
    if target not in prepared.targets: raise ValueError('Target outside declared candidate set')
    allowed = np.array([target not in m for m in prepared.lr_members], dtype=bool)
    blocked = [lr for lr, keep in zip(prepared.lr_ids, allowed) if not keep]
    x = prepared.standardized[:, [prepared.genes.index(tf) for tf in prepared.tfs]]
    y = prepared.standardized[:, [prepared.genes.index(target)]]
    d = build_design(x, y, prepared.spatial_basis, prepared.exposure[:, allowed], prepared.covariates, numerical_config)
    d.audit.update(target=target, tfs=prepared.tfs,
                   lr_ids=[lr for lr, keep in zip(prepared.lr_ids, allowed) if keep],
                   excluded_target_containing_lr=blocked,
                   exclusion_scope='LR main effects AND all TF interactions',
                   basis_names=prepared.basis_names, covariate_names=prepared.covariate_names)
    return d


def load_development(settings, dataset_id):
    """Read exactly whitelisted model inputs; never evaluate a prediction."""
    registry_path = Path(settings.development_registry)
    plan = json.loads(Path(settings.round01_plan).read_text(encoding='utf8'))
    if sha(registry_path) != plan['registry_sha256']: raise ValueError('Development registry changed')
    rows = pd.read_csv(registry_path)
    if len(rows) != 6 or not rows.split.eq('development').all() or rows.dataset_id.duplicated().any():
        raise ValueError('Only six locked development datasets are allowed')
    match = rows.loc[rows.dataset_id.eq(dataset_id)]
    if len(match) != 1: raise ValueError('Dataset is not in development registry')
    row = match.iloc[0]
    root = Path(row.dataset_dir)
    manifest = json.loads((root/'evidence_manifest.json').read_text(encoding='utf8'))
    if manifest['signature'] != row.signature or manifest['dataset_id'] != dataset_id:
        raise ValueError('Input manifest identity changed')
    names = ('expression.csv', 'metadata.csv', 'coordinates.csv', 'tf_list.csv', 'lr_pairs.csv')
    hashes = {}
    for name in names:
        path = root/name
        if path.stat().st_size > settings.adapter.maximum_input_file_bytes: raise MemoryError('Input file exceeds budget')
        hashes[name] = sha(path)
        if hashes[name] != manifest['files'][name]: raise ValueError(f'Changed input {name}')
    prepared = prepare_frames(pd.read_csv(root/'expression.csv', index_col=0),
        pd.read_csv(root/'metadata.csv', index_col=0), pd.read_csv(root/'coordinates.csv', index_col=0),
        pd.read_csv(root/'tf_list.csv', dtype=str).tf.tolist(), pd.read_csv(root/'lr_pairs.csv', dtype=str), settings.adapter)
    prepared.audit.update(dataset_id=dataset_id, input_hashes=hashes, registry_hash=sha(registry_path))
    return prepared
