"""Prepare frozen RNA/space/LR inputs without opening ChIP, Hi-C, ATAC or motif labels."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_resources(cfg, reserve: int = 256 * 1024**2) -> dict[str, int]:
    roots = cfg.budget_roots
    sizes = {str(root): sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
             for root in roots}
    used = sum(sizes.values())
    if used + reserve > cfg.cumulative_storage_limit:
        raise MemoryError(f'Cumulative storage plus reserve {used + reserve} exceeds 15 GiB')
    if psutil.virtual_memory().available < cfg.minimum_available_ram:
        raise MemoryError('Available RAM below 1.5 GiB')
    if shutil.disk_usage(cfg.output_root.anchor).free < cfg.minimum_disk_free:
        raise OSError('Free disk below 5 GiB')
    return {'used_bytes': used, 'reserve_bytes': reserve, 'limit_bytes': cfg.cumulative_storage_limit}


def _one_file(path: Path, frame: pd.DataFrame, *, index: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f'Frozen prepared file already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.writing')
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def _spatial_subset(coordinates: pd.DataFrame, count: int, seed: int) -> list[str]:
    """Round-robin through a fixed 10x10 quantile grid, blind to gene/reference data."""
    if len(coordinates) < count or coordinates.index.has_duplicates:
        raise ValueError('Insufficient or duplicate coordinate IDs')
    ordered = coordinates.sort_index()
    x = np.minimum(np.floor(ordered.x.rank(pct=True).to_numpy() * 10).astype(int), 9)
    y = np.minimum(np.floor(ordered.y.rank(pct=True).to_numpy() * 10).astype(int), 9)
    buckets: dict[tuple[int, int], list[str]] = {}
    for barcode, a, b in zip(ordered.index.astype(str), x, y):
        buckets.setdefault((int(a), int(b)), []).append(barcode)
    rng = np.random.default_rng(seed)
    for values in buckets.values():
        rng.shuffle(values)
    chosen: list[str] = []
    while len(chosen) < count:
        changed = False
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < count:
                chosen.append(buckets[key].pop())
                changed = True
        if not changed:
            raise ValueError('Spatial sampling ran out of spots')
    return sorted(chosen)


def _gene_id_to_symbol(gtf: Path) -> dict[str, str]:
    mapping: dict[str, set[str]] = {}
    with gzip.open(gtf, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            attrs = dict(part.strip().split(' ', 1) for part in fields[8].split(';') if ' ' in part.strip())
            gene_id = attrs.get('gene_id', '').strip('"').split('.')[0]
            symbol = attrs.get('gene_name', '').strip('"')
            if gene_id and symbol:
                mapping.setdefault(gene_id, set()).add(symbol)
    return {key: next(iter(values)) for key, values in mapping.items() if len(values) == 1}


def _liver(cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    coords_file = cfg.raw_root / 'gse165141_liver/GSE165141_spot_ST_CN73_Liver_C1_0.1.tsv.gz'
    coords = pd.read_csv(coords_file, sep='\t')
    coords.index = coords.x.astype(int).astype(str) + 'x' + coords.y.astype(int).astype(str)
    coords = coords[['new_x', 'new_y']].rename(columns={'new_x': 'x', 'new_y': 'y'})
    spots = _spatial_subset(coords, cfg.spots, cfg.seed)
    with tarfile.open(cfg.raw_root / 'gse165141_liver/GSE165141_RAW.tar') as archive:
        member = archive.extractfile('GSM5026931_Liver_CN73_C1_stdata.tsv.gz')
        if member is None:
            raise FileNotFoundError('Fixed liver expression member absent')
        with gzip.GzipFile(fileobj=member) as uncompressed:
            expression = pd.read_csv(uncompressed, sep='\t', index_col=0)
    missing = set(spots) - set(expression.index.astype(str))
    if missing:
        raise ValueError(f'{len(missing)} selected liver spots lack RNA')
    expression = expression.loc[spots]
    mapping = _gene_id_to_symbol(cfg.raw_root / 'mouse_annotations/Mus_musculus.GRCm38.86.gtf.gz')
    symbols = [mapping.get(str(value).split('.')[0]) if str(value).startswith('ENSMUSG') else None
               for value in expression.columns]
    keep = [i for i, value in enumerate(symbols) if value]
    expression = expression.iloc[:, keep]
    expression.columns = [symbols[i] for i in keep]
    expression = expression.T.groupby(level=0, sort=True).sum().T
    return expression, coords.loc[spots], dict(raw_spots=len(coords), gene_mapping='GRCm38.86 unique gene_id to symbol')


def _p22(cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    archive_path = cfg.external_raw / 'GSM6753043_MouseBrain_20um_100barcodes_ATAC_spatial.tar.gz'
    with tarfile.open(archive_path, 'r:gz') as archive:
        member = archive.extractfile('MouseBrain_20um_100barcodes_ATAC_spatial/tissue_positions_list.csv')
        if member is None:
            raise FileNotFoundError('P22 tissue positions absent')
        positions = pd.read_csv(member, header=None,
                                names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pixel_row', 'pixel_col'])
    positions = positions[positions.in_tissue.eq(1)].set_index('barcode')
    coords = positions[['array_col', 'array_row']].rename(columns={'array_col': 'x', 'array_row': 'y'})
    # A small fixed reserve permits replacement of GEO columns containing no
    # RNA at all. This QC uses only expression, never evaluation support.
    spots = _spatial_subset(coords, cfg.spots + 25, cfg.seed)
    rna_path = cfg.external_raw / 'GSM6753043_MouseBrain_20um_100barcodes_ATAC_matrix.tsv.gz'
    with gzip.open(rna_path, 'rt') as stream:
        header = stream.readline().rstrip('\n').split('\t')
    missing = set(spots) - set(header)
    if missing:
        raise ValueError(f'{len(missing)} selected P22 barcodes lack RNA')
    # GEO matrix is genes x barcodes, with an unlabeled first gene-name column.
    expression = pd.read_csv(rna_path, sep='\t', usecols=[0] + [header.index(value) for value in spots], index_col=0)
    expression = expression.loc[~expression.index.duplicated(keep=False)].T.loc[spots]
    empty = expression.isna().all(axis=1)
    if expression.loc[~empty].isna().any().any():
        raise ValueError('P22 RNA contains partially missing count values')
    valid = expression.index[~empty]
    chosen = _spatial_subset(coords.loc[valid], cfg.spots, cfg.seed)
    expression = expression.loc[chosen].apply(pd.to_numeric, errors='raise').astype(float)
    return expression, coords.loc[chosen], dict(raw_spots=len(coords),
        empty_rna_spots_excluded=int(empty.sum()), gene_mapping='GEO gene symbols')


def _cellchat_pairs(cfg) -> pd.DataFrame:
    output = cfg.output_root / '00_catalog/cellchat_single_gene.csv'
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        source = cfg.external_raw / 'CellChatDB.mouse.rda'
        wrapper = Path(__file__).with_name('export_cellchat.R')
        subprocess.run([str(cfg.rscript), '--vanilla', str(wrapper), str(source), str(output)], check=True,
                       creationflags=0x08000000)
    pairs = pd.read_csv(output, dtype=str)
    if not {'ligand', 'receptor'}.issubset(pairs) or pairs.duplicated(['ligand', 'receptor']).any():
        raise ValueError('Invalid single-gene CellChat export')
    return pairs


def prepare(cfg) -> Path:
    if cfg.spots != 500 or cfg.seed != 20260925:
        raise ValueError('Subset size and seed differ from frozen protocol')
    check_resources(cfg)
    catalog = pd.read_csv(cfg.tf_catalog, dtype=str).tf.dropna().astype(str).tolist()
    lr_catalog = _cellchat_pairs(cfg)
    for dataset, loader in [('gse165141_liver', _liver), ('gse205055_p22', _p22)]:
        root = cfg.output_root / '01_model_input' / dataset
        if root.exists():
            expected = ['expression.csv', 'coordinates.csv', 'metadata.csv', 'tf_list.csv',
                        'lr_pairs.csv', 'prior_tf_gene.csv', 'prior_lr_tf.csv',
                        'evaluation_tfs.csv', 'evaluation_targets.csv', 'prepare_audit.json']
            if all((root / name).is_file() for name in expected):
                print(f'[prepared cache] {dataset}: no overwrite', flush=True)
                continue
            raise FileExistsError(f'Incomplete prepared dataset exists: {root}; refusing overwrite')
        expression, coords, audit = loader(cfg)
        if expression.isna().any().any() or (expression.to_numpy() < 0).any():
            raise ValueError('RNA matrix contains missing or negative counts')
        detected = (expression > 0).mean()
        log = np.log1p(expression.div(expression.sum(axis=1).clip(lower=1), axis=0) * 10000)
        variance = log.var(ddof=0)
        eligible = detected[detected.ge(.02)].index
        if dataset == 'gse165141_liver':
            tfs = [name for name in ['Hnf4a', 'Cebpa'] if name in eligible]
        else:
            tfs = sorted((name for name in catalog if name in eligible),
                         key=lambda name: (-variance[name], name))[:20]
        if len(tfs) != (2 if dataset == 'gse165141_liver' else 20):
            raise ValueError(f'{dataset}: expected TF coverage absent: {tfs}')
        targets = sorted((name for name in eligible if name not in tfs),
                         key=lambda name: (-variance[name], name))[:120]
        if len(targets) != 120:
            raise ValueError(f'{dataset}: too few eligible RNA targets')
        pairs = lr_catalog[lr_catalog.ligand.isin(eligible) & lr_catalog.receptor.isin(eligible)].copy()
        pairs['rank'] = pairs.ligand.map(log.mean()) * pairs.receptor.map(log.mean())
        pairs = pairs.sort_values(['rank', 'ligand', 'receptor'], ascending=[False, True, True]).head(20)
        if pairs.empty:
            raise ValueError(f'{dataset}: no expressed single-gene LR pairs')
        pairs['lr_id'] = pairs.ligand + '__' + pairs.receptor
        genes = sorted(set(tfs) | set(targets) | set(pairs.ligand) | set(pairs.receptor))
        root.mkdir(parents=True)
        _one_file(root / 'expression.csv', expression.loc[:, genes], index=True)
        _one_file(root / 'coordinates.csv', coords, index=True)
        _one_file(root / 'metadata.csv', pd.DataFrame({'cell_id': expression.index,
                'sample_id': dataset, 'cell_type': 'unassigned', 'region': 'global'}))
        _one_file(root / 'tf_list.csv', pd.DataFrame({'tf': tfs}))
        _one_file(root / 'lr_pairs.csv', pairs[['lr_id', 'ligand', 'receptor']])
        _one_file(root / 'prior_tf_gene.csv', pd.DataFrame(columns=['tf', 'target', 'motif', 'atac', 'grn']))
        _one_file(root / 'prior_lr_tf.csv', pd.DataFrame(columns=['lr_id', 'tf', 'score']))
        _one_file(root / 'evaluation_tfs.csv', pd.DataFrame({'tf': tfs}))
        _one_file(root / 'evaluation_targets.csv', pd.DataFrame({'target': targets}))
        audit.update(dataset=dataset, selected_spots=len(expression), input_genes=len(genes),
                     tfs=tfs, targets=len(targets), lr_pairs=len(pairs),
                     reference_sources_read=False, seed=cfg.seed,
                     raw_input_hashes={str(path): sha(path) for path in (
                         [cfg.raw_root/'gse165141_liver/GSE165141_RAW.tar', cfg.raw_root/'gse165141_liver/GSE165141_spot_ST_CN73_Liver_C1_0.1.tsv.gz']
                         if dataset == 'gse165141_liver' else
                         [cfg.external_raw/'GSM6753043_MouseBrain_20um_100barcodes_ATAC_matrix.tsv.gz',
                          cfg.external_raw/'GSM6753043_MouseBrain_20um_100barcodes_ATAC_spatial.tar.gz'])})
        (root / 'prepare_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    return cfg.output_root / '01_model_input'
