"""Evaluation-only reference construction; never import from model workers."""

from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left
import gzip
import json
from pathlib import Path
import shutil
import time

import pandas as pd
import psutil

from .prepare import check_resources, sha


def _promoters_mm9(path: Path, targets: set[str]) -> dict[str, list[tuple[str, int, int]]]:
    promoters: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = line.rstrip('\n').split('\t')
            if len(row) < 13 or row[12] not in targets:
                continue
            tss = int(row[4] if row[3] == '+' else row[5])
            promoters[row[12]].append((row[2], max(0, tss - 2000), tss + 2000))
    return promoters


def _promoters_mm10(path: Path, targets: set[str]) -> dict[str, list[tuple[str, int, int]]]:
    promoters: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.startswith('#'):
                continue
            row = line.rstrip('\n').split('\t')
            if len(row) < 9 or row[2] != 'gene':
                continue
            attrs = dict(part.strip().split(' ', 1) for part in row[8].split(';') if ' ' in part.strip())
            symbol = attrs.get('gene_name', '').strip('"')
            if symbol not in targets:
                continue
            tss = int(row[3] if row[6] == '+' else row[4])
            # Ensembl GRCm38 uses "1" while CellRanger-ARC mm10 fragments use
            # "chr1". This naming normalization is not a coordinate lift-over.
            chrom = row[0] if row[0].startswith('chr') else 'chr' + row[0]
            promoters[symbol].append((chrom, max(0, tss - 2000), tss + 2000))
    return promoters


def _peaks(path: Path) -> dict[str, tuple[list[int], list[int]]]:
    peaks: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.startswith('#'):
                continue
            row = line.split('\t')
            if len(row) >= 4:
                peaks[row[1]].append((int(row[2]), int(row[3])))
    indexed = {}
    for chrom, intervals in peaks.items():
        intervals.sort()
        starts = [a for a, _ in intervals]
        prefix = []
        largest = -1
        for _, end in intervals:
            largest = max(largest, end)
            prefix.append(largest)
        indexed[chrom] = (starts, prefix)
    return indexed


def _overlap(interval: tuple[str, int, int], peaks: dict[str, tuple[list[int], list[int]]]) -> bool:
    chrom, start, end = interval
    indexed = peaks.get(chrom)
    if indexed is None:
        return False
    starts, prefix = indexed
    last = bisect_left(starts, end) - 1
    return last >= 0 and prefix[last] > start


def _resource_check(cfg, process: psutil.Process, started: float) -> None:
    if time.monotonic() - started > cfg.task_limit_seconds:
        raise TimeoutError('Reference preparation exceeded 4 hours')
    if process.memory_info().rss > cfg.worker_rss_limit:
        raise MemoryError('Reference process RSS exceeded 3.5 GiB')
    if psutil.virtual_memory().available < cfg.minimum_available_ram:
        raise MemoryError('Available RAM fell below 1.5 GiB')
    if shutil.disk_usage(cfg.output_root.anchor).free < cfg.minimum_disk_free:
        raise OSError('Free disk fell below 5 GiB')


def _contacts(path: Path) -> list[tuple[tuple[str, int, int], tuple[str, int, int]]]:
    def interval(value: str) -> tuple[str, int, int]:
        chrom, a, b = value.strip().strip('"').split(',')
        return chrom, int(a), int(b)
    contacts = []
    with gzip.open(path, 'rt') as stream:
        next(stream)
        for line in stream:
            row = line.rstrip('\n').split('\t')
            if len(row) < 7 or float(row[6]) < 5:
                continue
            first, second = interval(row[0]), interval(row[3])
            if first[0] != second[0] or abs((first[1] + first[2]) // 2 -
                                             (second[1] + second[2]) // 2) > 1_000_000:
                continue
            contacts.append((first, second))
    return contacts


def liver(cfg) -> Path:
    input_root = cfg.output_root / '01_model_input/gse165141_liver'
    out = cfg.output_root / '02_reference/gse165141_liver'
    if out.exists():
        raise FileExistsError(f'Reference already exists: {out}')
    tfs = pd.read_csv(input_root / 'evaluation_tfs.csv').tf.astype(str).tolist()
    targets = pd.read_csv(input_root / 'evaluation_targets.csv').target.astype(str).tolist()
    promoters = _promoters_mm9(cfg.raw_root / 'mouse_annotations/mm9_refGene.txt.gz', set(targets))
    root = cfg.raw_root / 'gse124463_liver_reference'
    filenames = {
        'Hnf4a': ['GSM3534498_LFD_HNF4a_ChIPSeq_replicate1.HOMER_peaks.txt.gz',
                   'GSM3534499_LFD_HNF4a_ChIPSeq_replicate2.HOMER_peaks.txt.gz'],
        'Cebpa': ['GSM3534506_LFD_CEBPa_ChIPSeq_replicate1.HOMER_peaks.txt.gz',
                   'GSM3534507_LFD_CEBPa_ChIPSeq_replicate2.HOMER_peaks.txt.gz'],
    }
    peaks = {tf: [_peaks(root / name) for name in filenames[tf]] for tf in tfs}
    contacts = _contacts(root / 'GSE124463_LF.minScore_5.maxDist_1Mb.txt.gz')
    rows = []
    for tf in tfs:
        first, second = peaks[tf]
        for target in targets:
            intervals = promoters.get(target, [])
            proximal = bool(intervals and any(_overlap(p, first) for p in intervals)
                            and any(_overlap(p, second) for p in intervals))
            distal = False
            for promoter in intervals:
                for a, b in contacts:
                    for near, far in ((a, b), (b, a)):
                        if (near[0] == promoter[0] and near[1] < promoter[2]
                                and promoter[1] < near[2] and _overlap(far, first)
                                and _overlap(far, second)):
                            distal = True
                            break
                    if distal:
                        break
                if distal:
                    break
            rows.append(dict(tf=tf, target=target, proximal=int(proximal), distal=int(distal),
                             label=int(proximal or distal), has_mm9_promoter=int(bool(intervals))))
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'support_edges.csv', index=False)
    audit = dict(dataset='gse165141_liver', genome='mm9', tfs=tfs, targets=len(targets),
                 n_candidates=len(frame), proximal=int(frame.proximal.sum()),
                 distal=int(frame.distal.sum()), union=int(frame.label.sum()),
                 promoter_coverage=int(frame.groupby('target').has_mm9_promoter.max().sum()),
                 contacts=len(contacts), coordinate_rule='TSS +/- 2kb, two ChIP replicates, CHiCAGO>=5, <=1Mb',
                 unobserved_not_true_negatives=True, model_prior_used=False,
                 source_hashes={str(path): sha(path) for path in
                    [cfg.raw_root / 'mouse_annotations/mm9_refGene.txt.gz',
                     root / 'GSE124463_LF.minScore_5.maxDist_1Mb.txt.gz',
                     *[root / name for names in filenames.values() for name in names]]})
    (out / 'reference_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return out


def p22(cfg) -> Path:
    input_root = cfg.output_root / '01_model_input/gse205055_p22'
    out = cfg.output_root / '02_reference/gse205055_p22'
    if (out / 'support_edges.csv').exists() or (out / 'reference_audit.json').exists():
        raise FileExistsError(f'Reference already exists: {out}')
    tfs = pd.read_csv(input_root / 'evaluation_tfs.csv').tf.astype(str).tolist()
    targets = pd.read_csv(input_root / 'evaluation_targets.csv').target.astype(str).tolist()
    gtf_path = getattr(cfg, 'gtf_path', cfg.raw_root / 'mouse_annotations/Mus_musculus.GRCm38.86.gtf.gz')
    promoters = _promoters_mm10(gtf_path, set(targets))
    spots = set(pd.read_csv(input_root / 'coordinates.csv', index_col=0).index.astype(str))
    motifs = pd.read_parquet(cfg.motif_catalog, columns=['tf', 'target'])
    motif_edges = set(zip(motifs.loc[motifs.tf.isin(tfs) & motifs.target.isin(targets), 'tf'],
                          motifs.loc[motifs.tf.isin(tfs) & motifs.target.isin(targets), 'target']))
    # Small promoter-bin index; the 6.9 GB decompressed fragments are streamed,
    # never extracted or held in memory. Only selected RNA spot barcodes count.
    bins: dict[tuple[str, int], set[str]] = defaultdict(set)
    for target, intervals in promoters.items():
        for chrom, start, end in intervals:
            for index in range(start // 1000, (end - 1) // 1000 + 1):
                bins[(chrom, index)].add(target)
    counts = dict.fromkeys(targets, 0)
    matching_fragments = 0
    process = psutil.Process()
    started = time.monotonic()
    inspected = 0
    fragment_path = getattr(cfg, 'fragment_path', cfg.raw_root / 'gse205055_p22/GSM6758285_MouseBrain_20um_100barcodes_ATAC_fragments.tsv.gz')
    with gzip.open(fragment_path, 'rt') as stream:
        for line in stream:
            inspected += 1
            if inspected % 250000 == 0:
                _resource_check(cfg, process, started)
            if line.startswith('#'):
                continue
            row = line.rstrip('\n').split('\t')
            if len(row) < 4 or row[3].split('-')[0] not in spots:
                continue
            matching_fragments += 1
            chrom, start, end = row[0], int(row[1]), int(row[2])
            candidates = set()
            for index in range(start // 1000, (end - 1) // 1000 + 1):
                candidates.update(bins.get((chrom, index), ()))
            for target in candidates:
                if any(c == chrom and a < end and start < b for c, a, b in promoters[target]):
                    counts[target] += 1
    rows = [dict(tf=tf, target=target, promoter_fragments=counts[target],
                 promoter_accessible=int(counts[target] >= 5),
                 gene_level_motif=int((tf, target) in motif_edges),
                 label=int(counts[target] >= 5 and (tf, target) in motif_edges),
                 has_mm10_promoter=int(bool(promoters.get(target))))
            for tf in tfs for target in targets]
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'support_edges.csv', index=False)
    audit = dict(dataset='gse205055_p22', genome='GRCm38/mm10', tfs=len(tfs), targets=len(targets),
                 n_candidates=len(frame), motif_support=int(frame.gene_level_motif.sum()),
                 accessible_targets=sum(value >= 5 for value in counts.values()),
                 joint_support=int(frame.label.sum()), matching_fragments=matching_fragments,
                 coordinate_rule='selected 500 RNA spots; target promoter TSS +/- 2kb; >=5 ATAC fragments',
                 motif_rule='preexisting cisTarget top-1000 gene-level 10kb motif ranking; not peak-level motif hit',
                 unobserved_not_true_negatives=True, model_prior_used=False,
                 source_hashes={str(path): sha(path) for path in [fragment_path, cfg.motif_catalog,
                    gtf_path]})
    (out / 'reference_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return out


def prepare(cfg) -> Path:
    check_resources(cfg)
    if not (cfg.output_root / '02_reference/gse165141_liver/reference_audit.json').exists():
        print('[reference] GSE165141 liver ChIP/Hi-C', flush=True)
        liver(cfg)
    else:
        print('[reference cache] GSE165141 liver', flush=True)
    if not (cfg.output_root / '02_reference/gse205055_p22/reference_audit.json').exists():
        print('[reference] GSE205055 P22 ATAC fragments + gene-level motif', flush=True)
        p22(cfg)
    else:
        print('[reference cache] GSE205055 P22', flush=True)
    return cfg.output_root / '02_reference'
