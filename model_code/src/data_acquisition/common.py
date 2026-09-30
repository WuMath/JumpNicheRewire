"""Reusable data-acquisition utilities with no command-line arguments."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    overwrite: bool,
    expected_md5: str | None = None,
) -> Path:
    """Download atomically and optionally validate an official checksum."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if expected_md5 and file_digest(destination, "md5").lower() != expected_md5.lower():
            raise ValueError(f"Existing file has the wrong MD5: {destination}")
        return destination

    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "JumpNicheRewire/0.1 data downloader"})
    downloaded = 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            downloaded += len(block)
            if downloaded and downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                print(f"  downloaded {downloaded / 1024**2:.1f} MiB: {destination.name}")
    if expected_md5 and file_digest(temporary, "md5").lower() != expected_md5.lower():
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file has the wrong MD5: {destination}")
    temporary.replace(destination)
    return destination


def safe_extract_tar(archive: Path, destination: Path) -> Path:
    """Extract regular files/directories while blocking traversal and links."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            resolved_member = (destination / member.name).resolve()
            try:
                resolved_member.relative_to(resolved_root)
            except ValueError as error:
                raise ValueError(f"Unsafe archive member: {member.name}") from error
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not accepted: {member.name}")
        bundle.extractall(destination)
    return destination


def find_file(root: Path, candidate_names: Iterable[str]) -> Path:
    for name in candidate_names:
        direct = Path(root) / name
        if direct.exists():
            return direct
        matches = list(Path(root).rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"None of {list(candidate_names)} was found under {root}")


def _read_compressed_tsv(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix.lower() == ".gz" else None
    return pd.read_csv(path, sep="\t", header=None, compression=compression)


def read_10x_matrix(matrix_root: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Read a 10x Matrix Market directory as genes x barcodes."""

    matrix_path = find_file(matrix_root, ("matrix.mtx.gz", "matrix.mtx"))
    features_path = find_file(matrix_root, ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"))
    barcodes_path = find_file(matrix_root, ("barcodes.tsv.gz", "barcodes.tsv"))

    if matrix_path.suffix.lower() == ".gz":
        with gzip.open(matrix_path, "rb") as handle:
            matrix = mmread(handle).tocsr()
    else:
        matrix = mmread(matrix_path).tocsr()
    features = _read_compressed_tsv(features_path)
    barcodes = _read_compressed_tsv(barcodes_path).iloc[:, 0].astype(str).to_numpy()
    if matrix.shape[0] != len(features) or matrix.shape[1] != len(barcodes):
        raise ValueError("10x matrix/features/barcodes dimensions do not agree")

    names = features.iloc[:, 1 if features.shape[1] > 1 else 0].astype(str).to_numpy()
    if features.shape[1] > 2:
        gene_expression = features.iloc[:, 2].astype(str).eq("Gene Expression").to_numpy()
        matrix = matrix[gene_expression]
        names = names[gene_expression]
    return matrix.astype(np.float64), names, barcodes


def read_visium_positions(path: Path, tissue_only: bool) -> pd.DataFrame:
    """Read Space Ranger v1 or v2 tissue-position files."""

    frame = pd.read_csv(path)
    if "barcode" not in frame.columns:
        frame = pd.read_csv(path, header=None, names=POSITION_COLUMNS)
    missing = set(POSITION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Position table is missing columns: {sorted(missing)}")
    if tissue_only:
        frame = frame.loc[pd.to_numeric(frame["in_tissue"], errors="coerce").eq(1)].copy()
    frame["barcode"] = frame["barcode"].astype(str)
    frame["x"] = pd.to_numeric(frame["pxl_col_in_fullres"], errors="raise")
    frame["y"] = pd.to_numeric(frame["pxl_row_in_fullres"], errors="raise")
    return frame[["barcode", "x", "y", "array_row", "array_col", "in_tissue"]]


def spatial_farthest_sample(coordinates: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    """Choose a spatially spread subset; return all rows when no cap is set."""

    n = len(coordinates)
    if maximum is None or maximum <= 0 or n <= maximum:
        return np.arange(n, dtype=int)
    random = np.random.default_rng(seed)
    scaled = np.asarray(coordinates, dtype=float)
    scale = np.ptp(scaled, axis=0)
    scaled = (scaled - scaled.min(axis=0)) / np.where(scale > 0, scale, 1.0)
    centre = np.mean(scaled, axis=0)
    first_candidates = np.flatnonzero(np.isclose(np.sum((scaled - centre) ** 2, axis=1), np.min(np.sum((scaled - centre) ** 2, axis=1))))
    first = int(random.choice(first_candidates))
    selected = np.empty(maximum, dtype=int)
    selected[0] = first
    minimum_distance = np.sum((scaled - scaled[first]) ** 2, axis=1)
    minimum_distance[first] = -1.0
    for index in range(1, maximum):
        chosen = int(np.argmax(minimum_distance))
        selected[index] = chosen
        distance = np.sum((scaled - scaled[chosen]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected[: index + 1]] = -1.0
    return np.sort(selected)


def read_network(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    if frame.empty:
        raise ValueError(f"Network resource is empty: {path}")
    return frame


def _symbol_columns(frame: pd.DataFrame) -> tuple[str, str]:
    source = "source_genesymbol" if "source_genesymbol" in frame.columns else "source"
    target = "target_genesymbol" if "target_genesymbol" in frame.columns else "target"
    if source not in frame.columns or target not in frame.columns:
        raise ValueError("Network requires source/target gene-symbol columns")
    return source, target


def _confidence(frame: pd.DataFrame) -> np.ndarray:
    if "curation_effort" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    values = pd.to_numeric(frame["curation_effort"], errors="coerce").fillna(0.0).to_numpy()
    maximum = float(np.log1p(values).max()) if len(values) else 0.0
    return 0.5 + 0.5 * np.log1p(values) / maximum if maximum > 0 else np.ones(len(frame))


def match_network_resources(
    gene_names: np.ndarray,
    expression: sparse.csr_matrix,
    collectri: pd.DataFrame,
    ligrec: pd.DataFrame,
    maximum_tfs: int,
    maximum_lr_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match mouse gene symbols and build TF, LR, and TF-target references."""

    lookup: dict[str, str] = {}
    for name in gene_names.astype(str):
        lookup.setdefault(name.upper(), name)
    detection = np.asarray((expression > 0).sum(axis=1)).ravel() / max(expression.shape[1], 1)
    mean = np.asarray(expression.mean(axis=1)).ravel()
    gene_score = {str(gene_names[i]).upper(): float(detection[i] + np.log1p(mean[i])) for i in range(len(gene_names))}

    tf_source, tf_target = _symbol_columns(collectri)
    tf_network = collectri.copy()
    tf_network["tf"] = tf_network[tf_source].astype(str).str.upper().map(lookup)
    tf_network["target"] = tf_network[tf_target].astype(str).str.upper().map(lookup)
    tf_network = tf_network.dropna(subset=["tf", "target"])
    if tf_network.empty:
        raise ValueError("No CollecTRI TF-target symbols overlap the expression matrix")
    tf_rank = (
        tf_network.groupby("tf")["target"].nunique().rename("target_count").reset_index()
    )
    tf_rank["expression_score"] = tf_rank["tf"].str.upper().map(gene_score).fillna(0.0)
    tf_rank = tf_rank.sort_values(["expression_score", "target_count", "tf"], ascending=[False, False, True])
    selected_tfs = tf_rank.head(maximum_tfs)["tf"].astype(str).tolist()
    tf_list = pd.DataFrame({"tf": selected_tfs})

    lr_source, lr_target = _symbol_columns(ligrec)
    lr_network = ligrec.copy()
    lr_network["ligand"] = lr_network[lr_source].astype(str).str.upper().map(lookup)
    lr_network["receptor"] = lr_network[lr_target].astype(str).str.upper().map(lookup)
    lr_network = lr_network.dropna(subset=["ligand", "receptor"])
    lr_network = lr_network.loc[lr_network["ligand"] != lr_network["receptor"]].copy()
    if lr_network.empty:
        raise ValueError("No OmniPath ligand-receptor symbols overlap the expression matrix")
    lr_network["confidence"] = _confidence(lr_network)
    lr_network["expression_score"] = [
        min(gene_score.get(str(ligand).upper(), 0.0), gene_score.get(str(receptor).upper(), 0.0))
        for ligand, receptor in zip(lr_network["ligand"], lr_network["receptor"])
    ]
    lr_network = (
        lr_network.sort_values(["expression_score", "confidence"], ascending=False)
        .drop_duplicates(["ligand", "receptor"])
        .head(maximum_lr_pairs)
        .copy()
    )
    lr_pairs = lr_network[["ligand", "receptor", "confidence"]].copy()
    lr_pairs.insert(0, "lr_id", lr_pairs["ligand"].astype(str) + "__" + lr_pairs["receptor"].astype(str))
    lr_pairs["signal_type"] = "unknown"
    lr_pairs["rho"] = 1.0
    lr_pairs = lr_pairs[["lr_id", "ligand", "receptor", "signal_type", "rho", "confidence"]]

    tf_network = tf_network.loc[tf_network["tf"].isin(selected_tfs)].copy()
    tf_network["grn"] = _confidence(tf_network)
    prior = (
        tf_network.groupby(["tf", "target"], as_index=False)["grn"].max()
        .assign(motif=np.nan, atac=np.nan)
        [["tf", "target", "motif", "atac", "grn"]]
    )
    return tf_list, lr_pairs, prior


def select_expression_genes(
    matrix: sparse.csr_matrix,
    gene_names: np.ndarray,
    required_genes: Iterable[str],
    maximum_genes: int,
    minimum_detected_spots: int,
) -> list[str]:
    """Retain all selected TF/LR genes and fill remaining slots with HVGs."""

    names = np.asarray(gene_names, dtype=str)
    unique_names = list(dict.fromkeys(names.tolist()))
    row_by_name: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        row_by_name.setdefault(name, []).append(index)
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
    mean = np.asarray(matrix.mean(axis=1)).ravel()
    second = np.asarray(matrix.power(2).mean(axis=1)).ravel()
    variance = np.maximum(second - mean**2, 0.0)
    dispersion = variance / np.maximum(mean, 1e-8)
    score_by_name = {
        name: max(float(dispersion[index]) for index in indices)
        for name, indices in row_by_name.items()
        if max(int(detected[index]) for index in indices) >= minimum_detected_spots
    }
    selected: list[str] = []
    for name in required_genes:
        if name in row_by_name and name not in selected:
            selected.append(name)
    ranked = sorted(score_by_name, key=lambda name: (-score_by_name[name], name))
    for name in ranked:
        if name not in selected:
            selected.append(name)
        if len(selected) >= maximum_genes:
            break
    if len(selected) < 2:
        raise ValueError("Too few genes remain after expression filtering")
    return selected[: max(maximum_genes, len(set(required_genes)))]


def dense_selected_expression(
    matrix: sparse.csr_matrix,
    gene_names: np.ndarray,
    selected_genes: list[str],
) -> np.ndarray:
    """Aggregate duplicate feature symbols and materialize only selected genes."""

    row_by_name: dict[str, list[int]] = {}
    for index, name in enumerate(np.asarray(gene_names, dtype=str)):
        row_by_name.setdefault(name, []).append(index)
    output = np.zeros((matrix.shape[1], len(selected_genes)), dtype=np.float64)
    for column, gene in enumerate(selected_genes):
        rows = row_by_name[gene]
        output[:, column] = np.asarray(matrix[rows].sum(axis=0)).ravel()
    return output


def write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path

