"""Shared, network-free preprocessing utilities.

All user-editable parameters live in dataclass configuration objects in the
entry modules.  This module contains only reusable parsing and validation code.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


from grn2_project.paths import PROCESSED_DATA_ROOT, PROJECT_ROOT, RAW_DATA_ROOT

SIMPLE_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")
# Official UniProtKB 6/10-character accession patterns.  When an OmniPath
# ``*_genesymbol`` value still matches this pattern, identifier mapping failed;
# it must not be treated as an expression-matrix gene symbol.
UNIPROT_ACCESSION = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def spatial_farthest_sample(
    coordinates: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    """Return a deterministic, spatially spread row subset."""

    points = np.asarray(coordinates, dtype=float)
    n_rows = len(points)
    if maximum is None or maximum <= 0 or n_rows <= maximum:
        return np.arange(n_rows, dtype=int)
    random = np.random.default_rng(seed)
    scale = np.ptp(points, axis=0)
    scaled = (points - points.min(axis=0)) / np.where(scale > 0, scale, 1.0)
    centre = scaled.mean(axis=0)
    centre_distance = np.sum((scaled - centre) ** 2, axis=1)
    first_candidates = np.flatnonzero(np.isclose(centre_distance, centre_distance.min()))
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


def _symbol_columns(frame: pd.DataFrame) -> tuple[str, str]:
    source = "source_genesymbol" if "source_genesymbol" in frame.columns else "source"
    target = "target_genesymbol" if "target_genesymbol" in frame.columns else "target"
    if source not in frame.columns or target not in frame.columns:
        raise ValueError("Reference table requires source/target gene-symbol columns")
    return source, target


def _simple_symbol(values: pd.Series) -> pd.Series:
    return values.notna() & values.astype(str).map(
        lambda value: bool(SIMPLE_SYMBOL.fullmatch(value.strip()))
    )


def is_uniprot_accession(values: pd.Series) -> pd.Series:
    """Identify unresolved UniProt accessions present in gene-symbol columns."""

    return values.notna() & values.astype(str).str.strip().str.fullmatch(
        UNIPROT_ACCESSION.pattern
    )


def confidence_from_curation(frame: pd.DataFrame) -> np.ndarray:
    if "curation_effort" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    values = pd.to_numeric(frame["curation_effort"], errors="coerce").fillna(0.0).to_numpy()
    maximum = float(np.log1p(values).max()) if len(values) else 0.0
    if maximum <= 0:
        return np.ones(len(frame), dtype=float)
    return 0.5 + 0.5 * np.log1p(values) / maximum


def standardize_reference_tables(
    tf_target_file: Path,
    ligand_receptor_file: Path,
    exclude_unmapped_uniprot_accessions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Convert the two local OmniPath TSVs into the project table contract."""

    tf_raw = pd.read_csv(tf_target_file, sep="\t", low_memory=False)
    lr_raw = pd.read_csv(ligand_receptor_file, sep="\t", low_memory=False)
    if tf_raw.empty or lr_raw.empty:
        raise ValueError("TF-target and ligand-receptor reference tables must be non-empty")

    tf_source, tf_target = _symbol_columns(tf_raw)
    tf_edges = tf_raw.copy()
    tf_edges["tf"] = tf_edges[tf_source].astype(str).str.strip()
    tf_edges["target"] = tf_edges[tf_target].astype(str).str.strip()
    tf_simple = _simple_symbol(tf_edges[tf_source]) & _simple_symbol(tf_edges[tf_target])
    tf_unmapped_accession = is_uniprot_accession(tf_edges["tf"]) | is_uniprot_accession(
        tf_edges["target"]
    )
    tf_keep = tf_simple
    if exclude_unmapped_uniprot_accessions:
        tf_keep = tf_keep & ~tf_unmapped_accession
    tf_edges = tf_edges.loc[tf_keep].copy()
    tf_edges["grn"] = confidence_from_curation(tf_edges)
    prior = (
        tf_edges.groupby(["tf", "target"], as_index=False)["grn"]
        .max()
        .assign(motif=np.nan, atac=np.nan)
        [["tf", "target", "motif", "atac", "grn"]]
    )
    tf_list = pd.DataFrame({"tf": sorted(prior["tf"].unique())})

    lr_source, lr_target = _symbol_columns(lr_raw)
    lr_edges = lr_raw.copy()
    lr_edges["ligand"] = lr_edges[lr_source].astype(str).str.strip()
    lr_edges["receptor"] = lr_edges[lr_target].astype(str).str.strip()
    lr_simple = _simple_symbol(lr_edges[lr_source]) & _simple_symbol(lr_edges[lr_target])
    lr_unmapped_accession = is_uniprot_accession(lr_edges["ligand"]) | is_uniprot_accession(
        lr_edges["receptor"]
    )
    lr_keep = lr_simple
    if exclude_unmapped_uniprot_accessions:
        lr_keep = lr_keep & ~lr_unmapped_accession
    lr_edges = lr_edges.loc[lr_keep].copy()
    lr_edges = lr_edges.loc[lr_edges["ligand"] != lr_edges["receptor"]].copy()
    lr_edges["confidence"] = confidence_from_curation(lr_edges)
    lr_pairs = (
        lr_edges.groupby(["ligand", "receptor"], as_index=False)["confidence"]
        .max()
        .sort_values(["confidence", "ligand", "receptor"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    lr_pairs.insert(0, "lr_id", lr_pairs["ligand"] + "__" + lr_pairs["receptor"])
    lr_pairs["signal_type"] = "unknown"
    lr_pairs["rho"] = 1.0
    lr_pairs = lr_pairs[
        ["lr_id", "ligand", "receptor", "signal_type", "rho", "confidence"]
    ]
    if tf_list.empty or prior.empty or lr_pairs.empty:
        raise ValueError("No usable simple-symbol TF/LR entries remain after standardization")
    filter_summary = {
        "tf_target_input_rows": int(len(tf_raw)),
        "tf_target_non_simple_rows_removed": int((~tf_simple).sum()),
        "tf_target_unmapped_uniprot_rows_found": int((tf_simple & tf_unmapped_accession).sum()),
        "tf_target_output_unique_edges": int(len(prior)),
        "ligand_receptor_input_rows": int(len(lr_raw)),
        "ligand_receptor_non_simple_rows_removed": int((~lr_simple).sum()),
        "ligand_receptor_unmapped_uniprot_rows_found": int(
            (lr_simple & lr_unmapped_accession).sum()
        ),
        "ligand_receptor_output_unique_pairs": int(len(lr_pairs)),
    }
    return tf_list, prior, lr_pairs, filter_summary


def _expression_gene_scores(
    gene_names: np.ndarray,
    expression_genes_by_cells: sparse.csr_matrix,
) -> dict[str, float]:
    detection = np.asarray((expression_genes_by_cells > 0).sum(axis=1)).ravel()
    detection = detection / max(expression_genes_by_cells.shape[1], 1)
    means = np.asarray(expression_genes_by_cells.mean(axis=1)).ravel()
    scores: dict[str, float] = {}
    for index, name in enumerate(np.asarray(gene_names, dtype=str)):
        key = name.upper()
        scores[key] = max(scores.get(key, 0.0), float(detection[index] + np.log1p(means[index])))
    return scores


def filter_references_for_expression(
    reference_dir: Path,
    gene_names: np.ndarray,
    expression_genes_by_cells: sparse.csr_matrix,
    maximum_tfs: int | None,
    maximum_lr_pairs: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match standardized reference symbols to one measured expression panel."""

    reference_dir = Path(reference_dir)
    tf_raw = pd.read_csv(reference_dir / "tf_list.csv")
    prior_raw = pd.read_csv(reference_dir / "prior_tf_gene.csv")
    lr_raw = pd.read_csv(reference_dir / "lr_pairs.csv")
    lookup: dict[str, str] = {}
    for name in np.asarray(gene_names, dtype=str):
        lookup.setdefault(name.upper(), name)
    gene_scores = _expression_gene_scores(gene_names, expression_genes_by_cells)

    prior = prior_raw.copy()
    prior["tf"] = prior["tf"].astype(str).str.upper().map(lookup)
    prior["target"] = prior["target"].astype(str).str.upper().map(lookup)
    prior = prior.dropna(subset=["tf", "target"]).copy()
    tf_candidates = tf_raw["tf"].astype(str).str.upper().map(lookup).dropna().drop_duplicates()
    target_counts = prior.groupby("tf")["target"].nunique()
    tf_rank = pd.DataFrame({"tf": tf_candidates})
    tf_rank["expression_score"] = tf_rank["tf"].str.upper().map(gene_scores).fillna(0.0)
    tf_rank["target_count"] = tf_rank["tf"].map(target_counts).fillna(0)
    tf_rank = tf_rank.sort_values(
        ["expression_score", "target_count", "tf"], ascending=[False, False, True]
    )
    if maximum_tfs is not None and maximum_tfs > 0:
        tf_rank = tf_rank.head(maximum_tfs)
    tf_list = tf_rank[["tf"]].reset_index(drop=True)
    if tf_list.empty:
        raise ValueError("No TF symbols in ModelData overlap this expression matrix")
    prior = prior.loc[prior["tf"].isin(tf_list["tf"])].drop_duplicates(["tf", "target"])

    lr_pairs = lr_raw.copy()
    lr_pairs["ligand"] = lr_pairs["ligand"].astype(str).str.upper().map(lookup)
    lr_pairs["receptor"] = lr_pairs["receptor"].astype(str).str.upper().map(lookup)
    lr_pairs = lr_pairs.dropna(subset=["ligand", "receptor"]).copy()
    lr_pairs["expression_score"] = [
        min(gene_scores.get(str(ligand).upper(), 0.0), gene_scores.get(str(receptor).upper(), 0.0))
        for ligand, receptor in zip(lr_pairs["ligand"], lr_pairs["receptor"])
    ]
    lr_pairs = lr_pairs.sort_values(
        ["expression_score", "confidence", "lr_id"], ascending=[False, False, True]
    ).drop_duplicates(["ligand", "receptor"])
    if maximum_lr_pairs is not None and maximum_lr_pairs > 0:
        lr_pairs = lr_pairs.head(maximum_lr_pairs)
    lr_pairs = lr_pairs.drop(columns=["expression_score"]).reset_index(drop=True)
    lr_pairs["lr_id"] = lr_pairs["ligand"] + "__" + lr_pairs["receptor"]
    if lr_pairs.empty:
        raise ValueError("No complete ligand-receptor pair in ModelData overlaps this expression matrix")
    return tf_list, prior.reset_index(drop=True), lr_pairs


def select_expression_genes(
    matrix: sparse.csr_matrix,
    gene_names: np.ndarray,
    required_genes: Iterable[str],
    maximum_genes: int | None,
    minimum_detected_cells: int,
) -> list[str]:
    """Keep all required TF/LR genes and fill remaining slots by dispersion."""

    names = np.asarray(gene_names, dtype=str)
    rows_by_name: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        rows_by_name.setdefault(name, []).append(index)
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
    means = np.asarray(matrix.mean(axis=1)).ravel()
    second = np.asarray(matrix.power(2).mean(axis=1)).ravel()
    dispersion = np.maximum(second - means**2, 0.0) / np.maximum(means, 1e-8)
    score_by_name = {
        name: max(float(dispersion[index]) for index in indices)
        for name, indices in rows_by_name.items()
        if max(int(detected[index]) for index in indices) >= minimum_detected_cells
    }
    selected: list[str] = []
    for name in required_genes:
        if name in rows_by_name and name not in selected:
            selected.append(name)
    limit = maximum_genes if maximum_genes is not None and maximum_genes > 0 else len(rows_by_name)
    for name in sorted(score_by_name, key=lambda value: (-score_by_name[value], value)):
        if name not in selected:
            selected.append(name)
        if len(selected) >= limit:
            break
    if len(selected) < 2:
        raise ValueError("Too few genes remain after expression/reference filtering")
    return selected[: max(limit, len(set(required_genes)))]


def dense_selected_expression(
    matrix: sparse.csr_matrix,
    gene_names: np.ndarray,
    selected_genes: list[str],
) -> np.ndarray:
    rows_by_name: dict[str, list[int]] = {}
    for index, name in enumerate(np.asarray(gene_names, dtype=str)):
        rows_by_name.setdefault(name, []).append(index)
    output = np.zeros((matrix.shape[1], len(selected_genes)), dtype=np.float64)
    for column, gene in enumerate(selected_genes):
        output[:, column] = np.asarray(matrix[rows_by_name[gene]].sum(axis=0)).ravel()
    return output


def required_lr_genes(lr_pairs: pd.DataFrame) -> list[str]:
    values = lr_pairs["ligand"].astype(str).tolist() + lr_pairs["receptor"].astype(str).tolist()
    return list(dict.fromkeys(values))


def filter_lr_tf_prior(
    reference_dir: Path,
    tf_list: pd.DataFrame,
    lr_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict an optional global LR-to-TF prior to one processed dataset."""

    path = Path(reference_dir) / "prior_lr_tf.csv"
    if not path.exists():
        return pd.DataFrame(columns=["lr_id", "tf", "score"])
    prior = pd.read_csv(path)
    if prior.empty:
        return pd.DataFrame(columns=["lr_id", "tf", "score"])
    required = {"lr_id", "tf", "score"}
    missing = required.difference(prior.columns)
    if missing:
        raise ValueError(f"Global LR-to-TF prior is missing columns: {sorted(missing)}")
    valid_lr = set(lr_pairs["lr_id"].astype(str))
    valid_tf = set(tf_list["tf"].astype(str))
    return prior.loc[
        prior["lr_id"].astype(str).isin(valid_lr)
        & prior["tf"].astype(str).isin(valid_tf)
    ].drop_duplicates(["lr_id", "tf"]).reset_index(drop=True)


def write_model_inputs(
    output_dir: Path,
    expression: pd.DataFrame,
    coordinates: pd.DataFrame,
    metadata: pd.DataFrame,
    tf_list: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    prior: pd.DataFrame,
    configuration: object,
    source_files: dict[str, Path],
    notes: list[str],
    lr_tf_prior: pd.DataFrame | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expression.to_csv(output / "expression.csv")
    coordinates.to_csv(output / "coordinates.csv")
    metadata.to_csv(output / "metadata.csv", index=False)
    tf_list.to_csv(output / "tf_list.csv", index=False)
    lr_pairs.to_csv(output / "lr_pairs.csv", index=False)
    prior.to_csv(output / "prior_tf_gene.csv", index=False)
    if lr_tf_prior is None:
        lr_tf_prior = pd.DataFrame(columns=["lr_id", "tf", "score"])
    lr_tf_prior.to_csv(output / "prior_lr_tf.csv", index=False)
    summary = pd.DataFrame(
        {
            "item": ["cells_or_spots", "genes", "expressed_tfs", "lr_pairs", "tf_gene_prior_edges"],
            "value": [len(expression), expression.shape[1], len(tf_list), len(lr_pairs), len(prior)],
        }
    )
    summary.to_csv(output / "data_summary.csv", index=False)
    write_json(
        output / "source_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "offline_preprocess",
            "configuration": asdict(configuration),
            "source_files": {
                name: {
                    "path": str(Path(path).resolve()),
                    "bytes": Path(path).stat().st_size,
                    "sha256": file_digest(Path(path)),
                }
                for name, path in source_files.items()
            },
            "notes": notes,
        },
    )
    print(summary.to_string(index=False))
    print(f"Model input directory: {output}")
    return output
