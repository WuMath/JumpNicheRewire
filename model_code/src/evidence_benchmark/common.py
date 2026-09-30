"""Shared, auditable data-contract and registry helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DATA_ROOT


MODEL_FILES = (
    "expression.csv", "coordinates.csv", "metadata.csv", "tf_list.csv",
    "lr_pairs.csv", "prior_tf_gene.csv", "prior_lr_tf.csv",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz") or suffixes.endswith(".txt"):
        return pd.read_csv(path, sep="\t")
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".xlsx") or suffixes.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table: {path}")


def first_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Cannot identify {label}; available columns: {frame.columns.tolist()}")


def split_sources(value: object) -> int:
    values = {part.strip() for part in str(value).replace("|", ";").split(";") if part.strip()}
    return len(values)


def copy_or_link(source: Path, destination: Path) -> None:
    """Create a hard link for large immutable inputs, falling back to a copy."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_empty_priors(root: Path) -> None:
    pd.DataFrame(columns=["tf", "target", "motif", "atac", "grn"]).to_csv(
        root / "prior_tf_gene.csv", index=False
    )
    pd.DataFrame(columns=["lr_id", "tf", "score"]).to_csv(root / "prior_lr_tf.csv", index=False)


def write_baseline_input(root: Path) -> None:
    output = root / "baseline_input"
    output.mkdir(parents=True, exist_ok=True)
    expression = pd.read_csv(root / "expression.csv", index_col=0)
    coordinates = pd.read_csv(root / "coordinates.csv", index_col=0).loc[expression.index]
    metadata = pd.read_csv(root / "metadata.csv")
    np.save(output / "counts.npy", expression.to_numpy(dtype=np.float32))
    np.save(output / "coordinates.npy", coordinates.to_numpy(dtype=np.float32))
    pd.DataFrame({"gene": expression.columns.astype(str)}).to_csv(output / "genes.csv", index=False)
    pd.DataFrame({"cell_id": expression.index.astype(str)}).to_csv(output / "cells.csv", index=False)
    expression.to_csv(output / "expression_cells_by_genes.csv")
    expression.T.to_csv(output / "expression_genes_by_cells.csv")
    coordinates.to_csv(output / "coordinates.csv")
    metadata.to_csv(output / "metadata.csv", index=False)


def validate_model_contract(root: Path) -> dict[str, object]:
    missing = [name for name in MODEL_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{root} misses model input files: {missing}")
    expression = pd.read_csv(root / "expression.csv", index_col=0)
    coordinates = pd.read_csv(root / "coordinates.csv", index_col=0)
    metadata = pd.read_csv(root / "metadata.csv")
    if expression.empty or coordinates.empty:
        raise ValueError(f"Empty expression/coordinates in {root}")
    if not expression.index.equals(coordinates.index):
        raise ValueError(f"Expression and coordinate IDs/order differ in {root}")
    if "cell_id" not in metadata.columns or metadata["cell_id"].astype(str).tolist() != expression.index.astype(str).tolist():
        raise ValueError(f"metadata.cell_id must equal expression row order in {root}")
    if not np.isfinite(expression.to_numpy(dtype=float)).all() or (expression.to_numpy(dtype=float) < 0).any():
        raise ValueError(f"Expression must be finite and non-negative in {root}")
    return {"cells": len(expression), "genes": expression.shape[1]}


def finalize_dataset(
    root: Path,
    *,
    dataset_id: str,
    evidence_track: str,
    source: str,
    evaluation_scope: str,
    truth_kind: str,
    species: str,
    tissue: str,
    truth_files: Iterable[str],
    notes: str,
) -> Path:
    dimensions = validate_model_contract(root)
    truth_dir = root / "truth"
    missing_truth = [name for name in truth_files if not (truth_dir / name).exists()]
    if missing_truth:
        raise FileNotFoundError(f"Missing truth files in {truth_dir}: {missing_truth}")
    signatures = {name: sha256_file(root / name) for name in MODEL_FILES}
    for name in truth_files:
        signatures[f"truth/{name}"] = sha256_file(truth_dir / name)
    manifest = {
        "dataset_id": dataset_id,
        "evidence_track": evidence_track,
        "source": source,
        "evaluation_scope": evaluation_scope,
        "truth_kind": truth_kind,
        "species": species,
        "tissue": tissue,
        "dataset_dir": str(root.resolve()),
        "truth_dir": str(truth_dir.resolve()),
        **dimensions,
        "notes": notes,
        "files": signatures,
    }
    manifest["signature"] = stable_hash(manifest)
    path = root / "evidence_manifest.json"
    write_json(path, manifest)
    return path


def rebuild_registry(data_root: Path = DATA_ROOT) -> Path:
    rows: list[dict[str, object]] = []
    existing = data_root / '03_registries/evidence_registry.csv'
    if existing.exists():
        previous = pd.read_csv(existing)
        if not previous.empty:
            # Retain explicitly reused, external read-only datasets. Newly
            # prepared local manifests override the same ID below.
            rows.extend(row for row in previous.to_dict('records')
                        if not Path(str(row['dataset_dir'])).resolve().is_relative_to(data_root.resolve()))
    new_ids: set[str] = set()
    for path in sorted((data_root / "02_prepared").rglob("evidence_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {key: payload[key] for key in (
            "dataset_id", "evidence_track", "source", "evaluation_scope", "truth_kind",
            "species", "tissue", "cells", "genes", "dataset_dir", "truth_dir", "signature",
        )}
        # Manifests may have been created on another computer.  The manifest's
        # parent is authoritative; embedded absolute paths are only historical
        # provenance and are not portable across Windows/Linux.
        row["dataset_dir"] = str(path.parent.resolve())
        row["truth_dir"] = str((path.parent / "truth").resolve())
        if row['dataset_id'] in new_ids:
            raise ValueError(f"Duplicate local dataset ID: {row['dataset_id']}")
        new_ids.add(row['dataset_id'])
        rows = [old for old in rows if old['dataset_id'] != row['dataset_id']]
        rows.append(row)
    registry = pd.DataFrame(rows)
    if not registry.empty and registry["dataset_id"].duplicated().any():
        duplicates = registry.loc[registry["dataset_id"].duplicated(False), "dataset_id"].tolist()
        raise ValueError(f"Duplicate dataset IDs: {duplicates}")
    output = data_root / "03_registries" / "evidence_registry.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output, index=False)
    return output


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    work = values[finite]
    order = np.argsort(work)
    ranked = work[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result[finite] = restored
    return result
