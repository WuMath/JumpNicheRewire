"""Input parser and validator for the project data contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig
from .types import InputBundle


def _read_table(path: Path, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file is missing: {path}")
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def _read_indexed_matrix(path: Path, required: bool) -> pd.DataFrame | None:
    frame = _read_table(path, required=required)
    if frame.empty and not required:
        return None
    if frame.shape[1] < 2:
        raise ValueError(f"Matrix requires an ID column plus values: {path}")
    id_column = frame.columns[0]
    frame[id_column] = frame[id_column].astype(str)
    if frame[id_column].duplicated().any():
        duplicated = frame.loc[frame[id_column].duplicated(), id_column].iloc[0]
        raise ValueError(f"Duplicated row ID '{duplicated}' in {path}")
    frame = frame.set_index(id_column)
    frame.index.name = "cell_id"
    frame.columns = frame.columns.astype(str)
    return frame


def _align_optional_matrix(frame: pd.DataFrame | None, cells: pd.Index, name: str) -> pd.DataFrame | None:
    if frame is None:
        return None
    missing = cells.difference(frame.index)
    if len(missing):
        raise ValueError(f"{name} is missing {len(missing)} expression cell IDs")
    return frame.loc[cells]


def _audit_row(item: str, available: bool, detail: str, action: str) -> dict[str, object]:
    return {"item": item, "available": bool(available), "detail": detail, "action": action}


def load_inputs(config: DataConfig) -> InputBundle:
    root = Path(config.input_dir)
    expression = _read_indexed_matrix(root / config.expression_file, required=True)
    # Never open fake/optional coordinates for a non-spatial experiment.
    coordinates = (
        _read_indexed_matrix(root / config.coordinates_file, required=True)
        if config.is_spatial else pd.DataFrame(0., index=expression.index, columns=["x", "y"])
    )
    assert expression is not None and coordinates is not None

    expression = expression.apply(pd.to_numeric, errors="raise").astype(float)
    coordinates = coordinates.apply(pd.to_numeric, errors="raise").astype(float)
    if expression.empty or coordinates.empty:
        raise ValueError("Expression and coordinate matrices must be non-empty")
    if expression.columns.duplicated().any():
        raise ValueError("Expression gene names must be unique")
    if (expression.to_numpy() < 0).any():
        raise ValueError("Expression values must be non-negative")
    if coordinates.shape[1] not in {2, 3}:
        raise ValueError("Coordinates must have exactly 2 or 3 numeric columns")

    missing_coordinates = expression.index.difference(coordinates.index)
    if len(missing_coordinates):
        raise ValueError(f"Coordinates are missing {len(missing_coordinates)} expression cell IDs")
    coordinates = coordinates.loc[expression.index]
    if not np.isfinite(expression.to_numpy()).all() or not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError("Expression and coordinate matrices must contain only finite values")

    metadata_path = root / config.metadata_file
    metadata_raw = _read_indexed_matrix(metadata_path, required=False)
    if metadata_raw is None:
        metadata = pd.DataFrame(index=expression.index)
    else:
        metadata = _align_optional_matrix(metadata_raw, expression.index, "metadata")
        assert metadata is not None
    if config.sample_column not in metadata.columns:
        metadata[config.sample_column] = "sample_1"
    metadata[config.sample_column] = (
        metadata[config.sample_column]
        .astype("object")
        .where(metadata[config.sample_column].notna(), "sample_1")
        .astype(str)
    )
    if config.cell_type_column not in metadata.columns:
        metadata[config.cell_type_column] = "NA"
    metadata[config.cell_type_column] = (
        metadata[config.cell_type_column]
        .astype("object")
        .where(metadata[config.cell_type_column].notna(), "NA")
        .astype(str)
    )
    if config.region_column not in metadata.columns:
        cell_types = metadata[config.cell_type_column]
        metadata[config.region_column] = np.where(cell_types.eq("NA"), "global", cell_types)
    metadata[config.region_column] = (
        metadata[config.region_column]
        .astype("object")
        .where(metadata[config.region_column].notna(), "global")
        .astype(str)
    )
    if config.condition_column not in metadata.columns:
        metadata[config.condition_column] = "unspecified"
    metadata[config.condition_column] = (
        metadata[config.condition_column]
        .astype("object")
        .where(metadata[config.condition_column].notna(), "unspecified")
        .astype(str)
    )

    tf_table = _read_table(root / config.tf_file, required=True)
    tf_column = "tf" if "tf" in tf_table.columns else tf_table.columns[0]
    tf_names = [str(value) for value in tf_table[tf_column].dropna().astype(str).unique()]
    tf_names = [name for name in tf_names if name in expression.columns]
    if not tf_names:
        raise ValueError("No TF names overlap the expression matrix")

    lr_pairs = (_read_table(root / config.lr_file, required=True) if config.is_spatial
                else pd.DataFrame(columns=["lr_id", "ligand", "receptor"]))
    required_lr = {"lr_id", "ligand", "receptor"}
    missing_lr = required_lr.difference(lr_pairs.columns)
    if missing_lr:
        raise ValueError(f"LR table is missing columns: {sorted(missing_lr)}")
    lr_pairs = lr_pairs.copy()
    lr_pairs["lr_id"] = lr_pairs["lr_id"].astype(str)
    lr_pairs["ligand"] = lr_pairs["ligand"].astype(str)
    lr_pairs["receptor"] = lr_pairs["receptor"].astype(str)
    if "signal_type" not in lr_pairs.columns:
        lr_pairs["signal_type"] = "unknown"
    if "rho" not in lr_pairs.columns:
        lr_pairs["rho"] = 1.0
    if "confidence" not in lr_pairs.columns:
        lr_pairs["confidence"] = np.nan
    if lr_pairs["lr_id"].duplicated().any():
        raise ValueError("lr_id values must be unique")

    tf_gene_prior = _read_table(root / config.tf_gene_prior_file, required=False)
    if tf_gene_prior.empty:
        tf_gene_prior = pd.DataFrame(columns=["tf", "target", "motif", "atac", "grn"])
    lr_tf_prior = _read_table(root / config.lr_tf_prior_file, required=False)
    if lr_tf_prior.empty:
        lr_tf_prior = pd.DataFrame(columns=["lr_id", "tf", "score"])

    ligand_activity = _align_optional_matrix(
        _read_indexed_matrix(root / config.ligand_activity_file, required=False), expression.index, "ligand_activity"
    )
    receptor_activity = _align_optional_matrix(
        _read_indexed_matrix(root / config.receptor_activity_file, required=False), expression.index, "receptor_activity"
    )
    tf_activity = _align_optional_matrix(
        _read_indexed_matrix(root / config.tf_activity_file, required=False), expression.index, "tf_activity"
    )
    perturbations = _read_table(root / config.perturbation_file, required=False)
    if perturbations.empty:
        perturbations = None

    audit = pd.DataFrame(
        [
            _audit_row("expression", True, f"{expression.shape[0]} cells x {expression.shape[1]} genes", "used"),
            _audit_row("coordinates", config.is_spatial, f"{coordinates.shape[1]}D" if config.is_spatial else "not applicable", "used" if config.is_spatial else "not read; internal zeros are shape placeholders only"),
            _audit_row("metadata", metadata_path.exists(), f"{metadata.shape[1]} columns", "defaulted where absent"),
            _audit_row("tf_catalog", True, f"{len(tf_names)} expressed TFs", "used"),
            _audit_row("lr_catalog", config.is_spatial, f"{len(lr_pairs)} pathways", "used" if config.is_spatial else "not read; non-spatial GRN-only"),
            _audit_row("tf_gene_prior", not tf_gene_prior.empty, f"{len(tf_gene_prior)} rows", "soft prior or omitted"),
            _audit_row("lr_tf_prior", not lr_tf_prior.empty, f"{len(lr_tf_prior)} rows", "soft prior or uniform penalty"),
            _audit_row("ligand_activity", ligand_activity is not None, "optional", "RNA proxy if absent"),
            _audit_row("receptor_activity", receptor_activity is not None, "optional", "RNA proxy if absent"),
            _audit_row("tf_activity", tf_activity is not None, "optional", "RNA proxy if absent"),
            _audit_row("perturbations", perturbations is not None, "optional", "no Level-3 evidence if absent"),
        ]
    )

    return InputBundle(
        expression=expression,
        coordinates=coordinates,
        metadata=metadata,
        tf_names=tf_names,
        lr_pairs=lr_pairs,
        tf_gene_prior=tf_gene_prior,
        lr_tf_prior=lr_tf_prior,
        ligand_activity=ligand_activity,
        receptor_activity=receptor_activity,
        tf_activity=tf_activity,
        perturbations=perturbations,
        audit=audit,
    )
