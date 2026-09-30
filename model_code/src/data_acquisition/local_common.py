"""Shared helpers for strictly local, offline data preparation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_acquisition.common import file_digest, match_network_resources, read_network, write_manifest


def read_local_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manually downloaded input is missing: {path}")
    lower = path.name.lower()
    if lower.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, low_memory=False)
    if lower.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported local table format: {path}")


def _map_definition(value: object, lookup: dict[str, str]) -> str | None:
    alternatives: list[str] = []
    for alternative in str(value).split("|"):
        components: list[str] = []
        for component in alternative.split("&"):
            matched = lookup.get(component.strip().upper())
            if matched is None:
                return None
            components.append(matched)
        alternatives.append("&".join(components))
    return "|".join(alternatives)


def prepared_reference_tables(
    gene_names: np.ndarray,
    tf_file: Path,
    lr_file: Path,
    prior_file: Path | None,
    maximum_tfs: int,
    maximum_lr_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter already prepared TF/LR tables against one expression panel."""

    lookup: dict[str, str] = {}
    for name in np.asarray(gene_names, dtype=str):
        lookup.setdefault(name.upper(), name)

    tf_raw = read_local_table(tf_file)
    tf_column = "tf" if "tf" in tf_raw.columns else tf_raw.columns[0]
    tf_names = [lookup.get(str(value).upper()) for value in tf_raw[tf_column].dropna()]
    tf_names = list(dict.fromkeys(value for value in tf_names if value is not None))[:maximum_tfs]
    if not tf_names:
        raise ValueError("No manually supplied TF names overlap the expression genes")
    tf_list = pd.DataFrame({"tf": tf_names})

    lr_pairs = read_local_table(lr_file).copy()
    missing = {"lr_id", "ligand", "receptor"}.difference(lr_pairs.columns)
    if missing:
        raise ValueError(f"Prepared LR table is missing columns: {sorted(missing)}")
    lr_pairs["ligand"] = lr_pairs["ligand"].map(lambda value: _map_definition(value, lookup))
    lr_pairs["receptor"] = lr_pairs["receptor"].map(lambda value: _map_definition(value, lookup))
    lr_pairs = lr_pairs.dropna(subset=["ligand", "receptor"]).drop_duplicates("lr_id").head(maximum_lr_pairs)
    if lr_pairs.empty:
        raise ValueError("No complete ligand-receptor definitions overlap the expression genes")
    if "signal_type" not in lr_pairs.columns:
        lr_pairs["signal_type"] = "unknown"
    if "rho" not in lr_pairs.columns:
        lr_pairs["rho"] = 1.0
    if "confidence" not in lr_pairs.columns:
        lr_pairs["confidence"] = np.nan
    lr_pairs = lr_pairs[["lr_id", "ligand", "receptor", "signal_type", "rho", "confidence"]]

    prior_columns = ["tf", "target", "motif", "atac", "grn"]
    if prior_file is None or not Path(prior_file).exists():
        prior = pd.DataFrame(columns=prior_columns)
    else:
        prior = read_local_table(prior_file).copy()
        if not {"tf", "target"}.issubset(prior.columns):
            raise ValueError("TF-gene prior requires tf and target columns")
        prior["tf"] = prior["tf"].astype(str).str.upper().map(lookup)
        prior["target"] = prior["target"].astype(str).str.upper().map(lookup)
        prior = prior.dropna(subset=["tf", "target"])
        prior = prior.loc[prior["tf"].isin(tf_names)].copy()
        for column in ("motif", "atac", "grn"):
            if column not in prior.columns:
                prior[column] = np.nan
        prior = prior[prior_columns].drop_duplicates(["tf", "target"])
    return tf_list, lr_pairs, prior


def build_reference_tables(
    mode: str,
    gene_names: np.ndarray,
    expression,
    maximum_tfs: int,
    maximum_lr_pairs: int,
    collectri_file: Path,
    ligand_receptor_network_file: Path,
    prepared_tf_file: Path,
    prepared_lr_file: Path,
    prepared_prior_file: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mode == "omnipath_raw":
        return match_network_resources(
            gene_names,
            expression,
            read_network(Path(collectri_file)),
            read_network(Path(ligand_receptor_network_file)),
            maximum_tfs,
            maximum_lr_pairs,
        )
    if mode == "prepared_tables":
        return prepared_reference_tables(
            gene_names,
            Path(prepared_tf_file),
            Path(prepared_lr_file),
            Path(prepared_prior_file) if prepared_prior_file is not None else None,
            maximum_tfs,
            maximum_lr_pairs,
        )
    raise ValueError("reference_mode must be omnipath_raw or prepared_tables")


def local_file_record(path: Path) -> dict[str, object]:
    path = Path(path)
    if path.is_file():
        return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_digest(path)}
    if path.is_dir():
        return {"path": str(path.resolve()), "type": "directory"}
    return {"path": str(path), "missing": True}


def write_project_inputs(
    output_dir: Path,
    expression: pd.DataFrame,
    coordinates: pd.DataFrame,
    metadata: pd.DataFrame,
    tf_list: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    prior: pd.DataFrame,
    configuration: Any,
    source_files: dict[str, Path],
    notes: list[str],
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expression.to_csv(output / "expression.csv")
    coordinates.to_csv(output / "coordinates.csv")
    metadata.to_csv(output / "metadata.csv", index=False)
    tf_list.to_csv(output / "tf_list.csv", index=False)
    lr_pairs.to_csv(output / "lr_pairs.csv", index=False)
    prior.to_csv(output / "prior_tf_gene.csv", index=False)
    pd.DataFrame(columns=["lr_id", "tf", "score"]).to_csv(output / "prior_lr_tf.csv", index=False)
    summary = pd.DataFrame(
        {
            "item": ["cells_or_spots", "genes", "expressed_tfs", "lr_pairs", "tf_gene_prior_edges"],
            "value": [len(expression), expression.shape[1], len(tf_list), len(lr_pairs), len(prior)],
        }
    )
    summary.to_csv(output / "data_summary.csv", index=False)
    write_manifest(
        output / "source_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "manual_download_local_processing",
            "configuration": asdict(configuration),
            "local_sources": {name: local_file_record(path) for name, path in source_files.items()},
            "notes": notes,
        },
    )
    print(summary.to_string(index=False))
    print(f"Project-ready data: {output}")
    return output

