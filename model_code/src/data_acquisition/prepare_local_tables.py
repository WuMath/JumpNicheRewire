"""Convert generic local expression, coordinate, metadata, TF, and LR tables.

Use this for GEO supplementary files, author-provided matrices, or laboratory
data already converted to flat tables. This script never accesses the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from grn2_project.paths import PROCESSED_DATA_ROOT, RAW_DATA_ROOT

from data_acquisition.common import (
    dense_selected_expression,
    select_expression_genes,
    spatial_farthest_sample,
)
from data_acquisition.local_common import prepared_reference_tables, read_local_table, write_project_inputs


from configs.real_data.settings.prepare_local_tables import LocalTableConfig, CONFIG





def _indexed(frame: pd.DataFrame, id_column: str | None, label: str) -> pd.DataFrame:
    column = id_column or str(frame.columns[0])
    if column not in frame.columns:
        raise ValueError(f"{label} ID column is absent: {column}")
    frame = frame.copy()
    frame[column] = frame[column].astype(str)
    if frame[column].duplicated().any():
        raise ValueError(f"{label} contains duplicated IDs")
    return frame.set_index(column)


def main(config: LocalTableConfig) -> Path:
    print("[1/4] Reading manually downloaded tables (offline)")
    raw_expression = read_local_table(Path(config.expression_file))
    expression = _indexed(raw_expression, config.expression_id_column, "expression")
    if config.expression_orientation == "genes_by_cells":
        expression = expression.T
    elif config.expression_orientation != "cells_by_genes":
        raise ValueError("expression_orientation must be cells_by_genes or genes_by_cells")
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    expression = expression.apply(pd.to_numeric, errors="raise").astype(float)
    if expression.columns.duplicated().any():
        expression = expression.T.groupby(level=0, sort=False).sum().T
    if expression.index.duplicated().any():
        raise ValueError("Expression cell IDs are duplicated after orientation conversion")
    if not np.isfinite(expression.to_numpy()).all() or (expression.to_numpy() < 0).any():
        raise ValueError("Expression must contain finite, non-negative values")

    raw_coordinates = read_local_table(Path(config.coordinates_file))
    coordinates = _indexed(raw_coordinates, config.coordinate_id_column, "coordinates")
    missing_coordinate_columns = set(config.coordinate_columns).difference(coordinates.columns)
    if missing_coordinate_columns:
        raise ValueError(f"Coordinate columns are missing: {sorted(missing_coordinate_columns)}")
    coordinates = coordinates.loc[:, list(config.coordinate_columns)].apply(pd.to_numeric, errors="raise")
    if len(config.coordinate_columns) not in {2, 3}:
        raise ValueError("Exactly two or three coordinate columns are required")
    missing_cells = expression.index.difference(coordinates.index)
    if len(missing_cells):
        raise ValueError(f"Coordinates are missing {len(missing_cells)} expression IDs")
    coordinates = coordinates.loc[expression.index]

    selected_cells = spatial_farthest_sample(
        coordinates.to_numpy(dtype=float), config.maximum_cells, config.random_seed
    )
    expression = expression.iloc[selected_cells]
    coordinates = coordinates.iloc[selected_cells]
    matrix = sparse.csr_matrix(expression.to_numpy(dtype=float).T)
    gene_names = expression.columns.astype(str).to_numpy()

    print("[2/4] Filtering manually supplied TF/LR tables")
    tf_list, lr_pairs, prior = prepared_reference_tables(
        gene_names,
        Path(config.tf_file),
        Path(config.lr_file),
        Path(config.prior_file) if config.prior_file is not None else None,
        config.maximum_tfs,
        config.maximum_lr_pairs,
    )
    required = tf_list["tf"].astype(str).tolist()
    for column in ("ligand", "receptor"):
        required.extend(
            lr_pairs[column].astype(str).str.replace("&", "|", regex=False).str.split("|", regex=False).explode().tolist()
        )
    maximum_genes = config.maximum_genes or len(gene_names)
    selected_genes = select_expression_genes(
        matrix,
        gene_names,
        required,
        maximum_genes,
        min(config.minimum_detected_cells, matrix.shape[1]),
    )
    prior = prior.loc[prior["target"].isin(selected_genes)].reset_index(drop=True)
    reduced_expression = pd.DataFrame(
        dense_selected_expression(matrix, gene_names, selected_genes),
        index=expression.index,
        columns=selected_genes,
    )
    reduced_expression.index.name = "cell_id"
    coordinates.index.name = "cell_id"
    coordinates.columns = ["x", "y", "z"][: len(config.coordinate_columns)]

    if config.metadata_file is not None and Path(config.metadata_file).exists():
        metadata = _indexed(
            read_local_table(Path(config.metadata_file)), config.metadata_id_column, "metadata"
        )
        missing_metadata = reduced_expression.index.difference(metadata.index)
        if len(missing_metadata):
            raise ValueError(f"Metadata is missing {len(missing_metadata)} selected IDs")
        metadata = metadata.loc[reduced_expression.index].copy()
    else:
        metadata = pd.DataFrame(index=reduced_expression.index)
    defaults = {
        "sample_id": config.default_sample_id,
        "cell_type": config.default_cell_type,
        "region": config.default_region,
        "condition": config.default_condition,
    }
    for column, value in defaults.items():
        if column not in metadata.columns:
            metadata[column] = value
    metadata.index.name = "cell_id"
    metadata = metadata.reset_index()

    print("[3/4] Writing model input files")
    sources = {
        "expression_file": Path(config.expression_file),
        "coordinates_file": Path(config.coordinates_file),
        "tf_file": Path(config.tf_file),
        "lr_file": Path(config.lr_file),
    }
    if config.metadata_file is not None:
        sources["metadata_file"] = Path(config.metadata_file)
    if config.prior_file is not None:
        sources["prior_file"] = Path(config.prior_file)
    destination = write_project_inputs(
        config.output_dir,
        reduced_expression,
        coordinates,
        metadata,
        tf_list,
        lr_pairs,
        prior,
        config,
        sources,
        [
            "No network request is made by this script.",
            "The source expression table is expected to be raw counts or non-negative normalized expression.",
            "Default metadata labels are placeholders and do not replace verified biological annotation.",
        ],
    )
    print("[4/4] Complete")
    return destination



