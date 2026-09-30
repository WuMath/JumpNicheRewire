"""Convert manually downloaded 10x Visium files into model inputs.

This module never accesses the network. Edit ``LocalVisiumConfig`` and run it
without command-line arguments.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grn2_project.paths import PROCESSED_DATA_ROOT, RAW_DATA_ROOT

from data_acquisition.common import (
    dense_selected_expression,
    find_file,
    read_10x_matrix,
    read_visium_positions,
    safe_extract_tar,
    select_expression_genes,
    spatial_farthest_sample,
)
from data_acquisition.local_common import build_reference_tables, write_project_inputs


from configs.real_data.settings.prepare_local_visium import LocalVisiumConfig, CONFIG





def _directory_from_source(source: Path, destination: Path, label: str) -> Path:
    source = Path(source)
    if source.is_dir():
        return source
    if not source.exists():
        raise FileNotFoundError(f"Manually downloaded {label} source is missing: {source}")
    if not tarfile.is_tarfile(source):
        raise ValueError(f"{label} source must be an extracted directory or tar/tar.gz archive: {source}")
    return safe_extract_tar(source, destination)


def main(config: LocalVisiumConfig) -> Path:
    print("[1/5] Reading manually downloaded files (offline)")
    matrix_root = _directory_from_source(
        config.matrix_source, Path(config.extraction_dir) / "matrix", "matrix"
    )
    spatial_source = Path(config.spatial_source)
    if spatial_source.is_file() and spatial_source.name in {"tissue_positions.csv", "tissue_positions_list.csv"}:
        position_file = spatial_source
    else:
        spatial_root = _directory_from_source(
            spatial_source, Path(config.extraction_dir) / "spatial", "spatial"
        )
        position_file = find_file(spatial_root, ("tissue_positions.csv", "tissue_positions_list.csv"))

    print("[2/5] Aligning expression barcodes and spatial positions")
    matrix, gene_names, barcodes = read_10x_matrix(matrix_root)
    positions = read_visium_positions(position_file, config.tissue_only)
    barcode_to_column = {barcode: index for index, barcode in enumerate(barcodes)}
    positions = positions.loc[positions["barcode"].isin(barcode_to_column)].reset_index(drop=True)
    if positions.empty:
        raise ValueError("No spatial barcodes overlap the expression matrix")
    selected_spots = spatial_farthest_sample(
        positions[["x", "y"]].to_numpy(dtype=float), config.maximum_spots, config.random_seed
    )
    positions = positions.iloc[selected_spots].reset_index(drop=True)
    matrix_columns = np.array([barcode_to_column[value] for value in positions["barcode"]], dtype=int)
    matrix = matrix[:, matrix_columns].tocsr()

    print("[3/5] Filtering local TF/LR references")
    tf_list, lr_pairs, prior = build_reference_tables(
        config.reference_mode,
        gene_names,
        matrix,
        config.maximum_tfs,
        config.maximum_lr_pairs,
        config.collectri_file,
        config.ligand_receptor_network_file,
        config.prepared_tf_file,
        config.prepared_lr_file,
        config.prepared_prior_file,
    )
    required = (
        tf_list["tf"].astype(str).tolist()
        + lr_pairs["ligand"].astype(str).str.replace("&", "|", regex=False).str.split("|", regex=False).explode().tolist()
        + lr_pairs["receptor"].astype(str).str.replace("&", "|", regex=False).str.split("|", regex=False).explode().tolist()
    )
    selected_genes = select_expression_genes(
        matrix,
        gene_names,
        required,
        config.maximum_genes,
        min(config.minimum_detected_spots, matrix.shape[1]),
    )
    prior = prior.loc[prior["target"].isin(selected_genes)].reset_index(drop=True)

    print("[4/5] Writing model input files")
    cell_ids = positions["barcode"].astype(str).to_numpy()
    expression = pd.DataFrame(
        dense_selected_expression(matrix, gene_names, selected_genes),
        index=cell_ids,
        columns=selected_genes,
    )
    expression.index.name = "cell_id"
    coordinates = positions.set_index("barcode")[["x", "y"]].copy()
    coordinates.index.name = "cell_id"
    metadata = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "sample_id": config.sample_id,
            "cell_type": config.cell_type_label,
            "region": config.region_label,
            "condition": config.condition_label,
            "array_row": positions["array_row"].to_numpy(),
            "array_col": positions["array_col"].to_numpy(),
            "source_dataset": config.dataset_name,
        }
    )
    sources = {"matrix_source": Path(config.matrix_source), "spatial_source": Path(config.spatial_source)}
    if config.reference_mode == "omnipath_raw":
        sources.update(
            {
                "collectri_file": Path(config.collectri_file),
                "ligand_receptor_network_file": Path(config.ligand_receptor_network_file),
            }
        )
    else:
        sources.update(
            {
                "prepared_tf_file": Path(config.prepared_tf_file),
                "prepared_lr_file": Path(config.prepared_lr_file),
                "prepared_prior_file": Path(config.prepared_prior_file),
            }
        )
    destination = write_project_inputs(
        config.output_dir,
        expression,
        coordinates,
        metadata,
        tf_list,
        lr_pairs,
        prior,
        config,
        sources,
        [
            "No network request is made by this script.",
            "Full-resolution pixel coordinates are used as one consistent distance unit.",
            "region=global is a placeholder and must not be described as an anatomical annotation.",
        ],
    )
    print("[5/5] Complete")
    return destination



