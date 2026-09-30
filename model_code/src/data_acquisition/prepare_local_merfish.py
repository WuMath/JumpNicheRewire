"""Convert a manually downloaded MERFISH CSV/CSV.GZ into model inputs.

This module never accesses the network. Edit ``LocalMERFISHConfig`` only.
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
from data_acquisition.local_common import build_reference_tables, read_local_table, write_project_inputs


from configs.real_data.settings.prepare_local_merfish import LocalMERFISHConfig, CONFIG





def main(config: LocalMERFISHConfig) -> Path:
    print("[1/4] Reading manually downloaded MERFISH table (offline)")
    raw = read_local_table(Path(config.expression_table))
    id_column = config.row_id_column or str(raw.columns[0])
    if id_column not in raw.columns:
        raise ValueError(f"row_id_column is absent: {id_column}")
    source_ids = raw[id_column].astype(str)
    raw = raw.drop(columns=[id_column])
    coordinate_names = ("STalign_x", "STalign_y") if config.use_aligned_coordinates else ("center_x", "center_y")
    missing = set(coordinate_names).difference(raw.columns)
    if missing:
        raise ValueError(f"MERFISH coordinate columns are missing: {sorted(missing)}")
    coordinates_all = raw.loc[:, list(coordinate_names)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(coordinates_all.to_numpy()).all(axis=1)
    raw = raw.loc[finite].reset_index(drop=True)
    source_ids = source_ids.loc[finite].reset_index(drop=True)
    coordinates_all = coordinates_all.loc[finite].reset_index(drop=True)
    gene_columns = [column for column in raw.columns if column not in config.non_gene_columns]
    expression_all = raw[gene_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (expression_all.to_numpy() < 0).any():
        raise ValueError("MERFISH expression must be non-negative")
    selected_cells = spatial_farthest_sample(
        coordinates_all.to_numpy(dtype=float), config.maximum_cells, config.random_seed
    )
    expression_all = expression_all.iloc[selected_cells].reset_index(drop=True)
    coordinates_all = coordinates_all.iloc[selected_cells].reset_index(drop=True)
    source_ids = source_ids.iloc[selected_cells].reset_index(drop=True)
    matrix = sparse.csr_matrix(expression_all.to_numpy(dtype=float).T)
    gene_names = expression_all.columns.astype(str).to_numpy()

    print("[2/4] Filtering local TF/LR references")
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
    required = tf_list["tf"].astype(str).tolist()
    for column in ("ligand", "receptor"):
        required.extend(
            lr_pairs[column].astype(str).str.replace("&", "|", regex=False).str.split("|", regex=False).explode().tolist()
        )
    selected_genes = select_expression_genes(
        matrix,
        gene_names,
        required,
        config.maximum_genes,
        min(config.minimum_detected_cells, matrix.shape[1]),
    )
    prior = prior.loc[prior["target"].isin(selected_genes)].reset_index(drop=True)

    print("[3/4] Writing model input files")
    cell_ids = np.array([f"merfish_cell_{index + 1:07d}" for index in range(len(expression_all))])
    expression = pd.DataFrame(
        dense_selected_expression(matrix, gene_names, selected_genes),
        index=cell_ids,
        columns=selected_genes,
    )
    expression.index.name = "cell_id"
    coordinates = pd.DataFrame(
        coordinates_all.to_numpy(dtype=float), index=cell_ids, columns=["x", "y"]
    )
    coordinates.index.name = "cell_id"
    metadata = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "source_cell_id": source_ids.astype(str).to_numpy(),
            "sample_id": config.sample_id,
            "cell_type": config.cell_type_label,
            "region": config.region_label,
            "condition": config.condition_label,
        }
    )
    sources = {"expression_table": Path(config.expression_table)}
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
            "A targeted MERFISH panel can contain far fewer TFs than whole-transcriptome data.",
            "region=global is a placeholder, not an anatomical label.",
        ],
    )
    print("[4/4] Complete")
    return destination



