"""Convert an existing local H5AD into JumpNicheRewire model inputs.

This module never accesses the network. Edit ``LocalH5ADConfig`` and run it
without command-line arguments. Use ``inspect_only=True`` first when the H5AD
layout is unknown.
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
from data_acquisition.local_common import build_reference_tables, write_project_inputs


from configs.real_data.settings.prepare_local_h5ad import LocalH5ADConfig, CONFIG





def _matrix_summary(values) -> dict[str, object]:
    if sparse.issparse(values):
        data = values.data
        return {
            "type": type(values).__name__,
            "shape": tuple(values.shape),
            "nonzero": int(values.nnz),
            "minimum_nonzero": float(data.min()) if len(data) else 0.0,
            "maximum": float(data.max()) if len(data) else 0.0,
        }
    shape = tuple(values.shape)
    sample_rows = min(shape[0], 200)
    sample_columns = min(shape[1], 500)
    sample = np.asarray(values[:sample_rows, :sample_columns], dtype=float)
    return {
        "type": type(values).__name__,
        "shape": shape,
        "sample_minimum": float(np.nanmin(sample)) if sample.size else np.nan,
        "sample_maximum": float(np.nanmax(sample)) if sample.size else np.nan,
    }


def inspect_h5ad(adata) -> None:
    print("H5AD shape:", adata.shape)
    print("obs columns:", list(adata.obs.columns))
    print("var columns:", list(adata.var.columns))
    print("layers:", list(adata.layers.keys()))
    print("obsm:", {key: tuple(value.shape) for key, value in adata.obsm.items()})
    print("raw available:", adata.raw is not None)
    print("X summary:", _matrix_summary(adata.X))
    for name in adata.layers.keys():
        print(f"layer[{name}] summary:", _matrix_summary(adata.layers[name]))


def _expression_source(adata, config: LocalH5ADConfig):
    if config.expression_source == "X":
        return adata.X, adata.var.copy(), np.asarray(adata.var_names.astype(str))
    if config.expression_source == "layer":
        if config.layer_name not in adata.layers:
            raise KeyError(f"H5AD layer is absent: {config.layer_name}; available={list(adata.layers.keys())}")
        return adata.layers[config.layer_name], adata.var.copy(), np.asarray(adata.var_names.astype(str))
    if config.expression_source == "raw":
        if adata.raw is None:
            raise ValueError("expression_source='raw' but adata.raw is absent")
        return adata.raw.X, adata.raw.var.copy(), np.asarray(adata.raw.var_names.astype(str))
    raise ValueError("expression_source must be X, layer, or raw")


def _to_csr(values) -> sparse.csr_matrix:
    if hasattr(values, "to_memory"):
        values = values.to_memory()
    if sparse.issparse(values):
        return values.tocsr().astype(np.float64)
    return sparse.csr_matrix(np.asarray(values, dtype=np.float64))


def _obs_value(obs: pd.DataFrame, column: str | None, default: object) -> pd.Series:
    if column is not None and column in obs.columns:
        return obs[column].copy()
    return pd.Series(default, index=obs.index)


def main(config: LocalH5ADConfig) -> Path:
    try:
        import anndata as ad
    except ImportError as error:
        raise ImportError("H5AD processing requires anndata; run: python -m pip install anndata>=0.10") from error

    h5ad_path = Path(config.h5ad_file)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file is missing: {h5ad_path}")
    
    print("[1/5] Reading local H5AD (offline)")
    adata = ad.read_h5ad(h5ad_path)
    inspect_h5ad(adata)
    if config.inspect_only:
        print("Inspection complete; set inspect_only=False after choosing expression and spatial keys.")
        return h5ad_path

    if adata.obs_names.duplicated().any():
        raise ValueError("adata.obs_names contains duplicated cell/spot IDs")
    if config.spatial_key not in adata.obsm:
        raise KeyError(f"obsm spatial key is absent: {config.spatial_key}; available={list(adata.obsm.keys())}")
    spatial_all = np.asarray(adata.obsm[config.spatial_key], dtype=float)
    if not config.spatial_dimensions or max(config.spatial_dimensions) >= spatial_all.shape[1]:
        raise ValueError("spatial_dimensions exceeds the selected obsm matrix width")
    spatial_all = spatial_all[:, list(config.spatial_dimensions)]
    if spatial_all.shape[1] not in {2, 3}:
        raise ValueError("Exactly two or three spatial dimensions must be selected")
    finite_coordinates = np.isfinite(spatial_all).all(axis=1)
    valid_indices = np.flatnonzero(finite_coordinates)
    if not len(valid_indices):
        raise ValueError("No H5AD observations have finite spatial coordinates")
    sampled_within_valid = spatial_farthest_sample(
        spatial_all[valid_indices], config.maximum_cells, config.random_seed
    )
    selected_obs = valid_indices[sampled_within_valid]

    print("[2/5] Selecting H5AD expression source")
    values, variable_metadata, default_names = _expression_source(adata, config)
    if config.gene_symbol_column is None:
        gene_names = default_names
    else:
        if config.gene_symbol_column not in variable_metadata.columns:
            raise KeyError(
                f"gene_symbol_column is absent: {config.gene_symbol_column}; "
                f"available={list(variable_metadata.columns)}"
            )
        symbols = variable_metadata[config.gene_symbol_column]
        usable = symbols.notna() & symbols.astype(str).str.strip().ne("")
        gene_names = symbols.where(usable, pd.Series(default_names, index=symbols.index)).astype(str).str.strip().to_numpy()
    if len(gene_names) != values.shape[1]:
        raise ValueError("Gene-symbol vector and expression columns differ in length")
    matrix_cells_genes = _to_csr(values[selected_obs, :])
    if matrix_cells_genes.data.size:
        if not np.isfinite(matrix_cells_genes.data).all():
            raise ValueError("Selected H5AD expression contains NaN or Inf")
        if (matrix_cells_genes.data < 0).any():
            raise ValueError(
                "Selected H5AD expression contains negative values. Do not use scaled/z-score X; "
                "choose a non-negative counts/log-normalized layer or raw matrix."
            )
    matrix = matrix_cells_genes.T.tocsr()

    print("[3/5] Filtering TF/LR references against H5AD genes")
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
    maximum_genes = config.maximum_genes or len(gene_names)
    selected_genes = select_expression_genes(
        matrix,
        gene_names,
        required,
        maximum_genes,
        min(config.minimum_detected_cells, matrix.shape[1]),
    )
    prior = prior.loc[prior["target"].isin(selected_genes)].reset_index(drop=True)

    print("[4/5] Writing model input files")
    cell_ids = np.asarray(adata.obs_names.astype(str))[selected_obs]
    expression = pd.DataFrame(
        dense_selected_expression(matrix, gene_names, selected_genes),
        index=cell_ids,
        columns=selected_genes,
    )
    expression.index.name = "cell_id"
    coordinate_names = ["x", "y", "z"][: spatial_all.shape[1]]
    coordinates = pd.DataFrame(spatial_all[selected_obs], index=cell_ids, columns=coordinate_names)
    coordinates.index.name = "cell_id"

    obs = adata.obs.iloc[selected_obs].copy()
    metadata = pd.DataFrame(index=obs.index)
    metadata["sample_id"] = _obs_value(obs, config.sample_column_in_obs, config.default_sample_id)
    metadata["cell_type"] = _obs_value(obs, config.cell_type_column_in_obs, config.default_cell_type)
    metadata["region"] = _obs_value(obs, config.region_column_in_obs, config.default_region)
    metadata["condition"] = _obs_value(obs, config.condition_column_in_obs, config.default_condition)
    for column in config.additional_obs_columns:
        if column not in obs.columns:
            raise KeyError(f"Requested additional obs column is absent: {column}")
        if column not in metadata.columns:
            metadata[column] = obs[column]
    metadata.index = cell_ids
    metadata.index.name = "cell_id"
    metadata = metadata.reset_index()

    sources = {"h5ad_file": h5ad_path}
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
            f"Expression source: {config.expression_source}; layer={config.layer_name}.",
            f"Set DataConfig.expression_is_normalized={config.expression_is_normalized} for the model run.",
            "H5AD X is not assumed to be raw counts; inspect layers/raw before choosing the source.",
        ],
    )
    print("[5/5] Complete")
    return destination



