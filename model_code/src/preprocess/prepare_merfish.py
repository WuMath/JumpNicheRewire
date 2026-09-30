"""Prepare the downloaded STalign MERFISH CSV.GZ for the model, offline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import sys
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


from preprocess.common import (
    PROJECT_ROOT,
    dense_selected_expression,
    filter_lr_tf_prior,
    filter_references_for_expression,
    required_lr_genes,
    select_expression_genes,
    spatial_farthest_sample,
    write_model_inputs,
)


from configs.real_data.settings.prepare_merfish import MERFISHConfig, CONFIG





def _read_selected_rows(
    path: Path,
    use_columns: list[str],
    id_column: str,
    source_rows: np.ndarray,
    chunk_rows: int,
) -> pd.DataFrame:
    selected_set = set(int(value) for value in source_rows)
    pieces: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(
        path,
        usecols=use_columns,
        dtype={id_column: "string"},
        chunksize=chunk_rows,
        low_memory=False,
    ):
        row_numbers = np.arange(offset, offset + len(chunk), dtype=int)
        keep = np.fromiter((int(value) in selected_set for value in row_numbers), dtype=bool)
        if keep.any():
            pieces.append(chunk.loc[keep].copy())
        offset += len(chunk)
    if not pieces:
        raise ValueError("No MERFISH rows were selected")
    return pd.concat(pieces, ignore_index=True)


def main(config: MERFISHConfig = CONFIG) -> Path:
    path = Path(config.expression_table)
    if not path.exists():
        raise FileNotFoundError(f"MERFISH CSV.GZ is missing: {path}")
    if not Path(config.reference_dir, "tf_list.csv").exists():
        raise FileNotFoundError(
            "Standardized references are missing. Run: python -m preprocess.prepare_references"
        )

    print("[1/5] Inspecting MERFISH columns and selecting spatially spread cells")
    columns = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    id_column = config.row_id_column or columns[0]
    coordinate_columns = (
        ["STalign_x", "STalign_y"]
        if config.use_aligned_coordinates
        else ["center_x", "center_y"]
    )
    required_columns = [id_column, "center_x", "center_y", "STalign_x", "STalign_y"]
    missing = set(required_columns).difference(columns)
    if missing:
        raise ValueError(f"MERFISH table is missing columns: {sorted(missing)}")
    coordinate_frame = pd.read_csv(
        path,
        usecols=required_columns,
        dtype={id_column: "string"},
        low_memory=False,
    )
    numeric_coordinates = coordinate_frame[coordinate_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(numeric_coordinates.to_numpy(dtype=float)).all(axis=1)
    valid_source_rows = np.flatnonzero(finite)
    if not len(valid_source_rows):
        raise ValueError("No MERFISH cell has finite selected coordinates")
    selected_within_valid = spatial_farthest_sample(
        numeric_coordinates.loc[finite].to_numpy(dtype=float),
        config.maximum_cells,
        config.random_seed,
    )
    selected_source_rows = valid_source_rows[selected_within_valid]

    non_gene = set(config.non_gene_columns) | {id_column}
    gene_columns = [
        column
        for column in columns
        if column not in non_gene
        and not any(column.startswith(prefix) for prefix in config.excluded_gene_prefixes)
    ]
    if not gene_columns:
        raise ValueError("No MERFISH gene-count columns remain")

    print("[2/5] Reading only selected MERFISH rows in chunks")
    selected = _read_selected_rows(
        path,
        list(dict.fromkeys(required_columns + gene_columns)),
        id_column,
        selected_source_rows,
        config.read_chunk_rows,
    )
    expression_values = selected[gene_columns].apply(pd.to_numeric, errors="coerce")
    if expression_values.isna().any().any():
        raise ValueError("Selected MERFISH expression contains non-numeric or missing values")
    expression_array = expression_values.to_numpy(dtype=float)
    if not np.isfinite(expression_array).all() or (expression_array < 0).any():
        raise ValueError("Selected MERFISH expression must be finite and non-negative")
    matrix = sparse.csr_matrix(expression_array.T)
    gene_names = np.asarray(gene_columns, dtype=str)

    print("[3/5] Matching ModelData TF/LR references to measured MERFISH genes")
    tf_list, prior, lr_pairs = filter_references_for_expression(
        config.reference_dir,
        gene_names,
        matrix,
        config.maximum_tfs,
        config.maximum_lr_pairs,
    )
    required = tf_list["tf"].astype(str).tolist() + required_lr_genes(lr_pairs)
    selected_genes = select_expression_genes(
        matrix,
        gene_names,
        required,
        config.maximum_genes,
        min(config.minimum_detected_cells, matrix.shape[1]),
    )
    prior = prior.loc[prior["target"].isin(selected_genes)].reset_index(drop=True)
    lr_tf_prior = filter_lr_tf_prior(config.reference_dir, tf_list, lr_pairs)

    print("[4/5] Writing unified model inputs")
    cell_ids = np.asarray(
        [f"merfish_cell_{index + 1:07d}" for index in range(len(selected))], dtype=str
    )
    expression = pd.DataFrame(
        dense_selected_expression(matrix, gene_names, selected_genes),
        index=cell_ids,
        columns=selected_genes,
    )
    expression.index.name = "cell_id"
    coordinates = pd.DataFrame(
        selected[coordinate_columns].to_numpy(dtype=float),
        index=cell_ids,
        columns=["x", "y"],
    )
    coordinates.index.name = "cell_id"
    metadata = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "source_cell_id": selected[id_column].astype(str).to_numpy(),
            "sample_id": config.sample_id,
            "cell_type": config.cell_type_label,
            "region": config.region_label,
            "condition": config.condition_label,
            "original_x": pd.to_numeric(selected["center_x"], errors="raise").to_numpy(),
            "original_y": pd.to_numeric(selected["center_y"], errors="raise").to_numpy(),
        }
    )
    destination = write_model_inputs(
        config.output_dir,
        expression,
        coordinates,
        metadata,
        tf_list,
        lr_pairs,
        prior,
        config,
        {
            "expression_table": path,
            "reference_tf": Path(config.reference_dir) / "tf_list.csv",
            "reference_lr": Path(config.reference_dir) / "lr_pairs.csv",
            "reference_prior": Path(config.reference_dir) / "prior_tf_gene.csv",
        },
        [
            "MERFISH gene columns are raw non-negative counts; use expression_is_normalized=False.",
            "Blank.* controls and Pmatch are excluded from expression genes.",
            "Coordinates use STalign_x/STalign_y when use_aligned_coordinates=True.",
            "The original scientific-notation cell identifier is retained in metadata.source_cell_id.",
            "region=global and cell_type=NA are placeholders, not inferred annotations.",
        ],
        lr_tf_prior=lr_tf_prior,
    )
    print("[5/5] Complete")
    return destination



