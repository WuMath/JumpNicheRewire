"""Prepare the downloaded 10x Visium H5 and spatial archive for the model.

The 10x ``filtered_feature_bc_matrix.h5`` is a feature-barcode matrix, not an
H5AD file.  It contains expression but not spatial coordinates.  Coordinates
are read separately from a Space Ranger spatial archive/directory and aligned
by barcode.  This module never accesses the network.
"""

from __future__ import annotations

import io
import tarfile
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


POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]


from configs.real_data.settings.prepare_visium import VisiumConfig, CONFIG





def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]
    )


def read_10x_h5(path: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Visium H5 processing requires h5py; install requirements.txt") from error

    with h5py.File(path, "r") as handle:
        if "matrix" not in handle:
            raise ValueError("Not a standard 10x feature-barcode H5: missing /matrix")
        group = handle["matrix"]
        shape = tuple(int(value) for value in group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
            dtype=np.float64,
        ).tocsr()
        feature_group = group["features"]
        gene_names = _decode(feature_group["name"][:])
        barcodes = _decode(group["barcodes"][:])
        if "feature_type" in feature_group:
            feature_types = _decode(feature_group["feature_type"][:])
            keep = feature_types == "Gene Expression"
            matrix = matrix[keep]
            gene_names = gene_names[keep]
    if matrix.shape != (len(gene_names), len(barcodes)):
        raise ValueError("10x H5 matrix, feature names, and barcodes have inconsistent dimensions")
    return matrix, gene_names, barcodes


def _read_position_stream(stream) -> pd.DataFrame:
    raw = stream.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    frame = pd.read_csv(io.StringIO(raw))
    if "barcode" not in frame.columns:
        frame = pd.read_csv(io.StringIO(raw), header=None, names=POSITION_COLUMNS)
    missing = set(POSITION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Spatial position table is missing columns: {sorted(missing)}")
    return frame


def read_positions(source: Path, tissue_only: bool) -> pd.DataFrame:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(
            f"Visium spatial source is missing: {source}. The expression H5 alone has no coordinates."
        )
    if source.is_file() and source.name in {"tissue_positions.csv", "tissue_positions_list.csv"}:
        with source.open("rb") as stream:
            frame = _read_position_stream(stream)
    elif source.is_dir():
        candidates = list(source.rglob("tissue_positions.csv")) + list(
            source.rglob("tissue_positions_list.csv")
        )
        if not candidates:
            raise FileNotFoundError(f"No tissue position CSV was found under {source}")
        with candidates[0].open("rb") as stream:
            frame = _read_position_stream(stream)
    elif tarfile.is_tarfile(source):
        with tarfile.open(source, "r:*") as archive:
            members = [
                member
                for member in archive.getmembers()
                if Path(member.name).name in {"tissue_positions.csv", "tissue_positions_list.csv"}
                and member.isfile()
            ]
            if not members:
                raise FileNotFoundError("Spatial archive contains no tissue position CSV")
            stream = archive.extractfile(members[0])
            if stream is None:
                raise OSError("Could not read tissue position CSV from spatial archive")
            frame = _read_position_stream(stream)
    else:
        raise ValueError("spatial_source must be a position CSV, directory, or tar/tar.gz archive")

    frame["barcode"] = frame["barcode"].astype(str)
    for column in POSITION_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if tissue_only:
        frame = frame.loc[frame["in_tissue"].eq(1)].copy()
    frame["x"] = frame["pxl_col_in_fullres"].astype(float)
    frame["y"] = frame["pxl_row_in_fullres"].astype(float)
    return frame


def main(config: VisiumConfig = CONFIG) -> Path:
    if not Path(config.matrix_h5).exists():
        raise FileNotFoundError(f"Visium H5 is missing: {config.matrix_h5}")
    if not Path(config.reference_dir, "tf_list.csv").exists():
        raise FileNotFoundError(
            "Standardized references are missing. Run: python -m preprocess.prepare_references"
        )
    print("[1/5] Reading 10x H5 expression and Space Ranger coordinates")
    matrix, gene_names, barcodes = read_10x_h5(Path(config.matrix_h5))
    positions = read_positions(Path(config.spatial_source), config.tissue_only)

    print("[2/5] Aligning expression and coordinates by barcode")
    barcode_to_column = {barcode: index for index, barcode in enumerate(barcodes)}
    positions = positions.loc[positions["barcode"].isin(barcode_to_column)].reset_index(drop=True)
    if positions.empty:
        raise ValueError("No spatial barcode overlaps the downloaded expression H5")
    selected_rows = spatial_farthest_sample(
        positions[["x", "y"]].to_numpy(dtype=float), config.maximum_spots, config.random_seed
    )
    positions = positions.iloc[selected_rows].reset_index(drop=True)
    columns = np.asarray([barcode_to_column[value] for value in positions["barcode"]], dtype=int)
    matrix = matrix[:, columns].tocsr()

    print("[3/5] Matching ModelData TF/LR references to expressed genes")
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
        min(config.minimum_detected_spots, matrix.shape[1]),
    )
    prior = prior.loc[prior["target"].isin(selected_genes)].reset_index(drop=True)
    lr_tf_prior = filter_lr_tf_prior(config.reference_dir, tf_list, lr_pairs)

    print("[4/5] Writing unified model inputs")
    spot_ids = positions["barcode"].astype(str).to_numpy()
    expression = pd.DataFrame(
        dense_selected_expression(matrix, gene_names, selected_genes),
        index=spot_ids,
        columns=selected_genes,
    )
    expression.index.name = "cell_id"
    coordinates = positions.set_index("barcode")[["x", "y"]].copy()
    coordinates.index.name = "cell_id"
    metadata = pd.DataFrame(
        {
            "cell_id": spot_ids,
            "sample_id": config.sample_id,
            "cell_type": config.cell_type_label,
            "region": config.region_label,
            "condition": config.condition_label,
            "array_row": positions["array_row"].to_numpy(),
            "array_col": positions["array_col"].to_numpy(),
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
            "matrix_h5": Path(config.matrix_h5),
            "spatial_source": Path(config.spatial_source),
            "reference_tf": Path(config.reference_dir) / "tf_list.csv",
            "reference_lr": Path(config.reference_dir) / "lr_pairs.csv",
            "reference_prior": Path(config.reference_dir) / "prior_tf_gene.csv",
        },
        [
            "Expression values are raw filtered 10x counts; use expression_is_normalized=False.",
            "Coordinates are full-resolution image pixels from Space Ranger.",
            "region=global and cell_type=NA are placeholders, not inferred annotations.",
        ],
        lr_tf_prior=lr_tf_prior,
    )
    print("[5/5] Complete")
    return destination



