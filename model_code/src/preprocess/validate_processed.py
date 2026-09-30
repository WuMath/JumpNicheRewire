"""Validate processed model-input directories without running the model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sys
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from preprocess.common import PROJECT_ROOT


from configs.real_data.settings.validate_processed import ValidationConfig, CONFIG





def validate_directory(input_dir: Path, config: ValidationConfig) -> pd.DataFrame:
    root = Path(input_dir)
    required = [
        "expression.csv",
        "coordinates.csv",
        "metadata.csv",
        "tf_list.csv",
        "lr_pairs.csv",
        "prior_tf_gene.csv",
        "prior_lr_tf.csv",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{root} is missing files: {missing}")

    expression = pd.read_csv(root / "expression.csv", index_col=0)
    coordinates = pd.read_csv(root / "coordinates.csv", index_col=0)
    metadata = pd.read_csv(root / "metadata.csv")
    tf_list = pd.read_csv(root / "tf_list.csv")
    lr_pairs = pd.read_csv(root / "lr_pairs.csv")
    prior = pd.read_csv(root / "prior_tf_gene.csv")
    lr_tf_prior = pd.read_csv(root / "prior_lr_tf.csv")

    expression.index = expression.index.astype(str)
    coordinates.index = coordinates.index.astype(str)
    expression.columns = expression.columns.astype(str)
    if expression.empty or coordinates.empty:
        raise ValueError(f"Expression/coordinates are empty: {root}")
    if expression.index.duplicated().any() or expression.columns.duplicated().any():
        raise ValueError(f"Duplicated expression cell or gene identifiers: {root}")
    if coordinates.index.duplicated().any():
        raise ValueError(f"Duplicated coordinate identifiers: {root}")
    if not expression.index.equals(coordinates.index):
        raise ValueError(f"Expression and coordinate IDs/order differ: {root}")
    if coordinates.shape[1] not in {2, 3}:
        raise ValueError(f"Coordinates must contain two or three dimensions: {root}")
    expression_values = expression.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    coordinate_values = coordinates.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(expression_values).all() or (expression_values < 0).any():
        raise ValueError(f"Expression must be finite and non-negative: {root}")
    if not np.isfinite(coordinate_values).all():
        raise ValueError(f"Coordinates must be finite: {root}")
    if metadata.empty or "cell_id" not in metadata.columns:
        raise ValueError(f"metadata.csv requires cell_id: {root}")
    metadata_ids = metadata["cell_id"].astype(str)
    if metadata_ids.duplicated().any() or set(metadata_ids) != set(expression.index):
        raise ValueError(f"Metadata cell IDs do not match expression: {root}")

    genes = set(expression.columns)
    expressed_tfs = tf_list["tf"].astype(str).isin(genes)
    if not expressed_tfs.all() or not len(tf_list):
        raise ValueError(f"TF table contains absent genes or is empty: {root}")
    lr_complete = lr_pairs["ligand"].astype(str).isin(genes) & lr_pairs["receptor"].astype(str).isin(genes)
    if not lr_complete.all() or not len(lr_pairs):
        raise ValueError(f"LR table contains incomplete pairs or is empty: {root}")
    if not prior.empty:
        valid_prior = prior["tf"].astype(str).isin(genes) & prior["target"].astype(str).isin(genes)
        if not valid_prior.all():
            raise ValueError(f"TF-target prior contains absent genes: {root}")
    if not lr_tf_prior.empty:
        required_lr_tf = {"lr_id", "tf", "score"}
        missing_lr_tf = required_lr_tf.difference(lr_tf_prior.columns)
        if missing_lr_tf:
            raise ValueError(f"LR-TF prior is missing columns {sorted(missing_lr_tf)}: {root}")
        valid_lr_tf = lr_tf_prior["lr_id"].astype(str).isin(lr_pairs["lr_id"].astype(str)) & lr_tf_prior[
            "tf"
        ].astype(str).isin(tf_list["tf"].astype(str))
        scores = pd.to_numeric(lr_tf_prior["score"], errors="raise")
        if not valid_lr_tf.all() or not np.isfinite(scores).all() or not scores.between(0, 1).all():
            raise ValueError(f"LR-TF prior contains invalid identifiers/scores: {root}")

    for activity_name in ("ligand_activity.csv", "receptor_activity.csv", "tf_activity.csv"):
        activity_path = root / activity_name
        if not activity_path.exists():
            continue
        activity = pd.read_csv(activity_path, index_col=0)
        activity.index = activity.index.astype(str)
        if not activity.index.equals(expression.index):
            raise ValueError(f"{activity_name} IDs/order differ from expression: {root}")
        activity_values = activity.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(activity_values).all():
            raise ValueError(f"{activity_name} contains non-finite values: {root}")

    composition_path = root / "cell_composition.csv"
    composition_coverage = 0
    if composition_path.exists():
        composition = pd.read_csv(composition_path, index_col=0)
        composition.index = composition.index.astype(str)
        if set(composition.index) != set(expression.index):
            raise ValueError(f"cell_composition IDs differ from expression: {root}")
        values = composition.apply(pd.to_numeric, errors="coerce")
        matched = values.notna().any(axis=1)
        totals = values.loc[matched].sum(axis=1)
        if (values.loc[matched].to_numpy() < 0).any() or not np.allclose(totals, 1.0, atol=1e-5):
            raise ValueError(f"cell_composition must be non-negative and sum to one: {root}")
        composition_coverage = int(matched.sum())

    region_values = metadata.get("region", pd.Series(index=metadata.index, dtype=object))
    cell_type_values = metadata.get("cell_type", pd.Series(index=metadata.index, dtype=object))
    region_coverage = int(
        (region_values.notna() & ~region_values.astype(str).str.lower().isin({"global", "na", "nan"})).sum()
    )
    cell_type_coverage = int(
        (
            cell_type_values.notna()
            & ~cell_type_values.astype(str).str.lower().isin({"global", "na", "nan"})
        ).sum()
    )

    summary = pd.DataFrame(
        {
            "check": [
                "cells_or_spots",
                "genes",
                "coordinate_dimensions",
                "metadata_rows",
                "expressed_tfs",
                "complete_lr_pairs",
                "tf_target_prior_edges",
                "lr_tf_prior_edges",
                "region_annotated",
                "cell_type_annotated",
                "composition_annotated",
                "expression_nonnegative_finite",
                "all_ids_aligned",
            ],
            "value": [
                len(expression),
                expression.shape[1],
                coordinates.shape[1],
                len(metadata),
                int(expressed_tfs.sum()),
                int(lr_complete.sum()),
                len(prior),
                len(lr_tf_prior),
                region_coverage,
                cell_type_coverage,
                composition_coverage,
                True,
                True,
            ],
        }
    )
    print(f"\nValidation: {root}")
    print(summary.to_string(index=False))
    if config.write_excel_report:
        with pd.ExcelWriter(root / config.report_name, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            tf_list.to_excel(writer, sheet_name="tf_list", index=False)
            lr_pairs.to_excel(writer, sheet_name="lr_pairs", index=False)
            prior.to_excel(writer, sheet_name="prior_tf_gene", index=False)
            lr_tf_prior.head(1_000_000).to_excel(writer, sheet_name="prior_lr_tf", index=False)
    return summary


def main(config: ValidationConfig = CONFIG) -> None:
    for input_dir in config.input_dirs:
        validate_directory(Path(input_dir), config)



