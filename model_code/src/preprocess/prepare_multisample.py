"""Combine two or more independently processed biological samples.

Do not list Visium and MERFISH from different animals as biological replicates.
Each configured directory must represent an independent sample from the same
platform/design.  Cell IDs are prefixed and spatial coordinates remain local;
the main model already prevents graph edges across ``sample_id`` values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from preprocess.common import PROJECT_ROOT, write_json
from preprocess.evidence_common import write_frame_atomic


from configs.real_data.settings.prepare_multisample import MultiSampleConfig, CONFIG





REQUIRED_FILES = (
    "expression.csv",
    "coordinates.csv",
    "metadata.csv",
    "tf_list.csv",
    "lr_pairs.csv",
    "prior_tf_gene.csv",
    "prior_lr_tf.csv",
)


def _read_sample(directory: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED_FILES if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"{directory} is missing: {missing}")
    return {
        "expression": pd.read_csv(directory / "expression.csv", index_col=0),
        "coordinates": pd.read_csv(directory / "coordinates.csv", index_col=0),
        "metadata": pd.read_csv(directory / "metadata.csv"),
        "tf": pd.read_csv(directory / "tf_list.csv"),
        "lr": pd.read_csv(directory / "lr_pairs.csv"),
        "tf_prior": pd.read_csv(directory / "prior_tf_gene.csv"),
        "lr_tf": pd.read_csv(directory / "prior_lr_tf.csv"),
    }


def main(config: MultiSampleConfig = CONFIG) -> Path:
    if len(config.sample_dirs) < config.minimum_samples:
        raise ValueError(
            "Configure at least two independent biological sample_dirs. "
            "Pseudo-replicates and cross-technology files are not accepted by default."
        )
    if len(config.sample_ids) != len(config.sample_dirs):
        raise ValueError("sample_ids must have exactly one value per sample_dirs entry")
    if len(set(config.sample_ids)) != len(config.sample_ids):
        raise ValueError("sample_ids must be unique")
    if config.require_same_platform:
        if len(config.platform_labels) != len(config.sample_dirs):
            raise ValueError("platform_labels must be supplied when require_same_platform=True")
        if len(set(config.platform_labels)) != 1:
            raise ValueError("Biological replicate cohort mixes different platforms")
    samples = [_read_sample(Path(directory)) for directory in config.sample_dirs]
    coordinate_shapes = {sample["coordinates"].shape[1] for sample in samples}
    coordinate_columns = {tuple(sample["coordinates"].columns.astype(str)) for sample in samples}
    if len(coordinate_shapes) != 1 or len(coordinate_columns) != 1:
        raise ValueError("All cohort samples must use the same coordinate dimensions/columns")
    gene_sets = [set(sample["expression"].columns.astype(str)) for sample in samples]
    if config.gene_join == "intersection":
        genes = sorted(set.intersection(*gene_sets))
    elif config.gene_join == "union":
        genes = sorted(set.union(*gene_sets))
    else:
        raise ValueError("gene_join must be 'intersection' or 'union'")
    if len(genes) < 2:
        raise ValueError("Too few genes remain after multi-sample alignment")

    expression_parts: list[pd.DataFrame] = []
    coordinate_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    for sample, sample_id in zip(samples, config.sample_ids):
        expression = sample["expression"].copy()
        expression.index = expression.index.astype(str)
        source_ids = expression.index.copy()
        prefixed = pd.Index([f"{sample_id}::{value}" for value in source_ids], name="cell_id")
        expression = expression.reindex(columns=genes, fill_value=0.0)
        expression.index = prefixed
        coordinates = sample["coordinates"].copy()
        coordinates.index = coordinates.index.astype(str)
        coordinates = coordinates.loc[source_ids]
        coordinates.index = prefixed
        metadata = sample["metadata"].copy()
        metadata["cell_id"] = metadata["cell_id"].astype(str)
        metadata = metadata.set_index("cell_id").loc[source_ids].reset_index()
        metadata["source_cell_id"] = metadata["cell_id"]
        metadata["cell_id"] = prefixed.to_numpy()
        metadata["sample_id"] = sample_id
        expression_parts.append(expression)
        coordinate_parts.append(coordinates)
        metadata_parts.append(metadata)

    expression_all = pd.concat(expression_parts, axis=0)
    coordinates_all = pd.concat(coordinate_parts, axis=0)
    metadata_all = pd.concat(metadata_parts, ignore_index=True)
    tf_all = pd.concat([sample["tf"] for sample in samples], ignore_index=True)
    tf_all = tf_all.loc[tf_all["tf"].astype(str).isin(genes)].drop_duplicates("tf")

    lr_tables = [sample["lr"].copy() for sample in samples]
    common_lr = set.intersection(*[set(table["lr_id"].astype(str)) for table in lr_tables])
    lr_all = lr_tables[0].loc[lr_tables[0]["lr_id"].astype(str).isin(common_lr)].copy()
    def all_components_measured(value: object) -> bool:
        components = [token for token in re.split(r"[&|]", str(value)) if token]
        return bool(components) and all(component in genes for component in components)

    lr_all = lr_all.loc[
        lr_all["ligand"].map(all_components_measured)
        & lr_all["receptor"].map(all_components_measured)
    ].copy()
    if config.require_same_lr_definition:
        for table in lr_tables[1:]:
            comparison = lr_all[["lr_id", "ligand", "receptor"]].merge(
                table[["lr_id", "ligand", "receptor"]],
                on="lr_id",
                suffixes=("_first", "_next"),
            )
            inconsistent = comparison[
                comparison["ligand_first"].ne(comparison["ligand_next"])
                | comparison["receptor_first"].ne(comparison["receptor_next"])
            ]
            if not inconsistent.empty:
                raise ValueError("Same lr_id has inconsistent definitions across samples")
    prior = pd.concat([sample["tf_prior"] for sample in samples], ignore_index=True)
    prior = prior.loc[prior["tf"].isin(tf_all["tf"]) & prior["target"].isin(genes)]
    prior = prior.groupby(["tf", "target"], as_index=False).agg(
        {column: "max" for column in ("motif", "atac", "grn") if column in prior.columns}
    )
    lr_tf = pd.concat([sample["lr_tf"] for sample in samples], ignore_index=True)
    if not lr_tf.empty:
        lr_tf = lr_tf.loc[
            lr_tf["lr_id"].isin(lr_all["lr_id"]) & lr_tf["tf"].isin(tf_all["tf"])
        ]
        lr_tf = lr_tf.sort_values("score", ascending=False).drop_duplicates(["lr_id", "tf"])

    output = Path(config.output_dir)
    write_frame_atomic(expression_all, output / "expression.csv", index=True)
    write_frame_atomic(coordinates_all, output / "coordinates.csv", index=True)
    write_frame_atomic(metadata_all, output / "metadata.csv")
    write_frame_atomic(tf_all, output / "tf_list.csv")
    write_frame_atomic(lr_all, output / "lr_pairs.csv")
    write_frame_atomic(prior, output / "prior_tf_gene.csv")
    write_frame_atomic(lr_tf, output / "prior_lr_tf.csv")
    summary = pd.DataFrame(
        {
            "item": ["samples", "cells_or_spots", "genes", "tfs", "lr_pairs"],
            "value": [len(samples), len(expression_all), len(genes), len(tf_all), len(lr_all)],
        }
    )
    write_frame_atomic(summary, output / "data_summary.csv")
    write_json(
        output / "multisample_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(config),
            "source_directories": [str(Path(path).resolve()) for path in config.sample_dirs],
            "note": "coordinates remain sample-local; sample_id prevents cross-sample spatial edges",
        },
    )
    print(summary.to_string(index=False))
    return output



