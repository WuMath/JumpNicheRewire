"""Match official STalign brain-region and cell-type labels to MERFISH S2R3.

The script builds a master annotation for all raw S2R3 cells and updates every
configured processed directory by ``metadata.source_cell_id``.  Exact IDs are
used first; a collision-checked rounded scientific-notation key is a fallback
for the precision mutation documented by STalign.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocess.common import PROJECT_ROOT, write_json
from preprocess.evidence_common import (
    canonical_numeric_id,
    read_csv_auto,
    write_frame_atomic,
)


from configs.real_data.settings.prepare_merfish_annotations import MERFISHAnnotationConfig, CONFIG





def _first_column_table(path: Path, value_names: list[str] | None = None) -> pd.DataFrame:
    frame = read_csv_auto(path, dtype="string", low_memory=False)
    if frame.empty:
        raise ValueError(f"Annotation table is empty: {path}")
    first = frame.columns[0]
    frame = frame.rename(columns={first: "annotation_cell_id"})
    if value_names is not None:
        missing = set(value_names).difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame = frame[["annotation_cell_id", *value_names]]
    frame["annotation_cell_id"] = frame["annotation_cell_id"].astype("string").str.strip()
    return frame.dropna(subset=["annotation_cell_id"]).drop_duplicates("annotation_cell_id")


def _match_values(
    base_ids: pd.Series,
    annotation: pd.DataFrame,
    value_columns: list[str],
    significant_digits: int,
) -> tuple[pd.DataFrame, pd.Series]:
    base = pd.DataFrame({"source_cell_id": base_ids.astype("string").str.strip()})
    values = annotation.set_index("annotation_cell_id")[value_columns]
    result = base[["source_cell_id"]].join(values, on="source_cell_id")
    method = pd.Series("unmatched", index=result.index, dtype="string")
    exact = result[value_columns].notna().any(axis=1)
    method.loc[exact] = "exact"

    unmatched = result.index[~exact]
    if len(unmatched):
        annotation_keys = annotation["annotation_cell_id"].map(
            lambda value: canonical_numeric_id(value, significant_digits)
        )
        key_table = annotation.assign(_numeric_key=annotation_keys).dropna(subset=["_numeric_key"])
        counts = key_table.groupby("_numeric_key")["annotation_cell_id"].nunique()
        key_table = key_table.loc[
            key_table["_numeric_key"].isin(counts[counts.eq(1)].index)
        ].drop_duplicates("_numeric_key")
        key_values = key_table.set_index("_numeric_key")[value_columns]
        base_keys = result.loc[unmatched, "source_cell_id"].map(
            lambda value: canonical_numeric_id(value, significant_digits)
        )
        fallback = pd.DataFrame({"_numeric_key": base_keys}, index=unmatched).join(
            key_values, on="_numeric_key"
        )
        matched = fallback[value_columns].notna().any(axis=1)
        matched_index = fallback.index[matched]
        result.loc[matched_index, value_columns] = fallback.loc[
            matched_index, value_columns
        ].to_numpy()
        method.loc[matched_index] = "rounded_numeric_fallback"
    return result[value_columns], method


def build_master_annotations(config: MERFISHAnnotationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_columns = read_csv_auto(config.raw_expression_table, nrows=0).columns.tolist()
    raw_id_column = raw_columns[0]
    raw_ids = read_csv_auto(
        config.raw_expression_table,
        usecols=[raw_id_column],
        dtype={raw_id_column: "string"},
        low_memory=False,
    )[raw_id_column]
    master = pd.DataFrame({"source_cell_id": raw_ids.astype("string").str.strip()})
    if master["source_cell_id"].duplicated().any():
        raise ValueError("Raw S2R3 cell IDs are not unique")

    structure_path = config.annotation_dir / config.structure_file_name
    cluster_path = config.annotation_dir / config.cluster_cell_type_file_name
    biological_path = config.annotation_dir / config.biological_cell_type_file_name
    structure = _first_column_table(
        structure_path,
        ["atlas_coord_0", "atlas_coord_1", "atlas_coord_2", "structure_id", "acronym"],
    )
    cluster = _first_column_table(cluster_path)
    cluster_value = [column for column in cluster.columns if column != "annotation_cell_id"]
    if len(cluster_value) != 1:
        raise ValueError("S2R3 cluster annotation must contain one label column")
    cluster = cluster.rename(columns={cluster_value[0]: "cell_type_cluster"})

    structure_values = [
        "atlas_coord_0",
        "atlas_coord_1",
        "atlas_coord_2",
        "structure_id",
        "acronym",
    ]
    matched_structure, structure_method = _match_values(
        master["source_cell_id"], structure, structure_values, config.id_significant_digits
    )
    matched_cluster, cluster_method = _match_values(
        master["source_cell_id"], cluster, ["cell_type_cluster"], config.id_significant_digits
    )
    master = pd.concat([master, matched_structure, matched_cluster], axis=1)

    biological_method = pd.Series("file_absent", index=master.index, dtype="string")
    if biological_path.exists():
        biological = _first_column_table(biological_path)
        preferred = (
            "celltype_merged"
            if "celltype_merged" in biological.columns
            else "celltype"
            if "celltype" in biological.columns
            else None
        )
        if preferred is None:
            raise ValueError("Biological cell-type table has no celltype column")
        matched_biological, biological_method = _match_values(
            master["source_cell_id"],
            biological,
            [preferred],
            config.id_significant_digits,
        )
        master["cell_type_biological"] = matched_biological[preferred]
    else:
        master["cell_type_biological"] = pd.NA

    background = set(value.lower() for value in config.background_acronyms)
    master["region"] = master["acronym"].astype("string").str.strip()
    master.loc[
        master["region"].isna()
        | master["region"].str.lower().isin(background)
        | pd.to_numeric(master["structure_id"], errors="coerce").fillna(0).eq(0),
        "region",
    ] = pd.NA
    master["cell_type"] = master["cell_type_biological"].combine_first(
        master["cell_type_cluster"]
    )
    master["region_match_method"] = structure_method
    master["cluster_match_method"] = cluster_method
    master["biological_type_match_method"] = biological_method
    master["region_source"] = np.where(master["region"].notna(), "Allen_CCF_STalign", "missing")
    master["cell_type_source"] = np.where(
        master["cell_type_biological"].notna(),
        "STalign_nine_slice_marker_annotation_v2",
        np.where(master["cell_type_cluster"].notna(), "STalign_S2R3_CT_cluster", "missing"),
    )
    audit = pd.DataFrame(
        {
            "item": [
                "raw_cells",
                "region_matched",
                "region_exact",
                "region_rounded_fallback",
                "cluster_cell_type_matched",
                "biological_cell_type_matched",
                "final_cell_type_matched",
            ],
            "value": [
                len(master),
                master["region"].notna().sum(),
                structure_method.eq("exact").sum(),
                structure_method.eq("rounded_numeric_fallback").sum(),
                master["cell_type_cluster"].notna().sum(),
                master["cell_type_biological"].notna().sum(),
                master["cell_type"].notna().sum(),
            ],
        }
    )
    return master, audit


def update_processed_metadata(
    processed_dir: Path,
    master: pd.DataFrame,
) -> dict[str, object]:
    metadata_path = Path(processed_dir) / "metadata.csv"
    metadata = pd.read_csv(metadata_path, dtype={"cell_id": "string", "source_cell_id": "string"})
    if "source_cell_id" not in metadata.columns:
        raise ValueError(f"Processed MERFISH metadata has no source_cell_id: {metadata_path}")
    annotations = master[
        [
            "source_cell_id",
            "cell_type",
            "cell_type_cluster",
            "cell_type_biological",
            "cell_type_source",
            "region",
            "structure_id",
            "region_source",
            "atlas_coord_0",
            "atlas_coord_1",
            "atlas_coord_2",
        ]
    ]
    old_columns = {
        column: f"{column}_old"
        for column in ("cell_type", "region")
        if column in metadata.columns
    }
    metadata = metadata.rename(columns=old_columns).merge(
        annotations, on="source_cell_id", how="left", validate="many_to_one"
    )
    for column in ("cell_type", "region"):
        old = f"{column}_old"
        if old in metadata.columns:
            old_values = metadata[old].astype("string")
            usable_old = old_values.where(~old_values.str.lower().isin({"na", "global", "nan"}))
            metadata[column] = metadata[column].combine_first(usable_old)
            metadata = metadata.drop(columns=old)
    write_frame_atomic(metadata, metadata_path)
    return {
        "processed_dir": str(processed_dir),
        "rows": len(metadata),
        "region_matched": int(metadata["region"].notna().sum()),
        "cell_type_matched": int(metadata["cell_type"].notna().sum()),
    }


def main(config: MERFISHAnnotationConfig = CONFIG) -> Path:
    required = [
        config.raw_expression_table,
        config.annotation_dir / config.structure_file_name,
        config.annotation_dir / config.cluster_cell_type_file_name,
    ]
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"MERFISH annotation input is missing: {path}. "
                "Run download_public_resources.py first."
            )
    print("[1/3] Matching official S2R3 region and cell-type annotations")
    master, audit = build_master_annotations(config)
    master_path = config.annotation_dir / config.master_output_name
    write_frame_atomic(master, master_path)
    write_frame_atomic(audit, config.annotation_dir / config.audit_output_name)
    print("[2/3] Updating available processed MERFISH metadata files")
    updates: list[dict[str, object]] = []
    for directory in config.processed_dirs:
        if not (Path(directory) / "metadata.csv").exists():
            if config.skip_missing_processed_dirs:
                continue
            raise FileNotFoundError(f"Processed MERFISH directory is missing: {directory}")
        updates.append(update_processed_metadata(directory, master))
    write_json(
        config.annotation_dir / config.manifest_name,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(config),
            "audit": dict(zip(audit["item"], audit["value"])),
            "updated_processed_directories": updates,
        },
    )
    print(audit.to_string(index=False))
    for update in updates:
        print(update)
    print("[3/3] Complete")
    return master_path



