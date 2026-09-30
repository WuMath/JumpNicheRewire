"""Add STalign/Allen regions and official cell compositions to Visium spots.

The official 2264 x 16 STalign file is already a Visium deconvolution result.
It is matched directly by barcode.  Brain regions are transferred from labeled
MERFISH S2R3 cells in the aligned Visium hires-image coordinate system.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from preprocess.common import PROJECT_ROOT, write_json
from preprocess.evidence_common import read_csv_auto, write_frame_atomic


from configs.real_data.settings.prepare_visium_annotations import VisiumAnnotationConfig, CONFIG





def read_scalefactors(source: Path) -> dict[str, float]:
    source = Path(source)
    if source.is_dir():
        candidates = list(source.rglob("scalefactors_json.json"))
        if not candidates:
            raise FileNotFoundError(f"No scalefactors_json.json under {source}")
        return json.loads(candidates[0].read_text(encoding="utf-8"))
    if source.name == "scalefactors_json.json":
        return json.loads(source.read_text(encoding="utf-8"))
    if tarfile.is_tarfile(source):
        with tarfile.open(source, "r:*") as archive:
            members = [
                member
                for member in archive.getmembers()
                if Path(member.name).name == "scalefactors_json.json" and member.isfile()
            ]
            if not members:
                raise FileNotFoundError("Visium spatial archive has no scalefactors_json.json")
            stream = archive.extractfile(members[0])
            if stream is None:
                raise OSError("Could not read Visium scale factors")
            return json.load(io.TextIOWrapper(stream, encoding="utf-8"))
    raise ValueError("visium_spatial_source must be a directory, JSON, or tar archive")


def _read_composition(path: Path) -> pd.DataFrame:
    frame = read_csv_auto(path, low_memory=False)
    first = frame.columns[0]
    frame = frame.rename(columns={first: "cell_id"})
    frame["cell_id"] = frame["cell_id"].astype(str)
    value_columns = [column for column in frame.columns if column != "cell_id"]
    values = frame[value_columns].apply(pd.to_numeric, errors="raise")
    if (values.to_numpy() < 0).any():
        raise ValueError("Official Visium composition contains negative values")
    totals = values.sum(axis=1)
    nonzero = totals.gt(0)
    values.loc[nonzero] = values.loc[nonzero].div(totals.loc[nonzero], axis=0)
    frame[value_columns] = values
    return frame


def _region_reference(
    master_path: Path,
    aligned_expression: Path,
) -> pd.DataFrame:
    master = pd.read_parquet(master_path, columns=["source_cell_id", "region"])
    header = read_csv_auto(aligned_expression, nrows=0).columns.tolist()
    id_column = header[0]
    aligned = read_csv_auto(
        aligned_expression,
        usecols=[id_column, "STalign_x", "STalign_y"],
        dtype={id_column: "string"},
        low_memory=False,
    ).rename(columns={id_column: "source_cell_id"})
    aligned["source_cell_id"] = aligned["source_cell_id"].astype("string").str.strip()
    reference = aligned.merge(master, on="source_cell_id", how="inner", validate="one_to_one")
    reference["STalign_x"] = pd.to_numeric(reference["STalign_x"], errors="coerce")
    reference["STalign_y"] = pd.to_numeric(reference["STalign_y"], errors="coerce")
    return reference.dropna(subset=["STalign_x", "STalign_y", "region"]).reset_index(drop=True)


def transfer_regions(
    coordinates_fullres: np.ndarray,
    reference: pd.DataFrame,
    hires_scale: float,
    spot_diameter_hires: float,
    config: VisiumAnnotationConfig,
) -> pd.DataFrame:
    tree = cKDTree(reference[["STalign_x", "STalign_y"]].to_numpy(dtype=float))
    query = np.asarray(coordinates_fullres, dtype=float) * hires_scale
    maximum_distance = spot_diameter_hires * config.maximum_distance_in_spot_diameters
    distances, indices = tree.query(
        query,
        k=min(config.nearest_merfish_cells, len(reference)),
        distance_upper_bound=maximum_distance,
    )
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    labels = reference["region"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    sigma = max(spot_diameter_hires, 1e-8)
    for distance_row, index_row in zip(distances, indices):
        valid = np.isfinite(distance_row) & (index_row < len(reference))
        if valid.sum() < config.minimum_region_neighbors:
            rows.append(
                {"region": pd.NA, "region_confidence": np.nan, "region_neighbors": int(valid.sum())}
            )
            continue
        selected_labels = labels[index_row[valid]]
        weights = np.exp(-0.5 * (distance_row[valid] / sigma) ** 2)
        scores: dict[str, float] = {}
        for label, weight in zip(selected_labels, weights):
            scores[label] = scores.get(label, 0.0) + float(weight)
        region, best = max(scores.items(), key=lambda item: (item[1], item[0]))
        confidence = best / max(sum(scores.values()), 1e-12)
        if confidence < config.minimum_region_confidence:
            region = pd.NA
        rows.append(
            {
                "region": region,
                "region_confidence": confidence,
                "region_neighbors": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def annotate_directory(
    processed_dir: Path,
    composition: pd.DataFrame,
    reference: pd.DataFrame,
    scale_factors: dict[str, float],
    config: VisiumAnnotationConfig,
) -> dict[str, object]:
    root = Path(processed_dir)
    metadata = pd.read_csv(root / "metadata.csv", dtype={"cell_id": "string"})
    coordinates = pd.read_csv(root / "coordinates.csv", index_col=0)
    coordinates.index = coordinates.index.astype(str)
    metadata["cell_id"] = metadata["cell_id"].astype(str)
    coordinates = coordinates.loc[metadata["cell_id"]]
    hires_scale = float(scale_factors["tissue_hires_scalef"])
    spot_diameter_hires = float(scale_factors["spot_diameter_fullres"]) * hires_scale
    region = transfer_regions(
        coordinates[["x", "y"]].to_numpy(dtype=float),
        reference,
        hires_scale,
        spot_diameter_hires,
        config,
    )
    region.index = metadata.index

    selected_composition = metadata[["cell_id"]].merge(
        composition, on="cell_id", how="left", validate="one_to_one"
    )
    value_columns = [column for column in composition.columns if column != "cell_id"]
    composition_values = selected_composition[value_columns]
    dominant = composition_values.idxmax(axis=1).where(composition_values.notna().any(axis=1))
    old_columns = {
        column: f"{column}_old"
        for column in ("cell_type", "region")
        if column in metadata.columns
    }
    metadata = metadata.rename(columns=old_columns)
    metadata["cell_type"] = dominant
    metadata["cell_type_source"] = np.where(
        dominant.notna(), "STalign_official_Visium_deconvolution", "missing"
    )
    metadata["region"] = region["region"]
    metadata["region_confidence"] = region["region_confidence"]
    metadata["region_neighbors"] = region["region_neighbors"]
    metadata["region_source"] = np.where(
        metadata["region"].notna(), "Allen_CCF_via_STalign_S2R3", "missing"
    )
    for column in value_columns:
        metadata[f"cellprop_{column.replace('-', '_')}"] = composition_values[column]
    for column in ("cell_type", "region"):
        old = f"{column}_old"
        if old in metadata.columns:
            old_values = metadata[old].astype("string")
            usable = old_values.where(~old_values.str.lower().isin({"na", "global", "nan"}))
            metadata[column] = metadata[column].combine_first(usable)
            metadata = metadata.drop(columns=old)
    write_frame_atomic(metadata, root / "metadata.csv")
    write_frame_atomic(selected_composition, root / config.composition_output_name)
    audit = pd.DataFrame(
        {
            "item": [
                "spots",
                "composition_matched",
                "region_assigned",
                "median_region_confidence",
                "hires_scale",
                "spot_diameter_hires_pixels",
            ],
            "value": [
                len(metadata),
                composition_values.notna().any(axis=1).sum(),
                metadata["region"].notna().sum(),
                metadata["region_confidence"].median(),
                hires_scale,
                spot_diameter_hires,
            ],
        }
    )
    write_frame_atomic(audit, root / config.audit_output_name)
    return {
        "processed_dir": str(root),
        **dict(zip(audit["item"], audit["value"])),
    }


def main(config: VisiumAnnotationConfig = CONFIG) -> list[dict[str, object]]:
    for path in (
        config.composition_file,
        config.merfish_master_annotations,
        config.merfish_aligned_expression,
        config.visium_spatial_source,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Visium annotation input is missing: {path}. "
                "Run download_public_resources.py and prepare_merfish_annotations.py first."
            )
    print("[1/3] Reading official Visium composition and aligned MERFISH region reference")
    composition = _read_composition(config.composition_file)
    reference = _region_reference(
        config.merfish_master_annotations, config.merfish_aligned_expression
    )
    scale_factors = read_scalefactors(config.visium_spatial_source)
    print("[2/3] Matching composition and transferring Allen regions")
    updates: list[dict[str, object]] = []
    for directory in config.processed_dirs:
        if not (Path(directory) / "metadata.csv").exists():
            if config.skip_missing_processed_dirs:
                continue
            raise FileNotFoundError(f"Processed Visium directory is missing: {directory}")
        updates.append(
            annotate_directory(directory, composition, reference, scale_factors, config)
        )
    manifest_path = config.composition_file.parent / config.manifest_name
    write_json(
        manifest_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(config),
            "region_reference_cells": len(reference),
            "official_composition_spots": len(composition),
            "updated_processed_directories": updates,
        },
    )
    for update in updates:
        print(update)
    print("[3/3] Complete")
    return updates



