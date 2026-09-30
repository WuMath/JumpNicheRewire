"""Unified Excel/CSV result contract shared with independent baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

from .config import OutputConfig, ProjectConfig
from .types import (
    BaseGRNResult,
    CommunicationResult,
    InputBundle,
    PriorResult,
    RewiringResult,
    SpatialResult,
)
from .utils import ensure_directory, log_message, safe_sheet_name


EXCEL_MAX_DATA_ROWS = 1_048_575


UNIFIED_SCHEMAS: dict[str, list[str]] = {
    "grn_edges": [
        "method", "unit_type", "cell_id", "region", "tf", "target", "score", "signed_score",
    ],
    "lr_gene": [
        "method", "receiver_id", "region", "lr_id", "ligand", "receptor", "target", "score", "signed_score",
    ],
    "receptor_paths": [
        "method", "sender_id", "receiver_id", "sender_region", "receiver_region", "lr_id",
        "ligand", "receptor", "tf", "target", "communication_score", "delta_beta", "path_score",
        "evidence_level", "evidence_label",
    ],
    "cci_edges": ["method", "sender_id", "receiver_id", "lr_id", "ligand", "receptor", "score"],
    "regulon_activity": ["method", "cell_id", "region", "tf", "activity"],
    "causal_edges": [
        "method", "region", "lr_id", "tf", "target", "gamma_estimate", "causal_estimand",
        "evidence_level", "evidence_label",
    ],
}


def _ensure_schema(frame: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in schema:
        if column not in output.columns:
            output[column] = np.nan
    leading = schema + [column for column in output.columns if column not in schema]
    return output.loc[:, leading]


def _top_k_flat_indices(indices: np.ndarray, scores: np.ndarray, limit: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    indices, scores = indices[valid], scores[valid]
    limit = max(int(limit), 0)
    if limit == 0 or not len(indices):
        return np.empty(0, dtype=np.int64)
    if len(indices) <= limit:
        return indices[np.lexsort((indices, -scores))]
    threshold = float(np.partition(scores, len(scores) - limit)[len(scores) - limit])
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)
    equal = equal[np.argsort(indices[equal], kind="stable")]
    selected = np.concatenate([greater, equal[: limit - len(greater)]])
    return indices[selected[np.lexsort((indices[selected], -scores[selected]))]]


def _base_grn_flat_indices(
    base: BaseGRNResult,
    maximum_rows: int,
    chunk_rows: int,
) -> np.ndarray:
    """Select deterministic top baseline edges without a full-volume index array."""
    if base.unit_type == "global":
        return np.flatnonzero(base.candidate_mask.ravel())
    beta = np.asarray(base.beta, dtype=float)
    n_cells, n_tfs, n_targets = beta.shape
    rows_per_cell = max(n_tfs * n_targets, 1)
    cells_per_chunk = max(1, int(chunk_rows) // rows_per_cell)
    retained = np.empty(0, dtype=np.int64)
    retained_scores = np.empty(0, dtype=float)
    for start in range(0, n_cells, cells_per_chunk):
        stop = min(start + cells_per_chunk, n_cells)
        coefficients = beta[start:stop]
        included = np.broadcast_to(base.candidate_mask[None, :, :], coefficients.shape) | (
            coefficients != 0.0
        )
        local = np.flatnonzero(included.ravel())
        if not len(local):
            continue
        local_indices = local + start * rows_per_cell
        local_scores = np.abs(coefficients.ravel()[local])
        local_indices = _top_k_flat_indices(local_indices, local_scores, maximum_rows)
        local_scores = np.abs(beta.ravel()[local_indices])
        combined_indices = np.concatenate([retained, local_indices])
        combined_scores = np.concatenate([retained_scores, local_scores])
        retained = _top_k_flat_indices(combined_indices, combined_scores, maximum_rows)
        retained_scores = np.abs(beta.ravel()[retained])
    return retained


def _base_grn_frame_from_indices(
    bundle: InputBundle,
    prior: PriorResult,
    base: BaseGRNResult,
    flat_indices: np.ndarray,
) -> pd.DataFrame:
    region_column = next((column for column in bundle.metadata.columns if column == "region"), None)
    region_values = np.asarray(
        bundle.metadata[region_column].astype(str).tolist()
        if region_column
        else ["global"] * len(bundle.expression),
        dtype=object,
    )
    beta = np.asarray(base.beta, dtype=float)
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    cell_index, tf_index, target_index = np.unravel_index(flat_indices, beta.shape)
    tf_names = np.asarray(base.tf_names, dtype=object)
    target_names = np.asarray(base.target_names, dtype=object)
    cell_ids = bundle.expression.index.astype(str).to_numpy(dtype=object)
    confidence = prior.confidence.reindex(
        index=base.tf_names, columns=base.target_names
    ).to_numpy(dtype=float)
    coefficients = beta[cell_index, tf_index, target_index]
    return pd.DataFrame(
        {
            "cell_id": np.full(len(cell_index), "__all__") if base.unit_type == "global" else cell_ids[cell_index],
            "region": np.full(len(cell_index), "__all__") if base.unit_type == "global" else region_values[cell_index],
            "tf": tf_names[tf_index],
            "target": target_names[target_index],
            "beta0": coefficients,
            "score": np.abs(coefficients),
            "prior_confidence": confidence[tf_index, target_index],
        }
    )


def _base_grn_frame(
    bundle: InputBundle,
    spatial: SpatialResult,
    prior: PriorResult,
    base: BaseGRNResult,
    maximum_rows: int,
) -> pd.DataFrame:
    flat_indices = _base_grn_flat_indices(base, maximum_rows, 250_000)
    return _base_grn_frame_from_indices(bundle, prior, base, flat_indices)


def _load_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "Parquet output requires pyarrow. Activate the jumpniche environment and run: "
            "python -m pip install pyarrow"
        ) from error
    return pa, pq


def validate_output_dependencies(config: OutputConfig) -> None:
    """Fail before model fitting when an enabled output backend is unavailable."""

    if config.save_parquet:
        _load_pyarrow()


def _write_frame_parquet(
    path: Path,
    frame: pd.DataFrame,
    compression: str,
    row_group_size: int,
) -> None:
    temporary_path = path.with_name(f".{path.stem}.writing{path.suffix}")
    parquet_frame = frame.copy(deep=False)
    for column in parquet_frame.select_dtypes(include=["object"]).columns:
        parquet_frame[column] = parquet_frame[column].astype("string")
    parquet_frame.to_parquet(
        temporary_path,
        engine="pyarrow",
        compression=compression,
        index=False,
        row_group_size=max(int(row_group_size), 1),
    )
    temporary_path.replace(path)


def _write_base_grn_parquet(
    path: Path,
    bundle: InputBundle,
    prior: PriorResult,
    base: BaseGRNResult,
    compression: str,
    row_group_size: int,
    maximum_rows: int,
) -> int:
    """Stream the bounded top baseline GRN into Parquet row groups."""

    pa, pq = _load_pyarrow()
    flat_indices = _base_grn_flat_indices(base, maximum_rows, row_group_size)
    chunk_size = max(int(row_group_size), 1)
    temporary_path = path.with_name(f".{path.stem}.writing{path.suffix}")
    writer = None
    try:
        for start in range(0, len(flat_indices), chunk_size):
            chunk = _base_grn_frame_from_indices(
                bundle,
                prior,
                base,
                flat_indices[start : start + chunk_size],
            )
            for column in chunk.select_dtypes(include=["object"]).columns:
                chunk[column] = chunk[column].astype("string")
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression=compression,
                )
            writer.write_table(table, row_group_size=chunk_size)
        if writer is None:
            empty = _base_grn_frame_from_indices(
                bundle, prior, base, np.empty(0, dtype=np.int64)
            )
            _write_frame_parquet(path, empty, compression, chunk_size)
            return 0
    finally:
        if writer is not None:
            writer.close()
    temporary_path.replace(path)
    return int(len(flat_indices))


def _write_parquet_outputs(
    parquet_dir: Path,
    frames: dict[str, pd.DataFrame],
    bundle: InputBundle,
    prior: PriorResult,
    base: BaseGRNResult,
    config: OutputConfig,
    verbose: bool,
) -> None:
    parquet_dir = ensure_directory(parquet_dir)
    manifest_rows: list[dict[str, object]] = []

    base_path = parquet_dir / "base_grn.parquet"
    base_rows = _write_base_grn_parquet(
        base_path,
        bundle,
        prior,
        base,
        config.parquet_compression,
        config.parquet_row_group_size,
        config.maximum_grn_rows,
    )
    manifest_rows.append(
        {
            "table": "base_grn",
            "file": base_path.name,
            "rows": base_rows,
            "columns": 7,
            "compression": config.parquet_compression,
        }
    )
    log_message(verbose, f"[output parquet] base_grn: {base_rows} rows")

    for table_name, frame in frames.items():
        path = parquet_dir / f"{table_name}.parquet"
        _write_frame_parquet(
            path,
            frame,
            config.parquet_compression,
            config.parquet_row_group_size,
        )
        manifest_rows.append(
            {
                "table": table_name,
                "file": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "compression": config.parquet_compression,
            }
        )
        log_message(verbose, f"[output parquet] {table_name}: {len(frame)} rows")

    manifest = pd.DataFrame(manifest_rows)
    _write_frame_parquet(
        parquet_dir / "manifest.parquet",
        manifest,
        config.parquet_compression,
        config.parquet_row_group_size,
    )


def _excel_sheet_chunks(
    sheet: str,
    frame: pd.DataFrame,
    maximum_data_rows: int,
) -> list[tuple[str, pd.DataFrame]]:
    """Split a table without exceeding Excel's hard worksheet row limit."""

    rows_per_sheet = max(1, min(int(maximum_data_rows), EXCEL_MAX_DATA_ROWS))
    safe_name = safe_sheet_name(sheet)
    if len(frame) <= rows_per_sheet:
        return [(safe_name, frame)]

    chunks: list[tuple[str, pd.DataFrame]] = []
    for part, start in enumerate(range(0, len(frame), rows_per_sheet), start=1):
        suffix = f"_{part:03d}"
        chunk_name = f"{safe_name[:31 - len(suffix)]}{suffix}"
        chunks.append((chunk_name, frame.iloc[start : start + rows_per_sheet]))
    return chunks


def _excel_cell(value: object) -> object:
    """Convert pandas/numpy scalar values to write-only openpyxl values."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or pd.isna(value):
        return None
    return value


def _write_streaming_workbook(
    workbook_path: Path,
    frames: dict[str, pd.DataFrame],
    maximum_data_rows: int,
    verbose: bool = False,
) -> None:
    """Write an atomic, memory-efficient workbook with automatic sheet splitting."""

    chunks: list[tuple[str, str, int, int, pd.DataFrame]] = []
    for table, frame in frames.items():
        start = 0
        for sheet_name, chunk in _excel_sheet_chunks(table, frame, maximum_data_rows):
            chunks.append((table, sheet_name, start, start + len(chunk), chunk))
            start += len(chunk)

    manifest = pd.DataFrame(
        [
            {
                "table": table,
                "sheet": sheet_name,
                "start_row_zero_based": start,
                "end_row_exclusive": end,
                "data_rows": len(chunk),
            }
            for table, sheet_name, start, end, chunk in chunks
        ]
    )
    temporary_path = workbook_path.with_name(
        f"{workbook_path.stem}.writing{workbook_path.suffix}"
    )
    workbook = Workbook(write_only=True)
    workbook_frames = [("excel_manifest", manifest)] + [
        (sheet_name, chunk) for _, sheet_name, _, _, chunk in chunks
    ]
    for sheet_name, frame in workbook_frames:
        worksheet = workbook.create_sheet(title=sheet_name)
        worksheet.append([str(column) for column in frame.columns])
        for row in frame.itertuples(index=False, name=None):
            worksheet.append([_excel_cell(value) for value in row])
        log_message(verbose, f"[output xlsx] {sheet_name}: {len(frame)} rows")
    workbook.save(temporary_path)
    temporary_path.replace(workbook_path)


def write_results(
    config: ProjectConfig,
    bundle: InputBundle,
    spatial: SpatialResult,
    prior: PriorResult,
    base: BaseGRNResult,
    communication: CommunicationResult,
    rewiring: RewiringResult,
    timings: dict[str, float],
) -> Path:
    output_dir = ensure_directory(Path(config.output.output_dir))
    workbook_path = output_dir / config.output.workbook_name
    run_info = pd.DataFrame(
        {
            "key": [
                "method", "version", "evidence_mode", "cells", "genes", "candidate_tfs",
                "candidate_targets", "lr_pathways", "parameterization", "random_seed",
            ],
            "value": [
                config.output.method_name, "0.1.0",
                "Perturbation-enhanced" if bundle.perturbations is not None else (
                    "Activity-enhanced" if any(x is not None for x in (bundle.ligand_activity, bundle.receptor_activity, bundle.tf_activity)) else "Core"
                ),
                len(bundle.expression), len(bundle.expression.columns), len(base.tf_names),
                len(base.target_names), len(bundle.lr_pairs), config.rewiring.parameterization,
                config.runtime.random_seed,
            ],
        }
    )
    run_info["value"] = run_info["value"].astype(str)
    history_rows = len(base.history) + len(rewiring.history)
    run_info = pd.concat(
        [
            run_info,
            pd.DataFrame(
                {
                    "key": [
                        "convergence_history_rows",
                        "convergence_history_excel_preview_limit",
                        "excel_rows_per_table_limit",
                        "convergence_history_full",
                    ],
                    "value": [
                        str(history_rows),
                        str(config.output.maximum_history_excel_rows),
                        str(config.output.maximum_excel_rows_per_table),
                        f"{config.output.parquet_dir_name}/convergence_history.parquet"
                        if config.output.save_parquet
                        else "Excel (complete when Parquet disabled)",
                    ],
                }
            ),
        ],
        ignore_index=True,
    )
    timing_frame = pd.DataFrame(
        [{"module": module, "seconds": seconds} for module, seconds in timings.items()]
    )
    graph = spatial.graph
    spatial_edges = pd.DataFrame(
        {
            "source_id": bundle.expression.index.to_numpy()[graph.source],
            "target_id": bundle.expression.index.to_numpy()[graph.target],
            "distance": graph.distance,
            "distance_weight": graph.distance_weight,
            "state_weight": graph.state_weight,
            "weight": graph.weight,
        }
    )
    if len(spatial_edges) > config.output.maximum_spatial_edge_rows:
        spatial_edges = spatial_edges.nlargest(config.output.maximum_spatial_edge_rows, "weight")

    frames: dict[str, pd.DataFrame] = {
        "run_info": run_info,
        "input_audit": bundle.audit,
        "timings": timing_frame,
        "spatial_genes": spatial.spatial_statistics,
        "spatial_edges": spatial_edges,
        "base_grn": _base_grn_frame(
            bundle, spatial, prior, base, config.output.maximum_grn_rows
        ),
        "grn_edges": _ensure_schema(rewiring.grn_edges, UNIFIED_SCHEMAS["grn_edges"]),
        "lr_gene": _ensure_schema(rewiring.lr_gene, UNIFIED_SCHEMAS["lr_gene"]),
        "receptor_paths": _ensure_schema(rewiring.receptor_paths, UNIFIED_SCHEMAS["receptor_paths"]),
        "cci_edges": _ensure_schema(communication.cci_edges, UNIFIED_SCHEMAS["cci_edges"]),
        "regulon_activity": _ensure_schema(rewiring.regulon_activity, UNIFIED_SCHEMAS["regulon_activity"]),
        "causal_edges": _ensure_schema(rewiring.causal_edges, UNIFIED_SCHEMAS["causal_edges"]),
        "convergence": pd.concat([base.diagnostics, rewiring.diagnostics], ignore_index=True),
        "convergence_history": pd.concat([base.history, rewiring.history], ignore_index=True),
    }
    if config.output.save_parquet:
        parquet_frames = {
            table_name: frame
            for table_name, frame in frames.items()
            if table_name != "base_grn"
        }
        _write_parquet_outputs(
            output_dir / config.output.parquet_dir_name,
            parquet_frames,
            bundle,
            prior,
            base,
            config.output,
            config.runtime.verbose,
        )
    if len(frames["grn_edges"]) > config.output.maximum_grn_rows:
        frames["grn_edges"] = frames["grn_edges"].nlargest(
            config.output.maximum_grn_rows, "score"
        )
    if config.output.save_intermediate_csv:
        csv_dir = ensure_directory(output_dir / "csv")
        for sheet, frame in frames.items():
            csv_path = csv_dir / f"{sheet}.csv"
            temporary_csv_path = csv_dir / f".{sheet}.writing.csv"
            frame.to_csv(temporary_csv_path, index=False)
            temporary_csv_path.replace(csv_path)
            log_message(
                config.runtime.verbose,
                f"[output csv] {sheet}: {len(frame)} rows",
            )
    if config.output.save_parquet:
        frames["convergence_history"] = frames["convergence_history"].head(max(0, config.output.maximum_history_excel_rows))
        excel_limit = max(int(config.output.maximum_excel_rows_per_table), 0)
        for table_name, frame in list(frames.items()):
            if len(frame) <= excel_limit:
                continue
            if excel_limit == 0:
                frames[table_name] = frame.head(0)
            elif "score" in frame.columns:
                # Stable ordering makes the human-readable preview reproducible.
                frames[table_name] = frame.sort_values(
                    "score", ascending=False, kind="mergesort"
                ).head(excel_limit)
            else:
                frames[table_name] = frame.head(excel_limit)
            log_message(
                config.runtime.verbose,
                f"[output xlsx preview] {table_name}: {len(frames[table_name])} of {len(frame)} rows; full table is Parquet",
            )
    _write_streaming_workbook(
        workbook_path,
        frames,
        config.output.maximum_excel_data_rows_per_sheet,
        config.runtime.verbose,
    )
    return workbook_path
