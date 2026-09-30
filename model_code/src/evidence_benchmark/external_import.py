"""Canonicalize files from methods that were run independently of this project."""

from __future__ import annotations

import json
import time
import traceback

import numpy as np
import pandas as pd

from .common import read_table, write_json
from .config import EXTERNAL_IMPORT_CONFIG, ExternalImportConfig, ExternalResultSpec


def _grn(spec: ExternalResultSpec) -> pd.DataFrame:
    frame = read_table(spec.grn_file)
    required = (spec.tf_column, spec.target_column, spec.score_column)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{spec.method} GRN misses columns {missing}")
    result = pd.DataFrame({
        "method": spec.method,
        "tf": frame[spec.tf_column].astype(str),
        "target": frame[spec.target_column].astype(str),
        "score": pd.to_numeric(frame[spec.score_column], errors="coerce"),
    })
    if spec.signed_score_column in frame:
        result["signed_score"] = pd.to_numeric(frame[spec.signed_score_column], errors="coerce")
    else:
        result["signed_score"] = np.nan
    return result.dropna(subset=["score"])


def _cci(spec: ExternalResultSpec) -> pd.DataFrame:
    frame = read_table(spec.cci_file)
    columns = (
        spec.sender_column, spec.receiver_column, spec.ligand_column,
        spec.receptor_column, spec.score_column,
    )
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{spec.method} CCI misses columns {missing}")
    if spec.cci_unit_type not in {"cell", "cell_type"}:
        raise ValueError("cci_unit_type must be 'cell' or 'cell_type'")
    sender_name = "sender_id" if spec.cci_unit_type == "cell" else "sender_cell_type"
    receiver_name = "receiver_id" if spec.cci_unit_type == "cell" else "receiver_cell_type"
    result = pd.DataFrame({
        "method": spec.method,
        sender_name: frame[spec.sender_column].astype(str),
        receiver_name: frame[spec.receiver_column].astype(str),
        "ligand": frame[spec.ligand_column].astype(str),
        "receptor": frame[spec.receptor_column].astype(str),
        "score": pd.to_numeric(frame[spec.score_column], errors="coerce"),
    })
    result["lr_id"] = result.ligand + "__" + result.receptor
    return result.dropna(subset=["score"])


def _one(spec: ExternalResultSpec, config: ExternalImportConfig) -> dict[str, object]:
    output = config.output_root / spec.method / spec.dataset_id
    parquet = output / "parquet"
    status_path = output / "run_status.json"
    if status_path.exists() and not config.overwrite:
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("state") == "completed":
            print(f"[unchanged external] {spec.method} / {spec.dataset_id}")
            return previous
    status: dict[str, object] = {
        "method": spec.method, "dataset_id": spec.dataset_id, "state": "running",
        "adapter": "external_file_only_no_jumpniche_import",
    }
    write_json(status_path, status)
    started = time.perf_counter()
    try:
        parquet.mkdir(parents=True, exist_ok=True)
        tasks = []
        if spec.grn_file is not None:
            _grn(spec).to_parquet(parquet / "grn_edges.parquet", index=False)
            tasks.append("grn_edges")
        if spec.cci_file is not None:
            _cci(spec).to_parquet(parquet / "cci_edges.parquet", index=False)
            tasks.append("cci_edges")
        if not tasks:
            raise ValueError("At least one of grn_file or cci_file is required")
        status.update(state="completed", tasks=tasks, runtime_seconds=time.perf_counter() - started)
    except Exception as error:
        status.update(
            state="failed", runtime_seconds=time.perf_counter() - started,
            error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc(),
        )
    write_json(status_path, status)
    return status


def run_external_import(config: ExternalImportConfig = EXTERNAL_IMPORT_CONFIG):
    if not config.specifications:
        raise ValueError("Add ExternalResultSpec entries in evidence_benchmark/config.py first")
    rows = [_one(spec, config) for spec in config.specifications]
    inventory = config.output_root / "external_import_inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    print(f"[external imports] {inventory}")
    return inventory



