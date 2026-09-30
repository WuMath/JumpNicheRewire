"""Shared Excel contract for baseline adapters only.

This module deliberately does not import ``jump_niche_rewire``.  Baseline jobs
can therefore run in a separate Python environment and only exchange workbooks
with the evaluation stage.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMAS = {
    "grn_edges": ["method", "unit_type", "cell_id", "region", "tf", "target", "score", "signed_score"],
    "lr_gene": ["method", "receiver_id", "region", "lr_id", "ligand", "receptor", "target", "score", "signed_score"],
    "receptor_paths": ["method", "sender_id", "receiver_id", "sender_region", "receiver_region", "lr_id", "ligand", "receptor", "tf", "target", "communication_score", "delta_beta", "path_score", "evidence_level", "evidence_label"],
    "cci_edges": ["method", "sender_id", "receiver_id", "lr_id", "ligand", "receptor", "score"],
    "regulon_activity": ["method", "cell_id", "region", "tf", "activity"],
    "causal_edges": ["method", "region", "lr_id", "tf", "target", "gamma_estimate", "causal_estimand", "evidence_level", "evidence_label"],
}


def run_official(command: tuple[str, ...], working_directory: Path, enabled: bool) -> None:
    if not enabled:
        return
    if not command:
        raise ValueError("run_official_command=True but official_command is empty")
    subprocess.run(list(command), cwd=working_directory, check=True, shell=False)


def read_table(path: Path, sheet: str | int = 0, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Expected official-method output is missing: {path}")
        return pd.DataFrame()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def rename_first_match(frame: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    output = frame.copy()
    lower = {str(column).lower(): column for column in output.columns}
    renames: dict[object, str] = {}
    for canonical, candidates in aliases.items():
        if canonical in output.columns:
            continue
        for candidate in candidates:
            if candidate.lower() in lower:
                renames[lower[candidate.lower()]] = canonical
                break
    return output.rename(columns=renames)


def canonical_task(
    frame: pd.DataFrame,
    method: str,
    task: str,
    aliases: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=SCHEMAS[task])
    output = rename_first_match(frame, aliases)
    output["method"] = method
    if "score" in output.columns:
        output["score"] = pd.to_numeric(output["score"], errors="coerce").fillna(0.0).abs()
    if task == "grn_edges":
        if "signed_score" not in output.columns and "score" in output.columns:
            output["signed_score"] = output["score"]
        if "unit_type" not in output.columns:
            output["unit_type"] = (
                np.where(output["cell_id"].notna(), "cell", "region")
                if "cell_id" in output.columns
                else "region"
            )
        if "region" not in output.columns:
            output["region"] = "global"
    if task == "lr_gene" and "signed_score" not in output.columns and "score" in output.columns:
        output["signed_score"] = output["score"]
    if task == "cci_edges":
        if "lr_id" not in output.columns:
            output["lr_id"] = "__all__"
        if "ligand" not in output.columns:
            output["ligand"] = "__all__"
        if "receptor" not in output.columns:
            output["receptor"] = "__all__"
    for column in SCHEMAS[task]:
        if column not in output.columns:
            output[column] = np.nan
    return output.loc[:, SCHEMAS[task] + [column for column in output.columns if column not in SCHEMAS[task]]]


def write_workbook(method: str, output_path: Path, tasks: dict[str, pd.DataFrame], metadata: dict[str, object]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_info = {"method": method, **metadata}
    run_info_frame = pd.DataFrame({"key": list(run_info), "value": list(run_info.values())})
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        run_info_frame.to_excel(writer, sheet_name="run_info", index=False)
        for task, schema in SCHEMAS.items():
            frame = tasks.get(task, pd.DataFrame(columns=schema))
            frame.to_excel(writer, sheet_name=task, index=False)
    return output_path
