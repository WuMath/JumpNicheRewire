"""Write leakage-safe input/output job sheets for independently run methods."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import JOB_CONFIG, EvidenceJobConfig, WORKSPACE_ROOT


def run_jobs(config: EvidenceJobConfig = JOB_CONFIG) -> Path:
    if not config.registry_path.exists():
        raise FileNotFoundError(f"Prepare evidence data first: {config.registry_path}")
    registry = pd.read_csv(config.registry_path)
    config.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for method_spec in config.methods:
        method = method_spec.method
        method_rows = []
        for row in registry.itertuples(index=False):
            if row.evaluation_scope == "sccigen_cci":
                dataset_tasks = ("cci_edges",)
            elif row.evaluation_scope == "real_pseudotruth":
                dataset_tasks = ("grn_edges", "cci_edges")
            else:
                dataset_tasks = ("grn_edges",)
            tasks = tuple(task for task in dataset_tasks if task in method_spec.tasks)
            if not tasks:
                continue
            item = {
                "method": method, "dataset_id": row.dataset_id,
                "evidence_track": row.evidence_track, "evaluation_scope": row.evaluation_scope,
                "input_dir": str(Path(row.dataset_dir) / "baseline_input"),
                "baseline_input_dir": str(Path(row.dataset_dir) / "baseline_input"),
                "required_output_tasks": ";".join(tasks),
                "raw_output_dir": str(config.output_root.parent / "02_runs/baselines_raw" / method / row.dataset_id),
                "unified_output_dir": str(config.output_root.parent / "02_runs/baselines" / method / row.dataset_id),
                "truth_exposed_to_method": False,
            }
            method_rows.append(item)
            rows.append(item)
        pd.DataFrame(method_rows).to_csv(config.output_root / f"{method}_evidence_jobs.csv", index=False)
    all_path = config.output_root / "all_evidence_jobs.csv"
    pd.DataFrame(rows).to_csv(all_path, index=False)
    print(f"[evidence jobs] {all_path}")
    return all_path



