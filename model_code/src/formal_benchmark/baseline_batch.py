"""Independent official-baseline batch adapter.

This module never imports ``jump_niche_rewire``.  Official methods run in their
own environments and exchange only files through ``data/formal_benchmark`` and
``02_runs/baselines_raw``.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import traceback

import pandas as pd

from baselines.common import canonical_task, read_table, write_workbook

from .config import BaselineBatchConfig
from .state import completion_matches, stable_hash, utc_now, write_completion, write_json_atomic


TASK_FILES = {
    "grn_edges": "raw_grn_name",
    "lr_gene": "raw_lr_gene_name",
    "receptor_paths": "raw_receptor_paths_name",
    "cci_edges": "raw_cci_name",
    "regulon_activity": "raw_activity_name",
    "causal_edges": "raw_causal_name",
}


DEFAULT_ALIASES = {
    "grn_edges": {
        "cell_id": ("cell", "cellid", "spot", "unit_id"),
        "region": ("cluster", "cell_type", "region"),
        "tf": ("regulator", "source", "tf"),
        "target": ("gene", "target"),
        "score": ("weight", "edge_weight", "importance", "score", "refined_weight"),
        "signed_score": ("signed_weight", "coefficient"),
    },
    "lr_gene": {
        "receiver_id": ("cell", "receiver", "receiver_id"),
        "region": ("cluster", "region"),
        "lr_id": ("lr", "lr_pair", "lr_id"),
        "ligand": ("ligand",),
        "receptor": ("receptor",),
        "target": ("gene", "target"),
        "score": ("regulatory_score", "weight", "score"),
    },
    "receptor_paths": {
        "receiver_region": ("cluster", "region", "receiver_region"),
        "lr_id": ("lr", "lr_id"),
        "ligand": ("ligand",),
        "receptor": ("receptor",),
        "tf": ("regulator", "tf"),
        "target": ("gene", "target"),
        "path_score": ("score", "importance", "weight"),
    },
    "cci_edges": {
        "sender_id": ("sender", "source", "cell1"),
        "receiver_id": ("receiver", "target", "target_cell", "cell2"),
        "lr_id": ("lr", "lr_id"),
        "score": ("probability", "weight", "score"),
    },
    "regulon_activity": {
        "cell_id": ("cell", "spot", "cell_id"),
        "region": ("cluster", "region"),
        "tf": ("regulator", "tf"),
        "activity": ("auc", "activity", "score"),
    },
    "causal_edges": {
        "region": ("cluster", "region"),
        "lr_id": ("lr", "lr_id"),
        "tf": ("regulator", "tf"),
        "target": ("gene", "target"),
        "gamma_estimate": ("effect", "coefficient", "gamma_estimate"),
    },
}


def _expanded_command(config: BaselineBatchConfig, input_dir: Path, raw_dir: Path, dataset_id: str) -> list[str]:
    substitutions = {
        "input_dir": str(input_dir.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "dataset_id": dataset_id,
    }
    return [token.format(**substitutions) for token in config.official_command_template]


def _freeze_protocol(config: BaselineBatchConfig) -> Path:
    path = config.profile_benchmark_root / "02_runs" / "frozen_baseline_protocols" / f"{config.method}.json"
    payload = {
        "benchmark_profile": config.profile,
        "method": config.method,
        "official_repository_dir": config.official_repository_dir,
        "official_command_template": config.official_command_template,
        "raw_file_contract": {task: getattr(config, field) for task, field in TASK_FILES.items()},
        "protocol_signature": stable_hash(asdict(config)),
        "frozen_at_utc": utc_now(),
        "test_metrics_inspected": False,
    }
    if path.exists() and not config.overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_signature") != payload["protocol_signature"]:
            raise FileExistsError(f"Different frozen protocol exists: {path}")
        return path
    write_json_atomic(path, payload)
    print(f"[baseline protocol frozen] {path}")
    return path


def run_baseline_batch(
    config: BaselineBatchConfig,
    required_tasks: tuple[str, ...],
    aliases: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> Path:
    if config.phase == "freeze":
        return _freeze_protocol(config)
    if config.phase == "test":
        lock = config.profile_benchmark_root / "02_runs" / "frozen_baseline_protocols" / f"{config.method}.json"
        if not config.unlock_test or not lock.exists():
            raise PermissionError("Freeze the baseline protocol, then set unlock_test=True before test")
    if config.phase not in {"train", "validation", "test"}:
        raise ValueError("Baseline phase must be train, validation, freeze, or test")
    registry_path = config.registry_path
    if not registry_path.exists():
        raise FileNotFoundError(f"Run stage 1 first: {registry_path}")
    registry = pd.read_csv(registry_path)
    if set(registry["benchmark_profile"].astype(str)) != {config.profile}:
        raise ValueError(f"Registry {registry_path} is not isolated to profile {config.profile!r}")
    registry = registry[registry["split"] == config.phase]
    aliases = aliases or DEFAULT_ALIASES
    statuses: list[dict[str, object]] = []
    for _, row in registry.iterrows():
        dataset_id = str(row["dataset_id"])
        input_dir = Path(str(row["dataset_dir"])) / "baseline_input"
        raw_dir = config.profile_benchmark_root / "02_runs" / "baselines_raw" / config.method / dataset_id
        output_dir = config.profile_benchmark_root / "02_runs" / "baselines" / config.method / dataset_id
        output_path = output_dir / f"{config.method}_results.xlsx"
        status_path = output_dir / "run_status.json"
        signature = stable_hash(
            {
                "dataset_signature": row["signature"],
                "method": config.method,
                "protocol": asdict(config),
            }
        )
        if not config.overwrite and completion_matches(output_dir / ".complete.json", signature, (status_path, output_path)):
            statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status: dict[str, object] = {
            "dataset_id": dataset_id,
            "benchmark_profile": config.profile,
            "split": config.phase,
            "method": config.method,
            "state": "running",
            "signature": signature,
            "raw_dir": str(raw_dir),
            "started_at_utc": utc_now(),
        }
        write_json_atomic(status_path, status)
        try:
            if config.launch_official:
                command = _expanded_command(config, input_dir, raw_dir, dataset_id)
                if not command:
                    raise ValueError("launch_official=True but official_command_template is empty")
                subprocess.run(command, cwd=config.official_repository_dir, check=True, shell=False)
            tasks: dict[str, pd.DataFrame] = {}
            found: set[str] = set()
            for task, field in TASK_FILES.items():
                raw_path = raw_dir / str(getattr(config, field))
                frame = read_table(raw_path, required=False)
                if not frame.empty:
                    tasks[task] = canonical_task(frame, config.method, task, aliases.get(task, DEFAULT_ALIASES[task]))
                    found.add(task)
            missing = sorted(set(required_tasks) - found)
            if missing:
                raise FileNotFoundError(
                    f"Missing required official outputs {missing}. Put raw files in {raw_dir} "
                    "or enable launch_official after configuring its fixed wrapper."
                )
            write_workbook(
                config.method,
                output_path,
                tasks,
                {
                    "dataset_id": dataset_id,
                    "split": config.phase,
                    "adapter": "official-output batch adapter",
                    "raw_dir": raw_dir,
                },
            )
            status.update(
                {
                    "state": "completed",
                    "runtime_seconds": time.perf_counter() - started,
                    "workbook_path": str(output_path),
                    "tasks_found": sorted(found),
                    "completed_at_utc": utc_now(),
                }
            )
            write_json_atomic(status_path, status)
            write_completion(output_dir / ".complete.json", signature, {"state": "completed"})
        except Exception as error:
            status.update(
                {
                    "state": "failed",
                    "runtime_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "completed_at_utc": utc_now(),
                }
            )
            write_json_atomic(status_path, status)
            print(f"[baseline missing/failed] {config.method} / {dataset_id}: {error}")
        statuses.append(status)
    status_frame = pd.json_normalize(statuses)
    registry_out = config.profile_benchmark_root / "02_runs" / f"{config.method}_{config.phase}_registry.csv"
    registry_out.parent.mkdir(parents=True, exist_ok=True)
    status_frame.to_csv(registry_out, index=False)
    print(f"[baseline batch] {config.method}: {registry_out}")
    return registry_out
