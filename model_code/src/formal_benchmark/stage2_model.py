"""Stage 2A: tune, freeze, and test JumpNicheRewire without test leakage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd

from .config import MODEL_CONFIG, ModelStageConfig, PROJECT_ROOT, ParameterSet
from .state import completion_matches, stable_hash, utc_now, write_completion, write_json_atomic


SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_niche_rewire import JumpNicheRewirePipeline
from jump_niche_rewire.config import CONFIG as BASE_CONFIG
from jump_niche_rewire.convergence import summarize_convergence, ConvergenceError


class _PeakRSS:
    """Sample resident memory, including NumPy allocations, when psutil exists."""

    def __init__(self) -> None:
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PeakRSS":
        try:
            import psutil
        except ImportError:
            return self
        process = psutil.Process()
        self.peak_bytes = process.memory_info().rss

        def sample() -> None:
            while not self._stop.wait(0.2):
                self.peak_bytes = max(self.peak_bytes or 0, process.memory_info().rss)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _read_registry(config: ModelStageConfig) -> pd.DataFrame:
    path = config.registry_path
    if not path.exists():
        raise FileNotFoundError(f"Run stage 1 first: {path}")
    registry = pd.read_csv(path)
    required = {"dataset_id", "benchmark_profile", "split", "dataset_dir", "signature"}
    if not required.issubset(registry.columns):
        raise ValueError(f"Dataset registry lacks columns: {sorted(required - set(registry.columns))}")
    profiles = set(registry["benchmark_profile"].astype(str))
    if profiles != {config.profile}:
        raise ValueError(f"Registry {path} contains profiles {sorted(profiles)}, expected only {config.profile!r}")
    return registry


def _convergence_summary(result: object, minimum_rate: float = 0.90) -> dict[str, object]:
    return summarize_convergence(result.base, result.rewiring, minimum_rate)


def _solver_source_signature() -> str:
    from .state import file_sha256
    files = {f"model/{path.name}": file_sha256(path) for path in sorted((SRC / "jump_niche_rewire").glob("*.py"))}
    for path in (SRC / "evaluation" / "metrics.py", SRC / "evaluation" / "evaluator.py"):
        files[f"evaluation/{path.name}"] = file_sha256(path)
    files["formal_stage2"] = file_sha256(Path(__file__))
    files["formal_config"] = file_sha256(Path(__file__).with_name("config.py"))
    # Do not hash active.py or editable per-dataset operation switches:
    # changing unlock_test must not invalidate a valid frozen experiment.
    for relative in ('configs/common/model.py', 'configs/self_sim_data/benchmark_settings.py'):
        path = PROJECT_ROOT / relative
        files[relative] = file_sha256(path)
    return stable_hash(files)


def _validation_grn_auprc(predictions: pd.DataFrame, truth_path: Path) -> float:
    # Imported here so test inference never imports or invokes the evaluation code.
    from evaluation.evaluator import evaluate_grn

    truth = pd.read_csv(truth_path)
    summary, _ = evaluate_grn("JumpNicheRewire", predictions, truth)
    values = [
        row["value"]
        for row in summary
        if row["task"] == "GRN recovery" and row["metric"] == "Average Precision"
    ]
    return float(values[0]) if values else np.nan


def _configure_model(
    config: ModelStageConfig,
    parameters: ParameterSet,
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    method_name: str,
) -> object:
    model = deepcopy(config.project_model if config.project_model is not None else BASE_CONFIG)
    model.data.input_dir = dataset_dir
    model.output.output_dir = output_dir
    model.output.workbook_name = f"{method_name}_results.xlsx"
    model.output.method_name = method_name
    model.output.save_parquet = config.save_parquet
    model.runtime.random_seed = seed
    model.runtime.verbose = config.verbose
    model.runtime.minimum_target_convergence_rate = config.minimum_convergence_rate
    model.runtime.require_convergence = True
    model.spatial.spatial_permutations = parameters.spatial_permutations
    model.base_grn.maximum_tfs = config.maximum_tfs
    model.base_grn.maximum_targets = config.maximum_targets
    model.base_grn.l1_penalty = parameters.l1_penalty
    model.base_grn.fusion_penalty = parameters.fusion_penalty
    model.base_grn.admm_rho = parameters.admm_rho
    model.base_grn.admm_max_iterations = parameters.admm_max_iterations
    model.rewiring.maximum_active_lr = config.maximum_active_lr
    model.rewiring.maximum_interaction_tfs = config.maximum_interaction_tfs
    model.rewiring.group_penalty = parameters.group_penalty
    model.rewiring.ridge_penalty = parameters.ridge_penalty
    model.rewiring.fista_max_iterations = parameters.fista_max_iterations
    model.rewiring.joint_iterations = parameters.joint_iterations
    model.rewiring.iteration_tolerance = parameters.iteration_tolerance
    return model


def _portable_effective_model_config(model: object) -> dict[str, object]:
    """Keep numerical settings in signatures while excluding machine paths."""

    payload = asdict(model)
    payload.get("data", {}).pop("input_dir", None)
    payload.get("output", {}).pop("output_dir", None)
    return payload


def _run_signature(
    config: ModelStageConfig,
    row: pd.Series,
    parameters: ParameterSet,
    method_name: str,
    model: object,
) -> tuple[str, str]:
    source_signature = _solver_source_signature()
    signature = stable_hash(
        {
            "dataset_signature": row["signature"],
            "parameters": parameters,
            "method": method_name,
            "benchmark_profile": config.profile,
            "stage2_version": "benchmark_validity_v3",
            "solver_source": source_signature,
            "effective_model_config": _portable_effective_model_config(model),
        }
    )
    return signature, source_signature


def _run_one(
    config: ModelStageConfig,
    row: pd.Series,
    parameters: ParameterSet,
    method_name: str,
    allow_validation_metric: bool,
) -> dict[str, object]:
    dataset_id = str(row["dataset_id"])
    dataset_dir = Path(str(row["dataset_dir"]))
    seed = int(row["seed"])
    output_dir = config.profile_benchmark_root / "02_runs" / "jumpniche" / parameters.parameter_id / dataset_id
    status_path = output_dir / "run_status.json"
    workbook_path = output_dir / f"{method_name}_results.xlsx"
    model = _configure_model(config, parameters, dataset_dir, output_dir, seed, method_name)
    signature, source_signature = _run_signature(
        config, row, parameters, method_name, model
    )
    required = (status_path, workbook_path, output_dir / "effective_config.json")
    if config.save_parquet:
        required += (output_dir / model.output.parquet_dir_name / "convergence_history.parquet",)
    if not config.overwrite and completion_matches(output_dir / ".complete.json", signature, required):
        cached = json.loads(status_path.read_text(encoding="utf-8"))
        if cached.get("state") == "completed" and cached.get("convergence_passed") is True:
            return cached
    output_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    started_clock = time.perf_counter()
    status: dict[str, object] = {
        "dataset_id": dataset_id,
        "benchmark_profile": config.profile,
        "split": row["split"],
        "scenario": row["scenario"],
        "seed": seed,
        "method": method_name,
        "parameter_id": parameters.parameter_id,
        "signature": signature,
        "started_at_utc": started,
        "state": "running",
        "solver_source_signature": source_signature,
        "parameter_signature": stable_hash(parameters),
    }
    write_json_atomic(status_path, status)
    monitor = _PeakRSS()
    try:
        write_json_atomic(output_dir / "effective_config.json", asdict(model))
        with monitor:
            result = JumpNicheRewirePipeline(model).run()
        convergence = _convergence_summary(result, config.minimum_convergence_rate)
        if not convergence["convergence_passed"]:
            raise ConvergenceError(result, convergence)
        validation_metric = (
            _validation_grn_auprc(result.rewiring.grn_edges, dataset_dir / "truth/truth_grn.csv")
            if allow_validation_metric
            else None
        )
        status.update(
            {
                "state": "completed",
                "completed_at_utc": utc_now(),
                "runtime_seconds": time.perf_counter() - started_clock,
                "peak_rss_bytes": monitor.peak_bytes,
                "workbook_path": str(result.workbook_path),
                "validation_grn_average_precision": validation_metric,
                **convergence,
            }
        )
        write_json_atomic(status_path, status)
        write_completion(output_dir / ".complete.json", signature, {"state": "completed"})
    except ConvergenceError as error:
        status.update({
            "state": "nonconverged", "completed_at_utc": utc_now(),
            "runtime_seconds": time.perf_counter() - started_clock, "peak_rss_bytes": monitor.peak_bytes,
            "workbook_path": str(error.result.workbook_path), "validation_grn_average_precision": None,
            "error_type": type(error).__name__, "error_message": str(error), "traceback": traceback.format_exc(),
            **error.summary,
        })
        write_json_atomic(status_path, status)
        print(f"[nonconverged] {dataset_id} / {parameters.parameter_id}; saved diagnostics, NOT accepted")
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "completed_at_utc": utc_now(),
                "runtime_seconds": time.perf_counter() - started_clock,
                "peak_rss_bytes": monitor.peak_bytes,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json_atomic(status_path, status)
        print(f"[failed] {dataset_id} / {parameters.parameter_id}: {error}")
    return status


def _collect_statuses(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("run_status.json") if root.exists() else []:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            row["status_path"] = str(path)
            rows.append(row)
        except (OSError, ValueError):
            rows.append({"state": "unreadable", "status_path": str(path)})
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def repair_status_write_failures(run_root: Path) -> int:
    """Recover completed model runs whose only failure was replacing status JSON."""

    repaired = 0
    if not run_root.exists():
        return repaired
    candidates = list(run_root.rglob(".run_status.json.writing"))
    candidates.extend(run_root.rglob(".run_status.json.*.writing"))
    for recovery_path in sorted(set(candidates)):
        try:
            status = json.loads(recovery_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        message = str(status.get("error_message", ""))
        workbook = Path(str(status.get("workbook_path", "")))
        completed_outputs_exist = (
            workbook.is_file()
            and "convergence_rate" in status
            and "validation_grn_average_precision" in status
        )
        status_write_only_failure = (
            status.get("error_type") == "PermissionError"
            and "run_status.json" in message
        )
        if not (completed_outputs_exist and status_write_only_failure and status.get("signature")):
            continue
        status["state"] = "completed"
        status["status_write_recovered"] = True
        status["status_write_recovery_source"] = str(recovery_path)
        for key in ("error_type", "error_message", "traceback"):
            status.pop(key, None)
        status_path = recovery_path.parent / "run_status.json"
        write_json_atomic(status_path, status)
        write_completion(
            recovery_path.parent / ".complete.json",
            str(status["signature"]),
            {"state": "completed", "status_write_recovered": True},
        )
        recovery_path.unlink(missing_ok=True)
        repaired += 1
        print(f"[status recovered] {status.get('dataset_id')} / {status.get('parameter_id')}")
    return repaired


def freeze_parameters(config: ModelStageConfig) -> Path:
    run_root = config.profile_benchmark_root / "02_runs" / "jumpniche"
    statuses = _collect_statuses(run_root)
    if statuses.empty:
        raise FileNotFoundError("No tuning run_status.json files were found")
    validation = statuses[
        (statuses.get("split").astype(str) == "validation")
        & (statuses.get("state").astype(str) == "completed")
    ].copy()
    if validation.empty:
        raise RuntimeError("No completed validation runs are available; do not unlock test")
    validation["metric"] = pd.to_numeric(validation["validation_grn_average_precision"], errors="coerce")
    validation["convergence"] = pd.to_numeric(validation["convergence_rate"], errors="coerce")
    summary = validation.groupby("parameter_id", as_index=False).agg(
        validation_grn_auprc=("metric", "mean"),
        convergence_rate=("convergence", "mean"),
        completed_runs=("dataset_id", "nunique"),
    )
    summary["eligible"] = summary["convergence_rate"] >= config.minimum_convergence_rate
    full_gate = validation.groupby("parameter_id")["convergence_passed"].agg(
        lambda values: values.fillna(False).astype(str).str.lower().isin(["true", "1", "1.0"]).all()
    ) if "convergence_passed" in validation else pd.Series(False, index=summary["parameter_id"])
    summary["all_validation_runs_passed"] = summary["parameter_id"].map(full_gate).fillna(False)
    expected_registry = _read_registry(config)
    expected_ids = set(expected_registry.loc[expected_registry["split"] == "validation", "dataset_id"].astype(str))
    completed_sets = validation.groupby("parameter_id")["dataset_id"].agg(lambda values: set(values.astype(str)))
    summary["validation_grid_complete"] = summary["parameter_id"].map(lambda value: bool(expected_ids) and completed_sets.get(value, set()) == expected_ids)
    current_parameters = {item.parameter_id: stable_hash(item) for item in config.parameter_grid()}
    current_source = _solver_source_signature()
    validation["current_configuration"] = (
        validation.get("parameter_signature", pd.Series("", index=validation.index)) == validation["parameter_id"].map(current_parameters)
    ) & (validation.get("solver_source_signature", pd.Series("", index=validation.index)) == current_source)
    current_groups = validation.groupby("parameter_id")["current_configuration"].all()
    summary["configuration_matches"] = summary["parameter_id"].map(current_groups).fillna(False)
    summary["eligible"] = summary["eligible"] & summary["all_validation_runs_passed"] & summary["validation_grid_complete"] & summary["configuration_matches"]
    eligible = summary[summary["eligible"] & summary["validation_grn_auprc"].notna()]
    if eligible.empty:
        audit_path = config.profile_benchmark_root / "02_runs" / "parameter_selection_audit.csv"
        summary.to_csv(audit_path, index=False)
        raise RuntimeError(
            f"No parameter set met convergence_rate >= {config.minimum_convergence_rate}; see {audit_path}"
        )
    winner_id = str(eligible.sort_values(["validation_grn_auprc", "convergence_rate"], ascending=False).iloc[0]["parameter_id"])
    parameter_map = {item.parameter_id: item for item in config.parameter_grid()}
    if winner_id not in parameter_map:
        raise ValueError(f"Validation winner {winner_id!r} is not in the current parameter grid")
    lock_path = config.profile_benchmark_root / "02_runs" / "frozen_parameters.json"
    selection_signature = stable_hash(
        validation[["dataset_id", "parameter_id", "metric", "convergence"]].sort_values(["dataset_id", "parameter_id"]).to_dict("records")
    )
    if lock_path.exists() and not config.overwrite:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("selection_signature") != selection_signature:
            raise FileExistsError(
                "A different frozen parameter lock already exists. Set overwrite=True only before any test run."
            )
        return lock_path
    summary_path = config.profile_benchmark_root / "02_runs" / "parameter_selection_audit.csv"
    summary.to_csv(summary_path, index=False)
    write_json_atomic(
        lock_path,
        {
            "benchmark_profile": config.profile,
            "parameter_id": winner_id,
            "parameters": asdict(parameter_map[winner_id]),
            "selection_metric": "validation GRN Average Precision",
            "minimum_convergence_rate": config.minimum_convergence_rate,
            "selection_signature": selection_signature,
            "validation_datasets": sorted(validation["dataset_id"].astype(str).unique()),
            "test_metrics_inspected": False,
            "frozen_at_utc": utc_now(),
        },
    )
    print(f"[parameters frozen] {winner_id} -> {lock_path}")
    return lock_path


def run_stage2(config: ModelStageConfig = MODEL_CONFIG) -> Path:
    repair_status_write_failures(config.profile_benchmark_root / "02_runs" / "jumpniche")
    if config.phase == "freeze":
        return freeze_parameters(config)
    registry = _read_registry(config)
    if config.phase == "tune":
        registry = registry[registry["split"].isin(["train", "validation"])]
        parameters = config.parameter_grid()
        method_names = {item.parameter_id: f"JumpNicheRewire[{item.parameter_id}]" for item in parameters}
    elif config.phase == "test":
        if not config.unlock_test:
            raise PermissionError("Set ModelStageConfig.unlock_test=True only after frozen_parameters.json is reviewed")
        lock_path = config.profile_benchmark_root / "02_runs" / "frozen_parameters.json"
        if not lock_path.exists():
            raise FileNotFoundError("Freeze validation-selected parameters before running test")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        parameters = (ParameterSet(**lock["parameters"]),)
        method_names = {parameters[0].parameter_id: "JumpNicheRewire"}
        registry = registry[registry["split"] == "test"]
    else:
        raise ValueError("phase must be tune, freeze, or test")
    statuses = [
        _run_one(
            config,
            row,
            parameter,
            method_names[parameter.parameter_id],
            allow_validation_metric=str(row["split"]) in {"train", "validation"},
        )
        for _, row in registry.iterrows()
        for parameter in parameters
    ]
    status_frame = pd.json_normalize(statuses)
    output_path = config.profile_benchmark_root / "02_runs" / f"jumpniche_{config.phase}_registry.csv"
    status_frame.to_csv(output_path, index=False)
    completed = int((status_frame.get("state") == "completed").sum()) if not status_frame.empty else 0
    print(f"[stage 2 model] {completed}/{len(status_frame)} completed -> {output_path}")
    return output_path



