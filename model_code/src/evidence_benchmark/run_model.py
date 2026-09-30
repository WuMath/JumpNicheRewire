"""Isolated, resumable JumpNicheRewire runs that never read benchmark truth."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

import pandas as pd

from .common import stable_hash, write_json
from .config import MODEL_CONFIG, EvidenceModelConfig, PROJECT_ROOT


SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jump_niche_rewire import JumpNicheRewirePipeline
from jump_niche_rewire.checkpoint import target_checkpoint_root
from jump_niche_rewire.config import CONFIG as BASE_CONFIG
from jump_niche_rewire.convergence import summarize_convergence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_settings(config: EvidenceModelConfig) -> dict[str, object]:
    payload = asdict(config)
    for key in (
        "registry_path", "output_root", "evidence_tracks", "dataset_ids",
        "isolate_datasets", "reconcile_stale_running_statuses",
        "overwrite",
    ):
        payload.pop(key, None)
    return payload


def _run_signature(config: EvidenceModelConfig, row: pd.Series) -> str:
    return stable_hash(
        {
            "dataset": str(row.signature),
            "scope": str(row.evaluation_scope),
            "model_settings": _model_settings(config),
        }
    )


def _configure(config: EvidenceModelConfig, row: pd.Series, output: Path):
    model = deepcopy(config.project_model if config.project_model is not None else BASE_CONFIG)
    model.data.input_dir = Path(str(row.dataset_dir))
    model.output.output_dir = output
    model.output.workbook_name = "JumpNicheRewire_results.xlsx"
    model.output.method_name = "JumpNicheRewire"
    model.output.save_parquet = True
    if 'species' in row and str(row['species']) not in {'', 'nan'}:
        model.data.species = str(row['species'])
    model.runtime.random_seed = int(stable_hash(str(row.dataset_id))[:8], 16)
    model.runtime.require_convergence = config.require_convergence
    model.runtime.minimum_target_convergence_rate = config.minimum_convergence_rate
    model.runtime.checkpoint_data_signature = str(row.signature)
    model.spatial.spatial_permutations = config.spatial_permutations
    model.base_grn.maximum_tfs = config.maximum_tfs
    model.base_grn.maximum_targets = config.maximum_targets
    model.base_grn.l1_penalty = config.l1_penalty
    model.base_grn.fusion_penalty = config.fusion_penalty
    model.base_grn.admm_max_iterations = config.admm_max_iterations
    model.rewiring.maximum_active_lr = config.maximum_active_lr
    model.rewiring.group_penalty = config.group_penalty
    model.rewiring.ridge_penalty = config.ridge_penalty
    model.rewiring.fista_max_iterations = config.fista_max_iterations
    model.rewiring.joint_iterations = config.joint_iterations
    model.rewiring.iteration_tolerance = config.iteration_tolerance
    model.rewiring.top_grn_rows = model.output.maximum_grn_rows
    if str(row.evaluation_scope) == "grn_only" or not model.data.is_spatial:
        model.rewiring.disable_communication_rewiring = True
        model.rewiring.joint_iterations = 1
    return model


class _PeakRSSMonitor:
    def __init__(self) -> None:
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self._stop.wait(0.25):
                try:
                    rss = int(process.memory_info().rss)
                except (psutil.Error, OSError):
                    return
                self.peak_bytes = rss if self.peak_bytes is None else max(self.peak_bytes, rss)

        self.peak_bytes = int(process.memory_info().rss)
        self._thread = threading.Thread(target=sample, name="peak-rss-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _run_one_dataset(
    config: EvidenceModelConfig,
    row: pd.Series,
    signature: str,
) -> dict[str, object]:
    output = config.output_root / str(row.evidence_track) / str(row.dataset_id)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "run_status.json"
    model = _configure(config, row, output)
    checkpoint_path = output / model.runtime.numerical_checkpoint_name
    target_checkpoint_path = target_checkpoint_root(model)
    status: dict[str, object] = {
        "dataset_id": str(row.dataset_id),
        "evidence_track": str(row.evidence_track),
        "evaluation_scope": str(row.evaluation_scope),
        "truth_kind": str(row.truth_kind),
        "signature": signature,
        "state": "running",
        "started_utc": _utc_now(),
        "worker_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "run_token": uuid.uuid4().hex,
        "checkpoint_present_at_start": checkpoint_path.exists(),
        "target_checkpoints_present_at_start": target_checkpoint_path.exists(),
        "truth_read_during_inference": False,
    }
    write_json(status_path, status)
    started = time.perf_counter()
    monitor = _PeakRSSMonitor()
    monitor.start()
    try:
        write_json(output / "effective_config.json", asdict(model))
        result = JumpNicheRewirePipeline(model).run()
        convergence = summarize_convergence(
            result.base,
            result.rewiring,
            config.minimum_convergence_rate,
            communication_rewiring_disabled=model.rewiring.disable_communication_rewiring,
        )
        status.update(
            {
                "state": "completed",
                "workbook_path": str(result.workbook_path),
                **convergence,
            }
        )
    except KeyboardInterrupt:
        status.update(
            {
                "state": "interrupted",
                "error_type": "KeyboardInterrupt",
                "error_message": "Dataset worker was interrupted by the user.",
                "traceback": traceback.format_exc(),
            }
        )
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        monitor.stop()
        status["runtime_seconds"] = time.perf_counter() - started
        status["peak_rss_bytes"] = monitor.peak_bytes
        status["peak_rss_gib"] = (
            monitor.peak_bytes / 1024**3 if monitor.peak_bytes is not None else None
        )
        status["finished_utc"] = _utc_now()
        status["checkpoint_retained"] = checkpoint_path.exists()
        status["target_checkpoints_retained"] = target_checkpoint_path.exists()
        status["target_checkpoint_file_count"] = (
            sum(1 for path in target_checkpoint_path.rglob("*.pkl") if path.is_file())
            if target_checkpoint_path.exists()
            else 0
        )
        write_json(status_path, status)
    print(f"[{status['state']}] {row.dataset_id}", flush=True)
    return status


def _pid_is_running(pid: object) -> bool:
    try:
        pid_value = int(pid)
        if pid_value <= 0:
            return False
        import psutil
        process = psutil.Process(pid_value)
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except Exception:
        return False


def reconcile_stale_running_statuses(config: EvidenceModelConfig) -> list[Path]:
    repaired: list[Path] = []
    if not config.output_root.exists():
        return repaired
    for status_path in config.output_root.rglob("run_status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("state") != "running" or _pid_is_running(status.get("worker_pid")):
            continue
        status.update(
            {
                "state": "interrupted",
                "finished_utc": _utc_now(),
                "error_type": "StaleRunningStatus",
                "error_message": "No live dataset worker exists for this running status.",
            }
        )
        write_json(status_path, status)
        repaired.append(status_path)
        print(f"[interrupted stale] {status.get('dataset_id', status_path.parent.name)}")
    return repaired


def _load_status(status_path: Path) -> dict[str, object] | None:
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _mark_abrupt_worker_exit(
    status_path: Path,
    row: pd.Series,
    signature: str,
    return_code: int,
) -> dict[str, object]:
    status = _load_status(status_path) or {
        "dataset_id": str(row.dataset_id),
        "evidence_track": str(row.evidence_track),
        "evaluation_scope": str(row.evaluation_scope),
        "truth_kind": str(row.truth_kind),
        "signature": signature,
        "truth_read_during_inference": False,
    }
    if status.get("state") in {"running", "queued", None}:
        status.update(
            {
                "state": "interrupted",
                "finished_utc": _utc_now(),
                "error_type": "WorkerProcessExit",
                "error_message": f"Dataset subprocess exited abruptly with code {return_code}.",
                "worker_return_code": int(return_code),
            }
        )
        write_json(status_path, status)
    return status


def _selected_registry(config: EvidenceModelConfig) -> pd.DataFrame:
    if not config.registry_path.exists():
        raise FileNotFoundError(f"Prepare data first: {config.registry_path}")
    registry = pd.read_csv(config.registry_path)
    registry = registry[registry.evidence_track.isin(config.evidence_tracks)].copy()
    if config.dataset_ids is not None:
        requested = list(dict.fromkeys(config.dataset_ids))
        if not requested:
            raise ValueError("dataset_ids is empty; list datasets explicitly or set it to None")
        missing = sorted(set(requested).difference(registry.dataset_id.astype(str)))
        if missing:
            raise ValueError(f"dataset_ids are absent from the registry: {missing}")
        order = {dataset_id: index for index, dataset_id in enumerate(requested)}
        registry = registry[registry.dataset_id.astype(str).isin(requested)].copy()
        registry["_selection_order"] = registry.dataset_id.astype(str).map(order)
        registry = registry.sort_values("_selection_order").drop(columns="_selection_order")
    if registry.empty:
        raise ValueError("No selected evidence datasets in registry")
    return registry


def run_models(config: EvidenceModelConfig = MODEL_CONFIG) -> Path:
    if config.reconcile_stale_running_statuses:
        reconcile_stale_running_statuses(config)
    registry = _selected_registry(config)
    statuses: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        output = config.output_root / str(row.evidence_track) / str(row.dataset_id)
        status_path = output / "run_status.json"
        signature = _run_signature(config, row_series)
        previous = _load_status(status_path)
        if (
            not config.overwrite
            and previous is not None
            and previous.get("state") == "completed"
            and previous.get("signature") == signature
        ):
            print(f"[unchanged] {row.dataset_id}")
            statuses.append(previous)
            continue
        if config.isolate_datasets:
            from main_jumpniche import main

            # 配置对象直接传给主函数，不再保存请求文件或设置请求路径环境变量。
            # spawn 为每份数据启动全新的 Python 进程；Windows/Linux 使用同一种方式。
            process = multiprocessing.get_context("spawn").Process(
                target=main,
                args=(config, row_series, signature),
                name=f"JumpNiche-{row.dataset_id}",
            )
            write_json(status_path, {
                "dataset_id": str(row.dataset_id), "evidence_track": str(row.evidence_track),
                "evaluation_scope": str(row.evaluation_scope), "truth_kind": str(row.truth_kind),
                "signature": signature, "state": "queued", "truth_read_during_inference": False,
            })
            try:
                process.start()
                process.join()
                status = _mark_abrupt_worker_exit(status_path, row_series, signature, process.exitcode)
            except KeyboardInterrupt:
                if process.is_alive():
                    process.terminate()
                    process.join()
                _mark_abrupt_worker_exit(status_path, row_series, signature, 130)
                raise
            except Exception as error:
                # 单个进程不能启动时，记录原因并继续下一个数据集。
                if process.is_alive():
                    process.terminate()
                    process.join()
                status = _load_status(status_path) or {}
                status.update(state="failed", error_type=type(error).__name__,
                              error_message=str(error), traceback=traceback.format_exc(),
                              finished_utc=_utc_now())
                write_json(status_path, status)
            finally:
                if process.pid is not None and not process.is_alive():
                    process.close()
        else:
            status = _run_one_dataset(config, row_series, signature)
        statuses.append(status)

    frame = pd.DataFrame(statuses)
    output_path = config.output_root / "run_inventory.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.writing")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output_path)
    return output_path



