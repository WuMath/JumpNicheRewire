"""Atomic numerical checkpoints written immediately before output expansion."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
from typing import Any

from .config import ProjectConfig


CHECKPOINT_VERSION = 2
TARGET_CHECKPOINT_VERSION = 1


def _numerical_configuration(config: ProjectConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("output", None)
    runtime = payload.get("runtime", {})
    for key in (
        "verbose",
        "require_convergence",
        "enable_numerical_checkpoint",
        "resume_numerical_checkpoint",
        "keep_numerical_checkpoint_after_success",
        "numerical_checkpoint_name",
        "enable_target_checkpoints",
        "resume_target_checkpoints",
        "target_checkpoint_dir_name",
        "keep_target_checkpoints_after_success",
    ):
        runtime.pop(key, None)
    rewiring = payload.get("rewiring", {})
    for key in ("top_grn_rows", "top_lr_gene_rows", "top_path_rows", "output_build_chunk_rows"):
        rewiring.pop(key, None)
    return payload


def numerical_signature(config: ProjectConfig) -> str:
    encoded = json.dumps(
        _numerical_configuration(config), sort_keys=True, default=str, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_path(config: ProjectConfig) -> Path:
    return Path(config.output.output_dir) / config.runtime.numerical_checkpoint_name


def load_numerical_checkpoint(config: ProjectConfig) -> dict[str, Any] | None:
    if not (
        config.runtime.enable_numerical_checkpoint
        and config.runtime.resume_numerical_checkpoint
    ):
        return None
    path = checkpoint_path(config)
    if not path.exists():
        return None
    try:
        with path.open("rb") as stream:
            checkpoint = pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        return None
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        return None
    if checkpoint.get("signature") != numerical_signature(config):
        return None
    state = checkpoint.get("state")
    return state if isinstance(state, dict) else None


def save_numerical_checkpoint(config: ProjectConfig, state: dict[str, Any]) -> Path | None:
    if not config.runtime.enable_numerical_checkpoint:
        return None
    path = checkpoint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    payload = {
        "version": CHECKPOINT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "signature": numerical_signature(config),
        "state": state,
    }
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def remove_numerical_checkpoint(config: ProjectConfig) -> None:
    path = checkpoint_path(config)
    if path.exists():
        path.unlink()


def target_checkpoint_root(config: ProjectConfig) -> Path:
    return Path(config.output.output_dir) / config.runtime.target_checkpoint_dir_name


def _target_checkpoint_path(
    config: ProjectConfig, stage: str, target_index: int
) -> Path:
    if not stage or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in stage):
        raise ValueError(f"Unsafe target checkpoint stage: {stage!r}")
    if int(target_index) < 0:
        raise ValueError("target checkpoint index must be nonnegative")
    return target_checkpoint_root(config) / stage / f"target_{int(target_index):05d}.pkl"


def load_target_checkpoint(
    config: ProjectConfig,
    stage: str,
    target_index: int,
    target_name: str,
) -> dict[str, Any] | None:
    if not (
        config.runtime.enable_target_checkpoints
        and config.runtime.resume_target_checkpoints
    ):
        return None
    path = _target_checkpoint_path(config, stage, target_index)
    if not path.exists():
        return None
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        return None
    if (
        payload.get("version") != TARGET_CHECKPOINT_VERSION
        or payload.get("signature") != numerical_signature(config)
        or payload.get("stage") != stage
        or payload.get("target_index") != int(target_index)
        or payload.get("target_name") != str(target_name)
    ):
        return None
    value = payload.get("value")
    return value if isinstance(value, dict) else None


def save_target_checkpoint(
    config: ProjectConfig,
    stage: str,
    target_index: int,
    target_name: str,
    value: dict[str, Any],
) -> Path | None:
    if not config.runtime.enable_target_checkpoints:
        return None
    path = _target_checkpoint_path(config, stage, target_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    payload = {
        "version": TARGET_CHECKPOINT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "signature": numerical_signature(config),
        "stage": stage,
        "target_index": int(target_index),
        "target_name": str(target_name),
        "value": value,
    }
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def remove_target_checkpoints(config: ProjectConfig) -> None:
    output = Path(config.output.output_dir).resolve()
    root = target_checkpoint_root(config).resolve()
    if root.parent != output:
        raise ValueError(f"Refusing to remove checkpoint directory outside output: {root}")
    if root.exists():
        shutil.rmtree(root)
