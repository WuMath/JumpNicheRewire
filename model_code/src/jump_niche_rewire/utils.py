"""Numerical and tabular utility functions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def soft_threshold(values: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def standardize(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axis, keepdims=True)
    scale = np.nanstd(values, axis=axis, ddof=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    output = (values - mean) / scale
    return output, np.squeeze(mean, axis=axis), np.squeeze(scale, axis=axis)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    result = np.full_like(p_values, np.nan)
    finite = np.isfinite(p_values)
    if not finite.any():
        return result
    p = np.clip(p_values[finite], 0.0, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    result[finite] = restored
    return result


def one_hot_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[np.ndarray, list[str]]:
    blocks: list[pd.DataFrame] = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").astype(float)
            finite = np.isfinite(numeric.to_numpy(dtype=float))
            if not finite.any():
                continue
            median = float(np.median(numeric.to_numpy(dtype=float)[finite]))
            values = numeric.where(finite, median)
            if values.nunique(dropna=False) <= 1:
                continue
            blocks.append(pd.DataFrame({column: values}, index=frame.index))
        else:
            values = series.astype("object").where(series.notna(), "NA").astype(str)
            if values.nunique(dropna=False) <= 1:
                continue
            encoded = pd.get_dummies(values, prefix=column, drop_first=True, dtype=float)
            if not encoded.empty:
                blocks.append(encoded)
    if not blocks:
        return np.empty((len(frame), 0), dtype=float), []
    combined = pd.concat(blocks, axis=1)
    values = combined.to_numpy(dtype=float)
    keep = np.isfinite(values).all(axis=0) & (np.std(values, axis=0) > 1e-12)
    combined = combined.loc[:, keep]
    if combined.empty:
        return np.empty((len(frame), 0), dtype=float), []
    return combined.to_numpy(dtype=float), combined.columns.astype(str).tolist()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataframe_memory_mb(frame: pd.DataFrame) -> float:
    return float(frame.memory_usage(index=True, deep=True).sum() / (1024**2))


def safe_sheet_name(name: str) -> str:
    forbidden = set("[]:*?/\\")
    cleaned = "".join("_" if char in forbidden else char for char in name)
    return cleaned[:31]


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


def finite_or_raise(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        count = int((~np.isfinite(values)).sum())
        raise ValueError(f"{name} contains {count} non-finite values")


def pair_key(left: str, right: str) -> str:
    return f"{left}::{right}"


def choose_top_indices(scores: np.ndarray, maximum: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    maximum = min(maximum, scores.size)
    if maximum <= 0:
        return np.array([], dtype=int)
    if maximum == scores.size:
        return np.argsort(-scores)
    partition = np.argpartition(-scores, maximum - 1)[:maximum]
    return partition[np.argsort(-scores[partition])]


def log_message(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)
