"""Read independent method workbooks and evaluate them on one truth contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import EvaluationConfig
from .metrics import binary_metrics, ranking_metrics


def _read_optional_sheet(path: Path, sheet: str) -> pd.DataFrame:
    try:
        with pd.ExcelFile(path) as excel:
            if sheet in excel.sheet_names:
                sheet_names = [sheet]
            else:
                prefix = f"{sheet}_"
                sheet_names = sorted(
                    name
                    for name in excel.sheet_names
                    if name.startswith(prefix) and name[len(prefix) :].isdigit()
                )
            if not sheet_names:
                return pd.DataFrame()
            frames = [pd.read_excel(excel, sheet_name=name) for name in sheet_names]
            return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    except ValueError:
        return pd.DataFrame()


def _method_name(path: Path) -> str:
    info = _read_optional_sheet(path, "run_info")
    if {"key", "value"}.issubset(info.columns):
        match = info.loc[info["key"].astype(str) == "method", "value"]
        if len(match):
            return str(match.iloc[0])
    return path.stem


def discover_workbooks(config: EvaluationConfig) -> list[Path]:
    paths = [Path(path) for path in config.result_workbooks if Path(path).exists()]
    if config.discover_additional_results:
        for directory in config.additional_result_directories:
            directory = Path(directory)
            if directory.exists():
                paths.extend(directory.rglob(config.result_pattern))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No unified method workbooks were found")
    return unique


def _summary_rows(method: str, task: str, metrics: dict[str, float]) -> list[dict[str, object]]:
    return [
        {"method": method, "task": task, "metric": metric, "value": value}
        for metric, value in metrics.items()
    ]


def evaluate_grn(method: str, predictions: pd.DataFrame, truth: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if predictions.empty:
        return [], pd.DataFrame()
    predictions = predictions.copy()
    predictions["score"] = pd.to_numeric(predictions["score"], errors="coerce").fillna(0.0)
    has_cells = "cell_id" in predictions.columns and predictions["cell_id"].notna().any()
    if has_cells:
        keys = ["cell_id", "tf", "target"]
        pred = predictions.groupby(keys, as_index=False)["score"].max()
        aligned = truth.merge(pred, on=keys, how="left").fillna({"score": 0.0})
    else:
        keys = ["region", "tf", "target"]
        truth_region = truth.groupby(keys, as_index=False).agg(label=("label", "max"), true_score=("true_score", "mean"))
        pred = predictions.groupby(keys, as_index=False)["score"].max()
        aligned = truth_region.merge(pred, on=keys, how="left").fillna({"score": 0.0})
    overall = binary_metrics(aligned["label"].to_numpy(), aligned["score"].to_numpy())
    summary = _summary_rows(method, "GRN recovery", overall)
    detail_rows: list[dict[str, object]] = []
    if "region" in aligned.columns:
        for region, group in aligned.groupby("region"):
            metrics = binary_metrics(group["label"].to_numpy(), group["score"].to_numpy())
            detail_rows.extend(
                {"method": method, "region": region, "metric": key, "value": value}
                for key, value in metrics.items()
            )
    return summary, pd.DataFrame(detail_rows)


def evaluate_lr_gene(method: str, predictions: pd.DataFrame, truth: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if predictions.empty:
        return [], pd.DataFrame()
    keys = ["receiver_id", "lr_id", "target"]
    pred = predictions.groupby(keys, as_index=False)["score"].max()
    aligned = truth.merge(pred, on=keys, how="left").fillna({"score": 0.0})
    detail: list[dict[str, object]] = []
    for (receiver, target), group in aligned.groupby(["receiver_id", "target"]):
        metrics = ranking_metrics(group["true_score"].to_numpy(), group["score"].to_numpy())
        detail.append({"method": method, "receiver_id": receiver, "target": target, **metrics})
    detail_frame = pd.DataFrame(detail)
    summary: list[dict[str, object]] = []
    for metric in ["Precision", "NDCG", "Spearman", "Kendall"]:
        summary.append(
            {
                "method": method,
                "task": "LR-gene attribution",
                "metric": metric,
                "value": float(detail_frame[metric].mean()) if not detail_frame.empty else np.nan,
            }
        )
    return summary, detail_frame


def evaluate_receptors(method: str, predictions: pd.DataFrame, truth: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if predictions.empty or truth.empty:
        return [], pd.DataFrame()
    region_column = "receiver_region" if "receiver_region" in predictions.columns else "region"
    predicted = predictions.loc[pd.to_numeric(predictions["path_score"], errors="coerce").fillna(0.0) > 0]
    detail: list[dict[str, object]] = []
    for region, group in truth.groupby("region"):
        true_receptors = set(group.loc[group["label"] == 1, "receptor"].astype(str))
        predicted_receptors = set(
            predicted.loc[predicted[region_column].astype(str) == str(region), "receptor"].astype(str)
        )
        ratio = len(true_receptors & predicted_receptors) / len(true_receptors) if true_receptors else np.nan
        detail.append(
            {
                "method": method,
                "region": region,
                "true_receptors": len(true_receptors),
                "predicted_receptors": len(predicted_receptors),
                "capture_ratio": ratio,
            }
        )
    detail_frame = pd.DataFrame(detail)
    summary = _summary_rows(
        method,
        "Receptor capture",
        {"Receptor capture ratio": float(detail_frame["capture_ratio"].mean())},
    )
    return summary, detail_frame


def evaluate_cci(method: str, predictions: pd.DataFrame, truth: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if predictions.empty:
        return [], pd.DataFrame()
    predictions = predictions.copy()
    if "lr_id" in predictions.columns and (predictions["lr_id"].astype(str) == "__all__").any():
        predictions = predictions[predictions["lr_id"].astype(str) == "__all__"]
    keys = ["sender_id", "receiver_id"]
    pred = predictions.groupby(keys, as_index=False)["score"].sum()
    aligned = truth.merge(pred, on=keys, how="left").fillna({"score": 0.0})
    metrics = binary_metrics(aligned["label"].to_numpy(), aligned["score"].to_numpy())
    summary = _summary_rows(method, "CCI reconstruction", {"Average Precision": metrics["Average Precision"], "AUROC": metrics["AUROC"]})
    return summary, aligned.assign(method=method)


def evaluate_causal(method: str, predictions: pd.DataFrame, truth: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    if predictions.empty or truth.empty:
        return [], pd.DataFrame()
    keys = ["region", "lr_id", "tf", "target"]
    pred = predictions.groupby(keys, as_index=False)["gamma_estimate"].mean()
    aligned = truth.merge(pred, on=keys, how="left").fillna({"gamma_estimate": 0.0})
    error = aligned["gamma_estimate"].to_numpy(dtype=float) - aligned["gamma_true"].to_numpy(dtype=float)
    if len(aligned) > 1:
        correlation = aligned[["gamma_estimate", "gamma_true"]].corr(method="spearman").iloc[0, 1]
    else:
        correlation = np.nan
    metrics = {
        "Bias": float(error.mean()),
        "MAE": float(np.abs(error).mean()),
        "Spearman": float(correlation) if np.isfinite(correlation) else np.nan,
        "Sign accuracy": float(
            np.mean(np.sign(aligned["gamma_estimate"]) == np.sign(aligned["gamma_true"]))
        ),
    }
    return _summary_rows(method, "CCC-to-GRN causal reconstruction", metrics), aligned.assign(method=method)


def run_evaluation(config: EvaluationConfig) -> Path:
    workbooks = discover_workbooks(config)
    truth_dir = Path(config.truth_dir)
    truth_grn = pd.read_csv(truth_dir / "truth_grn.csv")
    truth_lr = pd.read_csv(truth_dir / "truth_lr_gene.csv")
    truth_receptor = pd.read_csv(truth_dir / "truth_receptors.csv")
    truth_cci = pd.read_csv(truth_dir / "truth_cci.csv")
    truth_causal = pd.read_csv(truth_dir / "truth_causal.csv")

    summary_rows: list[dict[str, object]] = []
    detail_frames: dict[str, list[pd.DataFrame]] = {
        "grn_detail": [], "lr_gene_detail": [], "receptor_detail": [], "cci_detail": [], "causal_detail": []
    }
    for workbook in workbooks:
        method = _method_name(workbook)
        tasks = [
            ("grn_detail", evaluate_grn(method, _read_optional_sheet(workbook, "grn_edges"), truth_grn)),
            ("lr_gene_detail", evaluate_lr_gene(method, _read_optional_sheet(workbook, "lr_gene"), truth_lr)),
            ("receptor_detail", evaluate_receptors(method, _read_optional_sheet(workbook, "receptor_paths"), truth_receptor)),
            ("cci_detail", evaluate_cci(method, _read_optional_sheet(workbook, "cci_edges"), truth_cci)),
            ("causal_detail", evaluate_causal(method, _read_optional_sheet(workbook, "causal_edges"), truth_causal)),
        ]
        for detail_name, (summary, detail) in tasks:
            summary_rows.extend(summary)
            if not detail.empty:
                detail_frames[detail_name].append(detail)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.output_workbook
    summary_frame = pd.DataFrame(summary_rows)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_frame.to_excel(writer, sheet_name="summary", index=False)
        if not summary_frame.empty:
            pivot = summary_frame.pivot_table(index=["task", "metric"], columns="method", values="value")
            pivot.to_excel(writer, sheet_name="comparison")
        for name, frames in detail_frames.items():
            detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            detail.to_excel(writer, sheet_name=name[:31], index=False)
    return output_path
