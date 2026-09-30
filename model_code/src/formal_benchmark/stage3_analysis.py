"""Stage 3: reveal test truth once and create statistics, audits, and figures."""

from __future__ import annotations

import json
from html import escape
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import ANALYSIS_CONFIG, AnalysisStageConfig, PROJECT_ROOT
from .state import completion_matches, stable_hash, utc_now, write_completion


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluator import (
    _method_name,
    _read_optional_sheet,
    evaluate_causal,
    evaluate_cci,
    evaluate_grn,
    evaluate_lr_gene,
    evaluate_receptors,
)


def _workbook_inventory(config: AnalysisStageConfig, registry: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    runs_root = config.profile_benchmark_root / "02_runs"
    lock_path = runs_root / "frozen_parameters.json"
    if not lock_path.exists():
        raise FileNotFoundError("Frozen model parameters are missing; stage 3 must not select parameters")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("benchmark_profile") != config.profile:
        raise ValueError(f"Frozen parameters are for {lock.get('benchmark_profile')!r}, not {config.profile!r}")
    parameter_id = lock["parameter_id"]
    inventories: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for _, dataset in registry.iterrows():
        dataset_id = str(dataset["dataset_id"])
        main_path = runs_root / "jumpniche" / parameter_id / dataset_id / "JumpNicheRewire_results.xlsx"
        candidates: dict[str, Path] = {"JumpNicheRewire": main_path}
        for method in config.expected_methods:
            if method == "JumpNicheRewire":
                continue
            candidates[method] = runs_root / "baselines" / method / dataset_id / f"{method}_results.xlsx"
        for expected_method, workbook in candidates.items():
            status_path = workbook.parent / "run_status.json"
            status: dict[str, object] = {}
            if status_path.exists():
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    status = {"state": "unreadable"}
            state = str(status.get("state", "missing"))
            if workbook.exists() and state == "completed":
                inventories.append(
                    {
                        "dataset_id": dataset_id,
                        "scenario": dataset["scenario"],
                        "seed": int(dataset["seed"]),
                        "truth_dir": str(Path(str(dataset["dataset_dir"])) / "truth"),
                        "expected_method": expected_method,
                        "workbook": workbook,
                    }
                )
            audit_rows.append(
                {
                    "dataset_id": dataset_id,
                    "scenario": dataset["scenario"],
                    "seed": int(dataset["seed"]),
                    "method": expected_method,
                    "state": state,
                    "workbook_exists": workbook.exists(),
                    "status_path": str(status_path),
                    "workbook": str(workbook),
                    "error_type": status.get("error_type"),
                    "error_message": status.get("error_message"),
                    "runtime_seconds": status.get("runtime_seconds"),
                    "peak_rss_bytes": status.get("peak_rss_bytes"),
                }
            )
    return inventories, pd.DataFrame(audit_rows)


def _evaluate_one(item: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame]]:
    workbook = Path(item["workbook"])
    truth_dir = Path(str(item["truth_dir"]))
    method = _method_name(workbook)
    truth = {
        "grn": pd.read_csv(truth_dir / "truth_grn.csv"),
        "lr": pd.read_csv(truth_dir / "truth_lr_gene.csv"),
        "receptors": pd.read_csv(truth_dir / "truth_receptors.csv"),
        "cci": pd.read_csv(truth_dir / "truth_cci.csv"),
        "causal": pd.read_csv(truth_dir / "truth_causal.csv"),
    }
    tasks = (
        ("grn", evaluate_grn(method, _read_optional_sheet(workbook, "grn_edges"), truth["grn"])),
        ("lr_gene", evaluate_lr_gene(method, _read_optional_sheet(workbook, "lr_gene"), truth["lr"])),
        ("receptor", evaluate_receptors(method, _read_optional_sheet(workbook, "receptor_paths"), truth["receptors"])),
        ("cci", evaluate_cci(method, _read_optional_sheet(workbook, "cci_edges"), truth["cci"])),
        ("causal", evaluate_causal(method, _read_optional_sheet(workbook, "causal_edges"), truth["causal"])),
    )
    summary_rows: list[dict[str, object]] = []
    details: dict[str, pd.DataFrame] = {}
    for name, (rows, detail) in tasks:
        for row in rows:
            summary_rows.append(
                {
                    "dataset_id": item["dataset_id"],
                    "scenario": item["scenario"],
                    "seed": item["seed"],
                    **row,
                }
            )
        if not detail.empty:
            details[name] = detail.assign(
                dataset_id=item["dataset_id"], scenario=item["scenario"], seed=item["seed"]
            )
    return summary_rows, details


def _bootstrap_interval(values: np.ndarray, iterations: int, level: float, random: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = random.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _summary_with_ci(metrics: pd.DataFrame, config: AnalysisStageConfig) -> pd.DataFrame:
    random = np.random.default_rng(config.random_seed)
    rows = []
    for keys, group in metrics.groupby(["method", "task", "metric"], dropna=False):
        values = pd.to_numeric(group["value"], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        lower, upper = _bootstrap_interval(finite, config.bootstrap_iterations, config.confidence_level, random)
        rows.append(
            {
                "method": keys[0],
                "task": keys[1],
                "metric": keys[2],
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "standard_deviation": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                "ci_lower": lower,
                "ci_upper": upper,
                "datasets": len(finite),
            }
        )
    return pd.DataFrame(rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    output = [np.nan] * len(p_values)
    finite_indices = [index for index, value in enumerate(p_values) if np.isfinite(value)]
    ordered = sorted(finite_indices, key=lambda index: p_values[index])
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * p_values[index])
        running = max(running, adjusted)
        output[index] = running
    return output


def _paired_tests(metrics: pd.DataFrame, config: AnalysisStageConfig) -> pd.DataFrame:
    primary_rows = []
    for task, metric in config.primary_metric_by_task.items():
        subset = metrics[(metrics["task"] == task) & (metrics["metric"] == metric)].copy()
        reference = subset[subset["method"] == config.primary_method][["dataset_id", "value"]].rename(columns={"value": "primary_value"})
        for method in sorted(set(subset["method"]) - {config.primary_method}):
            comparator = subset[subset["method"] == method][["dataset_id", "value"]].rename(columns={"value": "comparator_value"})
            paired = reference.merge(comparator, on="dataset_id", how="inner").dropna()
            difference = paired["primary_value"].to_numpy(dtype=float) - paired["comparator_value"].to_numpy(dtype=float)
            p_value = np.nan
            if len(difference) >= config.minimum_pairs_for_test:
                p_value = 1.0 if np.allclose(difference, 0.0) else float(wilcoxon(difference, alternative="two-sided").pvalue)
            primary_rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "primary_method": config.primary_method,
                    "comparator": method,
                    "paired_datasets": len(difference),
                    "primary_mean": float(paired["primary_value"].mean()) if len(paired) else np.nan,
                    "comparator_mean": float(paired["comparator_value"].mean()) if len(paired) else np.nan,
                    "mean_paired_difference": float(np.mean(difference)) if len(difference) else np.nan,
                    "median_paired_difference": float(np.median(difference)) if len(difference) else np.nan,
                    "wins": int(np.sum(difference > 0)),
                    "ties": int(np.sum(np.isclose(difference, 0.0))),
                    "losses": int(np.sum(difference < 0)),
                    "wilcoxon_p": p_value,
                }
            )
    result = pd.DataFrame(primary_rows)
    if not result.empty:
        result["holm_adjusted_p"] = _holm_adjust(result["wilcoxon_p"].tolist())
    return result


def _convergence_audit(run_audit: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in run_audit[run_audit["method"] == "JumpNicheRewire"].iterrows():
        status_path = Path(str(row["status_path"]))
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        rate = pd.to_numeric(pd.Series([status.get("convergence_rate")]), errors="coerce").iloc[0]
        rows.append(
            {
                "dataset_id": row["dataset_id"],
                "scenario": row["scenario"],
                "state": status.get("state", "missing"),
                "convergence_rate": rate,
                "threshold": threshold,
                "passes": bool(np.isfinite(rate) and rate >= threshold),
                "failed_diagnostics": status.get("failed_diagnostics"),
                "diagnostics": status.get("diagnostics"),
                "module_rates": json.dumps(status.get("module_rates", {}), ensure_ascii=False, sort_keys=True),
                "error_message": status.get("error_message"),
            }
        )
    return pd.DataFrame(rows)


def _save_figures(summary: pd.DataFrame, metrics: pd.DataFrame, config: AnalysisStageConfig, figure_dir: Path) -> None:
    """Write dependency-free vector figures so plotting cannot lose the analysis."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    selections = []
    for task, metric in config.primary_metric_by_task.items():
        frame = summary[(summary["task"] == task) & (summary["metric"] == metric)].copy()
        if not frame.empty:
            selections.append((task, metric, frame))
    if selections:
        width = 1100
        panel_height = 300
        height = 40 + panel_height * len(selections)
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:18px;font-weight:bold}.tick{font-size:12px}.label{font-size:13px}</style>',
        ]
        palette = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2")
        for panel, (task, metric, frame) in enumerate(selections):
            frame = frame.sort_values("mean", ascending=False).reset_index(drop=True)
            top = 30 + panel * panel_height
            left, right = 85, width - 35
            plot_top, plot_bottom = top + 45, top + 225
            finite_bounds = pd.concat([frame["ci_lower"], frame["ci_upper"], frame["mean"]]).dropna().to_numpy(dtype=float)
            y_min = min(0.0, float(np.min(finite_bounds))) if len(finite_bounds) else 0.0
            y_max = max(1.0, float(np.max(finite_bounds))) if len(finite_bounds) else 1.0
            if math.isclose(y_min, y_max):
                y_max = y_min + 1.0

            def y_position(value: float) -> float:
                return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

            svg.append(f'<text class="title" x="{left}" y="{top + 20}">{escape(str(task))} — {escape(str(metric))}</text>')
            svg.append(f'<line x1="{left}" y1="{plot_bottom}" x2="{right}" y2="{plot_bottom}" stroke="#333"/>')
            svg.append(f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_bottom}" stroke="#333"/>')
            for tick in np.linspace(y_min, y_max, 5):
                y = y_position(float(tick))
                svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd"/>')
                svg.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:.2f}</text>')
            slot = (right - left) / max(len(frame), 1)
            bar_width = min(90.0, slot * 0.58)
            zero_y = y_position(0.0)
            for index, row in frame.iterrows():
                mean = float(row["mean"])
                lower = float(row["ci_lower"])
                upper = float(row["ci_upper"])
                x = left + slot * (index + 0.5)
                y_mean = y_position(mean)
                bar_y = min(zero_y, y_mean)
                bar_height = max(abs(zero_y - y_mean), 1.0)
                color = palette[index % len(palette)]
                svg.append(f'<rect x="{x - bar_width / 2:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
                svg.append(f'<line x1="{x:.1f}" y1="{y_position(upper):.1f}" x2="{x:.1f}" y2="{y_position(lower):.1f}" stroke="#111" stroke-width="2"/>')
                svg.append(f'<line x1="{x - 7:.1f}" y1="{y_position(upper):.1f}" x2="{x + 7:.1f}" y2="{y_position(upper):.1f}" stroke="#111"/>')
                svg.append(f'<line x1="{x - 7:.1f}" y1="{y_position(lower):.1f}" x2="{x + 7:.1f}" y2="{y_position(lower):.1f}" stroke="#111"/>')
                svg.append(f'<text class="label" x="{x:.1f}" y="{plot_bottom + 25}" text-anchor="middle">{escape(str(row["method"]))}</text>')
        svg.append("</svg>")
        (figure_dir / "primary_metrics_95ci.svg").write_text("\n".join(svg), encoding="utf-8")

    grn = metrics[(metrics["task"] == "GRN recovery") & (metrics["metric"] == "Average Precision")]
    if not grn.empty:
        pivot = grn.pivot_table(index="dataset_id", columns="method", values="value")
        methods = list(pivot.columns)
        width, height = max(700, 170 * len(methods)), 460
        left, right, top, bottom = 80, width - 30, 55, 370
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:19px;font-weight:bold}.tick{font-size:12px}.label{font-size:13px}</style>',
            f'<text class="title" x="{left}" y="30">GRN Average Precision across test datasets</text>',
        ]

        def y_position(value: float) -> float:
            return bottom - float(value) * (bottom - top)

        for tick in np.linspace(0.0, 1.0, 6):
            y = y_position(float(tick))
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd"/>')
            svg.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
        slot = (right - left) / max(len(methods), 1)
        for index, method in enumerate(methods):
            values = pivot[method].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            low, high = float(np.min(values)), float(np.max(values))
            mean = float(np.mean(values))
            x = left + slot * (index + 0.5)
            box_width = min(90.0, slot * 0.55)
            svg.append(f'<line x1="{x:.1f}" y1="{y_position(high):.1f}" x2="{x:.1f}" y2="{y_position(low):.1f}" stroke="#333"/>')
            svg.append(f'<rect x="{x - box_width / 2:.1f}" y="{y_position(q3):.1f}" width="{box_width:.1f}" height="{max(y_position(q1) - y_position(q3), 1):.1f}" fill="#A6CEE3" stroke="#333"/>')
            svg.append(f'<line x1="{x - box_width / 2:.1f}" y1="{y_position(median):.1f}" x2="{x + box_width / 2:.1f}" y2="{y_position(median):.1f}" stroke="#E45756" stroke-width="3"/>')
            svg.append(f'<circle cx="{x:.1f}" cy="{y_position(mean):.1f}" r="4" fill="#111"/>')
            svg.append(f'<text class="label" x="{x:.1f}" y="{bottom + 28}" text-anchor="middle">{escape(str(method))}</text>')
        svg.append("</svg>")
        (figure_dir / "grn_dataset_distribution.svg").write_text("\n".join(svg), encoding="utf-8")


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    run_audit: pd.DataFrame,
    convergence: pd.DataFrame,
) -> None:
    completed_fraction = float((run_audit["state"] == "completed").mean()) if len(run_audit) else 0.0
    convergence_fraction = float(convergence["passes"].mean()) if len(convergence) else 0.0
    lines = [
        "# Formal benchmark analysis",
        "",
        f"Generated (UTC): {utc_now()}",
        "",
        "## Publication-readiness gates",
        "",
        f"- Expected method-dataset runs completed: {completed_fraction:.1%}.",
        f"- JumpNicheRewire test datasets passing convergence threshold: {convergence_fraction:.1%}.",
        f"- Missing/failed runs: {int((run_audit['state'] != 'completed').sum())}; they were retained in `run_audit.csv` and not silently discarded.",
        "- Test metrics were computed only in this stage, after the parameter lock was present.",
        "",
        "## Primary metric summary",
        "",
    ]
    if summary.empty:
        lines.append("No evaluable completed workbooks were found.")
    else:
        view = summary[["method", "task", "metric", "mean", "ci_lower", "ci_upper", "datasets"]]
        lines.extend(["```text", view.to_string(index=False), "```"])
    lines.extend(["", "## Paired comparisons", ""])
    if paired.empty:
        lines.append("No task had paired method results.")
    else:
        lines.extend(["```text", paired.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Do not claim superiority from a larger mean alone. Require adequate paired datasets, a confidence interval, Holm-adjusted significance, convergence, and no selective omission of failures.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage3(config: AnalysisStageConfig = ANALYSIS_CONFIG) -> Path:
    registry_path = config.registry_path
    if not registry_path.exists():
        raise FileNotFoundError(f"Run stage 1 first: {registry_path}")
    registry = pd.read_csv(registry_path)
    if set(registry["benchmark_profile"].astype(str)) != {config.profile}:
        raise ValueError(f"Registry {registry_path} is not isolated to profile {config.profile!r}")
    registry = registry[registry["split"] == "test"].copy()
    inventories, run_audit = _workbook_inventory(config, registry)
    input_signature = stable_hash(
        {
            "config": config,
            "registry": registry[["dataset_id", "signature"]].to_dict("records"),
            "artifacts": [
                {
                    "path": str(item["workbook"]),
                    "size": Path(item["workbook"]).stat().st_size,
                    "mtime_ns": Path(item["workbook"]).stat().st_mtime_ns,
                }
                for item in inventories
            ],
            "run_states": run_audit[["dataset_id", "method", "state", "workbook_exists"]].to_dict("records"),
        }
    )
    stage_dir = config.profile_benchmark_root / "03_analysis"
    tables_dir = stage_dir / "tables"
    figures_dir = stage_dir / "figures"
    reports_dir = stage_dir / "reports"
    workbook_path = stage_dir / "benchmark_summary.xlsx"
    marker = stage_dir / ".stage_complete.json"
    if not config.overwrite and completion_matches(marker, input_signature, (workbook_path,)):
        print(f"[stage 3 unchanged] {workbook_path}")
        return workbook_path
    for directory in (tables_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    detail_frames: dict[str, list[pd.DataFrame]] = {}
    for item in inventories:
        rows, details = _evaluate_one(item)
        summary_rows.extend(rows)
        for task, frame in details.items():
            detail_frames.setdefault(task, []).append(frame)
    metrics = pd.DataFrame(summary_rows)
    summary = _summary_with_ci(metrics, config) if not metrics.empty else pd.DataFrame()
    paired = _paired_tests(metrics, config) if not metrics.empty else pd.DataFrame()
    convergence = _convergence_audit(run_audit, config.minimum_convergence_rate)
    metrics.to_csv(tables_dir / "dataset_metrics.csv", index=False)
    summary.to_csv(tables_dir / "summary_with_bootstrap_ci.csv", index=False)
    paired.to_csv(tables_dir / "paired_tests_holm.csv", index=False)
    run_audit.to_csv(tables_dir / "run_audit.csv", index=False)
    convergence.to_csv(tables_dir / "convergence_audit.csv", index=False)
    for name, frames in detail_frames.items():
        pd.concat(frames, ignore_index=True).to_parquet(tables_dir / f"{name}_detail.parquet", index=False)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="dataset_metrics", index=False)
        summary.to_excel(writer, sheet_name="summary_95ci", index=False)
        paired.to_excel(writer, sheet_name="paired_tests", index=False)
        run_audit.to_excel(writer, sheet_name="run_audit", index=False)
        convergence.to_excel(writer, sheet_name="convergence", index=False)
    _save_figures(summary, metrics, config, figures_dir)
    _write_report(reports_dir / "benchmark_report.md", summary, paired, run_audit, convergence)
    write_completion(
        marker,
        input_signature,
        {
            "completed_workbooks": len(inventories),
            "expected_method_dataset_runs": len(run_audit),
            "metrics": len(metrics),
        },
    )
    print(f"[stage 3 complete] {workbook_path}")
    return workbook_path



