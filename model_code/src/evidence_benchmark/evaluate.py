"""Scope-aware evaluation for exact, positive-unlabeled, and intervention truth."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .common import bh_fdr
from .config import EVALUATION_CONFIG, EvidenceEvaluationConfig


def _read_prediction(run_dir: Path, name: str) -> pd.DataFrame:
    parquet = run_dir / "parquet" / f"{name}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    workbooks = sorted(run_dir.glob("*_results.xlsx"))
    if workbooks:
        try:
            return pd.read_excel(workbooks[0], sheet_name=name)
        except ValueError:
            return pd.DataFrame()
    return pd.DataFrame()


def _tie_group_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold-based AP: every equal-score group enters in one step."""
    labels = np.asarray(labels, dtype=int)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    positives = int(labels.sum())
    if positives == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]) + 1, len(scores)]
    cumulative_tp = np.cumsum(labels)
    previous_recall = 0.0
    ap = 0.0
    for end in ends:
        tp = float(cumulative_tp[end - 1])
        recall = tp / positives
        precision = tp / end
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return float(ap)


def _tie_aware_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    if not positives or not negatives:
        return np.nan
    ranks = stats.rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _early_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    k = int(labels.sum())
    if k == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    boundary = scores[k - 1]
    above, tied = scores > boundary, scores == boundary
    slots = k - int(above.sum())
    expected = labels[above].sum() + slots * labels[tied].mean()
    return float(expected / k)


def _bootstrap_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    clusters: np.ndarray,
    metric,
    config: EvidenceEvaluationConfig,
    seed_key: str,
) -> tuple[float, float]:
    """Cluster bootstrap; repeated TF/LR groups, not individual edges, are resampled."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    clusters = np.asarray(clusters, dtype=object)
    units = np.unique(clusters)
    if len(units) < 2 or config.bootstrap_iterations < 1:
        return np.nan, np.nan
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16) ^ config.random_seed
    random = np.random.default_rng(seed)
    unit_rows = {unit: np.flatnonzero(clusters == unit) for unit in units}
    estimates: list[float] = []
    for _ in range(config.bootstrap_iterations):
        sampled = random.choice(units, size=len(units), replace=True)
        indices = np.concatenate([unit_rows[unit] for unit in sampled])
        value = metric(labels[indices], scores[indices])
        if np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return np.nan, np.nan
    alpha = (1.0 - config.confidence_level) / 2.0
    return tuple(float(value) for value in np.quantile(estimates, [alpha, 1.0 - alpha]))


def _rank_rows(
    method: str,
    dataset_id: str,
    track: str,
    task: str,
    truth_kind: str,
    labels,
    scores,
    clusters,
    config: EvidenceEvaluationConfig,
) -> list[dict[str, object]]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prefix = "PU-" if truth_kind == "positive_unlabeled" else ""
    metric_functions = {
        prefix + "Average Precision": _tie_group_average_precision,
        prefix + "AUROC": _tie_aware_auc,
        prefix + "Early Precision": _early_precision,
        "Positive prevalence": lambda values, _: float(np.mean(values)) if len(values) else np.nan,
    }
    rows = []
    for name, function in metric_functions.items():
        value = function(labels, scores)
        low, high = _bootstrap_interval(
            labels, scores, np.asarray(clusters), function, config,
            f"{dataset_id}|{task}|{name}",
        )
        rows.append({
            "method": method, "dataset_id": dataset_id, "evidence_track": track, "task": task,
            "truth_kind": truth_kind, "metric": name, "value": value,
            "ci_lower": low, "ci_upper": high,
            "ci_method": f"{config.confidence_level:.0%} cluster bootstrap",
            "bootstrap_iterations": config.bootstrap_iterations,
            "candidates": len(labels), "positives": int(labels.sum()),
        })
    return rows


def _aggregate_grn(prediction: pd.DataFrame) -> pd.DataFrame:
    if prediction.empty:
        return pd.DataFrame(columns=["tf", "target", "score", "signed_score"])
    prediction = prediction.copy()
    prediction["score"] = pd.to_numeric(prediction["score"], errors="coerce").fillna(0.0)
    if "signed_score" not in prediction:
        prediction["signed_score"] = prediction["score"]
    return prediction.groupby(["tf", "target"], as_index=False).agg(
        score=("score", "max"), signed_score=("signed_score", "mean")
    )


def _evaluate_grn(
    method: str, row, run_dir: Path, truth_name: str, task: str, config: EvidenceEvaluationConfig
):
    truth = pd.read_csv(Path(row.truth_dir) / truth_name)
    raw_prediction = _read_prediction(run_dir, "grn_edges")
    supports_direction = bool(
        "signed_score" in raw_prediction
        and pd.to_numeric(raw_prediction["signed_score"], errors="coerce").notna().any()
    )
    prediction = _aggregate_grn(raw_prediction)
    aligned = truth.merge(prediction, on=["tf", "target"], how="left").fillna({"score": 0.0, "signed_score": 0.0})
    metrics = _rank_rows(
        method, row.dataset_id, row.evidence_track, task, row.truth_kind,
        aligned.label, aligned.score, aligned.tf, config,
    )
    significant = aligned[
        aligned.label.eq(1) & pd.to_numeric(aligned.signed_truth, errors="coerce").ne(0)
    ] if supports_direction else aligned.iloc[0:0]
    if not significant.empty:
        direction = float((np.sign(significant.signed_score) == np.sign(significant.signed_truth)).mean())
        direction_labels = (
            np.sign(significant.signed_score.to_numpy(dtype=float))
            == np.sign(significant.signed_truth.to_numpy(dtype=float))
        ).astype(int)
        low, high = _bootstrap_interval(
            direction_labels, np.zeros(len(direction_labels)), significant.tf.to_numpy(),
            lambda values, _: float(np.mean(values)), config,
            f"{row.dataset_id}|{task}|Direction accuracy",
        )
        metrics.append({
            "method": method, "dataset_id": row.dataset_id,
            "evidence_track": row.evidence_track, "task": task,
            "truth_kind": row.truth_kind, "metric": "Direction accuracy", "value": direction,
            "ci_lower": low, "ci_upper": high,
            "ci_method": f"{config.confidence_level:.0%} TF-cluster bootstrap",
            "bootstrap_iterations": config.bootstrap_iterations,
            "candidates": len(aligned), "positives": int(aligned.label.sum()),
        })
    return metrics, aligned


def _evaluate_sccigen(method: str, row, run_dir: Path, config: EvidenceEvaluationConfig):
    truth = pd.read_csv(Path(row.truth_dir) / "truth_sccigen_cci.csv")
    prediction = _read_prediction(run_dir, "cci_edges")
    metadata = pd.read_csv(Path(row.dataset_dir) / "metadata.csv", dtype={"cell_id": str}).set_index("cell_id")
    if prediction.empty:
        grouped = pd.DataFrame(columns=["sender_cell_type", "receiver_cell_type", "ligand", "receptor", "score"])
    else:
        prediction = prediction[prediction.lr_id.astype(str).ne("__all__")].copy()
        if "sender_cell_type" not in prediction or "receiver_cell_type" not in prediction:
            prediction["sender_cell_type"] = prediction.sender_id.astype(str).map(metadata.cell_type)
            prediction["receiver_cell_type"] = prediction.receiver_id.astype(str).map(metadata.cell_type)
        grouped = prediction.groupby(
            ["sender_cell_type", "receiver_cell_type", "ligand", "receptor"], as_index=False, dropna=False
        ).score.max()
    keys = ["sender_cell_type", "receiver_cell_type", "ligand", "receptor"]
    aligned = truth.merge(grouped, on=keys, how="left").fillna({"score": 0.0})
    return _rank_rows(
        method, row.dataset_id, row.evidence_track, "sCCIgen implanted cross-cell gene-pair recovery",
        row.truth_kind, aligned.label, aligned.score, aligned.lr_id, config,
    ), aligned


def _evaluate_real_lr(method: str, row, run_dir: Path, config: EvidenceEvaluationConfig):
    truth = pd.read_csv(Path(row.truth_dir) / "truth_pseudo_lr.csv")
    prediction = _read_prediction(run_dir, "cci_edges")
    if prediction.empty:
        grouped = pd.DataFrame(columns=["lr_id", "score"])
    else:
        grouped = prediction[prediction.lr_id.astype(str).ne("__all__")].groupby("lr_id", as_index=False).score.max()
    aligned = truth.merge(grouped, on="lr_id", how="left").fillna({"score": 0.0})
    return _rank_rows(
        method, row.dataset_id, row.evidence_track, "Real curated LR retrieval", row.truth_kind,
        aligned.label, aligned.score, aligned.ligand, config,
    ), aligned


def _mean_interval(values: np.ndarray, config: EvidenceEvaluationConfig, seed_key: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or config.bootstrap_iterations < 1:
        return np.nan, np.nan
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16) ^ config.random_seed
    random = np.random.default_rng(seed)
    estimates = np.asarray([
        random.choice(values, size=len(values), replace=True).mean()
        for _ in range(config.bootstrap_iterations)
    ])
    alpha = (1.0 - config.confidence_level) / 2.0
    return tuple(float(value) for value in np.quantile(estimates, [alpha, 1.0 - alpha]))


def _summarize(metrics: pd.DataFrame, config: EvidenceEvaluationConfig) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    keys = ["method", "evidence_track", "task", "truth_kind", "metric"]
    for group_key, group in metrics.groupby(keys, dropna=False):
        values = group.value.to_numpy(dtype=float)
        low, high = _mean_interval(values, config, "summary|" + "|".join(map(str, group_key)))
        rows.append({
            **dict(zip(keys, group_key)), "mean": float(np.nanmean(values)),
            "ci_lower": low, "ci_upper": high, "datasets": int(group.dataset_id.nunique()),
            "ci_method": f"{config.confidence_level:.0%} dataset bootstrap",
        })
    return pd.DataFrame(rows)


def _paired_comparisons(metrics: pd.DataFrame, config: EvidenceEvaluationConfig) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    usable = metrics[metrics.metric.ne("Positive prevalence")]
    keys = ["evidence_track", "task", "truth_kind", "metric"]
    for group_key, group in usable.groupby(keys, dropna=False):
        pivot = group.pivot_table(index="dataset_id", columns="method", values="value", aggfunc="first")
        if config.method_name not in pivot:
            continue
        for comparator in sorted(set(pivot.columns) - {config.method_name}):
            paired = pivot[[config.method_name, comparator]].dropna()
            if paired.empty:
                continue
            differences = paired[config.method_name].to_numpy() - paired[comparator].to_numpy()
            low, high = _mean_interval(
                differences, config, "paired|" + "|".join(map(str, (*group_key, comparator)))
            )
            if len(differences) >= 2 and np.any(differences != 0):
                pvalue = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
            elif len(differences) >= 2:
                pvalue = 1.0
            else:
                pvalue = np.nan
            rows.append({
                **dict(zip(keys, group_key)), "primary_method": config.method_name,
                "comparator": comparator, "paired_datasets": len(paired),
                "mean_difference": float(np.mean(differences)),
                "ci_lower": low, "ci_upper": high, "wilcoxon_pvalue": pvalue,
                "primary_wins": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
                "comparator_wins": int((differences < 0).sum()),
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["wilcoxon_fdr"] = bh_fdr(result.wilcoxon_pvalue.to_numpy(dtype=float))
    return result


def run_evaluation(config: EvidenceEvaluationConfig = EVALUATION_CONFIG) -> Path:
    registry = pd.read_csv(config.registry_path)
    config.output_root.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    details: dict[str, list[pd.DataFrame]] = {}
    audit_rows = []
    for row in registry.itertuples(index=False):
        run_candidates = [(config.method_name, config.run_root / str(row.evidence_track) / str(row.dataset_id))]
        if config.external_result_root.exists():
            run_candidates.extend(
                (method_dir.name, method_dir / str(row.dataset_id))
                for method_dir in sorted(config.external_result_root.iterdir()) if method_dir.is_dir()
            )
        for method, run_dir in run_candidates:
            status_path = run_dir / "run_status.json"
            state = "missing"
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                state = status.get("state", "unknown")
            has_grn = not _read_prediction(run_dir, "grn_edges").empty if state == "completed" else False
            has_cci = not _read_prediction(run_dir, "cci_edges").empty if state == "completed" else False
            if state == "completed":
                if row.evaluation_scope == "sccigen_cci" and not has_cci:
                    state = "missing_required_task:cci_edges"
                elif row.evaluation_scope in {"grn_only", "perturbation_grn"} and not has_grn:
                    state = "missing_required_task:grn_edges"
                elif row.evaluation_scope == "real_pseudotruth" and not (has_grn or has_cci):
                    state = "missing_required_task:grn_edges_or_cci_edges"
            audit_rows.append({
                "method": method, "dataset_id": row.dataset_id,
                "evidence_track": row.evidence_track, "state": state,
                "has_grn_edges": has_grn, "has_cci_edges": has_cci,
            })
            if state != "completed":
                continue
            if row.evaluation_scope == "grn_only":
                rows, detail = _evaluate_grn(
                    method, row, run_dir, "truth_grn.csv", "SERGIO GRN recovery", config
                )
                details.setdefault("sergio_grn", []).append(
                    detail.assign(dataset_id=row.dataset_id, method=method)
                )
            elif row.evaluation_scope == "sccigen_cci":
                rows, detail = _evaluate_sccigen(method, row, run_dir, config)
                details.setdefault("sccigen_cci", []).append(
                    detail.assign(dataset_id=row.dataset_id, method=method)
                )
            elif row.evaluation_scope == "real_pseudotruth":
                rows = []
                if has_grn:
                    grn_rows, detail = _evaluate_grn(
                        method, row, run_dir, "truth_pseudo_grn.csv", "Real curated TF-target retrieval", config
                    )
                    rows.extend(grn_rows)
                    details.setdefault("real_grn", []).append(
                        detail.assign(dataset_id=row.dataset_id, method=method)
                    )
                if has_cci:
                    lr_rows, lr_detail = _evaluate_real_lr(method, row, run_dir, config)
                    rows.extend(lr_rows)
                    details.setdefault("real_lr", []).append(
                        lr_detail.assign(dataset_id=row.dataset_id, method=method)
                    )
            elif row.evaluation_scope == "perturbation_grn":
                rows, detail = _evaluate_grn(
                    method, row, run_dir, "truth_perturbation_grn.csv", "Held-out knockout response", config
                )
                details.setdefault("perturbation_grn", []).append(
                    detail.assign(dataset_id=row.dataset_id, method=method)
                )
            else:
                raise ValueError(f"Unknown evaluation scope: {row.evaluation_scope}")
            metric_rows.extend(rows)
    metrics = pd.DataFrame(metric_rows)
    audit = pd.DataFrame(audit_rows)
    summary = _summarize(metrics, config)
    paired = _paired_comparisons(metrics, config)
    metrics.to_csv(config.output_root / "evidence_metrics.csv", index=False)
    audit.to_csv(config.output_root / "evidence_run_audit.csv", index=False)
    summary.to_csv(config.output_root / "evidence_summary.csv", index=False)
    paired.to_csv(config.output_root / "evidence_paired_comparisons.csv", index=False)
    for name, frames in details.items():
        pd.concat(frames, ignore_index=True).to_parquet(config.output_root / f"{name}_detail.parquet", index=False)
    workbook = config.output_root / "evidence_benchmark_summary.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        paired.to_excel(writer, sheet_name="paired_comparisons", index=False)
        audit.to_excel(writer, sheet_name="run_audit", index=False)
    report = [
        "# Independent evidence benchmark", "",
        "Exact simulation, positive-unlabeled pseudo-truth, and intervention truth are reported separately.",
        "They must never be pooled into one accuracy number.", "",
    ]
    if not summary.empty:
        report.extend(["```text", summary.to_string(index=False), "```", ""])
    report.extend([
        "## Interpretation", "",
        "- SERGIO: GRN only; adapter coordinates are not spatial evidence.",
        "- sCCIgen: exact implanted cross-cell effects aggregated by cell type and synthetic gene pair; these are not curated biological LR pairs.",
        "- Real pseudo-truth: PU metrics; unlabeled edges are not proven negatives.",
        "- Perturbation: model saw safe controls only; knockout cells were revealed during evaluation.",
    ])
    (config.output_root / "EVIDENCE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[evidence evaluation] {workbook}")
    return workbook



