"""Recompute both paper tables and selected case summaries from frozen scores."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
METHODS = ("CeSpGRN", "CLR", "GENIE3", "GRNBoost2",
           "JumpNicheRewire", "PPCOR", "RegDiffusion")


def precision_at_k(label: np.ndarray, score: np.ndarray, k: int) -> tuple[float, float]:
    boundary = np.sort(score)[-k]
    above, tied = score > boundary, score == boundary
    hits = label[above].sum() + (k - above.sum()) * label[tied].mean()
    return float(hits / k), float(hits / label.sum())


def fbeta(precision: float, recall: float, beta: float) -> float:
    denominator = beta * beta * precision + recall
    return (1 + beta * beta) * precision * recall / denominator if denominator else 0.0


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    label = frame.label.to_numpy(dtype=int)
    score = frame.score.to_numpy(dtype=float)
    positives = int(label.sum())
    decision_k = math.ceil(0.1 * len(label))
    early, _ = precision_at_k(label, score, positives)
    precision, recall = precision_at_k(label, score, decision_k)
    return {"AP": float(average_precision_score(label, score)),
            "AUROC": float(roc_auc_score(label, score)),
            "Early_precision": early,
            "F1": fbeta(precision, recall, 1.0),
            "F0_1": fbeta(precision, recall, 0.1)}


def evaluate(dataset: str) -> None:
    rows = []
    expected = pd.read_csv(ROOT / "results" / dataset / "source_metrics.csv")
    for method in METHODS:
        path = ROOT / "results" / dataset / "scores" / f"{method}.csv"
        frame = pd.read_csv(path)
        if frame.duplicated(["tf", "target"]).any() or not np.isfinite(frame.score).all():
            raise ValueError(f"Invalid scores: {path}")
        observed = metrics(frame)
        native = "JumpNicheRewire_communication" if method == "JumpNicheRewire" and dataset == "p22" else method
        reference = expected.loc[expected.method.eq(native)]
        if len(reference) != 1:
            raise ValueError(f"Source row missing: {dataset}/{method}")
        ref = reference.iloc[0]
        for metric in ("AP", "AUROC", "Early_precision"):
            if not np.isclose(observed[metric], ref[metric], atol=1e-12, rtol=0):
                raise ValueError(f"Metric mismatch: {dataset}/{method}/{metric}")
        if dataset == "p22":
            for metric in ("F1", "F0_1"):
                if not np.isclose(observed[metric], ref[metric], atol=1e-12, rtol=0):
                    raise ValueError(f"Metric mismatch: {dataset}/{method}/{metric}")
        rows.append({"method": method, **observed})
    output = pd.DataFrame(rows)
    print(f"\n{dataset}: candidates={len(frame)}, positives={int(frame.label.sum())}")
    print(output.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def cases() -> None:
    for case in ("case_mouse_kidney", "case_human_breast_cancer"):
        folder = ROOT / "results/cases" / case
        values = pd.read_csv(folder / "selected_edge_spatial_values.csv")
        summary = json.loads((folder / "analysis_summary.json").read_text(encoding="utf8"))
        qspread = float(values.delta_beta.quantile(0.9) - values.delta_beta.quantile(0.1))
        mean_abs = float(values.delta_beta.abs().mean())
        if len(values) != summary["spots"] or not np.isclose(qspread, summary["selected_edge_delta_q90_minus_q10"], atol=1e-12) or not np.isclose(mean_abs, summary["selected_edge_mean_abs_delta"], atol=1e-12):
            raise ValueError(f"Case summary mismatch: {case}")
        print(f"{case}: spots={len(values)}, q90-q10={qspread:.4f}, mean|delta|={mean_abs:.4f}")
        edges = pd.read_csv(folder / "complete_edge_delta_summary.csv")
        if not edges.delta_q90_minus_q10.is_monotonic_decreasing:
            raise ValueError(f"Edge ordering mismatch: {case}")
        print(edges[["tf", "target", "delta_q90_minus_q10", "mean_abs_delta"]]
              .head(3).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    evaluate("scmultisim")
    evaluate("p22")
    cases()
    print("PASS: fixed-score paper results reproduced.")
