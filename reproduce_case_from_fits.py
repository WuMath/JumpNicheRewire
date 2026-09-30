"""Independently reconstruct reported spatial deltas from saved model coefficients."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CASES = ("case_mouse_kidney", "case_human_breast_cancer")


def check_case(case: str) -> None:
    fitted = ROOT / "model_fits" / case
    inputs = ROOT / "model_inputs" / case
    reported = ROOT / "results/cases" / case
    status = json.loads((fitted / "run_status.json").read_text(encoding="utf8"))
    if status.get("state") != "completed" or not status.get("convergence_passed"):
        raise ValueError(f"Fit status failed: {case}")
    cells = pd.read_csv(inputs / "expression.csv", index_col=0).index.astype(str)
    region = pd.read_csv(inputs / "metadata.csv", index_col=0).reindex(cells).region.astype(str)
    lr_ids = pd.read_csv(inputs / "lr_pairs.csv").lr_id.astype(str).tolist()
    cci = pd.read_parquet(fitted / "parquet/cci_edges.parquet",
                          columns=["receiver_id", "lr_id", "score"])
    cci = cci.loc[cci.lr_id.ne("__all__")]
    exposure = (cci.groupby(["receiver_id", "lr_id"], sort=False).score.sum()
                .unstack(fill_value=0).reindex(cells, fill_value=0)
                .reindex(columns=lr_ids, fill_value=0).to_numpy())
    with np.load(fitted / "mechanism_coefficients.npz", allow_pickle=False) as model:
        gamma = model["gamma"]
        active = model["active_lr_indices"]
        tfs = model["tf_names"].astype(str)
        targets = model["target_names"].astype(str)
        regions = model["regions"].astype(str)
    region_index = pd.Index(regions).get_indexer(region)
    if (region_index < 0).any():
        raise ValueError(f"Unmatched region: {case}")
    delta = np.einsum("ca,cakg->ckg", exposure[:, active], gamma[region_index], optimize=True)
    qspread = np.quantile(delta, 0.9, axis=0) - np.quantile(delta, 0.1, axis=0)
    mean_abs = np.abs(delta).mean(axis=0)
    edge_values = pd.DataFrame({"tf": np.repeat(tfs, len(targets)),
                                "target": np.tile(targets, len(tfs)),
                                "delta_q90_minus_q10": qspread.ravel(),
                                "mean_abs_delta": mean_abs.ravel()})
    old = pd.read_csv(reported / "complete_edge_delta_summary.csv")
    compared = old.merge(edge_values, on=["tf", "target"], validate="one_to_one",
                         suffixes=("_reported", "_recomputed"))
    if len(compared) != len(old):
        raise ValueError(f"Missing edge: {case}")
    for metric in ("delta_q90_minus_q10", "mean_abs_delta"):
        if not np.allclose(compared[f"{metric}_reported"],
                           compared[f"{metric}_recomputed"], atol=1e-12, rtol=0):
            raise ValueError(f"Mismatch: {case}/{metric}")
    selected = pd.read_csv(reported / "selected_edge_spatial_values.csv")
    top = old.iloc[0]
    k = pd.Index(tfs).get_loc(top.tf)
    g = pd.Index(targets).get_loc(top.target)
    if not np.allclose(delta[:, k, g], selected.delta_beta, atol=1e-12, rtol=0):
        raise ValueError(f"Spot-level delta mismatch: {case}")
    parts = exposure[:, active] * gamma[region_index, :, k, g]
    dominant = int(np.argmax(np.mean(np.abs(parts), axis=0)))
    if not np.allclose(parts[:, dominant], selected.dominant_lr_component,
                       atol=1e-12, rtol=0):
        raise ValueError(f"LR component mismatch: {case}")
    name = lr_ids[int(active[dominant])]
    summary = json.loads((reported / "analysis_summary.json").read_text(encoding="utf8"))
    if name != summary["dominant_lr_component"]:
        raise ValueError(f"LR name mismatch: {case}")
    print(f"{case}: {len(cells)} spots, {len(old)} edges, dominant LR {name}; PASS")


if __name__ == "__main__":
    for case in CASES:
        check_case(case)
