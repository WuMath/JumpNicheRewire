"""Compute manuscript Table 3 directly from fitted case coefficients and LR exposure.

The script never reads the reported table or its precomputed edge summaries.
Run from any directory with the Python environment used for JumpNicheRewire.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("case_mouse_kidney", "Mouse kidney"),
    ("case_human_breast_cancer", "Human breast cancer"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_case(name: str, inputs: Path, fits: Path) -> tuple[pd.DataFrame, dict]:
    source = inputs / name
    fit = fits / name
    status_file = fit / "run_status.json"
    coefficient_file = fit / "mechanism_coefficients.npz"
    cci_file = fit / "parquet" / "cci_edges.parquet"
    expression_file = source / "expression.csv"
    metadata_file = source / "metadata.csv"
    lr_file = source / "lr_pairs.csv"

    status = json.loads(status_file.read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"Incomplete fit: {name}")
    if status.get("convergence_passed") is False:
        raise ValueError(f"Fit failed convergence check: {name}")

    cells = pd.read_csv(expression_file, index_col=0).index.astype(str)
    if not cells.is_unique:
        raise ValueError(f"Duplicate spot identifiers: {name}")
    metadata = pd.read_csv(metadata_file, index_col=0).reindex(cells)
    if metadata.region.isna().any():
        raise ValueError(f"Missing region labels: {name}")
    lr_ids = pd.read_csv(lr_file).lr_id.astype(str).tolist()
    if len(lr_ids) != len(set(lr_ids)):
        raise ValueError(f"Duplicate LR identifiers: {name}")

    cci = pd.read_parquet(cci_file, columns=["receiver_id", "lr_id", "score"])
    cci = cci.loc[cci.lr_id.ne("__all__")].copy()
    if not set(cci.receiver_id.astype(str)).issubset(set(cells)):
        raise ValueError(f"Unknown receiver in LR exposure: {name}")
    if not set(cci.lr_id.astype(str)).issubset(set(lr_ids)):
        raise ValueError(f"Unknown LR identifier in exposure: {name}")
    if not np.isfinite(cci.score.to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite LR exposure: {name}")
    exposure = (cci.groupby(["receiver_id", "lr_id"], sort=False).score.sum()
                .unstack(fill_value=0).reindex(cells, fill_value=0)
                .reindex(columns=lr_ids, fill_value=0).to_numpy(dtype=float))

    with np.load(coefficient_file, allow_pickle=False) as model:
        gamma = model["gamma"]
        active = model["active_lr_indices"].astype(int)
        tfs = model["tf_names"].astype(str)
        targets = model["target_names"].astype(str)
        regions = model["regions"].astype(str)
    if gamma.shape != (len(regions), len(active), len(tfs), len(targets)):
        raise ValueError(f"Coefficient dimensions do not match edge identifiers: {name}")
    if (active < 0).any() or (active >= len(lr_ids)).any():
        raise ValueError(f"Active LR indices outside the input catalog: {name}")
    region_index = pd.Index(regions).get_indexer(metadata.region.astype(str))
    if (region_index < 0).any():
        raise ValueError(f"Unknown fitted region: {name}")

    # One value for each spot and fitted TF--target edge.
    delta = np.einsum("ia,iakg->ikg", exposure[:, active],
                      gamma[region_index], optimize=True)
    if not np.isfinite(delta).all():
        raise ValueError(f"Nonfinite edge change: {name}")
    q10 = np.quantile(delta, 0.10, axis=0, method="linear")
    q90 = np.quantile(delta, 0.90, axis=0, method="linear")
    mean_abs = np.abs(delta).mean(axis=0)
    edges = pd.DataFrame({
        "tf": np.repeat(tfs, len(targets)),
        "target": np.tile(targets, len(tfs)),
        "delta_q90_minus_q10": (q90 - q10).ravel(),
        "mean_abs_delta": mean_abs.ravel(),
    }).sort_values(["delta_q90_minus_q10", "tf", "target"],
                   ascending=[False, True, True]).reset_index(drop=True)
    audit = {
        "case": name,
        "spots": len(cells),
        "fitted_edges": len(edges),
        "input_expression_sha256": sha256(expression_file),
        "input_metadata_sha256": sha256(metadata_file),
        "input_lr_sha256": sha256(lr_file),
        "fit_coefficients_sha256": sha256(coefficient_file),
        "fit_exposure_sha256": sha256(cci_file),
        "quantile_method": "linear",
        "ranking": "descending q90-minus-q10; TF and target break ties",
    }
    return edges, audit


def latex_table(rows: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{The three TF--target edges with the largest spatial quantile differences in each section. Each difference is the 90th minus the 10th percentile of the fitted edge change over all spots. Mean absolute change is the arithmetic mean of its absolute value.}",
        r"\label{tab:caseedgevariation}",
        r"\begin{tabular}{llrr}\hline",
        r"Section & TF--target & Quantile difference & Mean absolute change\\\hline",
    ]
    for section, group in rows.groupby("section", sort=False):
        for index, row in enumerate(group.itertuples(index=False)):
            label = section if index == 0 else ""
            lines.append(
                f"{label} & \\textit{{{row.tf}}}$\\rightarrow$\\textit{{{row.target}}} "
                f"& {row.delta_q90_minus_q10:.4f} & {row.mean_abs_delta:.4f}\\\\"
            )
    lines.extend([r"\hline\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "model_inputs",
                        help="Root containing the two case input directories")
    parser.add_argument("--fits", type=Path, default=ROOT / "model_fits",
                        help="Root containing the two completed case fit directories")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reproduction_runs" / "table3_recomputed",
                        help="Directory for newly computed CSV, LaTeX, and audit files")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    top = []
    audits = []
    for name, section in CASES:
        edges, audit = compute_case(name, args.inputs, args.fits)
        edges.to_csv(args.output / f"{name}_all_edges.csv", index=False)
        top.append(edges.head(3).assign(section=section))
        audits.append(audit)
    table = pd.concat(top, ignore_index=True)
    table = table[["section", "tf", "target", "delta_q90_minus_q10", "mean_abs_delta"]]
    table.to_csv(args.output / "table3_top3.csv", index=False)
    (args.output / "table3_case_edges.tex").write_text(latex_table(table), encoding="utf-8")
    (args.output / "calculation_audit.json").write_text(
        json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
