"""Rebuild spatial case tables from a fresh fit without touching paper results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap


def analyze(name: str, rebuilt: bool) -> Path:
    inputs = (ROOT / "reproduction_runs/prepared" if rebuilt else ROOT / "model_inputs") / name
    fitted = ROOT / "reproduction_runs/fits" / name
    out = ROOT / "reproduction_runs/cases" / name
    if out.exists():
        raise FileExistsError(out)
    state = json.loads((fitted / "run_status.json").read_text(encoding="utf-8"))
    if state.get("state") != "completed":
        raise ValueError(f"Fresh fit is incomplete: {name}")
    expression = pd.read_csv(inputs / "expression.csv", index_col=0)
    cells = expression.index.astype(str)
    metadata = pd.read_csv(inputs / "metadata.csv", index_col="cell_id").reindex(cells)
    coords = pd.read_csv(inputs / "coordinates.csv", index_col=0).reindex(cells)
    lr_ids = pd.read_csv(inputs / "lr_pairs.csv").lr_id.astype(str).tolist()
    cci = pd.read_parquet(fitted / "parquet/cci_edges.parquet", columns=["receiver_id", "lr_id", "score"])
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
    ri = pd.Index(regions).get_indexer(metadata.region.astype(str))
    if (ri < 0).any():
        raise ValueError("Fitted region labels do not match the input")
    delta = np.einsum("ca,cakg->ckg", exposure[:, active], gamma[ri], optimize=True)
    spread = np.quantile(delta, .9, axis=0) - np.quantile(delta, .1, axis=0)
    mean_abs = np.abs(delta).mean(axis=0)
    edges = pd.DataFrame({"tf": np.repeat(tfs, len(targets)),
                          "target": np.tile(targets, len(tfs)),
                          "delta_q90_minus_q10": spread.ravel(),
                          "mean_abs_delta": mean_abs.ravel()})
    edges = edges.sort_values(["delta_q90_minus_q10", "tf", "target"],
                              ascending=[False, True, True]).reset_index(drop=True)
    top = edges.iloc[0]
    k, g = pd.Index(tfs).get_loc(top.tf), pd.Index(targets).get_loc(top.target)
    parts = exposure[:, active] * gamma[ri, :, k, g]
    strongest = int(np.argmax(np.abs(parts).mean(axis=0)))
    dominant = lr_ids[int(active[strongest])]
    out.mkdir(parents=True)
    edges.to_csv(out / "complete_edge_delta_summary.csv", index=False)
    pd.DataFrame({"cell_id": cells, "region": metadata.region.to_numpy(),
                  "x": coords.x.to_numpy(), "y": coords.y.to_numpy(),
                  "delta_beta": delta[:, k, g],
                  "dominant_lr_component": parts[:, strongest]}).to_csv(
                      out / "selected_edge_spatial_values.csv", index=False)
    region_rows = []
    for region in regions:
        mask = metadata.region.eq(region).to_numpy()
        region_rows.append({"region": region, "spots": int(mask.sum()),
                            "mean_delta_beta": float(delta[mask, k, g].mean()),
                            "median_delta_beta": float(np.median(delta[mask, k, g])),
                            "mean_abs_delta_beta": float(np.abs(delta[mask, k, g]).mean())})
    pd.DataFrame(region_rows).to_csv(out / "selected_edge_region_summary.csv", index=False)
    normalized = np.log1p(expression.div(expression.sum(axis=1).clip(lower=1), axis=0) * 10000)
    pcs = PCA(n_components=min(30, len(expression)-1, normalized.shape[1]),
              random_state=20260926).fit_transform(normalized)
    embedding = umap.UMAP(n_neighbors=30, min_dist=.3, random_state=20260926,
                          n_jobs=1).fit_transform(pcs)
    pd.DataFrame({"cell_id": cells, "UMAP1": embedding[:, 0],
                  "UMAP2": embedding[:, 1]}).to_csv(out / "expression_umap.csv", index=False)
    summary = {"case": name, "spots": len(cells), "fitted_tfs": len(tfs),
               "fitted_targets": len(targets), "selected_edge": [str(top.tf), str(top.target)],
               "selected_edge_delta_q90_minus_q10": float(top.delta_q90_minus_q10),
               "selected_edge_mean_abs_delta": float(top.mean_abs_delta),
               "dominant_lr_component": dominant}
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("case_mouse_kidney", "case_human_breast_cancer"))
    parser.add_argument("--rebuilt-input", action="store_true")
    args = parser.parse_args()
    print(analyze(args.case, args.rebuilt_input))
