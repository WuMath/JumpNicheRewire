"""Analyze two completed case fits; never refit or read benchmark truth."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
from publication_study.analysis import enrich, read_gmt
from configs.public_sim_data.publication_study import StudyConfig

SOURCE = ROOT / "data/result/publication_20260920"
INPUT = ROOT / "data/processed/publication_20260920"
OUT = ROOT / "data/result/case_communication_20260926"
CASES = ("case_mouse_kidney", "case_human_breast_cancer")
SEED = 20260926


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plot_case(dest, name, coords, embedding, labels, edge, values, lr_name, component, terms):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    groups = sorted(pd.unique(labels))
    colors = plt.get_cmap("tab10")
    for group_index, group in enumerate(groups):
        mask = np.asarray(labels == group)
        axes[0, 0].scatter(coords.x[mask], coords.y[mask], s=2.3, color=colors(group_index), label=group.replace("expression_cluster_", "C"), rasterized=True)
        axes[0, 1].scatter(embedding[mask, 0], embedding[mask, 1], s=2.3, color=colors(group_index), rasterized=True)
    axes[0, 0].legend(ncol=3, fontsize=7, markerscale=3, loc="best")
    axes[0, 0].set_title("A  Spatial: RNA-derived clusters")
    axes[0, 1].set_title("B  UMAP: RNA-derived clusters")
    for ax, x, y, val, title, cmap in (
        (axes[0, 2], coords.x, coords.y, values, "C  Spatial: communication-associated Δβ", "coolwarm"),
        (axes[1, 0], embedding[:, 0], embedding[:, 1], values, "D  UMAP: communication-associated Δβ", "coolwarm"),
        (axes[1, 1], coords.x, coords.y, component, f"E  Spatial: {lr_name} contribution", "coolwarm"),
    ):
        bound = float(np.quantile(np.abs(val), 0.99)) or 1.0
        dots = ax.scatter(x, y, c=val, s=3, cmap=cmap, vmin=-bound, vmax=bound, rasterized=True)
        ax.set_title(title)
        fig.colorbar(dots, ax=ax, shrink=0.72)
    for ax in (axes[0, 0], axes[0, 2], axes[1, 1]):
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("image x"); ax.set_ylabel("image y")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax = axes[1, 2]
    display = terms.sort_values(["overlap", "pathway"], ascending=[False, True]).head(6).iloc[::-1]
    if len(display):
        names = [str(s)[:37] for s in display.pathway]
        ax.barh(names, display.overlap, color="#4e79a7")
        for y, count in enumerate(display.overlap):
            ax.text(count + 0.05, y, str(int(count)), va="center", fontsize=8)
        ax.set_xlim(0, max(display.overlap) + 1)
    ax.set_xlabel("Overlap genes in top-50 Δβ targets")
    ax.set_title("F  Pathway overlap; none FDR-significant")
    fig.suptitle(f"{name}: {edge[0]} → {edge[1]} | model-internal difference, not causality", fontsize=12)
    fig.savefig(dest / "spatial_umap_communication_pathway.png", dpi=190)
    fig.savefig(dest / "spatial_umap_communication_pathway.pdf")
    plt.close(fig)


def analyze(name: str) -> dict:
    print(f"{name}: loading cached inputs", flush=True)
    source = SOURCE / "02_runs" / name / "JumpNicheRewire"
    model_input = INPUT / name
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    state = json.loads((source / "run_status.json").read_text(encoding="utf-8"))
    if state.get("state") != "completed" or not state.get("convergence_passed"):
        raise ValueError(f"Cached fit is not accepted: {name}")
    expression = pd.read_csv(model_input / "expression.csv", index_col=0)
    cells = expression.index.astype(str)
    coords = pd.read_csv(model_input / "coordinates.csv", index_col=0).reindex(cells)
    metadata = pd.read_csv(model_input / "metadata.csv", index_col=0).reindex(cells)
    if coords.isna().any().any() or metadata.region.isna().any():
        raise ValueError("Unmatched coordinates or region labels")
    lr_ids = pd.read_csv(model_input / "lr_pairs.csv").lr_id.astype(str).tolist()
    cci = pd.read_parquet(source / "parquet/cci_edges.parquet", columns=["receiver_id", "lr_id", "score"])
    cci = cci.loc[cci.lr_id.ne("__all__")]
    exposure = (cci.groupby(["receiver_id", "lr_id"], sort=False).score.sum().unstack(fill_value=0)
                .reindex(cells, fill_value=0).reindex(columns=lr_ids, fill_value=0).to_numpy())
    with np.load(source / "mechanism_coefficients.npz", allow_pickle=False) as model:
        gamma = model["gamma"]
        active = model["active_lr_indices"]
        tfs = model["tf_names"].astype(str)
        targets = model["target_names"].astype(str)
        regions = model["regions"].astype(str)
    region_index = pd.Index(regions).get_indexer(metadata.region.astype(str))
    if (region_index < 0).any() or gamma.shape != (len(regions), len(active), len(tfs), len(targets)):
        raise ValueError("Saved region/coefficient axes do not match inputs")
    delta = np.einsum("ca,cakg->ckg", exposure[:, active], gamma[region_index], optimize=True)
    print(f"{name}: reconstructed cached coefficients", flush=True)
    preview = pd.read_parquet(source / "parquet/grn_edges.parquet").head(1000)
    i = cells.get_indexer(preview.cell_id)
    k = pd.Index(tfs).get_indexer(preview.tf)
    g = pd.Index(targets).get_indexer(preview.target)
    reconstruction_error = float(np.max(np.abs(delta[i, k, g] - preview.delta_beta.to_numpy())))
    if reconstruction_error > 1e-8:
        raise ValueError(f"Communication coefficient reconstruction failed: {reconstruction_error}")
    robust_range = np.quantile(delta, 0.9, axis=0) - np.quantile(delta, 0.1, axis=0)
    mean_abs = np.abs(delta).mean(axis=0)
    edge_table = pd.DataFrame({"tf": np.repeat(tfs, len(targets)), "target": np.tile(targets, len(tfs)),
                               "delta_q90_minus_q10": robust_range.ravel(), "mean_abs_delta": mean_abs.ravel()})
    global_grn = pd.read_parquet(source / "global_grn.parquet")
    edge_table = edge_table.merge(global_grn[["tf", "target", "score"]].rename(columns={"score": "global_grn_score"}),
                                  on=["tf", "target"], validate="one_to_one")
    edge_table = edge_table.sort_values(["delta_q90_minus_q10", "tf", "target"], ascending=[False, True, True])
    edge_table.to_csv(dest / "complete_edge_delta_summary.csv", index=False)
    global_grn.sort_values("score", ascending=False).head(50).to_csv(dest / "top_global_grn_edges.csv", index=False)
    top = edge_table.iloc[0]
    ki = pd.Index(tfs).get_loc(top.tf)
    gi = pd.Index(targets).get_loc(top.target)
    selected = delta[:, ki, gi]
    parts = exposure[:, active] * gamma[region_index, :, ki, gi]
    lr_position = int(np.argmax(np.mean(np.abs(parts), axis=0)))
    lr_name = lr_ids[int(active[lr_position])]
    component = parts[:, lr_position]
    pd.DataFrame({"cell_id": cells, "region": metadata.region.to_numpy(), "x": coords.x.to_numpy(),
                  "y": coords.y.to_numpy(), "delta_beta": selected, "dominant_lr_component": component}).to_csv(
                      dest / "selected_edge_spatial_values.csv", index=False)
    region_rows = []
    for region in regions:
        mask = metadata.region.eq(region).to_numpy()
        region_rows.append({"region": region, "spots": int(mask.sum()), "mean_delta_beta": float(selected[mask].mean()),
                            "median_delta_beta": float(np.median(selected[mask])),
                            "mean_abs_delta_beta": float(np.abs(selected[mask]).mean())})
    pd.DataFrame(region_rows).to_csv(dest / "selected_edge_region_summary.csv", index=False)
    normalized = np.log1p(expression.div(expression.sum(axis=1).clip(lower=1), axis=0) * 10000)
    print(f"{name}: computing UMAP", flush=True)
    components = PCA(n_components=min(30, len(expression) - 1, normalized.shape[1]), random_state=SEED).fit_transform(normalized)
    embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED, n_jobs=1).fit_transform(components)
    print(f"{name}: UMAP complete", flush=True)
    pd.DataFrame({"cell_id": cells, "UMAP1": embedding[:, 0], "UMAP2": embedding[:, 1]}).to_csv(dest / "expression_umap.csv", index=False)
    target_ranking = pd.DataFrame({"target": targets, "mean_abs_delta": mean_abs.max(axis=0)})
    target_ranking = target_ranking.sort_values(["mean_abs_delta", "target"], ascending=[False, True])
    target_ranking.to_csv(dest / "communication_target_ranking.csv", index=False)
    library = ROOT / "data/raw/publication_20260920/pathways" / ("WikiPathways_2024_Human.gmt" if name.endswith("breast_cancer") else "WikiPathways_2024_Mouse.gmt")
    terms = enrich({"top50_communication_delta_targets": target_ranking.target.head(50).tolist()},
                   targets, read_gmt(library), StudyConfig())
    print(f"{name}: enrichment complete", flush=True)
    terms.to_csv(dest / "communication_target_pathway_all_tests.csv", index=False)
    terms.loc[terms.significant].to_csv(dest / "communication_target_pathway_fdr05.csv", index=False)
    plot_case(dest, name, coords, embedding, metadata.region.to_numpy(), (top.tf, top.target), selected, lr_name, component, terms)
    diagnostics = pd.read_parquet(source / "parquet/causal_edges.parquet", columns=["diagnostic_pass"])
    result = {"case": name, "spots": len(cells), "fitted_tfs": len(tfs), "fitted_targets": len(targets),
              "selected_edge": [top.tf, top.target], "selected_edge_delta_q90_minus_q10": float(top.delta_q90_minus_q10),
              "selected_edge_mean_abs_delta": float(top.mean_abs_delta), "dominant_lr_component": lr_name,
              "pathway_tests": len(terms), "pathway_fdr05": int(terms.significant.sum()),
              "best_pathway": str(terms.iloc[0].pathway) if len(terms) else None,
              "best_q": float(terms.q_value.min()) if len(terms) else None,
              "causal_diagnostics_passed": int(diagnostics.diagnostic_pass.fillna(False).sum()),
              "causal_diagnostics_total": len(diagnostics), "reconstruction_max_abs_error": reconstruction_error,
              "model_status_sha256": sha256(source / "run_status.json"),
              "coefficient_sha256": sha256(source / "mechanism_coefficients.npz"),
              "reference_scope": "RNA-only, 1 tissue section, expression-derived clusters, model-internal delta; no causal proof"}
    (dest / "analysis_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    results = [analyze(name) for name in CASES]
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for record in results:
        print(record)
