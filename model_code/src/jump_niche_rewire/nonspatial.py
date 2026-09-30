"""Non-spatial, global GRN ablation; no coordinate or region-derived operations.

This is NOT the full spatial/communication model. Reuse the weighted sparse
regression prior and the existing KKT-certified FISTA solver with singleton
groups (weighted L1), fitting one coefficient per TF-target across all cells.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .types import SpatialResult, SpatialGraph, BaseGRNResult, CommunicationResult
from .utils import standardize
from .spatial import normalize_expression
from .solvers import group_fista


def build_nonspatial_context(bundle, data_config):
    normalized = normalize_expression(bundle.expression, data_config)
    raw = bundle.expression.to_numpy(dtype=float)
    detection, totals = (raw > 0).mean(axis=0), raw.sum(axis=0)
    keep = ((detection >= data_config.minimum_detection_rate)
            & (totals >= data_config.minimum_total_count))
    genes = sorted(bundle.expression.columns[keep].astype(str))
    tfs = sorted(set(bundle.tf_names).intersection(genes))
    if not genes or not tfs:
        raise ValueError("Non-spatial expression QC left no genes or no expressed TFs")
    # Do not remove TF genes from the target set: TF_A -> TF_B is admissible.
    stats = pd.DataFrame({"gene": genes, "selected": True,
                          "selection_rank": np.arange(1, len(genes)+1),
                          "fisher_q": np.nan,
                          "selection_note": "expression_QC_only; spatial_test_not_applicable"})
    empty = np.empty(0, dtype=float)
    graph = SpatialGraph(source=np.empty(0, dtype=np.int64), target=np.empty(0, dtype=np.int64),
                         distance=empty, distance_weight=empty, state_weight=empty, weight=empty,
                         bandwidth=np.ones(len(raw)), state_embedding=np.empty((len(raw), 0)))
    bundle.audit = pd.concat([bundle.audit, pd.DataFrame([
        dict(item="model_variant", available=True, detail="nonspatial_global_grn_v1", action="weighted L1; spatial fusion and communication disabled"),
        dict(item="gene_selection", available=True, detail=f"{len(tfs)} TFs x {len(genes)} targets", action="expression QC only; all non-self pairs; no spatial cap"),
        dict(item="region", available=False, detail="not applicable", action="not used in global fit or evaluation"),
    ])], ignore_index=True)
    return SpatialResult(normalized, genes, [], stats, graph, tfs, genes, [])


def fit_global_grn(bundle, spatial, prior, data_config, config):
    tfs, targets = spatial.candidate_tfs, spatial.candidate_targets
    # Refuse silent truncation: the user must explicitly allow the whole panel.
    if len(tfs) > config.maximum_tfs or len(targets) > config.maximum_targets:
        raise ValueError(f"Non-spatial complete panel needs maximum_tfs >= {len(tfs)} and maximum_targets >= {len(targets)}")
    if config.hard_prior_threshold is not None:
        raise ValueError("Non-spatial common-panel benchmark requires hard_prior_threshold=None")
    x, _, _ = standardize(spatial.normalized_expression[tfs].to_numpy(float), axis=0)
    y, mean, scale = standardize(spatial.normalized_expression[targets].to_numpy(float), axis=0)
    covariate_prediction = np.zeros_like(y)
    if config.global_adjust_covariates:
        from .base_grn import _covariate_design
        cov = _covariate_design(bundle, data_config)
        inverse = np.linalg.solve(cov.T @ cov + 1e-6*np.eye(cov.shape[1]), cov.T)
        covariate_prediction = cov @ (inverse @ y)
        # Partial out covariates from BOTH predictors and targets.
        x = x - cov @ (inverse @ x)
    residual = y - covariate_prediction
    coefficients = np.zeros((len(tfs), len(targets)))
    mask = np.zeros_like(coefficients, dtype=bool)
    rows, history = [], []
    penalty = prior.penalty.reindex(index=tfs, columns=targets).to_numpy(float)
    for j, target in enumerate(targets):
        selected = np.array([i for i, tf in enumerate(tfs) if tf != target], dtype=int)
        mask[selected, j] = True
        weights = penalty[selected, j]
        weights = config.l1_penalty * weights / max(float(np.median(weights)), 1e-12) if len(weights) else weights
        fitted, diagnostic = group_fista(x[:, selected], residual[:, j], weights, 0.,
                                        config.global_max_iterations, config.global_tolerance)
        coefficients[selected, j] = fitted
        for record in diagnostic.pop("history"):
            history.append(dict(module="base_grn", target=target, solver="global_weighted_l1_fista", **record))
        rows.append(dict(module="base_grn", target=target, candidate_tfs=len(selected),
                         solver="global_weighted_l1_fista", **diagnostic))
    # Shape compatibility only; this view stores a single shared coefficient matrix.
    beta = np.broadcast_to(coefficients, (len(x), *coefficients.shape))
    return BaseGRNResult(beta, tfs, targets, x, y, mean, scale, covariate_prediction,
                         covariate_prediction + x @ coefficients, mask,
                         pd.DataFrame(rows), pd.DataFrame(history), unit_type="global")


def global_edges(base, prior=None):
    i, j = np.nonzero(base.candidate_mask)
    signed = base.beta[0, i, j]
    table = pd.DataFrame(dict(method="JumpNicheRewire", unit_type="global", cell_id="__all__",
                             region="__all__", tf=np.array(base.tf_names)[i],
                             target=np.array(base.target_names)[j], score=np.abs(signed),
                             signed_score=signed, beta0=signed, delta_beta=0.))
    if prior is not None:
        table["prior_confidence"] = prior.confidence.reindex(index=base.tf_names, columns=base.target_names).to_numpy()[i, j]
    return table.sort_values(["score", "tf", "target"], ascending=[False, True, True], kind="stable").reset_index(drop=True)


def no_communication(bundle):
    empty = pd.DataFrame(index=bundle.expression.index, dtype=float)
    contributions = pd.DataFrame(columns=["sender_index", "receiver_index", "sender_id", "receiver_id", "lr_index", "lr_id", "ligand", "receptor", "distance", "kernel_weight", "score"])
    edges = pd.DataFrame(columns=["method", "sender_id", "receiver_id", "lr_id", "ligand", "receptor", "score"])
    return CommunicationResult(empty, contributions, edges, empty.copy(), empty.copy())
