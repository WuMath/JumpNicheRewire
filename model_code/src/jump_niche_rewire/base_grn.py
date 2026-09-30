"""Module 2: graph-fused sparse, cell-resolved baseline GRN estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from .solvers import graph_fused_admm
from .checkpoint import load_target_checkpoint, save_target_checkpoint

from .config import BaseGRNConfig, DataConfig, ProjectConfig
from .types import BaseGRNResult, InputBundle, PriorResult, SpatialResult
from .utils import choose_top_indices, one_hot_frame, soft_threshold, standardize


def _covariate_design(bundle: InputBundle, data_config: DataConfig) -> np.ndarray:
    metadata_matrix, _ = one_hot_frame(bundle.metadata, data_config.confounder_columns)
    library = np.log1p(bundle.expression.sum(axis=1).to_numpy(dtype=float))[:, None]
    blocks = [np.ones((len(bundle.expression), 1), dtype=float), library]
    if metadata_matrix.shape[1]:
        blocks.append(metadata_matrix)
    design = np.concatenate(blocks, axis=1)
    design, _, _ = standardize(design[:, 1:], axis=0) if design.shape[1] > 1 else (np.empty((len(design), 0)), None, None)
    return np.column_stack([np.ones(len(bundle.expression)), design])


def _incidence_forward(beta: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return beta[source] - beta[target]


def _incidence_transpose(
    edge_values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    output = np.zeros((n_cells, edge_values.shape[1]), dtype=float)
    np.add.at(output, source, edge_values)
    np.add.at(output, target, -edge_values)
    return output


def _fit_graph_fused_target(
    x: np.ndarray,
    y: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    edge_weight: np.ndarray,
    l1_weight: np.ndarray,
    config: BaseGRNConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    return graph_fused_admm(x, y, source, target, edge_weight, l1_weight, config)


def fit_baseline_grn(
    bundle: InputBundle,
    spatial: SpatialResult,
    prior: PriorResult,
    data_config: DataConfig,
    config: BaseGRNConfig,
    project_config: ProjectConfig | None = None,
) -> BaseGRNResult:
    if not data_config.is_spatial:
        from .nonspatial import fit_global_grn
        return fit_global_grn(bundle, spatial, prior, data_config, config)
    tfs = spatial.candidate_tfs[: config.maximum_tfs]
    target_candidates = spatial.candidate_targets
    if len(target_candidates) > config.maximum_targets:
        target_statistics = spatial.spatial_statistics.set_index("gene")
        target_candidates = sorted(
            target_candidates,
            key=lambda gene: float(target_statistics.loc[gene, "fisher_q"]),
        )[: config.maximum_targets]
    targets = target_candidates
    if not targets:
        raise ValueError("No candidate target genes are available after spatial screening")

    normalized = spatial.normalized_expression
    tf_matrix, _, _ = standardize(normalized.loc[:, tfs].to_numpy(dtype=float), axis=0)
    target_matrix, target_mean, target_scale = standardize(
        normalized.loc[:, targets].to_numpy(dtype=float), axis=0
    )
    covariates = _covariate_design(bundle, data_config)
    ridge = 1e-6 * np.eye(covariates.shape[1])
    covariate_coefficients = np.linalg.solve(
        covariates.T @ covariates + ridge,
        covariates.T @ target_matrix,
    )
    covariate_prediction = covariates @ covariate_coefficients
    residual_targets = target_matrix - covariate_prediction

    n_cells = len(normalized)
    n_tfs = len(tfs)
    n_targets = len(targets)
    beta = np.zeros((n_cells, n_tfs, n_targets), dtype=float)
    candidate_mask = np.zeros((n_tfs, n_targets), dtype=bool)
    diagnostic_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    source = spatial.graph.source
    target_edge = spatial.graph.target
    edge_weight = spatial.graph.weight

    confidence = prior.confidence.reindex(index=tfs, columns=targets).fillna(0.5).to_numpy(dtype=float)
    penalty = prior.penalty.reindex(index=tfs, columns=targets).fillna(prior.penalty.stack().median()).to_numpy(dtype=float)
    for target_index, target_name in enumerate(targets):
        y = residual_targets[:, target_index]
        correlation = np.abs(tf_matrix.T @ y) / max(len(y), 1)
        candidate_score = correlation + confidence[:, target_index]
        if config.hard_prior_threshold is not None:
            eligible = np.flatnonzero(confidence[:, target_index] >= config.hard_prior_threshold)
            if len(eligible):
                local = choose_top_indices(
                    candidate_score[eligible], min(config.maximum_tfs, len(eligible))
                )
                selected = eligible[local]
            else:
                selected = choose_top_indices(candidate_score, 1)
        else:
            selected = choose_top_indices(candidate_score, min(config.maximum_tfs, n_tfs))
        candidate_mask[selected, target_index] = True
        selected_penalty = penalty[selected, target_index]
        median_penalty = max(float(np.median(selected_penalty)), 1e-12)
        l1_weight = config.l1_penalty * selected_penalty / median_penalty
        cached = (
            load_target_checkpoint(
                project_config, "base_grn", target_index, target_name
            )
            if project_config is not None
            else None
        )
        if (
            cached is not None
            and np.array_equal(np.asarray(cached.get("selected")), selected)
            and np.asarray(cached.get("fitted")).shape == (n_cells, len(selected))
            and isinstance(cached.get("diagnostics"), dict)
        ):
            fitted = np.asarray(cached["fitted"], dtype=float)
            diagnostics = dict(cached["diagnostics"])
        else:
            fitted, diagnostics = _fit_graph_fused_target(
                tf_matrix[:, selected],
                y,
                source,
                target_edge,
                edge_weight,
                l1_weight,
                config,
            )
            if project_config is not None:
                save_target_checkpoint(
                    project_config,
                    "base_grn",
                    target_index,
                    target_name,
                    {
                        "selected": selected,
                        "fitted": fitted,
                        "diagnostics": diagnostics,
                    },
                )
        diagnostics = dict(diagnostics)
        for record in diagnostics.pop("history"):
            history_rows.append({"module": "base_grn", "target": target_name, **record})
        beta[:, selected, target_index] = fitted
        diagnostic_rows.append(
            {
                "module": "base_grn",
                "target": target_name,
                "candidate_tfs": len(selected),
                **diagnostics,
            }
        )

    base_prediction = covariate_prediction + np.einsum("ik,ikg->ig", tf_matrix, beta)
    return BaseGRNResult(
        beta=beta,
        tf_names=tfs,
        target_names=targets,
        tf_matrix=tf_matrix,
        target_matrix=target_matrix,
        target_mean=target_mean,
        target_scale=target_scale,
        covariate_prediction=covariate_prediction,
        base_prediction=base_prediction,
        candidate_mask=candidate_mask,
        diagnostics=pd.DataFrame(diagnostic_rows),
        history=pd.DataFrame(history_rows),
    )
