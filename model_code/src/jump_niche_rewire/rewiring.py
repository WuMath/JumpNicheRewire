"""Module 4B: communication-conditioned GRN and aligned downstream outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .causal import build_causal_edges
from .solvers import group_fista, region_group_fista
from .checkpoint import load_target_checkpoint, save_target_checkpoint
from .config import CausalConfig, DataConfig, ProjectConfig, RewiringConfig
from .types import (
    BaseGRNResult,
    CommunicationResult,
    InputBundle,
    PriorResult,
    RewiringResult,
    SpatialResult,
)
from .utils import choose_top_indices, standardize


def _top_k_numeric(
    indices: np.ndarray,
    scores: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic top-k numeric candidates without Python row objects."""
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    indices, scores = indices[valid], scores[valid]
    limit = max(int(limit), 0)
    if limit == 0 or not len(indices):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    if len(indices) <= limit:
        order = np.lexsort((indices, -scores))
        return indices[order], scores[order]
    threshold = float(np.partition(scores, len(scores) - limit)[len(scores) - limit])
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)
    equal = equal[np.argsort(indices[equal], kind="stable")]
    selected = np.concatenate([greater, equal[: limit - len(greater)]])
    order = np.lexsort((indices[selected], -scores[selected]))
    selected = selected[order]
    return indices[selected], scores[selected]


def _merge_top_k_numeric(
    retained_indices: np.ndarray,
    retained_scores: np.ndarray,
    new_indices: np.ndarray,
    new_scores: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(retained_indices):
        return _top_k_numeric(new_indices, new_scores, limit)
    return _top_k_numeric(
        np.concatenate([retained_indices, np.asarray(new_indices, dtype=np.int64)]),
        np.concatenate([retained_scores, np.asarray(new_scores, dtype=float)]),
        limit,
    )


def _bounded_grn_edges(
    result: RewiringResult,
    bundle: InputBundle,
    base: BaseGRNResult,
    rewiring_config: RewiringConfig,
) -> pd.DataFrame:
    if base.unit_type == "global":
        from .nonspatial import global_edges
        return global_edges(base)
    beta_ccc = np.asarray(result.beta_ccc, dtype=float)
    n_cells, n_tfs, n_targets = beta_ccc.shape
    rows_per_cell = max(n_tfs * n_targets, 1)
    cells_per_chunk = max(1, int(rewiring_config.output_build_chunk_rows) // rows_per_cell)
    retained_indices = np.empty(0, dtype=np.int64)
    retained_scores = np.empty(0, dtype=float)
    limit = max(int(rewiring_config.top_grn_rows), 0)
    for start in range(0, n_cells, cells_per_chunk):
        stop = min(start + cells_per_chunk, n_cells)
        coefficients = beta_ccc[start:stop]
        included = np.broadcast_to(base.candidate_mask[None, :, :], coefficients.shape) | (
            np.abs(coefficients) >= rewiring_config.coefficient_threshold
        )
        local = np.flatnonzero(included.ravel())
        if not len(local):
            continue
        local_scores = np.abs(coefficients.ravel()[local])
        global_indices = local + start * rows_per_cell
        local_indices, local_scores = _top_k_numeric(global_indices, local_scores, limit)
        retained_indices, retained_scores = _merge_top_k_numeric(
            retained_indices,
            retained_scores,
            local_indices,
            local_scores,
            limit,
        )
    columns = [
        "method", "unit_type", "cell_id", "region", "tf", "target",
        "score", "signed_score", "beta0", "delta_beta",
    ]
    if not len(retained_indices):
        return pd.DataFrame(columns=columns)
    cell_index, tf_index, target_index = np.unravel_index(
        retained_indices, beta_ccc.shape
    )
    cells = bundle.expression.index.astype(str).to_numpy(dtype=object)
    region_labels = np.asarray(
        [result.regions[index] for index in result.region_index], dtype=object
    )
    tf_names = np.asarray(base.tf_names, dtype=object)
    target_names = np.asarray(base.target_names, dtype=object)
    signed = beta_ccc[cell_index, tf_index, target_index]
    return pd.DataFrame(
        {
            "method": "JumpNicheRewire",
            "unit_type": "cell",
            "cell_id": cells[cell_index],
            "region": region_labels[cell_index],
            "tf": tf_names[tf_index],
            "target": target_names[target_index],
            "score": retained_scores,
            "signed_score": signed,
            "beta0": base.beta[cell_index, tf_index, target_index],
            "delta_beta": (
                signed - base.beta[cell_index, tf_index, target_index]
            ),
        },
        columns=columns,
    )


def _bounded_lr_gene_edges(
    result: RewiringResult,
    bundle: InputBundle,
    base: BaseGRNResult,
    communication: CommunicationResult,
    tf_activity: np.ndarray,
    rewiring_config: RewiringConfig,
) -> pd.DataFrame:
    gamma = np.asarray(result.gamma, dtype=float)
    all_exposure = communication.exposures.to_numpy(dtype=float)
    stored_lr_indices = _result_active_lr_indices(result, gamma)
    exposure = all_exposure[:, stored_lr_indices]
    n_cells = len(bundle.expression)
    n_lr = exposure.shape[1]
    n_targets = len(base.target_names)
    active_lr = np.flatnonzero(
        np.any(np.abs(gamma) >= rewiring_config.coefficient_threshold, axis=(0, 2, 3))
    )
    columns = [
        "method", "receiver_id", "region", "lr_id", "ligand", "receptor",
        "target", "score", "signed_score",
    ]
    limit = max(int(rewiring_config.top_lr_gene_rows), 0)
    if limit == 0 or not len(active_lr):
        return pd.DataFrame(columns=columns)
    retained_indices = np.empty(0, dtype=np.int64)
    retained_scores = np.empty(0, dtype=float)
    rows_per_cell = n_lr * n_targets
    chunk_rows = max(int(rewiring_config.output_build_chunk_rows), 1)
    cells_per_chunk = max(1, chunk_rows // max(len(active_lr) * n_targets, 1))
    for region_position in range(len(result.regions)):
        region_cells = np.flatnonzero(result.region_index == region_position)
        for offset in range(0, len(region_cells), cells_per_chunk):
            cell_index = region_cells[offset : offset + cells_per_chunk]
            mediated = np.einsum(
                "ck,lkg->clg",
                tf_activity[cell_index],
                gamma[region_position, active_lr],
                optimize=True,
            )
            mediated *= exposure[np.ix_(cell_index, active_lr)][:, :, None]
            scores = np.abs(mediated)
            local = np.flatnonzero(np.isfinite(scores) & (scores > 0.0))
            if not len(local):
                continue
            local_cell, local_lr, local_target = np.unravel_index(local, scores.shape)
            global_indices = (
                cell_index[local_cell] * rows_per_cell
                + active_lr[local_lr] * n_targets
                + local_target
            )
            local_indices, local_scores = _top_k_numeric(
                global_indices, scores.ravel()[local], limit
            )
            retained_indices, retained_scores = _merge_top_k_numeric(
                retained_indices,
                retained_scores,
                local_indices,
                local_scores,
                limit,
            )
    if not len(retained_indices):
        return pd.DataFrame(columns=columns)
    cell_index, lr_index, target_index = np.unravel_index(
        retained_indices, (n_cells, n_lr, n_targets)
    )
    global_lr_index = stored_lr_indices[lr_index]
    region_position = result.region_index[cell_index]
    signed = exposure[cell_index, lr_index] * np.einsum(
        "ik,ik->i",
        tf_activity[cell_index],
        gamma[region_position, lr_index, :, target_index],
        optimize=True,
    )
    lr_table = bundle.lr_pairs.reset_index(drop=True)
    cells = bundle.expression.index.astype(str).to_numpy(dtype=object)
    regions = np.asarray(result.regions, dtype=object)
    targets = np.asarray(base.target_names, dtype=object)
    return pd.DataFrame(
        {
            "method": "JumpNicheRewire",
            "receiver_id": cells[cell_index],
            "region": regions[region_position],
            "lr_id": lr_table["lr_id"].astype(str).to_numpy()[global_lr_index],
            "ligand": lr_table["ligand"].astype(str).to_numpy()[global_lr_index],
            "receptor": lr_table["receptor"].astype(str).to_numpy()[global_lr_index],
            "target": targets[target_index],
            "score": retained_scores,
            "signed_score": signed,
        },
        columns=columns,
    )


def _bounded_receptor_paths(
    result: RewiringResult,
    bundle: InputBundle,
    base: BaseGRNResult,
    communication: CommunicationResult,
    tf_activity: np.ndarray,
    evidence_lookup: dict[tuple[str, str, str, str], tuple[int, str]],
    rewiring_config: RewiringConfig,
) -> pd.DataFrame:
    columns = [
        "method", "sender_id", "receiver_id", "sender_region", "receiver_region",
        "lr_id", "ligand", "receptor", "tf", "target", "communication_score",
        "delta_beta", "path_score", "evidence_level", "evidence_label",
    ]
    limit = max(int(rewiring_config.top_path_rows), 0)
    contributions = communication.sender_contributions.reset_index(drop=True)
    if limit == 0 or contributions.empty:
        return pd.DataFrame(columns=columns)
    gamma = np.asarray(result.gamma, dtype=float)
    n_regions, n_lr, n_tfs, n_targets = gamma.shape
    stored_lr_indices = _result_active_lr_indices(result, gamma)
    global_to_local_lr = {
        int(global_index): local_index
        for local_index, global_index in enumerate(stored_lr_indices)
    }
    contribution_global_lr = pd.to_numeric(
        contributions["lr_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    keep = np.fromiter(
        (int(index) in global_to_local_lr for index in contribution_global_lr),
        dtype=bool,
        count=len(contribution_global_lr),
    )
    contributions = contributions.loc[keep].reset_index(drop=True)
    contribution_global_lr = contribution_global_lr[keep]
    if contributions.empty:
        return pd.DataFrame(columns=columns)
    top_pairs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for region_position in range(n_regions):
        for lr_index in range(n_lr):
            flat = np.abs(gamma[region_position, lr_index]).ravel()
            nonzero = np.flatnonzero(flat > 0.0)
            if not len(nonzero):
                top_pairs[(region_position, lr_index)] = (
                    np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
                )
                continue
            # _top_k_numeric returns original flat indices; decode those directly.
            selected_flat, _ = _top_k_numeric(nonzero, flat[nonzero], min(50, len(nonzero)))
            tf_index, target_index = np.unravel_index(selected_flat, (n_tfs, n_targets))
            top_pairs[(region_position, lr_index)] = (
                np.asarray(tf_index, dtype=np.int64),
                np.asarray(target_index, dtype=np.int64),
            )

    receiver_index_all = pd.to_numeric(
        contributions["receiver_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    lr_index_all = np.asarray(
        [global_to_local_lr[int(index)] for index in contribution_global_lr],
        dtype=np.int64,
    )
    region_all = result.region_index[receiver_index_all]
    group_codes = region_all * n_lr + lr_index_all
    communication_scores = pd.to_numeric(
        contributions["score"], errors="coerce"
    ).to_numpy(dtype=float)
    retained_codes = np.empty(0, dtype=np.int64)
    retained_scores = np.empty(0, dtype=float)
    chunk_rows = max(int(rewiring_config.output_build_chunk_rows), 1)
    pair_space = n_tfs * n_targets
    for group_code in np.unique(group_codes):
        positions = np.flatnonzero(group_codes == group_code)
        region_position = int(group_code // n_lr)
        lr_index = int(group_code % n_lr)
        tf_index, target_index = top_pairs[(region_position, lr_index)]
        if not len(tf_index):
            continue
        contribution_chunk = max(1, chunk_rows // len(tf_index))
        edge_gamma = gamma[region_position, lr_index, tf_index, target_index]
        pair_code = tf_index * n_targets + target_index
        for offset in range(0, len(positions), contribution_chunk):
            row_index = positions[offset : offset + contribution_chunk]
            receivers = receiver_index_all[row_index]
            signed = (
                communication_scores[row_index, None]
                * edge_gamma[None, :]
                * tf_activity[receivers[:, None], tf_index[None, :]]
            )
            scores = np.abs(signed)
            local = np.flatnonzero(np.isfinite(scores) & (scores > 0.0))
            if not len(local):
                continue
            local_row, local_pair = np.unravel_index(local, scores.shape)
            codes = row_index[local_row] * pair_space + pair_code[local_pair]
            local_codes, local_scores = _top_k_numeric(
                codes, scores.ravel()[local], limit
            )
            retained_codes, retained_scores = _merge_top_k_numeric(
                retained_codes,
                retained_scores,
                local_codes,
                local_scores,
                limit,
            )
    if not len(retained_codes):
        return pd.DataFrame(columns=columns)
    contribution_index, pair_code = divmod(retained_codes, pair_space)
    tf_index, target_index = divmod(pair_code, n_targets)
    rows = contributions.iloc[contribution_index].reset_index(drop=True)
    receiver_index = receiver_index_all[contribution_index]
    sender_index = pd.to_numeric(rows["sender_index"], errors="raise").to_numpy(dtype=np.int64)
    lr_index = lr_index_all[contribution_index]
    global_lr_index = stored_lr_indices[lr_index]
    receiver_region_position = result.region_index[receiver_index]
    edge_gamma = gamma[receiver_region_position, lr_index, tf_index, target_index]
    communication_score = communication_scores[contribution_index]
    delta = communication_score * edge_gamma
    lr_table = bundle.lr_pairs.reset_index(drop=True)
    lr_ids = lr_table["lr_id"].astype(str).to_numpy(dtype=object)[global_lr_index]
    tf_names = np.asarray(base.tf_names, dtype=object)[tf_index]
    target_names = np.asarray(base.target_names, dtype=object)[target_index]
    regions = np.asarray(result.regions, dtype=object)
    region_labels = regions[result.region_index]
    levels_and_labels = [
        evidence_lookup.get(
            (regions[receiver_region_position[i]], str(lr_ids[i]), str(tf_names[i]), str(target_names[i])),
            (0, "CCC-associated"),
        )
        for i in range(len(retained_codes))
    ]
    cells = bundle.expression.index.astype(str).to_numpy(dtype=object)
    return pd.DataFrame(
        {
            "method": "JumpNicheRewire",
            "sender_id": cells[sender_index],
            "receiver_id": cells[receiver_index],
            "sender_region": region_labels[sender_index],
            "receiver_region": regions[receiver_region_position],
            "lr_id": lr_ids,
            "ligand": lr_table["ligand"].astype(str).to_numpy(dtype=object)[global_lr_index],
            "receptor": lr_table["receptor"].astype(str).to_numpy(dtype=object)[global_lr_index],
            "tf": tf_names,
            "target": target_names,
            "communication_score": communication_score,
            "delta_beta": delta,
            "path_score": retained_scores,
            "evidence_level": [item[0] for item in levels_and_labels],
            "evidence_label": [item[1] for item in levels_and_labels],
        },
        columns=columns,
    )


def _group_lasso_fista(
    design, response, group_penalties, ridge_penalty, max_iterations, tolerance, initial=None,
):
    return group_fista(design, response, group_penalties, ridge_penalty, max_iterations, tolerance, initial)


def _result_active_lr_indices(
    result: RewiringResult, gamma: np.ndarray
) -> np.ndarray:
    indices = getattr(result, "active_lr_indices", None)
    if indices is None or len(indices) != gamma.shape[1]:
        return np.arange(gamma.shape[1], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _assemble_beta_ccc(
    base_beta: np.ndarray,
    exposure_all: np.ndarray,
    gamma: np.ndarray,
    region_index: np.ndarray,
    active_lr_indices: np.ndarray,
    cell_chunk_size: int,
) -> np.ndarray:
    """Assemble communication-conditioned coefficients without gamma-per-cell."""

    beta_ccc = np.empty_like(base_beta, dtype=float)
    active_lr_indices = np.asarray(active_lr_indices, dtype=np.int64)
    cell_chunk_size = max(int(cell_chunk_size), 1)
    if gamma.shape[1] != len(active_lr_indices):
        raise ValueError("gamma active-LR axis does not match active_lr_indices")
    for region_position in range(gamma.shape[0]):
        region_cells = np.flatnonzero(region_index == region_position)
        for offset in range(0, len(region_cells), cell_chunk_size):
            cells = region_cells[offset : offset + cell_chunk_size]
            if len(active_lr_indices):
                delta = np.einsum(
                    "ca,akg->ckg",
                    exposure_all[np.ix_(cells, active_lr_indices)],
                    gamma[region_position],
                    optimize=True,
                )
                beta_ccc[cells] = base_beta[cells] + delta
            else:
                beta_ccc[cells] = base_beta[cells]
    return beta_ccc


def _regions(bundle: InputBundle, data_config: DataConfig) -> tuple[list[str], np.ndarray]:
    labels = bundle.metadata[data_config.region_column].fillna("global").astype(str)
    names = sorted(labels.unique().tolist())
    mapping = {name: index for index, name in enumerate(names)}
    return names, labels.map(mapping).to_numpy(dtype=int)


def _tf_activity(bundle: InputBundle, base: BaseGRNResult) -> np.ndarray:
    if bundle.tf_activity is None:
        return base.tf_matrix
    frame = bundle.tf_activity.reindex(index=bundle.expression.index, columns=base.tf_names)
    fallback = pd.DataFrame(base.tf_matrix, index=bundle.expression.index, columns=base.tf_names)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(fallback)
    standardized, _, _ = standardize(frame.to_numpy(dtype=float), axis=0)
    return standardized


def fit_rewiring(
    bundle: InputBundle,
    spatial: SpatialResult,
    prior: PriorResult,
    base: BaseGRNResult,
    communication: CommunicationResult,
    data_config: DataConfig,
    rewiring_config: RewiringConfig,
    causal_config: CausalConfig,
    initial_gamma: np.ndarray | None = None,
    build_outputs: bool = True,
    fixed_active_lr: np.ndarray | None = None,
    project_config: ProjectConfig | None = None,
    checkpoint_stage: str = "rewiring",
) -> RewiringResult:
    regions, region_index = (_regions(bundle, data_config) if data_config.is_spatial
                             else (["__all__"], np.zeros(len(bundle.expression), dtype=int)))
    exposure_all = communication.exposures.to_numpy(dtype=float)
    lr_ids_all = communication.exposures.columns.astype(str).tolist()
    active_lr_indices = choose_top_indices(
        np.var(exposure_all, axis=0),
        min(rewiring_config.maximum_active_lr, exposure_all.shape[1]),
    )
    if fixed_active_lr is not None:
        active_lr_indices = fixed_active_lr
    exposure = exposure_all[:, active_lr_indices]
    lr_ids = [lr_ids_all[index] for index in active_lr_indices]
    tf_activity = _tf_activity(bundle, base)
    residual_targets = base.target_matrix - base.base_prediction
    n_regions = len(regions)
    n_active_lr = len(active_lr_indices)
    n_tfs = len(base.tf_names)
    n_targets = len(base.target_names)
    # Only active LR coefficients are stored.  The previous all-LR tensor was
    # mostly structural zeros (1.9 GiB for Visium alone).
    gamma = np.zeros((n_regions, n_active_lr, n_tfs, n_targets), dtype=float)
    diagnostic_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    target_iterator = (
        [] if rewiring_config.disable_communication_rewiring else enumerate(base.target_names)
    )
    for target_index, target in target_iterator:
        base_strength = np.mean(np.abs(base.beta[:, :, target_index]), axis=0)
        selected_tfs = choose_top_indices(
            base_strength + np.abs(tf_activity.T @ residual_targets[:, target_index]) / len(tf_activity),
            min(rewiring_config.maximum_interaction_tfs, n_tfs),
        )
        pairs = [(local_lr, tf_index) for local_lr in range(len(lr_ids)) for tf_index in selected_tfs]
        pair_lr = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        pair_tf = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
        interactions = (
            exposure[:, pair_lr] * tf_activity[:, pair_tf]
            if len(pairs)
            else np.empty((len(bundle.expression), 0), dtype=float)
        )
        group_penalties = np.zeros(len(pairs), dtype=float)
        for pair_index, (local_lr, tf_index) in enumerate(pairs):
            lr_id = lr_ids[local_lr]
            penalty_value = float(
                prior.lr_tf_penalty.reindex(index=[lr_id], columns=[base.tf_names[tf_index]]).iloc[0, 0]
            )
            group_penalties[pair_index] = penalty_value
        if len(group_penalties):
            median_penalty = max(float(np.median(group_penalties)), 1e-12)
            group_penalties = rewiring_config.group_penalty * group_penalties / median_penalty
        cached = (
            load_target_checkpoint(
                project_config, checkpoint_stage, target_index, target
            )
            if project_config is not None
            else None
        )
        expected_size = len(pairs) * n_regions
        if (
            cached is not None
            and np.array_equal(np.asarray(cached.get("selected_tfs")), selected_tfs)
            and np.asarray(cached.get("coefficients")).shape == (expected_size,)
            and isinstance(cached.get("diagnostics"), dict)
        ):
            coefficients = np.asarray(cached["coefficients"], dtype=float)
            diagnostics = dict(cached["diagnostics"])
        else:
            coefficients, diagnostics = region_group_fista(
                interactions,
                residual_targets[:, target_index],
                region_index,
                n_regions,
                group_penalties,
                rewiring_config.ridge_penalty,
                rewiring_config.fista_max_iterations,
                rewiring_config.fista_tolerance,
                None if initial_gamma is None else np.concatenate([
                    initial_gamma[:, local_lr, tf_index, target_index]
                    for local_lr, tf_index in pairs
                ]) if pairs else None,
            )
            if project_config is not None:
                save_target_checkpoint(
                    project_config,
                    checkpoint_stage,
                    target_index,
                    target,
                    {
                        "selected_tfs": selected_tfs,
                        "coefficients": coefficients,
                        "diagnostics": diagnostics,
                    },
                )
        diagnostics = dict(diagnostics)
        for record in diagnostics.pop("history"):
            history_rows.append({"module": "communication_rewiring", "target": target, **record})
        for pair_index, (local_lr, tf_index) in enumerate(pairs):
            gamma[:, local_lr, tf_index, target_index] = coefficients[
                pair_index * n_regions : (pair_index + 1) * n_regions
            ]
        diagnostic_rows.append(
            {
                "module": "communication_rewiring",
                "target": target,
                "active_lr": len(lr_ids),
                "interaction_tfs": len(selected_tfs),
                **diagnostics,
            }
        )
    if rewiring_config.disable_communication_rewiring:
        diagnostic_rows.append(
            {
                "module": "communication_rewiring",
                "target": "__all__",
                "active_lr": 0,
                "interaction_tfs": 0,
                "iterations": 0.0,
                "converged": 1.0,
                "objective": 0.0,
                "note": "communication rewiring disabled by ablation config",
            }
        )

    beta_ccc = _assemble_beta_ccc(
        base.beta,
        exposure_all,
        gamma,
        region_index,
        active_lr_indices,
        rewiring_config.numerical_cell_chunk_size,
    )

    result = RewiringResult(
        gamma=gamma, regions=regions, region_index=region_index, beta_ccc=beta_ccc,
        grn_edges=pd.DataFrame(), lr_gene=pd.DataFrame(), receptor_paths=pd.DataFrame(),
        regulon_activity=pd.DataFrame(), causal_edges=pd.DataFrame(),
        diagnostics=pd.DataFrame(diagnostic_rows), history=pd.DataFrame(history_rows),
        active_lr_indices=np.asarray(active_lr_indices, dtype=np.int64),
    )
    if build_outputs:
        return populate_rewiring_outputs(result, bundle, spatial, base, communication, data_config, rewiring_config, causal_config)
    return result


def populate_rewiring_outputs(result, bundle, spatial, base, communication, data_config, rewiring_config, causal_config):
    gamma, regions, region_index, beta_ccc = result.gamma, result.regions, result.region_index, result.beta_ccc
    active_lr_indices = _result_active_lr_indices(result, np.asarray(gamma))
    tf_activity = _tf_activity(bundle, base)
    n_regions, n_lr, n_tfs, n_targets = gamma.shape
    causal_edges = build_causal_edges(
        gamma,
        regions,
        region_index,
        bundle,
        communication,
        base,
        data_config,
        causal_config,
        rewiring_config.coefficient_threshold,
        active_lr_indices,
    )
    evidence_lookup: dict[tuple[str, str, str, str], tuple[int, str]] = {}
    if not causal_edges.empty:
        for row in causal_edges.itertuples(index=False):
            evidence_lookup[(row.region, row.lr_id, row.tf, row.target)] = (
                int(row.evidence_level),
                str(row.evidence_label),
            )

    cells = bundle.expression.index.astype(str).tolist()
    region_labels = [regions[index] for index in region_index]
    grn_edges = _bounded_grn_edges(result, bundle, base, rewiring_config)
    lr_gene = _bounded_lr_gene_edges(
        result,
        bundle,
        base,
        communication,
        tf_activity,
        rewiring_config,
    )

    receptor_paths = _bounded_receptor_paths(
        result,
        bundle,
        base,
        communication,
        tf_activity,
        evidence_lookup,
        rewiring_config,
    )

    target_expression = spatial.normalized_expression.loc[:, base.target_names].to_numpy(dtype=float)
    activity_raw = np.einsum("ikg,ig->ik", np.abs(beta_ccc), target_expression)
    activity, _, _ = standardize(activity_raw, axis=0)
    regulon_activity = pd.DataFrame(
        {
            "method": "JumpNicheRewire",
            "cell_id": np.repeat(cells, n_tfs),
            "region": np.repeat(region_labels, n_tfs),
            "tf": np.tile(base.tf_names, len(cells)),
            "activity": activity.ravel(),
        }
    )
    return RewiringResult(
        gamma=gamma,
        regions=regions,
        region_index=region_index,
        beta_ccc=beta_ccc,
        grn_edges=grn_edges,
        lr_gene=lr_gene,
        receptor_paths=receptor_paths,
        regulon_activity=regulon_activity,
        causal_edges=causal_edges,
        diagnostics=result.diagnostics,
        history=result.history,
        active_lr_indices=active_lr_indices,
    )
