"""Module 4A: sender-resolved ligand-receptor communication exposure."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import CommunicationConfig, DataConfig
from .types import CommunicationResult, InputBundle, SpatialResult


def _split_complex(expression: str) -> list[list[str]]:
    alternatives: list[list[str]] = []
    for alternative in str(expression).split("|"):
        genes = [gene.strip() for gene in re.split(r"[&;+]", alternative) if gene.strip()]
        if genes:
            alternatives.append(genes)
    return alternatives


def _saturation_matrix(normalized: pd.DataFrame, quantile: float) -> pd.DataFrame:
    values = normalized.to_numpy(dtype=float)
    positive = np.where(values > 0, values, np.nan)
    kappa = np.nanquantile(positive, quantile, axis=0)
    global_positive = values[values > 0]
    fallback = float(np.median(global_positive)) if global_positive.size else 1.0
    kappa = np.where(np.isfinite(kappa) & (kappa > 0), kappa, fallback)
    saturated = values / (values + kappa[None, :] + 1e-12)
    return pd.DataFrame(saturated, index=normalized.index, columns=normalized.columns)


def _effective_complex(
    definition: str,
    saturated: pd.DataFrame,
    normalized: pd.DataFrame,
    completeness_power: float,
    detection_threshold: float,
    epsilon: float,
) -> np.ndarray:
    alternatives = _split_complex(definition)
    if not alternatives:
        return np.zeros(len(saturated), dtype=float)
    scores: list[np.ndarray] = []
    for genes in alternatives:
        available = [gene for gene in genes if gene in saturated.columns]
        if not available:
            scores.append(np.zeros(len(saturated), dtype=float))
            continue
        values = saturated.loc[:, available].to_numpy(dtype=float)
        geometric = np.exp(np.mean(np.log(values + epsilon), axis=1))
        detected = (normalized.loc[:, available].to_numpy(dtype=float) > detection_threshold).sum(axis=1)
        completeness = detected / max(len(genes), 1)
        scores.append((completeness**completeness_power) * geometric)
    return np.maximum.reduce(scores)


def _activity_or_complex(
    activity: pd.DataFrame | None,
    lr_id: str,
    definition: str,
    saturated: pd.DataFrame,
    normalized: pd.DataFrame,
    completeness_power: float,
    config: CommunicationConfig,
) -> np.ndarray:
    if activity is not None:
        if lr_id in activity.columns:
            return pd.to_numeric(activity[lr_id], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if definition in activity.columns:
            return pd.to_numeric(activity[definition], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return _effective_complex(
        definition,
        saturated,
        normalized,
        completeness_power,
        config.detection_threshold,
        config.complex_epsilon,
    )


def _directed_neighbor_edges(
    coordinates: np.ndarray,
    samples: np.ndarray,
    neighbor_count: int,
    bandwidth_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    senders: list[int] = []
    receivers: list[int] = []
    distances_all: list[float] = []
    bandwidth = np.ones(len(coordinates), dtype=float)
    for sample in pd.unique(samples):
        indices = np.flatnonzero(samples == sample)
        if len(indices) < 2:
            continue
        k = min(max(neighbor_count, bandwidth_neighbors), len(indices) - 1)
        tree = cKDTree(coordinates[indices])
        distances, neighbors = tree.query(coordinates[indices], k=k + 1)
        distances = np.atleast_2d(distances)
        neighbors = np.atleast_2d(neighbors)
        bandwidth_position = min(bandwidth_neighbors, k)
        bandwidth[indices] = np.maximum(distances[:, bandwidth_position], 1e-8)
        use_count = min(neighbor_count, k)
        for local_receiver in range(len(indices)):
            receiver = int(indices[local_receiver])
            for offset in range(1, use_count + 1):
                sender = int(indices[int(neighbors[local_receiver, offset])])
                senders.append(sender)
                receivers.append(receiver)
                distances_all.append(float(distances[local_receiver, offset]))
    positive = bandwidth[bandwidth > 0]
    if positive.size:
        bandwidth = np.clip(bandwidth, np.quantile(positive, 0.05), np.quantile(positive, 0.95))
    return (
        np.asarray(senders, dtype=int),
        np.asarray(receivers, dtype=int),
        np.asarray(distances_all, dtype=float),
        bandwidth,
    )


def build_communication_exposure(
    bundle: InputBundle,
    spatial: SpatialResult,
    data_config: DataConfig,
    config: CommunicationConfig,
) -> CommunicationResult:
    if not data_config.is_spatial:
        from .nonspatial import no_communication
        return no_communication(bundle)
    cells = bundle.expression.index.astype(str)
    lr_pairs = bundle.lr_pairs.reset_index(drop=True)
    normalized = spatial.normalized_expression
    saturated = _saturation_matrix(normalized, config.saturation_quantile)

    ligand_values: dict[str, np.ndarray] = {}
    receptor_values: dict[str, np.ndarray] = {}
    for row in lr_pairs.itertuples(index=False):
        ligand_values[row.lr_id] = _activity_or_complex(
            bundle.ligand_activity,
            row.lr_id,
            row.ligand,
            saturated,
            normalized,
            config.completeness_power_ligand,
            config,
        )
        receptor_values[row.lr_id] = _activity_or_complex(
            bundle.receptor_activity,
            row.lr_id,
            row.receptor,
            saturated,
            normalized,
            config.completeness_power_receptor,
            config,
        )
    ligand_effective = pd.DataFrame(ligand_values, index=bundle.expression.index)
    receptor_effective = pd.DataFrame(receptor_values, index=bundle.expression.index)

    senders, receivers, distance, bandwidth = _directed_neighbor_edges(
        bundle.coordinates.to_numpy(dtype=float),
        bundle.metadata[data_config.sample_column].astype(str).to_numpy(),
        config.communication_neighbors,
        config.bandwidth_neighbors,
    )
    if not config.allow_autocrine:
        keep = senders != receivers
        senders, receivers, distance = senders[keep], receivers[keep], distance[keep]

    exposure_values = np.zeros((len(cells), len(lr_pairs)), dtype=float)
    contribution_frames: list[pd.DataFrame] = []
    for lr_index, row in enumerate(lr_pairs.itertuples(index=False)):
        rho = float(row.rho) if np.isfinite(float(row.rho)) else config.default_range_factor
        rho = max(rho, 1e-6)
        signal_type = str(row.signal_type).lower()
        if signal_type in {"contact", "contact-dependent", "juxtacrine"}:
            kernel = np.zeros(len(senders), dtype=float)
            for receiver in np.unique(receivers):
                positions = np.flatnonzero(receivers == receiver)
                selected = positions[np.argsort(distance[positions])[: config.contact_neighbors]]
                kernel[selected] = 1.0
        else:
            denominator = 2.0 * rho**2 * np.maximum(bandwidth[senders] * bandwidth[receivers], 1e-12)
            kernel = np.exp(-(distance**2) / denominator)
        if config.normalize_sender_kernel and len(kernel):
            sums = np.bincount(receivers, weights=kernel, minlength=len(cells))
            kernel = kernel / np.maximum(sums[receivers], 1e-12)
        contribution = (
            kernel
            * ligand_effective[row.lr_id].to_numpy(dtype=float)[senders]
            * receptor_effective[row.lr_id].to_numpy(dtype=float)[receivers]
        )
        exposure_values[:, lr_index] = np.bincount(
            receivers, weights=contribution, minlength=len(cells)
        )
        keep = contribution > config.minimum_contribution
        if np.any(keep):
            contribution_frames.append(
                pd.DataFrame(
                    {
                        "sender_index": senders[keep],
                        "receiver_index": receivers[keep],
                        "sender_id": cells[senders[keep]],
                        "receiver_id": cells[receivers[keep]],
                        "lr_index": lr_index,
                        "lr_id": row.lr_id,
                        "ligand": row.ligand,
                        "receptor": row.receptor,
                        "distance": distance[keep],
                        "kernel_weight": kernel[keep],
                        "score": contribution[keep],
                    }
                )
            )

    exposures = pd.DataFrame(
        exposure_values,
        index=bundle.expression.index,
        columns=lr_pairs["lr_id"].astype(str),
    )
    if contribution_frames:
        sender_contributions = pd.concat(contribution_frames, ignore_index=True)
    else:
        sender_contributions = pd.DataFrame(
            columns=[
                "sender_index", "receiver_index", "sender_id", "receiver_id", "lr_index",
                "lr_id", "ligand", "receptor", "distance", "kernel_weight", "score",
            ]
        )
    cci_by_lr = sender_contributions[
        ["sender_id", "receiver_id", "lr_id", "ligand", "receptor", "score"]
    ].copy()
    if cci_by_lr.empty:
        aggregate = pd.DataFrame(columns=cci_by_lr.columns)
    else:
        aggregate = (
            cci_by_lr.groupby(["sender_id", "receiver_id"], as_index=False)["score"].sum()
            .assign(lr_id="__all__", ligand="__all__", receptor="__all__")
        )
        aggregate = aggregate[["sender_id", "receiver_id", "lr_id", "ligand", "receptor", "score"]]
    cci_edges = pd.concat([cci_by_lr, aggregate], ignore_index=True)
    cci_edges.insert(0, "method", "JumpNicheRewire")
    return CommunicationResult(
        exposures=exposures,
        sender_contributions=sender_contributions,
        cci_edges=cci_edges,
        ligand_effective=ligand_effective,
        receptor_effective=receptor_effective,
    )


def mix_communication_results(
    previous: CommunicationResult,
    candidate: CommunicationResult,
    old_weight: float,
) -> CommunicationResult:
    """Apply the paper's damped `C_(t+1)=rho*C_t+(1-rho)*C_tilde` update."""
    old_weight = float(np.clip(old_weight, 0.0, 1.0))
    new_weight = 1.0 - old_weight
    exposures = old_weight * previous.exposures + new_weight * candidate.exposures
    ligand_effective = old_weight * previous.ligand_effective + new_weight * candidate.ligand_effective
    receptor_effective = old_weight * previous.receptor_effective + new_weight * candidate.receptor_effective

    keys = [
        "sender_index", "receiver_index", "sender_id", "receiver_id", "lr_index",
        "lr_id", "ligand", "receptor",
    ]
    old = previous.sender_contributions.copy()
    new = candidate.sender_contributions.copy()
    merged = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"))
    if merged.empty:
        sender_contributions = previous.sender_contributions.iloc[0:0].copy()
    else:
        old_score = pd.to_numeric(merged.get("score_old"), errors="coerce").fillna(0.0)
        new_score = pd.to_numeric(merged.get("score_new"), errors="coerce").fillna(0.0)
        sender_contributions = merged.loc[:, keys].copy()
        for column in ("distance", "kernel_weight"):
            new_values = merged.get(f"{column}_new", pd.Series(np.nan, index=merged.index))
            old_values = merged.get(f"{column}_old", pd.Series(np.nan, index=merged.index))
            sender_contributions[column] = new_values.combine_first(old_values)
        sender_contributions["score"] = old_weight * old_score + new_weight * new_score
        sender_contributions = sender_contributions[sender_contributions["score"] > 0].reset_index(drop=True)

    cci_by_lr = sender_contributions[
        ["sender_id", "receiver_id", "lr_id", "ligand", "receptor", "score"]
    ].copy()
    if cci_by_lr.empty:
        aggregate = pd.DataFrame(columns=cci_by_lr.columns)
    else:
        aggregate = (
            cci_by_lr.groupby(["sender_id", "receiver_id"], as_index=False)["score"].sum()
            .assign(lr_id="__all__", ligand="__all__", receptor="__all__")
        )
        aggregate = aggregate[["sender_id", "receiver_id", "lr_id", "ligand", "receptor", "score"]]
    cci_edges = pd.concat([cci_by_lr, aggregate], ignore_index=True)
    cci_edges.insert(0, "method", "JumpNicheRewire")
    return CommunicationResult(
        exposures=exposures,
        sender_contributions=sender_contributions,
        cci_edges=cci_edges,
        ligand_effective=ligand_effective,
        receptor_effective=receptor_effective,
    )
