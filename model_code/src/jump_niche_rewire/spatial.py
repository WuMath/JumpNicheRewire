"""Module 1: spatial screening and jump-aware context graph."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import chi2

from .config import DataConfig, SpatialConfig
from .types import InputBundle, SpatialGraph, SpatialResult
from .utils import benjamini_hochberg, choose_top_indices, standardize


def normalize_expression(expression: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    values = expression.to_numpy(dtype=float)
    if config.expression_is_normalized:
        return pd.DataFrame(values, index=expression.index, columns=expression.columns)
    library = values.sum(axis=1)
    library = np.where(library > 0, library, 1.0)
    normalized = np.log1p(config.library_scale * values / library[:, None])
    return pd.DataFrame(normalized, index=expression.index, columns=expression.columns)


def _sample_knn(
    coordinates: np.ndarray,
    samples: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(coordinates)
    bandwidth = np.ones(n, dtype=float)
    edge_pairs: set[tuple[int, int]] = set()
    for sample in pd.unique(samples):
        indices = np.flatnonzero(samples == sample)
        if len(indices) < 2:
            continue
        k = min(neighbors, len(indices) - 1)
        tree = cKDTree(coordinates[indices])
        distances, local_neighbors = tree.query(coordinates[indices], k=k + 1)
        if k == 1:
            distances = np.asarray(distances).reshape(len(indices), 2)
            local_neighbors = np.asarray(local_neighbors).reshape(len(indices), 2)
        bandwidth[indices] = np.maximum(distances[:, -1], 1e-8)
        for local_i, neighbor_row in enumerate(local_neighbors[:, 1:]):
            i = int(indices[local_i])
            for local_j in np.atleast_1d(neighbor_row):
                j = int(indices[int(local_j)])
                if i != j:
                    edge_pairs.add((min(i, j), max(i, j)))
    if not edge_pairs:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
            bandwidth,
        )
    pairs = np.asarray(sorted(edge_pairs), dtype=int)
    source, target = pairs[:, 0], pairs[:, 1]
    distance = np.linalg.norm(coordinates[source] - coordinates[target], axis=1)
    return source, target, distance, bandwidth


def _clip_bandwidth(bandwidth: np.ndarray, config: SpatialConfig) -> np.ndarray:
    positive = bandwidth[np.isfinite(bandwidth) & (bandwidth > 0)]
    if positive.size == 0:
        return np.ones_like(bandwidth)
    lower = np.quantile(positive, config.bandwidth_lower_quantile)
    upper = np.quantile(positive, config.bandwidth_upper_quantile)
    lower = max(float(lower), 1e-8)
    upper = max(float(upper), lower)
    return np.clip(bandwidth, lower, upper)


def _distance_weights(
    source: np.ndarray,
    target: np.ndarray,
    distance: np.ndarray,
    bandwidth: np.ndarray,
) -> np.ndarray:
    denominator = 2.0 * np.maximum(bandwidth[source] * bandwidth[target], 1e-12)
    return np.exp(-(distance**2) / denominator)


def _statistic_matrix(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    method: str,
    gene_chunk_size: int = 32,
    spatial_operator: csr_matrix | None = None,
    spatial_degree: np.ndarray | None = None,
) -> np.ndarray:
    n, p = values.shape
    if len(source) == 0:
        return np.zeros(p)
    gene_chunk_size = max(int(gene_chunk_size), 1)
    weight_sum = 2.0 * weights.sum()
    if spatial_operator is None:
        spatial_operator = coo_matrix(
            (
                np.concatenate([weights, weights]),
                (
                    np.concatenate([source, target]),
                    np.concatenate([target, source]),
                ),
            ),
            shape=(n, n),
        ).tocsr()
    if spatial_degree is None:
        spatial_degree = np.asarray(spatial_operator.sum(axis=1)).ravel()
    output = np.empty(p, dtype=float)
    for start in range(0, p, gene_chunk_size):
        stop = min(start + gene_chunk_size, p)
        chunk = values[:, start:stop]
        centered = chunk - chunk.mean(axis=0, keepdims=True)
        denominator = np.maximum((centered**2).sum(axis=0), 1e-12)
        if method == "moran":
            numerator = np.sum(centered * (spatial_operator @ centered), axis=0)
            output[start:stop] = n * numerator / np.maximum(
                weight_sum * denominator, 1e-12
            )
        elif method == "geary":
            cross_product = np.sum(
                centered * (spatial_operator @ centered), axis=0
            )
            numerator = 2.0 * (
                np.sum(spatial_degree[:, None] * centered**2, axis=0)
                - cross_product
            )
            geary = (n - 1.0) * numerator / np.maximum(
                2.0 * weight_sum * denominator, 1e-12
            )
            output[start:stop] = 1.0 - geary
        elif method == "getis":
            positive = chunk - chunk.min(axis=0, keepdims=True)
            numerator = np.sum(
                positive * (spatial_operator @ positive), axis=0
            )
            total = positive.sum(axis=0)
            square = (positive**2).sum(axis=0)
            denominator_getis = np.maximum(total**2 - square, 1e-12)
            output[start:stop] = numerator / denominator_getis
        else:
            raise ValueError(f"Unknown spatial statistic: {method}")
    return output


def _stable_spatial_ranking(result: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    """Rank FDR ties by spatial effect sizes, never by input column order.

    Moran's I, 1-Geary's C and Getis statistics live on different numerical
    scales.  Their average percentile rank is therefore used as a comparable
    combined effect score.  Raw method effects and finally the gene identifier
    provide deterministic secondary tie breakers.
    """

    ranked = result.copy()
    statistic_columns = [f"{method}_statistic" for method in methods]
    if not statistic_columns or any(column not in ranked for column in statistic_columns):
        raise ValueError("Spatial ranking requires every configured effect statistic")
    effect_percentiles = [
        ranked[column].rank(method="average", pct=True).to_numpy(dtype=float)
        for column in statistic_columns
    ]
    ranked["spatial_effect_score"] = np.mean(effect_percentiles, axis=0)
    sort_columns = ["fisher_q", "spatial_effect_score", *statistic_columns, "gene"]
    ascending = [True, False, *([False] * len(statistic_columns)), True]
    ranked = ranked.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def spatial_gene_screen(
    raw_expression: pd.DataFrame,
    normalized: pd.DataFrame,
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    data_config: DataConfig,
    spatial_config: SpatialConfig,
    random: np.random.Generator,
) -> tuple[list[str], list[str], pd.DataFrame]:
    raw = raw_expression.to_numpy(dtype=float)
    detection = (raw > 0).mean(axis=0)
    totals = raw.sum(axis=0)
    qc_mask = (detection >= data_config.minimum_detection_rate) & (totals >= data_config.minimum_total_count)
    qc_genes = raw_expression.columns[qc_mask].astype(str).tolist()
    if not qc_genes:
        raise ValueError("No genes pass the configured expression QC thresholds")

    qc_values = normalized.loc[:, qc_genes].to_numpy(dtype=float)
    spatial_operator = coo_matrix(
        (
            np.concatenate([weights, weights]),
            (
                np.concatenate([source, target]),
                np.concatenate([target, source]),
            ),
        ),
        shape=(len(qc_values), len(qc_values)),
    ).tocsr()
    spatial_degree = np.asarray(spatial_operator.sum(axis=1)).ravel()
    result = pd.DataFrame({"gene": qc_genes})
    p_value_columns: list[str] = []
    for method in spatial_config.spatial_statistics:
        observed = _statistic_matrix(
            qc_values,
            source,
            target,
            weights,
            method,
            spatial_config.spatial_statistic_chunk_genes,
            spatial_operator,
            spatial_degree,
        )
        exceed = np.ones_like(observed, dtype=float)
        for _ in range(spatial_config.spatial_permutations):
            permuted = qc_values[random.permutation(len(qc_values))]
            permuted_stat = _statistic_matrix(
                permuted,
                source,
                target,
                weights,
                method,
                spatial_config.spatial_statistic_chunk_genes,
                spatial_operator,
                spatial_degree,
            )
            exceed += permuted_stat >= observed
        p_values = exceed / (spatial_config.spatial_permutations + 1.0)
        result[f"{method}_statistic"] = observed
        result[f"{method}_p"] = p_values
        result[f"{method}_q"] = benjamini_hochberg(p_values)
        p_value_columns.append(f"{method}_p")

    p_matrix = np.clip(result[p_value_columns].to_numpy(dtype=float), 1e-300, 1.0)
    fisher_statistic = -2.0 * np.log(p_matrix).sum(axis=1)
    fisher_p = chi2.sf(fisher_statistic, 2 * len(p_value_columns))
    result["fisher_statistic"] = fisher_statistic
    result["fisher_p"] = fisher_p
    result["fisher_q"] = benjamini_hochberg(fisher_p)
    result["selected"] = result["fisher_q"] < spatial_config.spatial_fdr

    result = _stable_spatial_ranking(result, tuple(spatial_config.spatial_statistics))

    selected = result.loc[result["selected"], "gene"].astype(str).tolist()
    if len(selected) < spatial_config.minimum_spatial_genes:
        count = min(spatial_config.minimum_spatial_genes, len(result))
        selected = result.head(count)["gene"].astype(str).tolist()
        result["selected"] = result["gene"].isin(selected)
        result["selection_note"] = "minimum_spatial_genes fallback"
    if len(selected) > spatial_config.maximum_spatial_genes:
        selected = result.loc[result["selected"]].head(spatial_config.maximum_spatial_genes)["gene"].astype(str).tolist()
        result["selected"] = result["gene"].isin(selected)
        result["selection_note"] = "maximum_spatial_genes cap"
    return qc_genes, selected, result


def _state_embedding(
    normalized: pd.DataFrame,
    qc_genes: list[str],
    config: SpatialConfig,
) -> np.ndarray:
    values = normalized.loc[:, qc_genes].to_numpy(dtype=float)
    variances = values.var(axis=0)
    hvg_count = min(config.hvg_count, len(qc_genes))
    selected = choose_top_indices(variances, hvg_count)
    standardized, _, _ = standardize(values[:, selected], axis=0)
    if standardized.shape[1] == 1:
        return standardized
    u, singular, _ = np.linalg.svd(standardized, full_matrices=False)
    maximum = min(config.pca_components, len(singular))
    variance = singular**2
    cumulative = np.cumsum(variance) / max(variance.sum(), 1e-12)
    threshold_count = int(np.searchsorted(cumulative, config.pca_variance_threshold) + 1)
    components = max(1, min(maximum, threshold_count))
    return u[:, :components] * singular[:components]


def _extract_catalog_genes(series: pd.Series) -> set[str]:
    genes: set[str] = set()
    for value in series.dropna().astype(str):
        genes.update(part.strip() for part in re.split(r"[&|;+]", value) if part.strip())
    return genes


def build_spatial_context(
    bundle: InputBundle,
    data_config: DataConfig,
    config: SpatialConfig,
    random: np.random.Generator,
) -> SpatialResult:
    if not data_config.is_spatial:
        from .nonspatial import build_nonspatial_context
        return build_nonspatial_context(bundle, data_config)
    normalized = normalize_expression(bundle.expression, data_config)
    coordinates = bundle.coordinates.to_numpy(dtype=float)
    samples = bundle.metadata[data_config.sample_column].astype(str).to_numpy()

    screen_source, screen_target, screen_distance, screen_bandwidth = _sample_knn(
        coordinates, samples, config.screening_neighbors
    )
    screen_bandwidth = _clip_bandwidth(screen_bandwidth, config)
    screen_weights = _distance_weights(
        screen_source, screen_target, screen_distance, screen_bandwidth
    )
    qc_genes, spatial_genes, statistics = spatial_gene_screen(
        bundle.expression,
        normalized,
        screen_source,
        screen_target,
        screen_weights,
        data_config,
        config,
        random,
    )

    source, target, distance, bandwidth = _sample_knn(
        coordinates, samples, config.graph_neighbors
    )
    bandwidth = _clip_bandwidth(bandwidth, config)
    if len(source):
        maximum_distance = config.long_edge_factor * np.sqrt(bandwidth[source] * bandwidth[target])
        keep = distance <= maximum_distance
        source, target, distance = source[keep], target[keep], distance[keep]
    distance_weight = _distance_weights(source, target, distance, bandwidth)

    embedding = _state_embedding(normalized, qc_genes, config)
    if len(source):
        state_distance = np.linalg.norm(embedding[source] - embedding[target], axis=1)
        median_distance = np.median(state_distance[state_distance > 0]) if np.any(state_distance > 0) else 1.0
        standardized_distance = state_distance / max(float(median_distance), 1e-8)
        state_weight = np.exp(-(standardized_distance**2) / (2.0 * config.state_bandwidth**2))
        cell_types = bundle.metadata[data_config.cell_type_column].astype(str).to_numpy()
        has_types = np.any(cell_types != "NA")
        if has_types:
            compatibility = np.where(
                cell_types[source] == cell_types[target],
                config.type_compatibility_same,
                config.type_compatibility_different,
            )
            state_weight *= compatibility
    else:
        state_weight = np.array([], dtype=float)
    weight = distance_weight * state_weight
    weight = np.where(weight >= config.edge_weight_floor, weight, 0.0)

    graph = SpatialGraph(
        source=source,
        target=target,
        distance=distance,
        distance_weight=distance_weight,
        state_weight=state_weight,
        weight=weight,
        bandwidth=bandwidth,
        state_embedding=embedding,
    )

    spatial_set = set(spatial_genes)
    qc_set = set(qc_genes)
    candidate_tfs = [tf for tf in bundle.tf_names if tf in spatial_set]
    if not candidate_tfs:
        tf_statistics = statistics[statistics["gene"].isin(bundle.tf_names)].sort_values(
            "selection_rank", kind="mergesort"
        ).head(min(len(bundle.tf_names), config.minimum_spatial_genes))
        candidate_tfs = tf_statistics["gene"].astype(str).tolist()
    candidate_targets = [gene for gene in spatial_genes if gene not in set(candidate_tfs)]
    ligand_genes = _extract_catalog_genes(bundle.lr_pairs["ligand"])
    receptor_genes = _extract_catalog_genes(bundle.lr_pairs["receptor"])
    communication_genes = sorted((ligand_genes | receptor_genes) & qc_set)

    return SpatialResult(
        normalized_expression=normalized,
        qc_genes=qc_genes,
        spatial_genes=spatial_genes,
        spatial_statistics=statistics,
        graph=graph,
        candidate_tfs=candidate_tfs,
        candidate_targets=candidate_targets,
        communication_genes=communication_genes,
    )
