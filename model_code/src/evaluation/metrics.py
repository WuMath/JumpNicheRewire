"""Metric implementations matching the paper's benchmark tasks."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, rankdata


def _binary_arrays(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(labels) & np.isfinite(scores)
    return labels[finite], scores[finite]


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Non-interpolated AP evaluated at distinct score thresholds.

    Processing a complete tied-score block at once makes the result invariant
    to the input row order.  Each block contributes its recall increment times
    the precision after accepting the whole block, matching the usual
    threshold-based definition of average precision.
    """

    labels, scores = _binary_arrays(labels, scores)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    boundaries = np.r_[np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]) + 1, len(ranked_scores)]
    cumulative_positives = np.cumsum(ranked_labels)
    previous_positives = 0
    ap = 0.0
    for end in boundaries:
        true_positives = int(cumulative_positives[end - 1])
        block_positives = true_positives - previous_positives
        if block_positives:
            ap += (block_positives / positives) * (true_positives / int(end))
        previous_positives = true_positives
    return float(ap)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _binary_arrays(labels, scores)
    positives = labels == 1
    negatives = labels == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    u_statistic = ranks[positives].sum() - positive_count * (positive_count + 1) / 2.0
    return float(u_statistic / (positive_count * negative_count))


def early_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Expected precision at k positives with an order-invariant tie boundary."""

    labels, scores = _binary_arrays(labels, scores)
    k = int(labels.sum())
    if k == 0:
        return float("nan")
    if k >= len(labels):
        return float(labels.mean())
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    boundary_score = ranked_scores[k - 1]
    above = ranked_scores > boundary_score
    tied = ranked_scores == boundary_score
    slots = k - int(above.sum())
    expected_tied_positives = slots * float(ranked_labels[tied].mean())
    return float((ranked_labels[above].sum() + expected_tied_positives) / k)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels, scores = _binary_arrays(labels, scores)
    prevalence = float(labels.mean()) if len(labels) else float("nan")
    ap = average_precision(labels, scores)
    return {
        "AUPRC": ap,
        "Average Precision": ap,
        "Early Precision": early_precision(labels, scores),
        "AUPRC ratio": ap / prevalence if prevalence > 0 and np.isfinite(ap) else float("nan"),
        "AUROC": auroc(labels, scores),
        "prevalence": prevalence,
        "candidate_count": float(len(labels)),
        "positive_count": float(labels.sum()),
    }


def ndcg(true_relevance: np.ndarray, predicted_score: np.ndarray) -> float:
    true_relevance = np.asarray(true_relevance, dtype=float)
    predicted_score = np.asarray(predicted_score, dtype=float)
    if len(true_relevance) == 0:
        return float("nan")
    ideal = np.argsort(-true_relevance, kind="mergesort")
    discounts = 1.0 / np.log2(np.arange(2, len(true_relevance) + 2))
    gains = 2.0 ** true_relevance - 1.0
    order = np.argsort(-predicted_score, kind="mergesort")
    ranked_scores = predicted_score[order]
    ranked_gains = gains[order]
    boundaries = np.r_[0, np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]) + 1, len(ranked_scores)]
    dcg = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # Average discount is the expectation over every ordering inside a tie.
        dcg += float(ranked_gains[start:end].sum() * discounts[start:end].mean())
    idcg = np.sum((2.0 ** true_relevance[ideal] - 1.0) * discounts)
    return float(dcg / idcg) if idcg > 0 else float("nan")


def _spearman_without_lapack(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman correlation without NumPy corrcoef/LAPACK on Windows MKL."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 2:
        return float("nan")
    first_rank = rankdata(first[finite], method="average").astype(float)
    second_rank = rankdata(second[finite], method="average").astype(float)
    first_rank -= float(first_rank.mean())
    second_rank -= float(second_rank.mean())
    denominator = float(
        np.sqrt(np.sum(first_rank * first_rank) * np.sum(second_rank * second_rank))
    )
    if not np.isfinite(denominator) or denominator <= 1e-15:
        return float("nan")
    return float(np.sum(first_rank * second_rank) / denominator)


def ranking_metrics(true_score: np.ndarray, predicted_score: np.ndarray) -> dict[str, float]:
    true_score = np.asarray(true_score, dtype=float)
    predicted_score = np.asarray(predicted_score, dtype=float)
    relevant = true_score > 0
    k = int(relevant.sum())
    if k:
        precision = early_precision(relevant.astype(int), predicted_score)
    else:
        precision = float("nan")
    variable = (
        len(true_score) > 1
        and np.nanstd(true_score) > 0
        and np.nanstd(predicted_score) > 0
    )
    spearman = _spearman_without_lapack(true_score, predicted_score) if variable else np.nan
    kendall = kendalltau(true_score, predicted_score).statistic if variable else np.nan
    return {
        "Precision": precision,
        "NDCG": ndcg(true_score, predicted_score),
        "Spearman": float(spearman) if np.isfinite(spearman) else float("nan"),
        "Kendall": float(kendall) if np.isfinite(kendall) else float("nan"),
    }
