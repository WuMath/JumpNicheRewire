"""Module 3: TF-gene and LR-TF adaptive soft priors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PriorConfig
from .types import InputBundle, PriorResult, SpatialResult
from .utils import sigmoid


def _scale_feature(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index)
    minimum = numeric.min()
    maximum = numeric.max()
    if maximum - minimum < 1e-12:
        return numeric.fillna(minimum).clip(0.0, 1.0)
    return ((numeric - minimum) / (maximum - minimum)).clip(0.0, 1.0)


def build_soft_priors(
    bundle: InputBundle,
    spatial: SpatialResult,
    config: PriorConfig,
) -> PriorResult:
    tfs = spatial.candidate_tfs
    targets = spatial.candidate_targets
    normalized = spatial.normalized_expression
    tf_values = normalized.loc[:, tfs].to_numpy(dtype=float)
    target_values = normalized.loc[:, targets].to_numpy(dtype=float)
    tf_centered = tf_values - tf_values.mean(axis=0, keepdims=True)
    target_centered = target_values - target_values.mean(axis=0, keepdims=True)
    numerator = tf_centered.T @ target_centered
    denominator = np.sqrt(
        np.maximum((tf_centered**2).sum(axis=0)[:, None], 1e-12)
        * np.maximum((target_centered**2).sum(axis=0)[None, :], 1e-12)
    )
    expression_evidence = np.abs(numerator / denominator)

    logit = config.intercept + config.expression_weight * expression_evidence
    prior = bundle.tf_gene_prior.copy()
    if not prior.empty and {"tf", "target"}.issubset(prior.columns):
        prior["tf"] = prior["tf"].astype(str)
        prior["target"] = prior["target"].astype(str)
        tf_index = {name: index for index, name in enumerate(tfs)}
        target_index = {name: index for index, name in enumerate(targets)}
        for feature, weight in (
            ("motif", config.motif_weight),
            ("atac", config.atac_weight),
            ("grn", config.grn_weight),
        ):
            if feature not in prior.columns:
                continue
            scaled = _scale_feature(prior[feature])
            for row_index, row in prior.iterrows():
                tf = row["tf"]
                target = row["target"]
                value = scaled.loc[row_index]
                if tf in tf_index and target in target_index and np.isfinite(value):
                    logit[tf_index[tf], target_index[target]] += weight * float(value)

    confidence_values = sigmoid(logit)
    penalty_values = config.prior_penalty_scale / (config.epsilon + confidence_values)
    confidence = pd.DataFrame(confidence_values, index=tfs, columns=targets)
    penalty = pd.DataFrame(penalty_values, index=tfs, columns=targets)

    lr_ids = bundle.lr_pairs["lr_id"].astype(str).tolist()
    lr_support = pd.DataFrame(0.0, index=lr_ids, columns=tfs)
    if not bundle.lr_tf_prior.empty and {"lr_id", "tf", "score"}.issubset(bundle.lr_tf_prior.columns):
        table = bundle.lr_tf_prior.copy()
        table["lr_id"] = table["lr_id"].astype(str)
        table["tf"] = table["tf"].astype(str)
        table["score"] = _scale_feature(table["score"])
        for row in table.itertuples(index=False):
            if row.lr_id in lr_support.index and row.tf in lr_support.columns and np.isfinite(row.score):
                lr_support.loc[row.lr_id, row.tf] = float(row.score)
    lr_confidence = sigmoid(config.intercept + lr_support.to_numpy(dtype=float))
    lr_penalty = pd.DataFrame(
        config.lr_penalty_scale / (config.epsilon + lr_confidence),
        index=lr_ids,
        columns=tfs,
    )
    return PriorResult(confidence=confidence, penalty=penalty, lr_tf_penalty=lr_penalty)

