"""Observational identification diagnostics and evidence-level assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg

from .config import CausalConfig, DataConfig
from .types import BaseGRNResult, CommunicationResult, InputBundle
from .utils import one_hot_frame


def _overlap_diagnostic(
    exposure: np.ndarray,
    covariates: np.ndarray,
    config: CausalConfig,
) -> dict[str, float | int | bool]:
    exposure = np.asarray(exposure, dtype=float).reshape(-1)
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)
    if covariates.ndim != 2 or covariates.shape[0] != exposure.shape[0]:
        raise ValueError("covariates must be a two-dimensional array aligned with exposure")

    finite_exposure = np.isfinite(exposure)
    exposure = exposure[finite_exposure]
    covariates = covariates[finite_exposure]
    if exposure.size < 2:
        return {
            "exposure_sd": float("nan"),
            "low_group_n": 0,
            "high_group_n": 0,
            "overlap_fraction": 0.0,
            "diagnostic_pass": False,
            "exposure_low": float("nan"),
            "exposure_high": float("nan"),
            "finite_sample_n": int(exposure.size),
            "adjustment_covariate_n": 0,
            "adjustment_fit_ok": False,
        }

    standard_deviation = float(np.std(exposure, ddof=1))
    lower_cut = float(np.quantile(exposure, config.lower_exposure_quantile))
    upper_cut = float(np.quantile(exposure, config.upper_exposure_quantile))
    low = exposure <= lower_cut
    high = exposure >= upper_cut

    clean_columns: list[np.ndarray] = []
    for column in covariates.T:
        finite = np.isfinite(column)
        if not finite.any():
            continue
        median = float(np.median(column[finite]))
        clean = np.where(finite, column, median)
        scale = float(np.std(clean, ddof=0))
        if not np.isfinite(scale) or scale <= 1e-8:
            continue
        clean_columns.append((clean - float(np.mean(clean))) / scale)

    centered_exposure = exposure - float(np.mean(exposure))
    adjustment_fit_ok = True
    if clean_columns:
        design = np.column_stack(clean_columns)
        ridge = max(float(config.overlap_ridge_penalty), 0.0)
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        right_hand_side = design.T @ centered_exposure
        # Avoid LAPACK ``solve``/``lstsq`` here.  Some Windows NumPy/MKL builds
        # abort the process instead of raising LinAlgError for degenerate small
        # systems.  The ridge Gram matrix is symmetric positive definite, so a
        # bounded conjugate-gradient solve is appropriate and exposes status.
        coefficients, solver_status = cg(
            gram,
            right_hand_side,
            rtol=1e-10,
            atol=0.0,
            maxiter=max(100, 20 * design.shape[1]),
        )
        residual = centered_exposure - design @ coefficients
        adjustment_fit_ok = bool(
            solver_status == 0
            and np.isfinite(coefficients).all()
            and np.isfinite(residual).all()
        )
        if not adjustment_fit_ok:
            residual = centered_exposure
    else:
        residual = centered_exposure

    finite_residual = np.isfinite(residual)
    low_residual = residual[low & finite_residual]
    high_residual = residual[high & finite_residual]
    if len(low_residual) >= 2 and len(high_residual) >= 2:
        low_interval = np.quantile(low_residual, [0.05, 0.95])
        high_interval = np.quantile(high_residual, [0.05, 0.95])
        intersection = max(0.0, min(low_interval[1], high_interval[1]) - max(low_interval[0], high_interval[0]))
        union = max(low_interval[1], high_interval[1]) - min(low_interval[0], high_interval[0])
        overlap = float(intersection / union) if union > 1e-12 else 1.0
    else:
        overlap = 0.0
    passed = (
        adjustment_fit_ok
        and standard_deviation >= config.minimum_exposure_standard_deviation
        and len(low_residual) >= config.minimum_group_size
        and len(high_residual) >= config.minimum_group_size
        and overlap >= config.minimum_overlap_fraction
    )
    return {
        "exposure_sd": standard_deviation,
        "low_group_n": int(len(low_residual)),
        "high_group_n": int(len(high_residual)),
        "overlap_fraction": overlap,
        "diagnostic_pass": bool(passed),
        "exposure_low": lower_cut,
        "exposure_high": upper_cut,
        "finite_sample_n": int(exposure.size),
        "adjustment_covariate_n": int(len(clean_columns)),
        "adjustment_fit_ok": adjustment_fit_ok,
    }


def build_causal_edges(
    gamma: np.ndarray,
    regions: list[str],
    region_index: np.ndarray,
    bundle: InputBundle,
    communication: CommunicationResult,
    base: BaseGRNResult,
    data_config: DataConfig,
    config: CausalConfig,
    coefficient_threshold: float,
    active_lr_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    covariates, covariate_names = one_hot_frame(bundle.metadata, data_config.confounder_columns)
    all_lr_ids = communication.exposures.columns.astype(str).tolist()
    if active_lr_indices is None:
        active_lr_indices = np.arange(len(all_lr_ids), dtype=np.int64)
    active_lr_indices = np.asarray(active_lr_indices, dtype=np.int64)
    if gamma.shape[1] != len(active_lr_indices):
        raise ValueError("causal gamma axis does not match active LR indices")
    lr_ids = [all_lr_ids[index] for index in active_lr_indices]
    diagnostics = {
        lr_id: _overlap_diagnostic(
            communication.exposures[lr_id].to_numpy(dtype=float), covariates, config
        )
        for lr_id in lr_ids
    }
    activity_available = any(
        matrix is not None
        for matrix in (bundle.ligand_activity, bundle.receptor_activity, bundle.tf_activity)
    )
    perturbations = bundle.perturbations
    perturbation_lookup: dict[tuple[str, str, str, str], float] = {}
    if perturbations is not None and {"lr_id", "tf", "target", "delta_observed"}.issubset(perturbations.columns):
        for row in perturbations.itertuples(index=False):
            region = str(getattr(row, "region", "global"))
            perturbation_lookup[(str(row.lr_id), str(row.tf), str(row.target), region)] = float(row.delta_observed)

    rows: list[dict[str, object]] = []
    for region_position, region in enumerate(regions):
        cells = np.flatnonzero(region_index == region_position)
        if len(cells) == 0:
            continue
        for lr_index, lr_id in enumerate(lr_ids):
            diagnostic = diagnostics[lr_id]
            exposure = communication.exposures[lr_id].to_numpy(dtype=float)[cells]
            exposure_iqr = float(np.quantile(exposure, 0.75) - np.quantile(exposure, 0.25))
            for tf_index, tf in enumerate(base.tf_names):
                tf_values = base.tf_matrix[cells, tf_index]
                tf_iqr = float(np.quantile(tf_values, 0.75) - np.quantile(tf_values, 0.25))
                for target_index, target in enumerate(base.target_names):
                    estimate = float(gamma[region_position, lr_index, tf_index, target_index])
                    if abs(estimate) < coefficient_threshold:
                        continue
                    level = 0
                    label = "CCC-associated"
                    has_adjustment = len(covariate_names) > 0 or not config.require_adjustment_covariates_for_level1
                    if bool(diagnostic["diagnostic_pass"]) and has_adjustment:
                        level = 1
                        label = "observationally-identified"
                    if activity_available and level >= 1:
                        level = config.activity_evidence_level
                        label = "activity-supported"
                    observed_delta = perturbation_lookup.get((lr_id, tf, target, region))
                    if observed_delta is None:
                        observed_delta = perturbation_lookup.get((lr_id, tf, target, "global"))
                    direction_consistent = np.nan
                    if observed_delta is not None:
                        direction_consistent = bool(np.sign(observed_delta) == np.sign(estimate))
                        if direction_consistent:
                            level = config.perturbation_evidence_level
                            label = "perturbation-supported"
                    rows.append(
                        {
                            "method": "JumpNicheRewire",
                            "region": region,
                            "lr_id": lr_id,
                            "tf": tf,
                            "target": target,
                            "gamma_estimate": estimate,
                            "causal_estimand": estimate * exposure_iqr * tf_iqr,
                            "adjustment_set": ";".join(covariate_names) if covariate_names else "none",
                            **diagnostic,
                            "observed_delta": observed_delta,
                            "direction_consistent": direction_consistent,
                            "evidence_level": level,
                            "evidence_label": label,
                        }
                    )
    return pd.DataFrame(rows)
