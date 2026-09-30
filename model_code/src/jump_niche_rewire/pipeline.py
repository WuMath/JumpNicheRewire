"""End-to-end four-module JumpNicheRewire pipeline."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .base_grn import fit_baseline_grn
from .checkpoint import (
    load_numerical_checkpoint,
    remove_numerical_checkpoint,
    remove_target_checkpoints,
    save_numerical_checkpoint,
)
from .communication import build_communication_exposure, mix_communication_results
from .config import ProjectConfig
from .io import load_inputs
from .prior import build_soft_priors
from .results import validate_output_dependencies, write_results
from .rewiring import fit_rewiring, populate_rewiring_outputs
from .convergence import summarize_convergence, ConvergenceError
from threadpoolctl import threadpool_limits
from .spatial import build_spatial_context
from .types import PipelineResult
from .utils import format_seconds, log_message, choose_top_indices


def _adapt_communication_mixing(
    current: float,
    residual: float,
    previous_residual: float | None,
    direction: np.ndarray,
    previous_direction: np.ndarray | None,
    config: object,
) -> tuple[float, float, bool]:
    """Increase old-state weight when the raw fixed-point updates oscillate."""

    cosine = np.nan
    if not config.adaptive_communication_mixing or previous_residual is None or previous_direction is None:
        return current, cosine, False
    denominator = float(np.linalg.norm(direction) * np.linalg.norm(previous_direction))
    if denominator > 1e-15:
        cosine = float(np.sum(direction * previous_direction) / denominator)
    stalled = residual >= config.mixing_stall_ratio * previous_residual
    oscillating = np.isfinite(cosine) and cosine <= config.mixing_oscillation_cosine
    diverging = residual > 1.25 * previous_residual
    if not ((stalled and oscillating) or diverging):
        return current, cosine, False
    maximum = max(current, float(config.maximum_communication_mixing))
    updated = min(maximum, 1.0 - (1.0 - current) * config.mixing_step_contraction)
    return float(updated), cosine, bool(updated > current)


def _mix_normalized_expression(
    previous: pd.DataFrame,
    candidate: pd.DataFrame,
    old_weight: float,
) -> pd.DataFrame:
    """Damp expression with the same weight as the coupled communication state."""

    if not previous.index.equals(candidate.index) or not previous.columns.equals(candidate.columns):
        raise ValueError("cannot mix normalized expression with different labels")
    old_weight = float(np.clip(old_weight, 0.0, 1.0))
    return old_weight * previous + (1.0 - old_weight) * candidate


class JumpNicheRewirePipeline:
    """Run the paper's four core modules with auditable evidence fallbacks."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.random = np.random.default_rng(config.runtime.random_seed)

    def run(self) -> PipelineResult:
        try:
            limiter = threadpool_limits(
                limits=self.config.runtime.blas_threads, user_api="blas"
            )
        except (OSError, RuntimeError) as error:
            # Some Windows environments load incompatible MKL/OpenMP discovery
            # DLLs.  Evidence workers already set the vendor thread environment
            # variables before Python starts, so controller failure is safe to
            # bypass and must not abort the numerical run.
            log_message(
                self.config.runtime.verbose,
                f"[threads] threadpool controller unavailable; using process "
                f"environment limits ({type(error).__name__}: {error})",
            )
            return self._run()
        with limiter:
            return self._run()

    def _run(self) -> PipelineResult:
        if not self.config.data.is_spatial and not self.config.rewiring.disable_communication_rewiring:
            raise ValueError("Non-spatial input requires disable_communication_rewiring=True")
        validate_output_dependencies(self.config.output)
        checkpoint = load_numerical_checkpoint(self.config)
        if checkpoint is not None:
            log_message(
                self.config.runtime.verbose,
                "[checkpoint] Reusing completed numerical state; rebuilding outputs only",
            )
            return self._finish_numerical_state(checkpoint)
        timings: dict[str, float] = {}
        start = time.perf_counter()
        bundle = load_inputs(self.config.data)
        timings["input"] = time.perf_counter() - start
        log_message(self.config.runtime.verbose, f"[input] {format_seconds(timings['input'])}")

        start = time.perf_counter()
        spatial = build_spatial_context(
            bundle, self.config.data, self.config.spatial, self.random
        )
        timings["module1_spatial"] = time.perf_counter() - start
        log_message(
            self.config.runtime.verbose,
            f"[module 1] mode={'spatial' if self.config.data.is_spatial else 'nonspatial_global'}; "
            f"{len(spatial.candidate_tfs)} TFs, {len(spatial.candidate_targets)} targets, {len(spatial.graph.source)} spatial edges, "
            f"{format_seconds(timings['module1_spatial'])}",
        )

        start = time.perf_counter()
        prior = build_soft_priors(bundle, spatial, self.config.prior)
        timings["module3_prior_build"] = time.perf_counter() - start
        log_message(
            self.config.runtime.verbose,
            f"[module 3 priors] {format_seconds(timings['module3_prior_build'])}",
        )

        start = time.perf_counter()
        base = fit_baseline_grn(
            bundle,
            spatial,
            prior,
            self.config.data,
            self.config.base_grn,
            self.config,
        )
        timings["module2_base_grn"] = time.perf_counter() - start
        log_message(
            self.config.runtime.verbose,
            f"[module 2] {len(base.tf_names)} TFs x {len(base.target_names)} targets, "
            f"{format_seconds(timings['module2_base_grn'])}",
        )

        start = time.perf_counter()
        communication = build_communication_exposure(
            bundle, spatial, self.config.data, self.config.communication
        )
        timings["module4_communication"] = time.perf_counter() - start
        log_message(
            self.config.runtime.verbose,
            f"[module 4 exposure] {format_seconds(timings['module4_communication'])}",
        )

        start = time.perf_counter()
        working_spatial = spatial
        history_frames = []
        joint_rows = []
        previous_gamma = None
        previous_beta = None
        previous_communication_direction = None
        previous_fixed_point_residual = None
        communication_mixing = float(self.config.rewiring.communication_mixing)
        joint_converged = False
        active_lr = choose_top_indices(
            np.var(communication.exposures.to_numpy(dtype=float), axis=0),
            min(self.config.rewiring.maximum_active_lr, communication.exposures.shape[1]),
        )
        if not 0 <= self.config.rewiring.communication_mixing < 1:
            raise ValueError("communication_mixing must be in [0, 1); 1 would fake convergence")
        if not 0 <= self.config.rewiring.maximum_communication_mixing < 1:
            raise ValueError("maximum_communication_mixing must be in [0, 1)")
        if not 0 < self.config.rewiring.mixing_step_contraction < 1:
            raise ValueError("mixing_step_contraction must be in (0, 1)")
        if self.config.rewiring.mixing_stall_ratio <= 0 or not -1 <= self.config.rewiring.mixing_oscillation_cosine <= 1:
            raise ValueError("invalid adaptive communication mixing thresholds")
        if self.config.rewiring.joint_iterations < 1:
            raise ValueError("joint_iterations must be positive")
        for iteration in range(1, self.config.rewiring.joint_iterations + 1):
            rewiring = fit_rewiring(
                bundle, working_spatial, prior, base, communication,
                self.config.data, self.config.rewiring, self.config.causal,
                initial_gamma=previous_gamma, build_outputs=False, fixed_active_lr=active_lr,
                project_config=self.config,
                checkpoint_stage=f"rewiring_joint_{iteration:03d}",
            )
            history_frames.append(rewiring.history.assign(joint_iteration=iteration))
            inner_rate = float(rewiring.diagnostics["converged"].mean())
            if self.config.rewiring.disable_communication_rewiring:
                joint_converged = True
                joint_rows.append(
                    dict(
                        module="joint_iteration",
                        target="__all__",
                        iteration=iteration,
                        relative_change=np.nan,
                        fixed_point_residual=np.nan,
                        damped_change=np.nan,
                        communication_mixing=np.nan,
                        mixing_adapted=0,
                        oscillation_cosine=np.nan,
                        beta_relative_change=np.nan,
                        tolerance=self.config.rewiring.iteration_tolerance,
                        inner_convergence_rate=inner_rate,
                        objective=float(rewiring.diagnostics["objective"].sum()),
                        converged=np.nan,
                        applicable=0,
                        note="not applicable: communication rewiring is disabled",
                    )
                )
                log_message(
                    self.config.runtime.verbose,
                    f"[joint] not applicable (GRN-only); inner={inner_rate:.1%}",
                )
                break
            predicted_standardized = base.covariate_prediction + np.einsum("ik,ikg->ig", base.tf_matrix, rewiring.beta_ccc)
            predicted_normalized = predicted_standardized * base.target_scale[None, :] + base.target_mean[None, :]
            updated_expression = spatial.normalized_expression.copy()
            updated_expression.loc[:, base.target_names] = np.clip(predicted_normalized, 0., None)
            candidate_spatial = replace(spatial, normalized_expression=updated_expression)
            candidate_communication = build_communication_exposure(
                bundle, candidate_spatial, self.config.data, self.config.communication,
            )
            old_values = communication.exposures.to_numpy(dtype=float)
            candidate_values = candidate_communication.exposures.to_numpy(dtype=float)
            communication_direction = candidate_values - old_values
            # Certify the UN-DAMPED fixed-point residual, never alpha*residual.
            iteration_change = float(np.linalg.norm(communication_direction) / max(np.linalg.norm(old_values), 1e-12))
            communication_mixing, oscillation_cosine, mixing_adapted = _adapt_communication_mixing(
                communication_mixing, iteration_change, previous_fixed_point_residual,
                communication_direction, previous_communication_direction, self.config.rewiring,
            )
            beta_change = 0. if previous_beta is None else float(np.linalg.norm(rewiring.beta_ccc - previous_beta) / max(np.linalg.norm(previous_beta), 1.))
            joint_converged = bool(
                np.isfinite(iteration_change) and iteration_change <= self.config.rewiring.iteration_tolerance
                and beta_change <= self.config.rewiring.iteration_tolerance
                and inner_rate >= self.config.runtime.minimum_target_convergence_rate
            )
            joint_rows.append(dict(
                module="joint_iteration", target="__all__", iteration=iteration,
                relative_change=iteration_change, fixed_point_residual=iteration_change,
                damped_change=(1-communication_mixing)*iteration_change,
                communication_mixing=communication_mixing,
                mixing_adapted=int(mixing_adapted), oscillation_cosine=oscillation_cosine,
                beta_relative_change=beta_change, tolerance=self.config.rewiring.iteration_tolerance,
                inner_convergence_rate=inner_rate, objective=float(rewiring.diagnostics["objective"].sum()),
                converged=int(joint_converged),
            ))
            log_message(self.config.runtime.verbose, f"[joint {iteration}] fixed_point={iteration_change:.3g}, beta_change={beta_change:.3g}, inner={inner_rate:.1%}")
            if mixing_adapted:
                log_message(self.config.runtime.verbose, f"[joint {iteration}] oscillation detected; communication_mixing -> {communication_mixing:.3f}")
            # On exit, coefficients and exposure remain from the SAME fitted state.
            if joint_converged or iteration == self.config.rewiring.joint_iterations or inner_rate < self.config.runtime.minimum_target_convergence_rate:
                break
            # Both arrays are immutable inputs to the next numerical solve.  Keep
            # references instead of duplicating what can be a multi-GiB beta cube.
            previous_gamma, previous_beta = rewiring.gamma, rewiring.beta_ccc
            previous_communication_direction = communication_direction.copy()
            previous_fixed_point_residual = iteration_change
            communication = mix_communication_results(communication, candidate_communication, communication_mixing)
            mixed_expression = _mix_normalized_expression(
                working_spatial.normalized_expression,
                candidate_spatial.normalized_expression,
                communication_mixing,
            )
            working_spatial = replace(working_spatial, normalized_expression=mixed_expression)
        joint_summary = dict(joint_rows[-1])
        if self.config.rewiring.disable_communication_rewiring:
            stop_reason = "not_applicable"
        else:
            stop_reason = "converged" if joint_converged else (
                "inner_not_converged"
                if inner_rate < self.config.runtime.minimum_target_convergence_rate
                else "iteration_limit"
            )
        joint_summary.update(iterations=iteration, stop_reason=stop_reason)
        rewiring.diagnostics = pd.concat([rewiring.diagnostics, pd.DataFrame([joint_summary])], ignore_index=True)
        rewiring.history = pd.concat(history_frames + [pd.DataFrame(joint_rows)], ignore_index=True)
        timings["module4_rewiring_numerical"] = time.perf_counter() - start
        state = {
            "bundle": bundle,
            "spatial": spatial,
            "prior": prior,
            "base": base,
            "communication": communication,
            "working_spatial": working_spatial,
            "rewiring": rewiring,
            "timings": timings,
        }
        checkpoint_path = save_numerical_checkpoint(self.config, state)
        if checkpoint_path is not None:
            log_message(
                self.config.runtime.verbose,
                f"[checkpoint] {checkpoint_path}",
            )
        return self._finish_numerical_state(state)

    def _finish_numerical_state(self, state: dict[str, object]) -> PipelineResult:
        bundle = state["bundle"]
        spatial = state["spatial"]
        prior = state["prior"]
        base = state["base"]
        communication = state["communication"]
        working_spatial = state["working_spatial"]
        rewiring = state["rewiring"]
        timings = dict(state["timings"])

        start = time.perf_counter()
        rewiring = populate_rewiring_outputs(
            rewiring, bundle, working_spatial, base, communication, self.config.data, self.config.rewiring, self.config.causal,
        )
        timings["module4_output_build"] = time.perf_counter() - start
        log_message(
            self.config.runtime.verbose,
            f"[module 4 output build] {format_seconds(timings['module4_output_build'])}",
        )

        start = time.perf_counter()
        workbook_path = write_results(
            self.config,
            bundle,
            spatial,
            prior,
            base,
            communication,
            rewiring,
            timings,
        )
        timings["output"] = time.perf_counter() - start
        log_message(self.config.runtime.verbose, f"[output] {workbook_path}")
        result = PipelineResult(
            spatial=spatial,
            prior=prior,
            base=base,
            communication=communication,
            rewiring=rewiring,
            workbook_path=str(workbook_path),
        )
        summary = summarize_convergence(
            base,
            rewiring,
            self.config.runtime.minimum_target_convergence_rate,
            communication_rewiring_disabled=self.config.rewiring.disable_communication_rewiring,
        )
        if self.config.runtime.require_convergence and not summary["convergence_passed"]:
            raise ConvergenceError(result, summary)
        if not self.config.runtime.keep_numerical_checkpoint_after_success:
            remove_numerical_checkpoint(self.config)
        if not self.config.runtime.keep_target_checkpoints_after_success:
            remove_target_checkpoints(self.config)
        return result
