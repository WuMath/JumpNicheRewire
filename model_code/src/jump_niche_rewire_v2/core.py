"""Auditable convex numerical prototype; no paths, labels, or benchmark access.

Generic sparse-group proximal gradient, not a new optimization algorithm.
See docs/DECOMPOSED_MODEL_V2_DESIGN.md for citations and limitations.
The eventual dataset adapter must mask target-containing LR exposures.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DecompositionConfig:
    l1_backbone: float = 0.01
    l1_spatial: float = 0.02
    l1_lr: float = 0.02
    group_penalty: float = 0.01
    ridge_backbone: float = 0.01
    ridge_spatial: float = 0.1
    ridge_lr: float = 0.1
    max_iterations: int = 10000
    absolute_tolerance: float = 1e-7
    relative_tolerance: float = 1e-6
    svd_relative_tolerance: float = 1e-10
    max_estimated_matrix_bytes: int = 256 * 1024**2

    def validate(self):
        for name in ('l1_backbone', 'l1_spatial', 'l1_lr', 'group_penalty',
                     'ridge_backbone', 'ridge_spatial', 'ridge_lr',
                     'absolute_tolerance', 'relative_tolerance', 'svd_relative_tolerance'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if min(self.ridge_backbone, self.ridge_spatial, self.ridge_lr) <= 0:
            raise ValueError('Strictly positive ridge penalties are required')
        if not 0 < self.svd_relative_tolerance < 1:
            raise ValueError('SVD tolerance must be between zero and one')
        if self.absolute_tolerance + self.relative_tolerance <= 0:
            raise ValueError('At least one convergence tolerance must be positive')
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError('max_iterations must be a positive integer')
        if self.max_estimated_matrix_bytes < 1:
            raise ValueError('A positive matrix memory budget is required')


@dataclass
class Design:
    x: np.ndarray
    y: np.ndarray
    h: np.ndarray
    z: np.ndarray
    matrix: np.ndarray
    projected_y: np.ndarray
    column_scales: np.ndarray
    inactive_columns: np.ndarray
    nuisance: np.ndarray
    nuisance_pinv: np.ndarray
    l1: np.ndarray
    ridge: np.ndarray
    audit: dict

    @property
    def block_size(self):
        return 1 + self.h.shape[1] + self.z.shape[1]


@dataclass
class FitResult:
    theta: np.ndarray
    backbone: np.ndarray
    spatial: np.ndarray
    lr: np.ndarray
    nuisance_coefficients: np.ndarray
    converged: np.ndarray
    residual: np.ndarray
    tolerance: np.ndarray
    iterations: int
    lipschitz: float
    history: list


def _matrix(value, name, rows=None):
    if value is None:
        return np.empty((rows, 0), dtype=float)
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or (rows is not None and value.shape[0] != rows):
        raise ValueError(f'{name} must be a two-dimensional aligned matrix')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} has non-finite values')
    return value


def _center_modifiers(value):
    centered = value - value.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=0))
    active = rms > 1e-12
    centered[:, ~active] = 0
    centered[:, active] /= rms[active]
    return centered


def build_design(x, y, h=None, z=None, covariates=None, config=None):
    """Build from already standardized X/Y and fixed modifiers, not raw counts.

    Intercept and modifier main effects are unpenalized nuisance columns.
    Use the SAME nuisance projection for Y and all interaction design columns.
    """
    cfg = config or DecompositionConfig()
    cfg.validate()
    x, y = _matrix(x, 'x'), _matrix(y, 'y')
    n, t = x.shape
    if n < 2 or t < 1 or y.shape[0] != n or y.shape[1] < 1:
        raise ValueError('Need aligned X/Y with n>=2, T>=1 and G>=1')
    h, z = _matrix(h, 'h', n), _matrix(z, 'z', n)
    c = _matrix(covariates, 'covariates', n)
    width = 1 + h.shape[1] + z.shape[1]
    p, g, nc = t * width, y.shape[1], 1 + c.shape[1] + h.shape[1] + z.shape[1]
    estimate = 8 * (6*n*p + 4*p*p + 6*p*g + 6*n*nc + 4*nc*nc + 6*n*g)
    if estimate > cfg.max_estimated_matrix_bytes:
        raise MemoryError(f'Estimated numerical workspace {estimate} exceeds matrix budget')
    h, z = _center_modifiers(h), _center_modifiers(z)
    nuisance = np.column_stack([np.ones(n), c, h, z])
    left, singular, right = np.linalg.svd(nuisance, full_matrices=False)
    active = singular > cfg.svd_relative_tolerance * singular[0]
    q = left[:, active]
    pinv = (right[active].T / singular[active]) @ q.T
    modifiers = np.column_stack([np.ones(n), h, z])
    raw = (x[:, :, None] * modifiers[:, None, :]).reshape(n, p)
    matrix = raw - q @ (q.T @ raw)
    projected_y = y - q @ (q.T @ y)
    before = np.sqrt(np.mean(raw**2, axis=0))
    after = np.sqrt(np.mean(matrix**2, axis=0))
    inactive = after <= 1e-12 * np.maximum(1., before)
    scales = np.where(inactive, 1., after)
    matrix /= scales
    matrix[:, inactive] = 0
    l1 = np.tile([cfg.l1_backbone] + [cfg.l1_spatial]*h.shape[1] + [cfg.l1_lr]*z.shape[1], t)
    ridge = np.tile([cfg.ridge_backbone] + [cfg.ridge_spatial]*h.shape[1] + [cfg.ridge_lr]*z.shape[1], t)
    values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(values > cfg.svd_relative_tolerance * max(float(values[0]), 1.)))
    condition = float(values[0] / values[-1]) if rank == p else None
    audit = dict(n=n, t=t, g=g, p=p, nuisance_rank=int(active.sum()),
                 nuisance_columns=nc, design_rank=rank, full_column_rank=(rank == p),
                 design_condition_number=condition, inactive_columns=np.flatnonzero(inactive).tolist(),
                 projection_energy_ratio=(after / np.maximum(before, 1e-12)).tolist(),
                 estimated_matrix_bytes=estimate,
                 coefficient_identification='penalty-dependent; not causal identification')
    return Design(x, y, h, z, matrix, projected_y, scales, inactive,
                  nuisance, pinv, l1, ridge, audit)


def prox_sparse_group(value, l1, group_penalty, step, t, width):
    """Disjoint groups: weighted elementwise soft threshold, then L2 shrink."""
    shrunk = np.sign(value) * np.maximum(np.abs(value) - step*l1[:, None], 0.)
    grouped = shrunk.reshape(t, width, -1)
    norm = np.linalg.norm(grouped, axis=1, keepdims=True)
    factor = np.maximum(1. - step*group_penalty / np.maximum(norm, 1e-300), 0.)
    return (grouped * factor).reshape(value.shape)


def smooth_value_gradient(theta, gram, cross, constant):
    """Quadratic loss including ridge; one value per target."""
    product = gram @ theta
    return constant + .5*np.sum(theta*product, axis=0) - np.sum(theta*cross, axis=0), product-cross


def prediction_parts(design, backbone, spatial, lr):
    """Reconstruct n×G components without building cell×TF×target arrays."""
    base = design.x @ backbone
    local = np.zeros_like(base)
    communication = np.zeros_like(base)
    for tf in range(design.x.shape[1]):
        local += design.x[:, tf, None] * (design.h @ spatial[tf])
        communication += design.x[:, tf, None] * (design.z @ lr[tf])
    return base, local, communication


def fit(design, config=None, resource_callback=None):
    cfg = config or DecompositionConfig()
    cfg.validate()
    if resource_callback is not None:
        resource_callback()
    # Prevent accidentally fitting a design with penalties from another config.
    expected_l1 = np.tile([cfg.l1_backbone] + [cfg.l1_spatial]*design.h.shape[1] + [cfg.l1_lr]*design.z.shape[1], design.x.shape[1])
    expected_ridge = np.tile([cfg.ridge_backbone] + [cfg.ridge_spatial]*design.h.shape[1] + [cfg.ridge_lr]*design.z.shape[1], design.x.shape[1])
    if not (np.array_equal(expected_l1, design.l1) and np.array_equal(expected_ridge, design.ridge)):
        raise ValueError('Design penalties and fit configuration differ')
    a, y = design.matrix, design.projected_y
    n, p = a.shape
    gram = a.T @ a / n + np.diag(design.ridge)
    cross = a.T @ y / n
    constant = np.sum(y*y, axis=0) / (2*n)
    lipschitz = float(np.linalg.eigvalsh(gram)[-1]) * (1 + 1e-12)
    step = 1/lipschitz
    theta = np.zeros((p, y.shape[1]))
    threshold = cfg.absolute_tolerance + cfg.relative_tolerance*np.maximum(1., np.max(np.abs(cross), axis=0))
    history = []
    for iteration in range(cfg.max_iterations + 1):
        if resource_callback is not None and iteration % 25 == 0:
            resource_callback()
        smooth, gradient = smooth_value_gradient(theta, gram, cross, constant)
        candidate = prox_sparse_group(theta-step*gradient, design.l1, cfg.group_penalty,
                                      step, design.x.shape[1], design.block_size)
        mapping = (theta-candidate) / step
        residual = np.max(np.abs(mapping), axis=0)
        objective = smooth + np.sum(design.l1[:, None]*np.abs(theta), axis=0)
        objective += cfg.group_penalty*np.linalg.norm(theta.reshape(design.x.shape[1], design.block_size, -1), axis=1).sum(axis=0)
        if not (np.isfinite(objective).all() and np.isfinite(residual).all()):
            raise FloatingPointError('Non-finite objective or proximal mapping')
        history.append(dict(iteration=iteration, objective=objective.tolist(),
                            proximal_mapping_inf=residual.tolist()))
        converged = residual <= threshold
        if converged.all() or iteration == cfg.max_iterations:
            break
        theta = candidate
    raw = (theta / design.column_scales[:, None]).reshape(design.x.shape[1], design.block_size, -1)
    backbone = raw[:, 0, :].copy()
    spatial = raw[:, 1:1+design.h.shape[1], :].copy()
    lr = raw[:, 1+design.h.shape[1]:, :].copy()
    components = prediction_parts(design, backbone, spatial, lr)
    nuisance_coef = design.nuisance_pinv @ (design.y - sum(components))
    return FitResult(theta, backbone, spatial, lr, nuisance_coef, converged,
                     residual, threshold, iteration, lipschitz, history)
