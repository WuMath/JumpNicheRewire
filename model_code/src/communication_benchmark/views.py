"""Explicit edge-coefficient views, not expression-contribution networks.

One target at a time: arrays are n_cells x n_TFs, never n_cells x genes^2.
Reference exposure is mandatory. A centered zero denotes mean exposure, not
absence of communication. No labels, parameter selection or fitting occur here.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NetworkViews:
    reference: np.ndarray
    total: np.ndarray
    communication_delta: np.ndarray
    reference_name: str

    def global_scores(self):
        # Fixed mean absolute cell coefficient: do NOT abs(mean) away changes.
        return {name: np.mean(np.abs(getattr(self, name)), axis=0)
                for name in ('reference', 'total', 'communication_delta')}


def _finite(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f'{name}: invalid shape or nonfinite values')
    return value


def coefficient_views(backbone, spatial, lr, h, z, reference_z, *, reference_name):
    """W_ref=B+HU+z_ref*Gamma; W_total=B+HU+Z*Gamma.

    Coefficients must already be on the same TF expression scale. For v2 these
    are the unscaled backbone/spatial/lr arrays, not penalized theta. Spatial
    state stays fixed across views. reference_z uses the fitted Z coordinates.
    """
    if not isinstance(reference_name, str) or not reference_name.strip():
        raise ValueError('An explicit reference name is required')
    b = _finite(backbone, 1, 'backbone')
    u, g = _finite(spatial, 2, 'spatial'), _finite(lr, 2, 'lr')
    h, z = _finite(h, 2, 'h'), _finite(z, 2, 'z')
    r = _finite(reference_z, 1, 'reference_z')
    if not len(b) or not len(h) or len(h) != len(z):
        raise ValueError('Empty or misaligned cells/TFs')
    if u.shape != (len(b), h.shape[1]) or g.shape != (len(b), z.shape[1]):
        raise ValueError('TF/modifier dimensions do not match')
    if r.shape != (z.shape[1],):
        raise ValueError('Reference LR dimensions do not match')
    with np.errstate(over='raise', invalid='raise'):
        ref = b[None, :] + h @ u.T + (r @ g.T)[None, :]
        total = b[None, :] + h @ u.T + z @ g.T
        delta = (z-r[None, :]) @ g.T
    if not all(np.isfinite(a).all() for a in (ref, total, delta)):
        raise ValueError('Nonfinite derived networks')
    if not np.allclose(ref+delta, total, atol=1e-10, rtol=1e-10):
        raise ValueError('Reference + communication delta != total')
    return NetworkViews(ref, total, delta, reference_name)


def reference_in_centered_coordinates(raw_exposure, raw_reference, fitted_z):
    """Rebuild v2 centering/RMS and VERIFY against checkpoint before using it.

    The caller must align cells and target-specific permitted LR IDs first.
    Zero-variance modifiers have zero contribution under v2's transformation.
    outside_marginal_support marks extrapolation; false does not prove joint
    support or causal validity.
    """
    raw = _finite(raw_exposure, 2, 'raw_exposure')
    ref = _finite(raw_reference, 1, 'raw_reference')
    z = _finite(fitted_z, 2, 'fitted_z')
    if not len(raw) or raw.shape != z.shape or ref.shape != (raw.shape[1],):
        raise ValueError('Exposure/reference alignment differs')
    mean = raw.mean(axis=0)
    centered = raw-mean
    rms = np.sqrt(np.mean(centered**2, axis=0))
    active = rms > 1e-12
    expected = np.zeros_like(raw)
    expected[:, active] = centered[:, active]/rms[active]
    if not np.allclose(expected, z, atol=1e-10, rtol=1e-10):
        raise ValueError('Raw exposures do not reproduce fitted modifiers')
    normalized = np.zeros_like(ref)
    normalized[active] = (ref[active]-mean[active])/rms[active]
    outside = (ref < raw.min(axis=0)) | (ref > raw.max(axis=0))
    return normalized, {'outside_marginal_support': outside.tolist(),
                        'inactive_reference_changed': ((~active) & (ref != mean)).tolist(),
                        'causal_interpretation': False}
