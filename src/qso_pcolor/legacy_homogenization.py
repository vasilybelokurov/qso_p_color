"""Experimental Legacy photometric maps; no implicit use by active scoring.

Maps act on complete g/r/z magnitude or luptitude vectors in mag. Applying a
published magnitude relation to luptitudes is an explicit faint-flux extension,
not an algebraically exact reproduction of that flux relation. Forward affine
maps of mixture coordinates allow subsequent exact missing-band marginalisation.
"""
from __future__ import annotations

import numpy as np

from .gaussmix import GaussianMixture


def affine_photometry(x: np.ndarray, covariance: np.ndarray, matrix: np.ndarray,
                      offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform complete vectors (mag) and covariance (mag squared) linearly."""
    x, covariance = np.asarray(x, float), np.asarray(covariance, float)
    matrix, offset = np.asarray(matrix, float), np.asarray(offset, float)
    if x.shape[-1] != matrix.shape[1] or covariance.shape != x.shape + (x.shape[-1],):
        raise ValueError('photometry/covariance/map shape mismatch')
    if not np.isfinite(x).all() or not np.isfinite(covariance).all():
        raise ValueError('complete finite vectors required; do not impute missing bands')
    return x @ matrix.T + offset, matrix @ covariance @ matrix.T


def cubic_photometry(x: np.ndarray, coefficients: np.ndarray,
                     colour_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Subtract fitted N-S offsets versus northern g-z; return local Jacobian.

Coefficients have shape (polynomial powers, 3), in increasing power order.
Only the colour used for the correction is bounded; measurements are retained.
The returned Jacobian propagates full covariance by J S J.T to first order.
"""
    x = np.asarray(x, float)
    if x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError('complete finite grz vectors required')
    raw = x[:, 0] - x[:, 2]
    colour = np.clip(raw, *colour_range)
    powers = np.polynomial.polynomial.polyvander(colour, len(coefficients)-1)
    delta = powers @ coefficients
    dc = sum(i * colour[:, None] ** (i-1) * coefficients[i]
             for i in range(1, len(coefficients)))
    dc *= ((raw > colour_range[0]) & (raw < colour_range[1]))[:, None]
    jac = np.broadcast_to(np.eye(3), (len(x), 3, 3)).copy()
    jac[:, :, 0] -= dc
    jac[:, :, 2] += dc
    return x - delta, jac


def tied_native_mixture(mixture: GaussianMixture, north_indices: np.ndarray,
                        south_indices: np.ndarray, north_from_south: np.ndarray,
                        offset: np.ndarray, residual_covariance: np.ndarray) -> GaussianMixture:
    """Replace northern coordinates by a noisy affine view of shared south ones.

All other coordinates and their mutual covariances remain unchanged. This is
a diagnostic projection of an existing fit, NOT a fit using pooled data.
Residual covariance (mag squared) must be positive definite. Observed subsets
remain exact Gaussian marginals; missing correction colours need no imputation.
"""
    ni, si = np.asarray(north_indices, int), np.asarray(south_indices, int)
    a, b, c = (np.asarray(v, float) for v in (north_from_south, offset, residual_covariance))
    if len(ni) != len(si) or set(ni) & set(si):
        raise ValueError('disjoint paired northern/southern coordinates required')
    if a.shape != (len(ni), len(si)) or b.shape != (len(ni),) or c.shape != a.shape:
        raise ValueError('map shape mismatch')
    if not np.isfinite(c).all() or not np.allclose(c, c.T):
        raise ValueError('finite symmetric residual covariance required')
    np.linalg.cholesky(c)
    retain = np.array([i for i in range(mixture.n_dim) if i not in ni])
    projection = np.zeros((mixture.n_dim, len(retain)))
    projection[retain, np.arange(len(retain))] = 1
    projection[np.ix_(ni, [int(np.flatnonzero(retain == j)[0]) for j in si])] = a
    shift = np.zeros(mixture.n_dim); shift[ni] = b
    means = mixture.means[:, retain] @ projection.T + shift
    covs = projection @ mixture.covs[:, retain][:, :, retain] @ projection.T
    covs[:, ni[:, None], ni] += c
    return GaussianMixture(mixture.weights.copy(), means, covs, labels=mixture.labels)
