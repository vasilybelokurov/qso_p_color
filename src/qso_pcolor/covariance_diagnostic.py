"""Isolated covariance-constraint experiments; never used by production fits."""
from __future__ import annotations

import numpy as np

from .gaussmix import GaussianMixture
from .streaming_xd import accumulate_batches


def floor_covariance(scatter: np.ndarray, floor: float) -> np.ndarray:
    """Constrained Gaussian M step with eigenvalues >= floor (feature units squared).

    Minimize log det(V) + tr(S V^-1) over V >= floor I, where S is
    the conditional centred scatter. Eigenvectors follow S and each eigenvalue
    is max(eigenvalue(S), floor). No covariance inverse is formed.
    """
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('positive finite covariance floor required')
    s = np.asarray(scatter, float)
    if not np.isfinite(s).all() or s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError('finite square scatter required')
    values, vectors = np.linalg.eigh(.5*(s+s.T))
    result = (vectors*np.maximum(values, floor)) @ vectors.T
    return .5*(result+result.T)


def constrained_step(source, mix: GaussianMixture, *, floor: float,
                     expected_rows: int) -> tuple[GaussianMixture, dict]:
    """One diagnostic XD update using all rows and the exact existing E step."""
    if np.linalg.eigvalsh(mix.covs).min() < floor*(1-1e-10):
        raise ValueError('starting model must satisfy the proposed fixed constraint')
    q, dm, v, ll, weight, seen = accumulate_batches(source, mix)
    if seen != expected_rows or weight <= 0:
        raise ValueError('diagnostic row accounting changed')
    means, covs = mix.means.copy(), mix.covs.copy()
    for j in np.flatnonzero(q > 0):
        shift = dm[j]/q[j]
        means[j] += shift
        covs[j] = floor_covariance(v[j]/q[j]-np.outer(shift, shift), floor)
    candidate = GaussianMixture(q/q.sum(), means, covs, mix.labels)
    after = total_weight = 0.; rows = 0
    for x, covariance, observed, weights in source():
        after += float(weights @ candidate.log_prob(x, covariance, observed=observed))
        total_weight += weights.sum(); rows += len(x)
    if rows != expected_rows or not np.isclose(total_weight, weight):
        raise ValueError('diagnostic evaluation rows or weights changed')
    return candidate, dict(rows=rows, floor=floor, before=ll/weight, after=after/weight,
        change=(after-ll)/weight, min_eigenvalue=float(np.linalg.eigvalsh(candidate.covs).min()),
        production_model_changed=False)
