"""Experimental streamed XD with known linear observation operators.

Latent coordinates and observations are in mag; covariances in mag squared.
Densities are normalized over the observed native coordinates, including when
the observation matrix is rectangular or rank deficient. No bands are imputed.
This module is opt-in and does not change the production trainer.
"""
from __future__ import annotations

from collections.abc import Callable
import numpy as np
from scipy.special import logsumexp

from .gaussmix import GaussianMixture, log_gauss_batch
from .xd import XDFitResult, _mask_groups


def native_mixture(mix: GaussianMixture, matrix: np.ndarray, offset: np.ndarray,
                   extra_covariance: np.ndarray, labels: tuple[str, ...] = ()) -> GaussianMixture:
    """Project a latent mixture into native mag coordinates before masking."""
    h, b, t = map(np.asarray, (matrix, offset, extra_covariance))
    if h.ndim != 2 or h.shape[1] != mix.n_dim or b.shape != (h.shape[0],) or t.shape != (len(b), len(b)):
        raise ValueError('incompatible observation operator')
    if not all(np.isfinite(v).all() for v in (h, b, t)) or not np.allclose(t, t.T):
        raise ValueError('finite operator and symmetric extra covariance required')
    if np.linalg.eigvalsh(t).min() < -1e-12:
        raise ValueError('extra covariance must be positive semidefinite')
    return GaussianMixture(mix.weights, mix.means @ h.T + b,
                           h @ mix.covs @ h.T + t, labels)


def projected_moments(mix: GaussianMixture, y: np.ndarray, noise: np.ndarray,
                      matrix: np.ndarray, offset: np.ndarray) -> tuple:
    """Return log component densities and latent conditional mean shifts/covariance.

    Inputs contain observed coordinates only; noise includes extra scatter.
    Component densities exclude mixture weights, with units mag**(-n_observed).
    """
    h = np.asarray(matrix)
    cross = mix.covs @ h.T
    total = (h @ cross)[None] + noise[:, None]
    mean = mix.means @ h.T + offset
    lp = log_gauss_batch(y[:, None], mean[None], total)
    chol = np.linalg.cholesky(total)
    rhs = np.broadcast_to(cross.swapaxes(-1, -2)[None],
                          (len(y), mix.n_components, h.shape[0], mix.n_dim))
    gain_t = np.linalg.solve(chol.swapaxes(-1, -2), np.linalg.solve(chol, rhs))
    gain = gain_t.swapaxes(-1, -2)
    shift = np.einsum('nkdp,nkp->nkd', gain, y[:, None] - mean[None])
    conditional = mix.covs[None] - gain @ cross.swapaxes(-1, -2)[None]
    return lp, shift, conditional


def accumulate_projected(source: Callable, mix: GaussianMixture, operators: dict) -> tuple:
    """Accumulate one complete pass of (y, covariance, observed, system) batches."""
    k, d = mix.means.shape
    count, first, second = np.zeros(k), np.zeros((k, d)), np.zeros((k, d, d))
    total_ll, seen = 0., 0
    for y, cov, observed, system in source():
        if not observed.any(axis=1).all() or not np.isfinite(y[observed]).all():
            raise ValueError('every row needs finite observed measurements')
        seen += len(y)
        for s in np.unique(system):
            if int(s) not in operators:
                raise ValueError('unknown observation system')
            h, b, t = operators[int(s)]
            for local, dims in _mask_groups(observed[system == s]):
                rows = np.flatnonzero(system == s)[local]
                c = cov[rows][:, dims][:, :, dims] + t[np.ix_(dims, dims)]
                lp, shift, conditional = projected_moments(
                    mix, y[rows][:, dims], c, h[dims], b[dims])
                with np.errstate(divide='ignore'):
                    weighted = lp + np.log(mix.weights)
                norm = logsumexp(weighted, axis=1)
                resp = np.exp(weighted - norm[:, None])
                count += resp.sum(axis=0)
                first += np.einsum('nk,nkd->kd', resp, shift)
                second += np.einsum('nk,nkde->kde', resp, conditional)
                second += np.einsum('nk,nkd,nke->kde', resp, shift, shift)
                total_ll += norm.sum()
    return count, first, second, float(total_ll), seen


def fit_projected(source: Callable, *, init: GaussianMixture, operators: dict,
                  expected_rows: int, max_iter: int, tol: float,
                  regularization: float, progress: Callable | None = None) -> XDFitResult:
    """Fit all rows using fixed operators; check row accounting on every pass.

    ``progress(iteration, model, mean_log_density, rows)`` receives each updated
    mixture. The reported history is the pre-update density in native mag units.
    """
    if expected_rows < 1 or max_iter < 1 or tol < 0 or regularization < 0:
        raise ValueError('invalid projected fit settings')
    for h, b, t in operators.values():
        native_mixture(init, h, b, t)
    mix, history, converged = init, [], False
    for it in range(1, max_iter + 1):
        count, first, second, ll, n = accumulate_projected(source, mix, operators)
        if n != expected_rows:
            raise ValueError(f'row accounting: expected {expected_rows}, saw {n}')
        alive = count > 1e-10
        shift = first / np.maximum(count[:, None], 1e-300)
        means, covs = mix.means.copy(), mix.covs.copy()
        means[alive] += shift[alive]
        c = second[alive] / count[alive, None, None] - shift[alive, :, None] * shift[alive, None, :]
        covs[alive] = .5 * (c + c.swapaxes(-1, -2)) + regularization * np.eye(mix.n_dim)
        np.linalg.cholesky(covs)
        mix = GaussianMixture(count / count.sum(), means, covs, init.labels)
        history.append(ll / n)
        if progress is not None:
            progress(it, mix, history[-1], n)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol * max(1., abs(history[-2])):
            converged = True
            break
    ll, n = 0., 0
    natives = {s: native_mixture(mix, *op) for s, op in operators.items()}
    for y, cov, obs, system in source():
        n += len(y)
        for s in np.unique(system):
            rows = system == s
            ll += natives[int(s)].log_prob(y[rows], cov[rows], observed=obs[rows]).sum()
    if n != expected_rows:
        raise ValueError('row accounting changed during final evaluation')
    return XDFitResult(mix, it, float(ll/n), converged, history)
