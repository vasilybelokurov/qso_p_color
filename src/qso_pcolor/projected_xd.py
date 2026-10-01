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
                  regularization: float, progress: Callable | None = None,
                  initial_history: tuple[float, ...] = (),
                  accumulator: Callable | None = None,
                  covariance_update: str = 'additive',
                  prior_strength: float = 1.0,
                  final_evaluation: bool = True,
                  fixed_coordinate: dict | None = None) -> XDFitResult:
    """Fit all rows using fixed operators; check row accounting on every pass.

    ``progress(iteration, model, mean_objective, rows)`` receives each updated
    mixture. The reported history is the pre-update objective per row in native
    mag units. A restart supplies that history with its updated mixture.
    An optional accumulator parallelizes the E step, retaining one global update.

    ``covariance_update='additive'`` reproduces the historical update, which
    adds ``regularization`` (mag^2) to every covariance after the M step. It
    maximises no fixed objective, so its log likelihood can decline; it stops
    only on a non-negative change below ``tol``. ``'map'`` is the conjugate
    prior update of Bovy, Hogg & Roweis (2011, eqs. 19-20):
    ``V = (q S + nu w I) / (q + nu)`` with ``w = regularization`` and
    ``nu = prior_strength``, from the per-component log prior
    ``-(nu/2) [log|V| + w tr(V^-1)]``. Then the history is the mean log
    posterior (likelihood plus log prior, per row), which EM cannot decrease;
    the fit stops when its absolute relative change is below ``tol``.
    ``mean_loglike`` is always the plain log likelihood of the returned model;
    ``final_evaluation=False`` skips that extra pass and returns NaN, for
    callers that fit in blocks and evaluate only the model they keep.

    ``fixed_coordinate=dict(index=i, mean=m, variance=s2)`` holds latent
    coordinate ``i`` at the same mean and variance in every component, with
    zero covariance to all others. With that block structure the component
    likelihood factorises, so the exact constrained M step is the unconstrained
    (or MAP) update of the remaining block. Used to make QSO colours independent
    of magnitude: coordinate ``i`` is a magnitude, the others colours. The
    initial mixture must already satisfy the constraint.
    """
    if expected_rows < 1 or max_iter < 1 or tol < 0 or regularization < 0:
        raise ValueError('invalid projected fit settings')
    if covariance_update not in ('additive', 'map'):
        raise ValueError('covariance_update must be additive or map')
    if covariance_update == 'map' and not (regularization > 0 and np.isfinite(prior_strength) and prior_strength > 0):
        raise ValueError('map covariance update needs positive regularization and prior strength')
    use_map = covariance_update == 'map'
    if fixed_coordinate is not None:
        i_fix, m_fix, v_fix = (int(fixed_coordinate['index']), float(fixed_coordinate['mean']),
                               float(fixed_coordinate['variance']))
        others = np.arange(init.n_dim) != i_fix
        if not (0 <= i_fix < init.n_dim and v_fix > 0 and np.isfinite(m_fix)
                and np.allclose(init.means[:, i_fix], m_fix) and np.allclose(init.covs[:, i_fix, i_fix], v_fix)
                and np.allclose(init.covs[:, i_fix, others], 0) and np.allclose(init.covs[:, others, i_fix], 0)):
            raise ValueError('initial mixture violates the fixed-coordinate constraint')

    def stationary(new, old):
        change = new - old
        scale = tol * max(1., abs(old))
        return abs(change) < scale if use_map else 0 <= change < scale
    for h, b, t in operators.values():
        native_mixture(init, h, b, t)
    history = list(initial_history)
    if len(history) > max_iter or not np.isfinite(history).all():
        raise ValueError('invalid projected restart history')
    mix, converged, it = init, False, len(history)
    accumulate = accumulate_projected if accumulator is None else accumulator
    if len(history) > 1:
        converged = stationary(history[-1], history[-2])
    for it in range(it + 1, (it if converged else max_iter) + 1):
        count, first, second, ll, n = accumulate(source, mix, operators)
        if n != expected_rows:
            raise ValueError(f'row accounting: expected {expected_rows}, saw {n}')
        alive = count > 1e-10
        shift = first / np.maximum(count[:, None], 1e-300)
        means, covs = mix.means.copy(), mix.covs.copy()
        means[alive] += shift[alive]
        c = second[alive] / count[alive, None, None] - shift[alive, :, None] * shift[alive, None, :]
        c = .5 * (c + c.swapaxes(-1, -2))
        if use_map:
            # Scatter about the new mean, S = q c; prior adds nu w I and nu counts.
            q = count[alive, None, None]
            covs[alive] = (q*c + prior_strength*regularization*np.eye(mix.n_dim)) / (q + prior_strength)
            objective = ll + log_covariance_prior(mix.covs, regularization, prior_strength)
        else:
            covs[alive] = c + regularization * np.eye(mix.n_dim)
            objective = ll
        if fixed_coordinate is not None:
            means[:, i_fix] = m_fix
            covs[:, i_fix, :] = 0.; covs[:, :, i_fix] = 0.; covs[:, i_fix, i_fix] = v_fix
        np.linalg.cholesky(covs)
        mix = GaussianMixture(count / count.sum(), means, covs, init.labels)
        history.append(objective / n)
        if progress is not None:
            progress(it, mix, history[-1], n)
        if len(history) > 1 and stationary(history[-1], history[-2]):
            converged = True
            break
    value = mean_log_likelihood(source, mix, operators, expected_rows) if final_evaluation else float('nan')
    return XDFitResult(mix, it, value, converged, history)


def mean_log_likelihood(source: Callable, mix: GaussianMixture, operators: dict,
                        expected_rows: int) -> float:
    """Plain mean log likelihood per row (native mag units) over one full pass."""
    ll, n = 0., 0
    natives = {s: native_mixture(mix, *op) for s, op in operators.items()}
    for y, cov, obs, system in source():
        n += len(y)
        for s in np.unique(system):
            rows = system == s
            ll += natives[int(s)].log_prob(y[rows], cov[rows], observed=obs[rows]).sum()
    if n != expected_rows:
        raise ValueError('row accounting changed during final evaluation')
    return float(ll/n)


def log_covariance_prior(covs: np.ndarray, regularization: float, prior_strength: float) -> float:
    """Summed log prior ``-(nu/2)[log|V| + w tr(V^-1)]`` over components (unnormalised).

    ``covs`` in mag^2 with shape (K, d, d); ``regularization`` is ``w`` in mag^2.
    Uses Cholesky factors only; the additive constant is omitted.
    """
    chol = np.linalg.cholesky(covs)
    logdet = 2*np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
    eye = np.broadcast_to(np.eye(covs.shape[-1]), covs.shape)
    trace_inv = (np.linalg.solve(chol, eye)**2).sum(axis=(-2, -1))
    return float(-.5*prior_strength*(logdet + regularization*trace_inv).sum())
