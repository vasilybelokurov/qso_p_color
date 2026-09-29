"""Full-data XD passes with memory bounded by an input batch.

The mixture is held fixed throughout each pass; sufficient statistics from
every row are summed before the M step. This is ordinary batch EM, not online
EM or a subsampled objective. Densities are over observed feature coordinates.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
import numpy as np

from .gaussmix import GaussianMixture
from .xd import XDFitResult, _estep_chunk, _mask_groups

Batch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
BatchSource = Callable[[], Iterable[Batch]]


def accumulate_batches(source: BatchSource, mix: GaussianMixture) -> tuple:
    """Sum one complete E step with fixed parameters; densities use observed coordinates.

    Returns component counts, centred first/second moments, weighted log
    likelihood, total weight and row count. No parameters are updated here.
    """
    k, d = mix.means.shape
    aq, adm, av = np.zeros(k), np.zeros((k, d)), np.zeros((k, d, d))
    total_ll = weight_sum = 0.
    seen = 0
    for x, cov, obs, weights in checked_batches(source, d):
        seen += len(x)
        weight_sum += weights.sum()
        for rows, dims in _mask_groups(obs):
            pq, _, pdm, pv, pll = _estep_chunk(rows, dims, mix, x, cov, weights)
            aq += pq; adm += pdm; av += pv; total_ll += pll
    return aq, adm, av, total_ll, weight_sum, seen


def checked_batches(source: BatchSource, dimensions: int) -> Iterable[Batch]:
    """Validate features, full noise covariances and dimensionless row weights."""
    for x, cov, observed, weights in source():
        x, cov = np.asarray(x, float), np.asarray(cov, float)
        observed, weights = np.asarray(observed, bool), np.asarray(weights, float)
        n = len(x)
        if (x.shape != (n, dimensions) or observed.shape != x.shape or
                cov.shape != (n, dimensions, dimensions) or weights.shape != (n,)):
            raise ValueError('invalid streamed batch shapes')
        if (not np.isfinite(x[observed]).all() or not observed.any(axis=1).all() or
                not np.isfinite(cov).all() or not np.isfinite(weights).all() or
                (weights < 0).any()):
            raise ValueError('invalid streamed measurements or weights')
        if not np.allclose(cov, cov.swapaxes(1, 2)):
            raise ValueError('noise covariance must be symmetric')
        if n:
            yield x, cov, observed, weights


def initialise_batches(source: BatchSource, n_components: int, *,
                       labels: tuple[str, ...], seed: int) -> GaussianMixture:
    """Initialise from moments of ALL fitting rows, excluding other roles.

    Means and variances are accumulated stably per observed band. A random
    initial centre is not a training sample cap; every pass still uses all rows.
    """
    d = len(labels)
    count, mean, m2 = np.zeros(d), np.zeros(d), np.zeros(d)
    for x, _, obs, w in checked_batches(source, d):
        for j in range(d):
            use = obs[:, j] & (w > 0)
            n = w[use].sum()
            if not n:
                continue
            mu = np.average(x[use, j], weights=w[use])
            scatter = np.sum(w[use] * (x[use, j] - mu)**2)
            total = count[j] + n
            delta = mu - mean[j]
            m2[j] += scatter + delta**2 * count[j] * n / total
            mean[j] += delta * n / total
            count[j] = total
    if (count == 0).any() or n_components < 1:
        raise ValueError('initialisation needs fitting observations in every model band')
    variance = m2 / count
    variance = np.where(variance > 0, variance, 1.)
    rng = np.random.default_rng(seed)
    means = mean + rng.normal(size=(n_components, d)) * np.sqrt(variance)
    return GaussianMixture(np.full(n_components, 1 / n_components), means,
                           np.tile(np.diag(variance), (n_components, 1, 1)), labels)


def fit_xd_batches(source: BatchSource, *, init: GaussianMixture,
                   expected_rows: int, max_iter: int, tol: float,
                   regularization: float, progress: Callable | None = None,
                   initial_history: tuple[float, ...] = (),
                   accumulator: Callable | None = None) -> XDFitResult:
    """Fit all streamed rows; features and covariance retain their native units.

    ``source`` must replay the same selected rows on every call. Memory is
    O(batch_size * K * D**2), independent of total row count. ``expected_rows``
    is checked on EVERY pass, including the final returned-model likelihood.
    The callback receives iteration, updated mixture, pre-update likelihood,
    and row count; it may save a restart checkpoint without changing the fit.
    An optional accumulator may distribute the E step, but must account for
    every row before returning sufficient statistics for the single M step.
    """
    if (expected_rows < 1 or max_iter < 1 or not np.isfinite(tol) or tol < 0 or
            not np.isfinite(regularization) or regularization < 0):
        raise ValueError('invalid streamed fit settings')
    mix = init
    k, d = mix.means.shape
    history = list(initial_history)
    if len(history) > max_iter or not np.isfinite(history).all():
        raise ValueError('invalid restart history')
    converged = len(history) > 1 and abs(history[-1]-history[-2]) < tol*max(1., abs(history[-2]))
    reference_weight = None
    accumulate = accumulate_batches if accumulator is None else accumulator
    it = len(history)
    for it in range(len(history)+1, (len(history) if converged else max_iter)+1):
        aq, adm, av, total_ll, weight_sum, seen = accumulate(source, mix)
        if seen != expected_rows or weight_sum <= 0:
            raise ValueError(f'stream row accounting: expected {expected_rows}, saw {seen}')
        if reference_weight is not None and not np.isclose(weight_sum, reference_weight):
            raise ValueError('stream weights changed between iterations')
        reference_weight = weight_sum
        alive = aq > 1e-10
        means = np.where(alive[:, None], mix.means + adm / np.maximum(aq, 1e-300)[:, None], mix.means)
        covs = mix.covs.copy()
        for j in np.flatnonzero(alive):
            shift = adm[j] / aq[j]
            c = av[j] / aq[j] - np.outer(shift, shift)
            covs[j] = .5 * (c + c.T) + regularization * np.eye(d)
        if aq.sum() <= 0:
            raise RuntimeError('EM collapsed')
        mix = GaussianMixture(aq / aq.sum(), means, covs, labels=init.labels)
        mean_ll = total_ll / weight_sum
        history.append(mean_ll)
        if progress is not None:
            progress(it, mix, mean_ll, seen)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol * max(1., abs(history[-2])):
            converged = True
            break
    total_ll = weight_sum = 0.
    seen = 0
    for x, cov, obs, weights in checked_batches(source, d):
        seen += len(x); weight_sum += weights.sum()
        total_ll += float(weights @ mix.log_prob(x, cov, observed=obs))
    if seen != expected_rows or weight_sum <= 0 or (reference_weight is not None and not np.isclose(weight_sum, reference_weight)):
        raise ValueError('stream changed during final likelihood pass')
    return XDFitResult(mix, it, total_ll / weight_sum, converged, history)
