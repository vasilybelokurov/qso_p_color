"""Extreme deconvolution: EM fit of an intrinsic Gaussian mixture to noisy data.

Implements the algorithm of Bovy, Hogg & Roweis (2011), arXiv:0905.2979, in
pure numpy.  We do not depend on the C ``extreme_deconvolution`` library or on
``astroML``: the algorithm is short, the dependency is awkward to build, and a
local implementation lets us handle per-object missing dimensions exactly, which
the reference implementation does only through projection matrices.

For observation :math:`i` with observed sub-vector :math:`\\mathbf{w}_i`, noise
covariance :math:`\\mathbf{S}_i` and component :math:`j`,

.. math::
    \\mathbf{T}_{ij} &= \\mathbf{V}_j^{OO} + \\mathbf{S}_i \\\\
    q_{ij} &\\propto \\alpha_j\\,\\mathcal{N}(\\mathbf{w}_i \\mid
        \\boldsymbol{\\mu}_j^{O}, \\mathbf{T}_{ij}) \\\\
    \\mathbf{b}_{ij} &= \\boldsymbol{\\mu}_j +
        \\mathbf{V}_j^{\\cdot O}\\mathbf{T}_{ij}^{-1}
        (\\mathbf{w}_i - \\boldsymbol{\\mu}_j^{O}) \\\\
    \\mathbf{B}_{ij} &= \\mathbf{V}_j -
        \\mathbf{V}_j^{\\cdot O}\\mathbf{T}_{ij}^{-1}\\mathbf{V}_j^{O\\cdot}

and the M step is the weighted moment update given in the paper.  With
:math:`\\mathbf{S}_i = 0` and no missing dimensions this reduces exactly to
ordinary Gaussian-mixture EM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .gaussmix import GaussianMixture, log_gauss_batch

__all__ = ["XDFitResult", "fit_xd", "select_n_components"]

log = logging.getLogger(__name__)


@dataclass
class XDFitResult:
    """Outcome of one XD fit."""

    mixture: GaussianMixture
    n_iter: int
    mean_loglike: float
    converged: bool
    history: list[float]


def _mask_groups(observed: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group rows by their observed-dimension pattern.

    Returns a list of ``(row_indices, observed_dim_indices)``.
    """
    packed = np.packbits(observed, axis=1)
    _, inverse = np.unique(packed, axis=0, return_inverse=True)
    groups = []
    for g in range(inverse.max() + 1):
        rows = np.flatnonzero(inverse == g)
        dims = np.flatnonzero(observed[rows[0]])
        groups.append((rows, dims))
    return groups


def _init_mixture(
    x: np.ndarray,
    observed: np.ndarray,
    n_components: int,
    rng: np.random.Generator,
) -> GaussianMixture:
    """Initialise from a random subset of complete-ish rows plus the global scatter."""
    d = x.shape[1]
    # Use dimension-wise means/variances computed only over observed entries, so
    # that missing data do not bias the starting point.
    filled = np.where(observed, x, np.nan)
    mu0 = np.nanmean(filled, axis=0)
    var0 = np.nanvar(filled, axis=0)
    var0 = np.where(np.isfinite(var0) & (var0 > 0), var0, 1.0)

    # Component centres: jittered draws around random data rows.
    idx = rng.choice(x.shape[0], size=n_components, replace=x.shape[0] < n_components)
    means = np.where(observed[idx], x[idx], mu0[None, :])
    means = means + rng.normal(scale=0.1 * np.sqrt(var0), size=means.shape)

    covs = np.repeat(np.diag(var0)[None, :, :], n_components, axis=0)
    weights = np.full(n_components, 1.0 / n_components)
    return GaussianMixture(weights, means, covs)


def fit_xd(
    x: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    n_components: int = 10,
    observed: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    regularization: float = 0.0,
    init: GaussianMixture | None = None,
    seed: int = 0,
    labels: tuple[str, ...] = (),
) -> XDFitResult:
    """Fit an intrinsic Gaussian mixture by extreme deconvolution.

    Parameters
    ----------
    x : ndarray, shape (n, d)
        Observed feature vectors.  Unobserved entries are ignored and may be NaN.
    cov : ndarray, shape (n, d, d) or (d, d), optional
        Per-object measurement covariance.  ``None`` fits an ordinary GMM.
    n_components : int
        Number of mixture components ``K``.  Choose it with
        :func:`select_n_components`, not by assertion.
    observed : ndarray of bool, shape (n, d), optional
        Usable dimensions per object.  ``None`` means all.
    weights : ndarray, shape (n,), optional
        Non-negative per-object sample weights, e.g. inverse selection
        probabilities.  They enter every sum in the M step.
    max_iter, tol : int, float
        EM stops when the weighted mean log-likelihood improves by less than
        ``tol`` between iterations.
    regularization : float
        Added to the diagonal of each component covariance in the M step.  This
        is a variance floor in the units of the features; it prevents a
        component from collapsing onto a single point when the measurement
        errors are small.  It is a *model choice* and is recorded in the result.
    init : GaussianMixture, optional
        Starting point.  ``None`` initialises from the data.
    seed : int
        Seed for the initialisation RNG.  Fits are reproducible given this.
    labels : tuple of str
        Feature names carried into the fitted mixture.

    Returns
    -------
    XDFitResult

    Notes
    -----
    The returned ``mean_loglike`` is the weighted mean of
    :math:`\\log p(\\mathbf{w}_i)` over the training set, which decreases in
    magnitude as dimensions go missing; it is comparable across fits only for a
    fixed data set.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n, d = x.shape

    if observed is None:
        obs = np.ones((n, d), dtype=bool)
    else:
        obs = np.broadcast_to(np.asarray(observed, dtype=bool), (n, d)).copy()
    # A NaN in an observed slot is a caller error; catch it rather than
    # propagating NaN into the fit.
    if np.isnan(x[obs]).any():
        raise ValueError("x contains NaN in a dimension flagged as observed")

    if cov is None:
        s = np.zeros((n, d, d))
    else:
        s = np.asarray(cov, dtype=float)
        s = np.broadcast_to(s.reshape(-1, d, d), (n, d, d))

    if weights is None:
        w_i = np.ones(n)
    else:
        w_i = np.asarray(weights, dtype=float)
        if w_i.shape != (n,) or (w_i < 0).any():
            raise ValueError("weights must be non-negative with shape (n,)")

    rng = np.random.default_rng(seed)
    mix = init if init is not None else _init_mixture(x, obs, n_components, rng)
    k = mix.n_components

    groups = _mask_groups(obs)
    history: list[float] = []
    converged = False
    prev = -np.inf
    it = 0

    for it in range(1, max_iter + 1):
        # Accumulators for the M step.
        acc_q = np.zeros(k)
        acc_mu = np.zeros((k, d))
        acc_dmu = np.zeros((k, d))
        acc_v = np.zeros((k, d, d))
        total_ll = 0.0

        for rows, dims in groups:
            if dims.size == 0:
                continue
            xs = x[np.ix_(rows, dims)]                                # (m, p)
            ss = s[np.ix_(rows, dims, dims)]                          # (m, p, p)
            ws = w_i[rows]                                            # (m,)
            mu_o = mix.means[:, dims]                                 # (K, p)
            v_do = mix.covs[:, :, dims]                               # (K, d, p)
            v_oo = v_do[:, dims, :]                                   # (K, p, p)

            t = v_oo[None, :, :, :] + ss[:, None, :, :]               # (m, K, p, p)
            lp = log_gauss_batch(xs[:, None, :], mu_o[None, :, :], t)  # (m, K)
            lq = lp + np.log(np.maximum(mix.weights, 1e-300))[None, :]
            ll = logsumexp(lq, axis=1)                                # (m,)
            total_ll += float((ws * ll).sum())
            q = np.exp(lq - ll[:, None])                              # (m, K)

            resid = xs[:, None, :] - mu_o[None, :, :]                 # (m, K, p)
            # gain = V^{.O} T^{-1}  -> (m, K, d, p); solve instead of inverting.
            rhs = np.broadcast_to(v_do[None], (rows.size, k, d, dims.size))
            gain = np.linalg.solve(
                np.swapaxes(t, -1, -2), np.swapaxes(rhs, -1, -2)
            )
            gain = np.swapaxes(gain, -1, -2)                          # (m, K, d, p)

            b = mix.means[None, :, :] + np.einsum("mkdp,mkp->mkd", gain, resid)
            # B = V - V^{.O} T^-1 V^{O.}; v_do[k, e, p] is already V^{O.}[p, e].
            bmat = mix.covs[None, :, :, :] - np.einsum("mkdp,kep->mkde", gain, v_do)

            qw = q * ws[:, None]                                      # (m, K)
            acc_q += qw.sum(axis=0)
            acc_mu += np.einsum("mk,mkd->kd", qw, b)
            acc_v += np.einsum("mk,mkde->kde", qw, bmat)
            # Second moment about the *current* mean rather than about zero.
            # Accumulating E[bb^T] and subtracting mu mu^T later loses all
            # precision when |mu| greatly exceeds the spread: for data at
            # 1e8 +/- 1 it returned a variance of 0 instead of 1.
            db = b - mix.means[None, :, :]
            acc_v += np.einsum("mk,mkd,mke->kde", qw, db, db)
            acc_dmu += np.einsum("mk,mkd->kd", qw, db)

        if acc_q.sum() <= 0:
            raise RuntimeError("EM collapsed: total responsibility is zero")

        new_weights = acc_q / acc_q.sum()
        alive = acc_q > 1e-10
        if not alive.all():
            log.warning("dropping %d empty component(s)", int((~alive).sum()))
        new_means = np.where(
            alive[:, None],
            mix.means + acc_dmu / np.maximum(acc_q, 1e-300)[:, None],
            mix.means,
        )
        new_covs = np.empty_like(acc_v)
        for j in range(k):
            if not alive[j]:
                new_covs[j] = mix.covs[j]
                continue
            # acc_v holds sum q (B + (b - mu_old)(b - mu_old)^T); shifting the
            # centre from mu_old to mu_new is an exact rank-one correction.
            shift = acc_dmu[j] / acc_q[j]
            c = acc_v[j] / acc_q[j] - np.outer(shift, shift)
            c = 0.5 * (c + c.T)
            if regularization:
                c = c + regularization * np.eye(d)
            new_covs[j] = c

        mix = GaussianMixture(new_weights, new_means, new_covs, labels=labels)

        mean_ll = total_ll / w_i.sum()
        history.append(mean_ll)
        if it > 1 and abs(mean_ll - prev) < tol * max(1.0, abs(prev)):
            converged = True
            prev = mean_ll
            break
        prev = mean_ll

    final = GaussianMixture(mix.weights, mix.means, mix.covs, labels=labels)
    # The likelihood accumulated inside the loop belongs to the mixture *before*
    # that iteration's M step, so reporting it would describe a model we are not
    # returning.  Recompute it for the model we actually hand back.
    final_ll = float(
        (w_i * final.log_prob(x, cov, observed=obs)).sum() / w_i.sum()
    )
    return XDFitResult(
        mixture=final,
        n_iter=it,
        mean_loglike=final_ll,
        converged=converged,
        history=history,
    )


def select_n_components(
    x: np.ndarray,
    cov: np.ndarray | None,
    candidates: list[int],
    *,
    observed: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    n_folds: int = 4,
    seed: int = 0,
    **fit_kwargs,
) -> tuple[int, dict[int, float]]:
    """Choose ``K`` by held-out predictive log density.

    Folds are formed by splitting on ``groups`` (e.g. a coarse HEALPix index),
    so that spatially adjacent objects never straddle the train/validation
    boundary.  With ``groups=None`` the split is a plain random partition, which
    is appropriate only when the rows are genuinely exchangeable.

    Returns
    -------
    best_k : int
        The ``K`` with the highest mean held-out log density.
    scores : dict
        ``K -> mean held-out log density per object``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[0]
    rng = np.random.default_rng(seed)

    if groups is None:
        fold_of = rng.integers(0, n_folds, size=n)
    else:
        g = np.asarray(groups)
        uniq = np.unique(g)
        assign = rng.permutation(uniq.size) % n_folds
        fold_of = assign[np.searchsorted(uniq, g)]

    scores: dict[int, float] = {}
    for k in candidates:
        total, count = 0.0, 0.0
        for f in range(n_folds):
            tr, va = fold_of != f, fold_of == f
            if tr.sum() < k * 5 or va.sum() == 0:
                continue
            res = fit_xd(
                x[tr],
                None if cov is None else np.asarray(cov)[tr],
                n_components=k,
                observed=None if observed is None else observed[tr],
                weights=None if weights is None else weights[tr],
                seed=seed + f,
                **fit_kwargs,
            )
            lp = res.mixture.log_prob(
                x[va],
                None if cov is None else np.asarray(cov)[va],
                observed=None if observed is None else observed[va],
            )
            w_va = np.ones(int(va.sum())) if weights is None else weights[va]
            total += float((w_va * lp).sum())
            count += float(w_va.sum())
        scores[k] = total / count if count else -np.inf
        log.info("K=%d held-out mean log density %.4f", k, scores[k])

    best_k = max(scores, key=scores.get)
    return best_k, scores
