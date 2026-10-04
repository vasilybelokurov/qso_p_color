"""Calibrated 'outside the model' test for the nearest-component distance.

The scorer's tail diagnostic is the nearest-component Mahalanobis distance of an object's non-reference
bands given its reference band (``_conditional_min_mahalanobis``), with the object's measurement noise
added. Its distribution depends on how many and which bands are observed, on the noise and on the mixture
geometry, so a fixed cut in sigma means different things for different objects (in 20 bands a typical
member of the population lies beyond 4 sigma).

Here the same statistic is calibrated per object by a parametric bootstrap: draw ``draws`` objects from the
class model conditioned on the object's own reference value, in its own observed bands and with its own
noise covariance, compute the same statistic for each, and return

    p = (1 + #{T_fake >= T_obs}) / (draws + 1),

the probability that a member of the class is at least as far from every component. Small p means the
object is outside that class model. Draws are seeded from the object's inputs, so results do not depend on
batch order.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import logsumexp


def _components(mixtures, mixture_weights=None):
    """Concatenate (log weight, mean, cov) over mixtures; ``mixture_weights`` weight each mixture."""
    mw = np.full(len(mixtures), 1/len(mixtures)) if mixture_weights is None else np.asarray(mixture_weights, float)
    lw, mu, cv = [], [], []
    for m, w in zip(mixtures, mw):
        keep = m.weights > 0
        lw.append(np.log(w) + np.log(m.weights[keep])); mu.append(m.means[keep]); cv.append(m.covs[keep])
    return np.concatenate(lw), np.concatenate(mu), np.concatenate(cv)


def tail_pvalue(components, x, s, observed, anchor, *, draws: int, seed: int) -> tuple[float, float]:
    """(p, T_obs) for one object. ``components`` from :func:`_components`; x (d,), s (d, d), observed (d,).

    T is the squared nearest-component conditional Mahalanobis distance (as in the scorer, which reports its
    square root). Returns (nan, nan) when the reference is unobserved or no other band is observed.
    """
    lw, mus, vs = components
    obs = np.asarray(observed, bool)
    dims = np.flatnonzero(obs & (np.arange(len(x)) != anchor))
    if not obs[anchor] or not len(dims):
        return np.nan, np.nan
    j = np.concatenate([[anchor], dims])
    t = vs[:, j][:, :, j] + s[np.ix_(j, j)][None]                       # (K, J, J), noise added
    va = t[:, 0, 0]; cross = t[:, 1:, 0]
    da = x[anchor] - mus[:, anchor]
    mean = mus[:, dims] + cross/va[:, None]*da[:, None]                # (K, J-1)
    cond = t[:, 1:, 1:] - cross[:, :, None]*cross[:, None, :]/va[:, None, None]
    cond = .5*(cond + cond.swapaxes(1, 2))
    chol = np.linalg.cholesky(cond)
    inv = np.linalg.inv(chol)                                          # (K, J-1, J-1), L^-1

    def stat(y):                                                       # y (B, J-1) -> (B,)
        delta = y[None] - mean[:, None, :]                             # (K, B, J-1)
        z = np.matmul(inv, delta.swapaxes(1, 2))                       # (K, J-1, B)
        return np.einsum('kib,kib->kb', z, z).min(axis=0)

    t_obs = float(stat(x[dims][None])[0])
    # class members with this reference value: components re-weighted by the reference likelihood
    lwa = lw - .5*np.log(2*np.pi*va) - .5*da**2/va
    p = np.exp(lwa - logsumexp(lwa))
    rng = np.random.default_rng(seed)
    k = rng.choice(len(p), size=draws, p=p)
    eps = rng.standard_normal((draws, len(dims)))
    y = mean[k] + np.einsum('bij,bj->bi', chol[k], eps)
    t_fake = stat(y)
    return float((1 + np.count_nonzero(t_fake >= t_obs))/(draws + 1)), t_obs


def row_seed(seed: int, x, s, observed) -> int:
    """Deterministic per-object seed from its inputs."""
    payload = np.asarray(x, float).tobytes() + np.asarray(s, float).tobytes() + np.asarray(observed, bool).tobytes()
    return int(np.random.SeedSequence([seed, int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')]).generate_state(1)[0])
