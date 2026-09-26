"""Batched Gaussian-mixture evaluation with per-object noise and missing dimensions.

All densities are returned as natural logs. Nothing in this module ever forms an
explicit matrix inverse: every quadratic form and log-determinant comes from a
Cholesky factor.

The central object is a mixture

.. math::
    p(\\mathbf{v}) = \\sum_k \\alpha_k \\,\\mathcal{N}(\\mathbf{v}\\mid
    \\boldsymbol{\\mu}_k, \\mathbf{V}_k)

interpreted as the *intrinsic* (noise-deconvolved) distribution.  An observation
carries its own covariance :math:`\\mathbf{S}_i`, so the density of the observed
value is evaluated with :math:`\\mathbf{V}_k + \\mathbf{S}_i`.  A missing
dimension is handled by exact marginalisation, i.e. by dropping the
corresponding rows and columns of both matrices.

Units: this module is unit-agnostic.  Densities are per unit volume of whatever
feature space the caller supplies, so a density over ``d`` observed dimensions
has units of (feature unit)^-d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import logsumexp

__all__ = [
    "GaussianMixture",
    "assert_positive_definite",
    "log_gauss_batch",
    "condition_joint",
]

_LOG2PI = float(np.log(2.0 * np.pi))


def log_gauss_batch(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    jitter: float = 0.0,
) -> np.ndarray:
    """Log density of a multivariate normal, broadcast over leading axes.

    Parameters
    ----------
    x : ndarray, shape (..., d)
        Points at which to evaluate.
    mean : ndarray, shape (..., d)
        Means, broadcastable against ``x``.
    cov : ndarray, shape (..., d, d)
        Covariances, broadcastable against ``x[..., None]``.
    jitter : float, optional
        Added to the diagonal before factorisation.  Use only to rescue a
        marginally non-positive-definite matrix; it changes the density.

    Returns
    -------
    ndarray, shape (...)
        Natural log of the density.

    Notes
    -----
    Uses a Cholesky factorisation, so ``cov`` must be positive definite.  A
    non-PD input raises ``numpy.linalg.LinAlgError`` rather than silently
    returning garbage.
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    d = x.shape[-1]
    if d == 0:
        # No observed dimensions: the density of an empty vector is 1.
        return np.zeros(np.broadcast_shapes(x.shape[:-1], mean.shape[:-1]))

    delta = x - mean
    shape = np.broadcast_shapes(delta.shape[:-1], cov.shape[:-2])
    delta = np.broadcast_to(delta, shape + (d,))
    cov = np.broadcast_to(cov, shape + (d, d))

    flat_delta = delta.reshape(-1, d)
    flat_cov = cov.reshape(-1, d, d)
    if jitter:
        flat_cov = flat_cov + jitter * np.eye(d)

    # np.linalg.cholesky is batched; scipy's is not.
    chol = np.linalg.cholesky(flat_cov)  # lower triangular, (n, d, d)
    # Solve L y = delta for y, then maha = |y|^2.
    y = np.linalg.solve(chol, flat_delta[..., None])[..., 0]
    maha = np.einsum("ni,ni->n", y, y)
    logdet = 2.0 * np.log(np.einsum("nii->ni", chol)).sum(axis=-1)

    out = -0.5 * (d * _LOG2PI + logdet + maha)
    return out.reshape(shape)


@dataclass(frozen=True)
class GaussianMixture:
    """An intrinsic (noise-deconvolved) Gaussian mixture in ``d`` dimensions.

    Attributes
    ----------
    weights : ndarray, shape (K,)
        Mixture weights, summing to one.
    means : ndarray, shape (K, d)
    covs : ndarray, shape (K, d, d)
    labels : tuple of str, optional
        Names of the feature dimensions.  Used to check that a caller's feature
        vector is in the same order as the fitted model, and to resolve
        ``observed_mask`` by name.
    """

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        mu = np.asarray(self.means, dtype=float)
        v = np.asarray(self.covs, dtype=float)
        if mu.ndim != 2 or v.ndim != 3:
            raise ValueError("means must be (K, d) and covs must be (K, d, d)")
        if w.shape != (mu.shape[0],) or v.shape[:2] != mu.shape:
            raise ValueError("inconsistent mixture shapes")
        if v.shape[1] != v.shape[2]:
            raise ValueError("covariances must be square")
        # numpy's Cholesky reads only the lower triangle, so an asymmetric
        # matrix would be silently accepted and quietly reinterpreted.
        if not np.allclose(v, np.swapaxes(v, -1, -2), atol=1e-10, rtol=1e-8):
            raise ValueError("covariances must be symmetric")
        if not np.isclose(w.sum(), 1.0, atol=1e-8):
            raise ValueError(f"weights must sum to 1, got {w.sum()!r}")
        if self.labels and len(self.labels) != mu.shape[1]:
            raise ValueError("labels length does not match dimensionality")
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "means", mu)
        object.__setattr__(self, "covs", v)
        object.__setattr__(self, "labels", tuple(self.labels))

    # -- basic properties -------------------------------------------------

    @property
    def n_components(self) -> int:
        return self.means.shape[0]

    @property
    def n_dim(self) -> int:
        return self.means.shape[1]

    # -- evaluation -------------------------------------------------------

    def log_prob(
        self,
        x: np.ndarray,
        cov: np.ndarray | None = None,
        *,
        observed: np.ndarray | None = None,
    ) -> np.ndarray:
        """log p(x_obs) with per-object noise and exact missing-band marginalisation.

        Parameters
        ----------
        x : ndarray, shape (n, d) or (d,)
            Observed feature vectors.  Entries flagged unobserved are ignored
            and may hold any value, including NaN.
        cov : ndarray, shape (n, d, d) or (d, d), optional
            Per-object measurement covariance :math:`\\mathbf{S}_i`.  ``None``
            means noiseless, i.e. evaluate the intrinsic mixture itself.
        observed : ndarray of bool, shape (n, d) or (d,), optional
            Which dimensions are usable.  ``None`` means all.

        Returns
        -------
        ndarray, shape (n,)
            Log density of the observed sub-vector.  Because the number of
            observed dimensions can differ between objects, these values are
            **not** mutually comparable unless the same dimensions are observed.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n, d = x.shape
        if d != self.n_dim:
            raise ValueError(f"expected {self.n_dim} features, got {d}")

        if cov is None:
            s = np.zeros((n, d, d))
        else:
            s = np.asarray(cov, dtype=float)
            s = np.broadcast_to(s.reshape(-1, d, d), (n, d, d))

        if observed is None:
            obs = np.ones((n, d), dtype=bool)
        else:
            obs = np.broadcast_to(np.asarray(observed, dtype=bool), (n, d))

        out = np.empty(n)
        # Group objects by their observed-dimension pattern so that each group
        # is one vectorised call.  In practice there are only a handful of
        # distinct patterns (all bands, one band missing, ...).
        packed = np.packbits(obs, axis=1)
        _, inverse = np.unique(packed, axis=0, return_inverse=True)
        for g in range(inverse.max() + 1 if n else 0):
            sel = inverse == g
            idx = np.flatnonzero(obs[np.flatnonzero(sel)[0]])
            if idx.size == 0:
                out[sel] = 0.0  # nothing observed: density of an empty vector
                continue
            xs = x[np.ix_(sel, idx)]                       # (m, dg)
            ss = s[np.ix_(sel, idx, idx)]                  # (m, dg, dg)
            mus = self.means[:, idx]                       # (K, dg)
            vs = self.covs[np.ix_(np.arange(self.n_components), idx, idx)]

            # (m, K, dg) and (m, K, dg, dg)
            lp = log_gauss_batch(
                xs[:, None, :],
                mus[None, :, :],
                vs[None, :, :, :] + ss[:, None, :, :],
            )
            out[sel] = logsumexp(lp + np.log(self.weights)[None, :], axis=1)
        return out

    def min_mahalanobis(
        self,
        x: np.ndarray,
        cov: np.ndarray | None = None,
        *,
        observed: np.ndarray | None = None,
        chunk: int = 4096,
    ) -> np.ndarray:
        """Distance to the nearest component, in sigma, shape (n,).

        :math:`\\min_k \\sqrt{(\\mathbf{x}-\\boldsymbol{\\mu}_k)^T
        (\\mathbf{V}_k+\\mathbf{S}_i)^{-1}(\\mathbf{x}-\\boldsymbol{\\mu}_k)}`
        in each object's observed subspace; NaN where nothing is observed.
        The number of observed dimensions differs between objects, so the same
        distance is a different tail probability in 3 and in 4 dimensions.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n, d = x.shape
        s = (
            np.zeros((n, d, d))
            if cov is None
            else np.broadcast_to(np.asarray(cov, float).reshape(-1, d, d), (n, d, d))
        )
        obs = (
            np.ones((n, d), dtype=bool)
            if observed is None
            else np.broadcast_to(np.asarray(observed, bool), (n, d))
        )
        out = np.full(n, np.nan)
        packed = np.packbits(obs, axis=1)
        _, inverse = np.unique(packed, axis=0, return_inverse=True)
        for g in range(inverse.max() + 1 if n else 0):
            rows = np.flatnonzero(inverse == g)
            idx = np.flatnonzero(obs[rows[0]])
            if idx.size == 0:
                continue
            mus = self.means[:, idx]
            vs = self.covs[np.ix_(np.arange(self.n_components), idx, idx)]
            for lo in range(0, rows.size, chunk):
                r = rows[lo:lo + chunk]
                delta = x[np.ix_(r, idx)][:, None, :] - mus[None]          # (m, K, dg)
                chol = np.linalg.cholesky(vs[None] + s[np.ix_(r, idx, idx)][:, None])
                y = np.linalg.solve(chol, delta[..., None])[..., 0]
                out[r] = np.sqrt(np.einsum("mki,mki->mk", y, y).min(axis=1))
        return out

    def responsibilities(
        self,
        x: np.ndarray,
        cov: np.ndarray | None = None,
        *,
        observed: np.ndarray | None = None,
    ) -> np.ndarray:
        """Posterior component probabilities, shape (n, K), rows summing to one."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n, d = x.shape
        s = (
            np.zeros((n, d, d))
            if cov is None
            else np.broadcast_to(np.asarray(cov, float).reshape(-1, d, d), (n, d, d))
        )
        obs = (
            np.ones((n, d), dtype=bool)
            if observed is None
            else np.broadcast_to(np.asarray(observed, bool), (n, d))
        )
        log_r = np.full((n, self.n_components), -np.inf)
        packed = np.packbits(obs, axis=1)
        _, inverse = np.unique(packed, axis=0, return_inverse=True)
        for g in range(inverse.max() + 1 if n else 0):
            sel = inverse == g
            idx = np.flatnonzero(obs[np.flatnonzero(sel)[0]])
            if idx.size == 0:
                log_r[sel] = np.log(self.weights)[None, :]
                continue
            lp = log_gauss_batch(
                x[np.ix_(sel, idx)][:, None, :],
                self.means[:, idx][None, :, :],
                self.covs[np.ix_(np.arange(self.n_components), idx, idx)][None]
                + s[np.ix_(sel, idx, idx)][:, None, :, :],
            )
            log_r[sel] = lp + np.log(self.weights)[None, :]
        log_r -= logsumexp(log_r, axis=1, keepdims=True)
        return np.exp(log_r)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` samples from the intrinsic mixture."""
        k = rng.choice(self.n_components, size=n, p=self.weights)
        out = np.empty((n, self.n_dim))
        for j in range(self.n_components):
            sel = k == j
            m = int(sel.sum())
            if m:
                out[sel] = rng.multivariate_normal(self.means[j], self.covs[j], size=m)
        return out

    # -- structural operations -------------------------------------------

    def marginal(self, idx: np.ndarray) -> "GaussianMixture":
        """Exact marginal mixture over the dimensions in ``idx``."""
        idx = np.asarray(idx, dtype=int)
        return GaussianMixture(
            self.weights,
            self.means[:, idx],
            self.covs[np.ix_(np.arange(self.n_components), idx, idx)],
            labels=tuple(self.labels[i] for i in idx) if self.labels else (),
        )

    def mixture_with(self, other: "GaussianMixture", weight: float) -> "GaussianMixture":
        """Convex combination ``weight * self + (1 - weight) * other``.

        Used both for redshift-bin interpolation and for hierarchical shrinkage
        of a sparse sky cell towards its parent.  The result is a valid density
        because both inputs are normalised.
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must lie in [0, 1]")
        if self.n_dim != other.n_dim:
            raise ValueError("cannot mix models of different dimensionality")
        return GaussianMixture(
            np.concatenate([weight * self.weights, (1.0 - weight) * other.weights]),
            np.concatenate([self.means, other.means]),
            np.concatenate([self.covs, other.covs]),
            labels=self.labels or other.labels,
        )

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "covs": self.covs.tolist(),
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianMixture":
        covs = np.asarray(d["covs"], float)
        # A serialised model is data from disk: check it here rather than
        # discovering a corrupt covariance as a NaN likelihood much later.
        assert_positive_definite(covs, "deserialised covariance")
        return cls(
            np.asarray(d["weights"], float),
            np.asarray(d["means"], float),
            covs,
            labels=tuple(d.get("labels", ())),
        )


def condition_joint(
    mix: GaussianMixture,
    y: np.ndarray,
    y_idx: np.ndarray,
    *,
    y_var: np.ndarray | None = None,
) -> tuple[np.ndarray, GaussianMixture]:
    """Condition a joint mixture on exact or noisy values of some dimensions.

    Given a mixture over :math:`(\\mathbf{a}, \\mathbf{b})` and an observed
    :math:`\\mathbf{b} = \\mathbf{y}` with covariance :math:`\\mathbf{S}_y`,
    return the mixture over :math:`\\mathbf{a}\\mid\\mathbf{y}`.

    Component weights become

    .. math::
        w_k \\propto \\alpha_k\\,\\mathcal{N}(\\mathbf{y}\\mid
        \\boldsymbol{\\mu}_{b,k}, \\mathbf{V}_{bb,k} + \\mathbf{S}_y),

    which is where the redshift uncertainty of the primary enters: it widens the
    matching kernel rather than being ignored.

    Parameters
    ----------
    mix : GaussianMixture
        Joint mixture.
    y : ndarray, shape (p,)
        Observed values of the conditioning dimensions.
    y_idx : ndarray of int, shape (p,)
        Indices of the conditioning dimensions within the joint vector.
    y_var : ndarray, shape (p, p), optional
        Covariance of ``y``.  ``None`` means an exact value.

    Returns
    -------
    weights : ndarray, shape (K,)
        Conditional component weights.
    mixture : GaussianMixture
        Mixture over the remaining dimensions, carrying those weights.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    y_idx = np.atleast_1d(np.asarray(y_idx, dtype=int))
    a_idx = np.setdiff1d(np.arange(mix.n_dim), y_idx)
    sy = np.zeros((y.size, y.size)) if y_var is None else np.asarray(y_var, float)

    k = mix.n_components
    mu_a = mix.means[:, a_idx]
    mu_b = mix.means[:, y_idx]
    v_aa = mix.covs[np.ix_(np.arange(k), a_idx, a_idx)]
    v_ab = mix.covs[np.ix_(np.arange(k), a_idx, y_idx)]
    v_bb = mix.covs[np.ix_(np.arange(k), y_idx, y_idx)] + sy

    log_w = np.log(mix.weights) + log_gauss_batch(y[None, :], mu_b, v_bb)
    log_w -= logsumexp(log_w)
    w = np.exp(log_w)

    means = np.empty_like(mu_a)
    covs = np.empty_like(v_aa)
    for j in range(k):
        c = cho_factor(v_bb[j], lower=True)
        gain = cho_solve(c, v_ab[j].T).T            # V_ab V_bb^-1
        means[j] = mu_a[j] + gain @ (y - mu_b[j])
        covs[j] = v_aa[j] - gain @ v_ab[j].T
        covs[j] = 0.5 * (covs[j] + covs[j].T)       # enforce symmetry
    labels = tuple(mix.labels[i] for i in a_idx) if mix.labels else ()
    return w, GaussianMixture(w, means, covs, labels=labels)


def assert_positive_definite(covs: np.ndarray, name: str = "covariance") -> None:
    """Raise if any matrix in a stack is not positive definite.

    Called after deserialising a model, so that a corrupted file fails at load
    time rather than producing silently wrong likelihoods.
    """
    covs = np.asarray(covs, dtype=float)
    for j, c in enumerate(covs.reshape(-1, covs.shape[-2], covs.shape[-1])):
        try:
            cholesky(c, lower=True)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise ValueError(f"{name}[{j}] is not positive definite") from exc
