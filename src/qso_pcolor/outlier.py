"""A broad "unmodelled" component of the field population.

Why it exists
-------------
Both colour models are Gaussian mixtures.  Far from every component of both,
their log ratio is a difference of quadratic forms,

.. math::
    \\ln\\frac{p_Q}{p_B} \\approx -\\tfrac12\\,\\Delta\\mathbf{c}^T\\bigl[
    (\\mathbf{V}_Q+\\mathbf{S})^{-1}-(\\mathbf{V}_B+\\mathbf{S})^{-1}\\bigr]
    \\Delta\\mathbf{c},

which grows without bound in whichever direction the quasar components happen
to be wider.  An object that neither model describes -- an artefact, a moving
object, a population missing from the training sets -- is then called a quasar
with probability one because of how the tails were fitted, not because of its
colours.

The remedy is a fourth, explicitly broad hypothesis :math:`U`.  The field is
written as

.. math::
    p(\\mathbf{c}\\mid\\text{field}, m) = [1-\\eta(m)]\\,p(\\mathbf{c}\\mid B, m)
        + \\eta(m)\\,p(\\mathbf{c}\\mid U),

with :math:`p(\\mathbf{c}\\mid U)=\\mathcal{N}(\\bar{\\mathbf{c}},\\,
\\kappa^2\\mathbf{E})` in the same feature coordinates.  If
:math:`\\kappa^2\\mathbf{E}-\\mathbf{V}_k` is positive definite for
every component :math:`k` of both models, :math:`U` dominates every other
density far enough out in every direction -- with or without measurement noise,
and in every observed subspace, since a principal submatrix of a positive
definite matrix is positive definite.  The object then goes to "unmodelled",
not to "quasar".

What sets the numbers
---------------------
:math:`\\bar{\\mathbf{c}}` and :math:`\\boldsymbol{\\Sigma}` are the moments of
a reference mixture (the background model).  :math:`\\mathbf{E}` is an
*envelope*: a covariance at least :math:`\\boldsymbol{\\Sigma}` and at least
every component, built by :func:`envelope_covariance`.  Scaling
:math:`\\boldsymbol{\\Sigma}` alone would not do: in relative-flux coordinates a
few quasar components are ~70 times wider than the field in ``w1/r`` and
``w2/r``, so an isotropically scaled field covariance would be 70 times too
broad in ``g/r`` as well.  :math:`\\kappa` and :math:`\\eta(m)` are fitted by
held-out field likelihood (``scripts/fit_outlier_model.py``).  None of them is a
constant in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit

from .gaussmix import GaussianMixture

__all__ = [
    "OutlierModel",
    "mixture_moments",
    "min_dominant_kappa",
    "envelope_covariance",
    "fit_outlier_fraction",
]


def mixture_moments(mixtures: list[GaussianMixture],
                    weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of a (weighted) pool of intrinsic mixtures.

    Parameters
    ----------
    mixtures : list of GaussianMixture
        All in the same feature coordinates.
    weights : ndarray, optional
        Relative weight of each mixture in the pool; equal by default.

    Returns
    -------
    mean : ndarray, shape (d,)
    cov : ndarray, shape (d, d)
        The exact second central moment of the pooled mixture.
    """
    w = np.ones(len(mixtures)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    alpha = np.concatenate([wi * m.weights for wi, m in zip(w, mixtures)])
    mu = np.concatenate([m.means for m in mixtures])
    v = np.concatenate([m.covs for m in mixtures])
    mean = alpha @ mu
    dev = mu - mean
    cov = np.einsum("k,kij->ij", alpha, v) + np.einsum("k,ki,kj->ij", alpha, dev, dev)
    return mean, 0.5 * (cov + cov.T)


def min_dominant_kappa(base_cov: np.ndarray, mixtures: list[GaussianMixture],
                       extra_covs: np.ndarray | None = None) -> float:
    """Smallest kappa for which kappa^2 * base_cov exceeds every component.

    Returns :math:`\\sqrt{\\max_k \\lambda_{\\max}(\\mathbf{V}_k,
    \\boldsymbol{\\Sigma})}`, the largest generalised eigenvalue of any
    component covariance against ``base_cov`` (solved with a Cholesky factor of
    ``base_cov``, never an inverse).  Any kappa strictly above it gives a
    density that eventually dominates every component in every direction.
    """
    worst = 0.0
    covs = [v for m in mixtures for v in m.covs]
    if extra_covs is not None:
        covs += list(extra_covs)
    for vk in covs:
        worst = max(worst, float(eigh(vk, base_cov, eigvals_only=True)[-1]))
    return float(np.sqrt(worst))


def envelope_covariance(base_cov: np.ndarray, mixtures: list[GaussianMixture],
                        extra_covs: np.ndarray | None = None) -> np.ndarray:
    """A covariance >= ``base_cov`` and >= every component, in the Loewner order.

    Construction, all with Cholesky factors:

    1. Whiten every component by ``base_cov = L L^T``: :math:`A_k = L^{-1}
       \\mathbf{V}_k L^{-T}`.
    2. Rotate to the eigenbasis :math:`\\mathbf{Q}` of the mean :math:`A_k`, so the
       wide directions are axes.
    3. On each axis take :math:`D_i = \\max(1, \\max_k \\sum_j |(Q^TA_kQ)_{ij}|)`.
       A diagonal matrix whose entries are the absolute row sums of a symmetric
       matrix exceeds it (the difference is diagonally dominant with a
       non-negative diagonal), so :math:`\\mathrm{diag}(D)\\succeq Q^TA_kQ` for
       every :math:`k`, and :math:`D_i\\ge1` gives :math:`\\succeq I`.
    4. Return :math:`L\\mathbf{Q}\\,\\mathrm{diag}(D)\\,\\mathbf{Q}^TL^T`.

    The bound is anisotropic: it is only as wide as the widest component *in
    each direction*.  :func:`min_dominant_kappa` against it is at most one.

    ``extra_covs`` (shape (n, d, d)) are further matrices to dominate that are
    not components of a full-dimensional mixture -- e.g. a marginal fit in a few
    bands, embedded with zeros elsewhere.  Positive semi-definite is enough.
    """
    lo = np.linalg.cholesky(base_cov)
    covs = np.concatenate([m.covs for m in mixtures]
                          + ([np.asarray(extra_covs, float)] if extra_covs is not None else []))
    wk = np.linalg.solve(lo, np.swapaxes(np.linalg.solve(lo, covs), -1, -2))
    wk = 0.5 * (wk + np.swapaxes(wk, -1, -2))
    _, q = np.linalg.eigh(wk.mean(axis=0))
    ak = np.einsum("ia,kij,jb->kab", q, wk, q)
    diag = np.maximum(np.abs(ak).sum(axis=2).max(axis=0), 1.0)
    env = lo @ q @ np.diag(diag) @ q.T @ lo.T
    return 0.5 * (env + env.T)


def fit_outlier_fraction(log_p_bkg: np.ndarray, log_p_out: np.ndarray, *,
                         tol: float = 1e-10, max_iter: int = 10_000) -> float:
    """Maximum-likelihood eta for p = (1 - eta) p_B + eta p_U, by EM.

    Parameters
    ----------
    log_p_bkg, log_p_out : ndarray, shape (n,)
        Natural-log densities of the same objects under the two components.

    Returns
    -------
    float
        The fraction in [0, 1).  EM for a single mixing weight increases the
        likelihood monotonically and converges to the global maximum, because
        the log-likelihood is concave in eta.
    """
    a = np.asarray(log_p_bkg, float)
    b = np.asarray(log_p_out, float)
    good = np.isfinite(a) | np.isfinite(b)
    a, b = a[good], b[good]
    if a.size == 0:
        raise ValueError("no finite densities to fit")
    # Responsibility of U at eta: expit(log(eta / (1 - eta)) - (a - b)), in log
    # space so a tiny eta cannot overflow.
    d = np.where(np.isfinite(a - b), a - b, np.where(np.isfinite(a), 745.0, -745.0))
    eta = 0.5
    for _ in range(max_iter):
        new = float(np.mean(expit(np.log(eta) - np.log1p(-eta) - d)))
        if abs(new - eta) < tol * max(eta, 1e-300):
            eta = new
            break
        eta = new
        if eta <= 0.0:
            return 0.0
    return eta


@dataclass
class OutlierModel:
    """The unmodelled component: a broad normalised Gaussian and its fraction.

    Attributes
    ----------
    mean : ndarray, shape (d,)
    cov : ndarray, shape (d, d)
        Intrinsic covariance, ``kappa**2`` times the envelope.
        Measurement noise is added per object, as for every other density.
    fraction : ndarray, shape (M,)
        :math:`\\eta` in each reference-magnitude bin: the share of the field
        surface density :math:`\\Sigma_B(m)` assigned to :math:`U`.
    mag_edges : ndarray, shape (M+1,)
    kappa, kappa_min : float
        The breadth used, and the smallest breadth that dominates every
        component it was checked against.
    system : str
    labels : tuple of str
    meta : dict
        Provenance: which models were checked, which fields fitted eta.
    """

    mean: np.ndarray
    cov: np.ndarray
    fraction: np.ndarray
    mag_edges: np.ndarray
    kappa: float
    kappa_min: float
    system: str
    labels: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, float)
        self.cov = np.asarray(self.cov, float)
        self.fraction = np.atleast_1d(np.asarray(self.fraction, float))
        self.mag_edges = np.asarray(self.mag_edges, float)
        self.labels = tuple(self.labels)
        if self.fraction.size != self.mag_edges.size - 1:
            raise ValueError("one outlier fraction per magnitude bin is required")
        if ((self.fraction < 0) | (self.fraction >= 1)).any():
            raise ValueError("outlier fractions must lie in [0, 1)")
        if not self.kappa > self.kappa_min:
            raise ValueError(
                f"kappa={self.kappa:.4g} does not exceed kappa_min={self.kappa_min:.4g}: "
                f"the density would not dominate every component's tail")
        self._mix = GaussianMixture(np.ones(1), self.mean[None], self.cov[None],
                                    labels=self.labels)

    @classmethod
    def build(cls, reference: list[GaussianMixture], check: list[GaussianMixture], *,
              kappa: float, fraction: np.ndarray, mag_edges: np.ndarray, system: str,
              labels: tuple[str, ...], meta: dict | None = None) -> "OutlierModel":
        """Centre from ``reference``, shape from the envelope over ``check``.

        ``check`` must contain every component of every model the scorer will
        compare against -- all quasar slices and all background mixtures --
        or the dominance guarantee does not hold.
        """
        mean, base = mixture_moments(reference)
        env = envelope_covariance(base, check)
        kmin = min_dominant_kappa(env, check)
        return cls(mean, kappa**2 * env, fraction, mag_edges, float(kappa), kmin,
                   system, labels, dict(meta or {}))

    def check_system(self, system: str) -> None:
        if system != self.system:
            raise ValueError(
                f"photometric system mismatch: outlier model is {self.system!r}, "
                f"candidate is {system!r}.")

    def log_prob(self, x: np.ndarray, cov: np.ndarray | None = None, *,
                 observed: np.ndarray | None = None) -> np.ndarray:
        """log p(c_obs | U), per unit observed feature volume, shape (n,)."""
        return self._mix.log_prob(x, cov, observed=observed)

    def fraction_at(self, ref_mag: np.ndarray) -> np.ndarray:
        """eta at each magnitude, the bin clipped to the fitted range."""
        m = np.asarray(ref_mag, float)
        idx = np.clip(np.digitize(m, self.mag_edges) - 1, 0, self.fraction.size - 1)
        return self.fraction[idx]

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "kind": "outlier_gaussian",
            "mean": self.mean.tolist(), "cov": self.cov.tolist(),
            "fraction": self.fraction.tolist(), "mag_edges": self.mag_edges.tolist(),
            "kappa": self.kappa, "kappa_min": self.kappa_min,
            "system": self.system, "labels": list(self.labels), "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OutlierModel":
        if d.get("kind") != "outlier_gaussian":
            raise ValueError(f"not an outlier model: kind={d.get('kind')!r}")
        return cls(d["mean"], d["cov"], d["fraction"], d["mag_edges"], d["kappa"],
                   d["kappa_min"], d["system"], tuple(d["labels"]), d.get("meta", {}))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "OutlierModel":
        return cls.from_dict(json.loads(Path(path).read_text()))
