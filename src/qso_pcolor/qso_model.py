"""QSO colour locus as a function of redshift.

Model
-----
We represent the quasar colour distribution as a set of mixtures conditional on
redshift,

.. math::
    p(\\mathbf{c}\\mid Q, z) \\approx \\sum_j \\omega_j(z)\\,
    p_j(\\mathbf{c}),

where :math:`p_j` is an extreme-deconvolution mixture fitted to spectroscopic
quasars in redshift slice :math:`j` and :math:`\\omega_j(z)` are linear
interpolation weights between the two neighbouring slice centres.  Each
:math:`p_j` is normalised over colour space, so the result is a genuine
conditional density: the redshift distribution of the *training sample* does not
leak into it.

Why not a single joint mixture in :math:`(\\mathbf{c}, z)`
--------------------------------------------------------
A joint Gaussian mixture, conditioned analytically on :math:`z`, is the XDQSOz
construction and is also supported here via :func:`JointColourRedshiftModel`.
It has two properties that are awkward for this problem:

1. Within a component the colour-redshift relation is *linear*.  The quasar
   track is strongly non-monotonic (Lyman-alpha and the major broad lines cross
   band edges), so components must be numerous and narrow in ``z`` before the
   conditional stops extrapolating along straight lines.
2. Conditioning on ``z`` reweights components by their ``z``-marginal, which
   carries the training sample's redshift distribution.  The conditional density
   is still correct, but the redshift prior is implicit rather than declared.

The slice model makes both the resolution in ``z`` and the assumed redshift
prior explicit and separately testable.  Which one predicts held-out quasar
colours better is an empirical question; :mod:`qso_pcolor.validation` measures
it rather than assuming an answer.

Units
-----
``log_p_colour_given_z`` returns a log density per unit volume of feature space.
Redshift densities are per unit ``z``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .gaussmix import GaussianMixture, condition_joint
from .xd import fit_xd

__all__ = [
    "RedshiftMatch",
    "SlicedColourRedshiftModel",
    "JointColourRedshiftModel",
    "fit_sliced_model",
]

C_KM_S = 299792.458


@dataclass(frozen=True)
class RedshiftMatch:
    """Definition of "the same redshift as the primary".

    Exactly one of ``half_width_kms`` or ``dz_half_width`` must be set.  There is
    no default: the width of the match window changes the meaning of every
    probability the package reports, so it is always the user's declaration.

    Parameters
    ----------
    half_width_kms : float, optional
        Rest-frame velocity half-width, converted with
        :math:`\\Delta z = (1+z_0)\\,\\Delta v / c`.
    dz_half_width : float, optional
        Half-width directly in redshift.
    z_primary_err : float
        1-sigma uncertainty on the primary redshift.  It is added in quadrature
        to the window when ``kernel='gaussian'`` and ignored for a top hat.
    kernel : {'tophat', 'gaussian'}
        Shape of the match weight over redshift.
    """

    half_width_kms: float | None = None
    dz_half_width: float | None = None
    z_primary_err: float = 0.0
    kernel: str = "tophat"

    def __post_init__(self) -> None:
        if (self.half_width_kms is None) == (self.dz_half_width is None):
            raise ValueError(
                "specify exactly one of half_width_kms or dz_half_width"
            )
        if self.kernel not in ("tophat", "gaussian"):
            raise ValueError("kernel must be 'tophat' or 'gaussian'")

    def half_width(self, z_primary: float) -> float:
        """Half-width in redshift at the primary redshift."""
        if self.dz_half_width is not None:
            return float(self.dz_half_width)
        return float((1.0 + z_primary) * self.half_width_kms / C_KM_S)

    def weight(self, z: np.ndarray, z_primary: float) -> np.ndarray:
        """Match weight in [0, 1] over a redshift grid.

        A top hat gives a hard window; a Gaussian gives a kernel of the same
        1-sigma width, both broadened by ``z_primary_err``.  The weight is a
        *selection* function, not a normalised density: it multiplies the
        quasar intensity inside the same-redshift integral.
        """
        z = np.asarray(z, dtype=float)
        w = self.half_width(z_primary)
        s = float(np.hypot(w, self.z_primary_err))
        if self.kernel == "tophat":
            lo, hi = z_primary - s, z_primary + s
            return ((z >= lo) & (z <= hi)).astype(float)
        return np.exp(-0.5 * ((z - z_primary) / s) ** 2)

    def describe(self) -> dict:
        return {
            "half_width_kms": self.half_width_kms,
            "dz_half_width": self.dz_half_width,
            "z_primary_err": self.z_primary_err,
            "kernel": self.kernel,
        }


@dataclass
class SlicedColourRedshiftModel:
    """Redshift-conditional colour mixtures on a redshift grid.

    Attributes
    ----------
    z_centres : ndarray, shape (J,)
        Centres of the redshift slices, ascending.
    mixtures : list of GaussianMixture
        One per slice, each normalised over colour space.
    n_train : ndarray, shape (J,)
        Effective number of training objects per slice.  Slices with too few
        objects are unreliable and are reported through ``support``.
    system : str
        Photometric-system identifier (e.g. ``'ls_dr11_south_grz'``).  Scoring
        raises if the candidate's system differs: a Legacy model must never be
        applied to SDSS features.
    labels : tuple of str
        Feature names, in column order.
    meta : dict
        Provenance: training catalogue, cuts, config hash, fit settings.
    """

    z_centres: np.ndarray
    mixtures: list[GaussianMixture]
    n_train: np.ndarray
    system: str
    labels: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.z_centres = np.asarray(self.z_centres, dtype=float)
        if self.z_centres.ndim != 1 or (np.diff(self.z_centres) <= 0).any():
            raise ValueError("z_centres must be 1-D and strictly ascending")
        if len(self.mixtures) != self.z_centres.size:
            raise ValueError("one mixture per redshift slice is required")
        self.n_train = np.asarray(self.n_train, dtype=float)

    @property
    def n_dim(self) -> int:
        return self.mixtures[0].n_dim

    @property
    def z_range(self) -> tuple[float, float]:
        return float(self.z_centres[0]), float(self.z_centres[-1])

    def check_system(self, system: str) -> None:
        """Raise if ``system`` is not the one this model was trained on."""
        if system != self.system:
            raise ValueError(
                f"photometric system mismatch: model is {self.system!r}, "
                f"candidate is {system!r}. Train a model for that system or "
                f"apply an explicit, validated transformation."
            )

    # -- slice densities --------------------------------------------------

    def _log_p_slices(
        self,
        x: np.ndarray,
        cov: np.ndarray | None,
        observed: np.ndarray | None,
    ) -> np.ndarray:
        """log p(c_obs | Q, slice j) for every object and slice, shape (n, J)."""
        return np.stack(
            [m.log_prob(x, cov, observed=observed) for m in self.mixtures], axis=1
        )

    def log_p_colour_given_z(
        self,
        x: np.ndarray,
        cov: np.ndarray | None,
        z: np.ndarray,
        *,
        observed: np.ndarray | None = None,
        _log_slices: np.ndarray | None = None,
    ) -> np.ndarray:
        """log p(c_obs | Q, z), shape (n, n_z).

        Interpolates linearly *in density* between neighbouring slices, so the
        result stays a normalised density for any ``z``.  Outside the trained
        redshift range the nearest slice is used; callers should consult
        :meth:`in_support` rather than trusting an extrapolated value.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        log_slices = (
            self._log_p_slices(x, cov, observed) if _log_slices is None else _log_slices
        )
        n = log_slices.shape[0]

        zc = self.z_centres
        idx = np.clip(np.searchsorted(zc, z) - 1, 0, zc.size - 2)
        lo, hi = zc[idx], zc[idx + 1]
        frac = np.clip((z - lo) / (hi - lo), 0.0, 1.0)  # clamps outside the grid

        with np.errstate(divide="ignore"):
            log_w_lo = np.log(1.0 - frac)
            log_w_hi = np.log(frac)
        a = log_slices[:, idx] + log_w_lo[None, :]
        b = log_slices[:, idx + 1] + log_w_hi[None, :]
        out = np.logaddexp(a, b)
        assert out.shape == (n, z.size)
        return out

    def in_support(self, z: np.ndarray) -> np.ndarray:
        """True where ``z`` lies inside the trained redshift range."""
        z = np.asarray(z, dtype=float)
        return (z >= self.z_centres[0]) & (z <= self.z_centres[-1])

    # -- derived quantities ----------------------------------------------

    def log_p_colour_at_primary(
        self,
        x: np.ndarray,
        cov: np.ndarray | None,
        z_primary: float,
        *,
        observed: np.ndarray | None = None,
    ) -> np.ndarray:
        """log p(c_obs | Q, z_primary), shape (n,).

        This is a class-conditional likelihood.  It is **not** a probability that
        the object is a quasar, and must not be reported as one.
        """
        return self.log_p_colour_given_z(
            x, cov, np.array([z_primary]), observed=observed
        )[:, 0]

    def redshift_posterior(
        self,
        x: np.ndarray,
        cov: np.ndarray | None,
        z_grid: np.ndarray,
        *,
        observed: np.ndarray | None = None,
        log_z_prior: np.ndarray | None = None,
    ) -> np.ndarray:
        """p(z | c_obs, Q) on ``z_grid``, normalised by trapezoid, shape (n, n_z).

        Parameters
        ----------
        log_z_prior : ndarray, shape (n_z,), optional
            Log of the assumed quasar redshift prior, up to a constant.  ``None``
            means flat in ``z`` over the grid, which is a declaration, not a
            physical statement: for a field-quasar calculation pass the log of
            :math:`\\Sigma_Q(z, m)` instead so that the redshift prior and the
            magnitude selection are the ones you intend.
        """
        z_grid = np.asarray(z_grid, dtype=float)
        log_like = self.log_p_colour_given_z(x, cov, z_grid, observed=observed)
        if log_z_prior is not None:
            log_like = log_like + np.asarray(log_z_prior, dtype=float)[None, :]
        # Normalise by trapezoid in linear space, in a numerically safe way.
        shift = log_like.max(axis=1, keepdims=True)
        p = np.exp(log_like - shift)
        norm = np.trapezoid(p, z_grid, axis=1)[:, None]
        return p / norm

    def p_zmatch_given_qso(
        self,
        x: np.ndarray,
        cov: np.ndarray | None,
        z_primary: float,
        match: RedshiftMatch,
        z_grid: np.ndarray,
        *,
        observed: np.ndarray | None = None,
        log_z_prior: np.ndarray | None = None,
    ) -> np.ndarray:
        """P(z matches the primary | c_obs, Q), shape (n,).

        The quantity is *conditional on the object being a quasar*.  A value near
        1 means "if this is a quasar, its redshift agrees with the primary" — it
        says nothing about whether it is a quasar at all.
        """
        z_grid = np.asarray(z_grid, dtype=float)
        post = self.redshift_posterior(
            x, cov, z_grid, observed=observed, log_z_prior=log_z_prior
        )
        w = match.weight(z_grid, z_primary)
        return np.trapezoid(post * w[None, :], z_grid, axis=1)

    def ood_score(
        self,
        x: np.ndarray,
        cov: np.ndarray | None,
        z: float,
        *,
        observed: np.ndarray | None = None,
    ) -> np.ndarray:
        """Distance to the nearest mixture component, in sigma, shape (n,).

        Defined as :math:`\\min_k \\sqrt{(\\mathbf{c}-\\boldsymbol{\\mu}_k)^T
        (\\mathbf{V}_k+\\mathbf{S})^{-1}(\\mathbf{c}-\\boldsymbol{\\mu}_k)}` over
        the components of the slice nearest ``z``, evaluated in the observed
        subspace.  Large values mean the object sits outside the region the model
        was trained on, where the density is an extrapolation.  Reported
        alongside every score; never used to silently modify one.
        """
        j = int(np.argmin(np.abs(self.z_centres - z)))
        mix = self.mixtures[j]
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
        for i in range(n):
            idx = np.flatnonzero(obs[i])
            if idx.size == 0:
                continue
            best = np.inf
            for k in range(mix.n_components):
                delta = x[i, idx] - mix.means[k, idx]
                cc = mix.covs[np.ix_([k], idx, idx)][0] + s[np.ix_([i], idx, idx)][0]
                y = np.linalg.solve(np.linalg.cholesky(cc), delta)
                best = min(best, float(np.sqrt(y @ y)))
            out[i] = best
        return out

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "kind": "sliced_colour_redshift",
            "z_centres": self.z_centres.tolist(),
            "mixtures": [m.to_dict() for m in self.mixtures],
            "n_train": self.n_train.tolist(),
            "system": self.system,
            "labels": list(self.labels),
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, d: dict) -> "SlicedColourRedshiftModel":
        return cls(
            np.asarray(d["z_centres"], float),
            [GaussianMixture.from_dict(m) for m in d["mixtures"]],
            np.asarray(d["n_train"], float),
            d["system"],
            tuple(d["labels"]),
            d.get("meta", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SlicedColourRedshiftModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_sliced_model(
    x: np.ndarray,
    cov: np.ndarray | None,
    z: np.ndarray,
    z_edges: np.ndarray,
    *,
    observed: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    n_components: int = 8,
    min_per_slice: int = 200,
    overlap: float = 0.5,
    system: str = "unspecified",
    labels: tuple[str, ...] = (),
    meta: dict | None = None,
    **fit_kwargs,
) -> SlicedColourRedshiftModel:
    """Fit one extreme-deconvolution mixture per redshift slice.

    Parameters
    ----------
    x, cov, observed, weights
        Training features, per-object covariances, usability mask and sample
        weights, as accepted by :func:`qso_pcolor.xd.fit_xd`.
    z : ndarray, shape (n,)
        Spectroscopic redshifts.
    z_edges : ndarray, shape (J+1,)
        Slice boundaries.  Slice centres are the midpoints.
    overlap : float
        Fraction of a slice width added to each side when collecting training
        objects.  Overlapping slices smooth the model along ``z`` at the cost of
        correlated fits; ``0`` gives disjoint slices.
    min_per_slice : int
        A slice with fewer usable objects is fitted with a single component and
        marked in ``n_train``, rather than being silently dropped.

    Returns
    -------
    SlicedColourRedshiftModel
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    z = np.asarray(z, dtype=float)
    z_edges = np.asarray(z_edges, dtype=float)
    if (np.diff(z_edges) <= 0).any():
        raise ValueError("z_edges must be strictly ascending")
    centres = 0.5 * (z_edges[:-1] + z_edges[1:])

    mixtures: list[GaussianMixture] = []
    counts = np.zeros(centres.size)
    for j in range(centres.size):
        w = z_edges[j + 1] - z_edges[j]
        lo = z_edges[j] - overlap * w
        hi = z_edges[j + 1] + overlap * w
        sel = (z >= lo) & (z < hi)
        counts[j] = int(sel.sum())
        if counts[j] < 2:
            raise ValueError(
                f"redshift slice {j} ({lo:.3f}-{hi:.3f}) has {int(counts[j])} "
                f"training objects; widen the slice or narrow the redshift range"
            )
        k = n_components if counts[j] >= min_per_slice else 1
        res = fit_xd(
            x[sel],
            None if cov is None else np.asarray(cov)[sel],
            n_components=k,
            observed=None if observed is None else observed[sel],
            weights=None if weights is None else np.asarray(weights)[sel],
            labels=labels,
            **fit_kwargs,
        )
        mixtures.append(res.mixture)

    return SlicedColourRedshiftModel(
        centres,
        mixtures,
        counts,
        system,
        labels,
        meta={"overlap": overlap, "min_per_slice": min_per_slice, **(meta or {})},
    )


@dataclass
class JointColourRedshiftModel:
    """XDQSOz-style joint mixture in (colour, redshift), conditioned analytically.

    Kept as an alternative backend so that the two representations can be
    compared on held-out data.  The joint vector is ``[c_1, ..., c_d, z]``, with
    redshift as the last dimension.
    """

    mixture: GaussianMixture
    system: str
    labels: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    @property
    def n_dim(self) -> int:
        return self.mixture.n_dim - 1

    def check_system(self, system: str) -> None:
        if system != self.system:
            raise ValueError(
                f"photometric system mismatch: model is {self.system!r}, "
                f"candidate is {system!r}."
            )

    def log_p_colour_given_z(
        self,
        x: np.ndarray,
        cov: np.ndarray | None,
        z: np.ndarray,
        *,
        observed: np.ndarray | None = None,
        z_err: float = 0.0,
    ) -> np.ndarray:
        """log p(c_obs | Q, z), shape (n, n_z).

        ``z_err`` widens the component weights, which is how a primary redshift
        uncertainty enters: the conditional weight becomes
        :math:`\\alpha_k\\mathcal{N}(z_0\\mid\\mu_{z,k}, V_{zz,k}+\\sigma_z^2)`.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        zi = np.array([self.mixture.n_dim - 1])
        out = np.empty((np.atleast_2d(x).shape[0], z.size))
        yv = None if z_err == 0 else np.array([[z_err**2]])
        for i, zz in enumerate(z):
            _, cond = condition_joint(self.mixture, np.array([zz]), zi, y_var=yv)
            out[:, i] = cond.log_prob(x, cov, observed=observed)
        return out

    def redshift_posterior(
        self,
        x: np.ndarray,
        cov: np.ndarray | None,
        z_grid: np.ndarray,
        *,
        observed: np.ndarray | None = None,
    ) -> np.ndarray:
        """p(z | c_obs, Q) on ``z_grid``.

        Note that this inherits the redshift distribution of the training
        sample through the mixture's ``z`` marginal.  That distribution is the
        spectroscopic *target selection*, not the quasar population.
        """
        z_grid = np.asarray(z_grid, dtype=float)
        x = np.atleast_2d(np.asarray(x, dtype=float))
        d = self.mixture.n_dim - 1
        c_idx = np.arange(d)
        n = x.shape[0]

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
        colour_mix = self.mixture.marginal(c_idx)
        resp = colour_mix.responsibilities(x, s, observed=obs)  # (n, K)

        k = self.mixture.n_components
        mu_z = self.mixture.means[:, d]
        v_zz = self.mixture.covs[:, d, d]
        out = np.zeros((n, z_grid.size))
        for i in range(n):
            idx = np.flatnonzero(obs[i])
            if idx.size == 0:
                out[i] = 1.0 / (z_grid[-1] - z_grid[0])
                continue
            for j in range(k):
                cc = self.mixture.covs[np.ix_([j], idx, idx)][0] + s[i][np.ix_(idx, idx)]
                v_zc = self.mixture.covs[j, d, idx]
                sol = np.linalg.solve(cc, x[i, idx] - self.mixture.means[j, idx])
                mu = mu_z[j] + v_zc @ sol
                var = v_zz[j] - v_zc @ np.linalg.solve(cc, v_zc)
                var = max(var, 1e-12)
                out[i] += resp[i, j] * np.exp(
                    -0.5 * (z_grid - mu) ** 2 / var
                ) / np.sqrt(2 * np.pi * var)
        norm = np.trapezoid(out, z_grid, axis=1)[:, None]
        return out / np.where(norm > 0, norm, 1.0)

    def to_dict(self) -> dict:
        return {
            "kind": "joint_colour_redshift",
            "mixture": self.mixture.to_dict(),
            "system": self.system,
            "labels": list(self.labels),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JointColourRedshiftModel":
        return cls(
            GaussianMixture.from_dict(d["mixture"]),
            d["system"],
            tuple(d["labels"]),
            d.get("meta", {}),
        )
