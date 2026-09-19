"""Turn calibrated fluxes into colour/shape features with a full covariance.

Two design rules follow from the science and are enforced here rather than left
to the caller:

1. **Colours are correlated.**  Any feature vector built from a shared set of
   band fluxes has off-diagonal covariance.  Every transform in this module
   returns a full matrix, never a vector of independent errors.
2. **A negative or low-significance flux is a measurement, not a failure.**  We
   never clip, floor or drop a band because its flux came out negative.  A band
   is unusable only when the survey says so (``flux_ivar <= 0``, zero exposures,
   or a mask bit), and that is expressed through the ``observed`` mask, which the
   likelihood marginalises over exactly.

Flux unit throughout: nanomaggies, i.e. ``m = 22.5 - 2.5 log10(f)``, which is the
Legacy Surveys and SDSS convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "FeatureSet",
    "FeatureTransform",
    "RelativeFluxTransform",
    "AsinhColourTransform",
    "deredden",
    "flux_to_mag",
]

_2P5_OVER_LN10 = 2.5 / np.log(10.0)


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    """AB magnitude from nanomaggies; non-positive flux gives NaN, not an error."""
    flux = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 22.5 - 2.5 * np.log10(np.where(flux > 0, flux, np.nan))


def deredden(
    flux: np.ndarray,
    ivar: np.ndarray,
    transmission: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Galactic transmission corrections to fluxes and their variances.

    ``f_dered = f / T`` and ``var_dered = var / T^2``.  Propagating the
    correction into the variance matters: at low Galactic latitude the
    correction is large and a fixed fractional error would otherwise be
    understated.  Bands with ``T <= 0`` are returned as unusable (variance
    ``inf``).

    Parameters
    ----------
    flux, ivar, transmission : ndarray, shape (n, d)
        Observed flux, its inverse variance, and ``mw_transmission_*``.

    Returns
    -------
    flux_dered : ndarray, shape (n, d)
    var_dered : ndarray, shape (n, d)
        Variance, ``inf`` where the measurement is unusable.
    """
    flux = np.asarray(flux, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    t = np.asarray(transmission, dtype=float)

    good = np.isfinite(ivar) & (ivar > 0) & np.isfinite(t) & (t > 0) & np.isfinite(flux)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(good, flux / t, np.nan)
        v = np.where(good, 1.0 / (ivar * t**2), np.inf)
    return f, v


@dataclass
class FeatureSet:
    """Features, covariance and usability for a batch of objects.

    Attributes
    ----------
    x : ndarray, shape (n, d)
        Feature vector.  Entries where ``observed`` is False are NaN.
    cov : ndarray, shape (n, d, d)
        Full feature covariance.  Rows/columns of unobserved dimensions are
        zeroed; the likelihood never reads them.
    observed : ndarray of bool, shape (n, d)
    ref_flux : ndarray, shape (n,)
        Flux in the reference band, nanomaggies (dereddened if requested).
    ref_mag : ndarray, shape (n,)
        ``22.5 - 2.5 log10(ref_flux)``; NaN where the reference flux is
        non-positive.
    ref_snr : ndarray, shape (n,)
    labels : tuple of str
        Feature names, in the order of the columns.
    flags : dict of str -> ndarray of bool
        Per-object diagnostic flags.  These never modify ``x``; they travel with
        the score so that a high probability from suspect photometry is
        distinguishable from a high probability from clean photometry.
    """

    x: np.ndarray
    cov: np.ndarray
    observed: np.ndarray
    ref_flux: np.ndarray
    ref_mag: np.ndarray
    ref_snr: np.ndarray
    labels: tuple[str, ...]
    flags: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return self.x.shape[0]

    def usable(self, min_dims: int = 2) -> np.ndarray:
        """Objects with at least ``min_dims`` usable feature dimensions."""
        return self.observed.sum(axis=1) >= min_dims

    def flag_summary(self) -> dict[str, int]:
        return {k: int(v.sum()) for k, v in self.flags.items()}


class FeatureTransform:
    """Interface for flux -> feature transforms."""

    name: str = "base"
    labels: tuple[str, ...] = ()

    def __call__(
        self, flux: np.ndarray, var: np.ndarray, bands: tuple[str, ...]
    ) -> FeatureSet:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class RelativeFluxTransform(FeatureTransform):
    """Fluxes divided by a reference band: ``r_j = f_j / f_ref``.

    This is the XDQSO-style representation.  It separates SED shape from overall
    brightness exactly, and unlike colours it stays finite and well defined when
    a flux is negative — which is common for faint sources and is real
    information about the SED.

    The one genuine weakness is that the denominator is itself noisy.  The
    Jacobian propagation below is a first-order expansion and degrades once
    ``f_ref`` is not significantly positive, so objects with
    ``ref_snr < min_ref_snr`` are flagged ``low_ref_snr``.  We do not clip or
    replace the reference flux; the caller decides whether to drop those objects
    or to score them with :class:`AsinhColourTransform` instead.

    Parameters
    ----------
    reference_band : str
        Band used as the denominator.
    min_ref_snr : float
        Below this the linearised covariance is not trustworthy and the object
        is flagged.  It is a *diagnostic* threshold, not a cut.
    """

    reference_band: str = "r"
    min_ref_snr: float = 5.0
    name: str = "relative_flux"

    def __call__(
        self, flux: np.ndarray, var: np.ndarray, bands: tuple[str, ...]
    ) -> FeatureSet:
        flux = np.atleast_2d(np.asarray(flux, dtype=float))
        var = np.atleast_2d(np.asarray(var, dtype=float))
        n, nb = flux.shape
        if nb != len(bands):
            raise ValueError("flux columns do not match the band list")
        if self.reference_band not in bands:
            raise ValueError(f"reference band {self.reference_band!r} not in {bands}")

        a = bands.index(self.reference_band)
        others = [i for i in range(nb) if i != a]
        labels = tuple(f"{bands[i]}/{self.reference_band}" for i in others)
        d = len(others)

        f_ref = flux[:, a]
        v_ref = var[:, a]
        with np.errstate(divide="ignore", invalid="ignore"):
            ref_snr = np.where(v_ref > 0, f_ref / np.sqrt(v_ref), np.nan)

        band_ok = np.isfinite(flux) & np.isfinite(var) & (var > 0)
        ref_ok = band_ok[:, a] & (f_ref != 0.0)

        x = np.full((n, d), np.nan)
        cov = np.zeros((n, d, d))
        observed = np.zeros((n, d), dtype=bool)

        safe_ref = np.where(ref_ok, f_ref, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = flux[:, others] / safe_ref[:, None]
        observed[:] = band_ok[:, others] & ref_ok[:, None]
        x = np.where(observed, r, np.nan)

        # C_r = J C_f J^T with J_{j,j} = 1/f_a and J_{j,a} = -r_j/f_a.
        # Band fluxes are independent, so C_f is diagonal and the result is
        #   C_r[j,k] = delta_jk v_j / f_a^2 + r_j r_k v_a / f_a^2.
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_fa2 = np.where(ref_ok, 1.0 / safe_ref**2, 0.0)
            v_others = np.where(observed, var[:, others], 0.0)
            r_safe = np.where(observed, r, 0.0)
            cov = (
                np.einsum("nj,n->nj", v_others, inv_fa2)[:, :, None] * np.eye(d)[None]
                + np.einsum("nj,nk,n->njk", r_safe, r_safe, v_ref * inv_fa2)
            )
        cov = np.where(
            (observed[:, :, None] & observed[:, None, :]), cov, 0.0
        )

        ref_mag = flux_to_mag(np.where(ref_ok, f_ref, np.nan))
        flags = {
            "low_ref_snr": ~(ref_snr >= self.min_ref_snr),
            "no_reference_band": ~ref_ok,
            "nonpositive_ref_flux": ref_ok & (f_ref <= 0),
        }
        return FeatureSet(x, cov, observed, f_ref, ref_mag, ref_snr, labels, flags)


@dataclass
class AsinhColourTransform(FeatureTransform):
    """Asinh ("luptitude") colours of adjacent bands.

    Defined per band as

    .. math::
        m_b = 22.5 - \\frac{2.5}{\\ln 10}
              \\left[\\operatorname{asinh}\\!\\frac{f_b}{2 b_b} + \\ln b_b\\right],

    which stays finite for zero and negative flux, then differenced into
    adjacent-band colours.  The softening parameters ``b_b`` are a property of
    the survey's depth and must be supplied explicitly: hiding them as constants
    inside the code would make the feature space undocumented.

    Colours built from shared bands are correlated by construction; the full
    matrix ``A C_m A^T`` is returned.
    """

    softening: dict[str, float]
    name: str = "asinh_colour"

    def __call__(
        self, flux: np.ndarray, var: np.ndarray, bands: tuple[str, ...]
    ) -> FeatureSet:
        flux = np.atleast_2d(np.asarray(flux, dtype=float))
        var = np.atleast_2d(np.asarray(var, dtype=float))
        n, nb = flux.shape
        missing = [b for b in bands if b not in self.softening]
        if missing:
            raise ValueError(f"no softening parameter for band(s) {missing}")

        b = np.array([self.softening[x] for x in bands], dtype=float)
        band_ok = np.isfinite(flux) & np.isfinite(var) & (var > 0)

        # Substitute a finite placeholder in unusable bands *before* the colour
        # matrix multiply.  A NaN there would propagate through 0 * NaN into
        # every colour, including ones built only from good bands, and those
        # colours would still be flagged observed.
        flux = np.where(band_ok, flux, 0.0)
        var = np.where(band_ok, var, 0.0)

        m = 22.5 - _2P5_OVER_LN10 * (np.arcsinh(flux / (2 * b)) + np.log(b))
        # dm/df = -(2.5/ln10) / sqrt(4 b^2 + f^2)
        dmdf = -_2P5_OVER_LN10 / np.sqrt(4 * b**2 + flux**2)
        var_m = np.where(band_ok, dmdf**2 * var, 0.0)

        # Adjacent-band colours: A has +1 and -1 on shared bands.
        d = nb - 1
        a_mat = np.zeros((d, nb))
        for j in range(d):
            a_mat[j, j] = 1.0
            a_mat[j, j + 1] = -1.0
        labels = tuple(f"{bands[j]}-{bands[j + 1]}" for j in range(d))

        x = np.einsum("jb,nb->nj", a_mat, m)
        cov = np.einsum("jb,nb,kb->njk", a_mat, var_m, a_mat)
        observed = band_ok[:, :-1] & band_ok[:, 1:]
        x = np.where(observed, x, np.nan)
        cov = np.where(observed[:, :, None] & observed[:, None, :], cov, 0.0)

        # Reference brightness: use the first band with a usable flux as the
        # anchor is fragile, so require the caller's chosen reference instead.
        ref_i = nb // 2
        with np.errstate(divide="ignore", invalid="ignore"):
            ref_snr = np.where(
                band_ok[:, ref_i], flux[:, ref_i] / np.sqrt(var[:, ref_i]), np.nan
            )
        return FeatureSet(
            x,
            cov,
            observed,
            flux[:, ref_i],
            m[:, ref_i],
            ref_snr,
            labels,
            {"nonpositive_ref_flux": ~(flux[:, ref_i] > 0)},
        )
