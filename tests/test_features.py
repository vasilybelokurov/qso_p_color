"""Feature transforms checked against Monte Carlo error propagation.

The covariance returned by each transform is compared against the empirical
covariance of transformed flux realisations.  This is the only way to catch a
missing off-diagonal term, which is exactly the mistake that would quietly bias
every likelihood downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from qso_pcolor.features import (
    AsinhColourTransform,
    RelativeFluxTransform,
    deredden,
    flux_to_mag,
)

BANDS = ("g", "r", "i", "z")


def test_relative_flux_covariance_matches_monte_carlo():
    rng = np.random.default_rng(0)
    truth = np.array([3.0, 6.0, 8.0, 11.0])          # nanomaggies, high S/N
    var = np.array([0.04, 0.05, 0.06, 0.09])
    tr = RelativeFluxTransform(reference_band="r")

    n = 400000
    draws = truth + rng.normal(0, np.sqrt(var), size=(n, 4))
    fs = tr(draws, np.broadcast_to(var, (n, 4)), BANDS)
    empirical = np.cov(fs.x.T)

    fs0 = tr(truth[None, :], var[None, :], BANDS)
    assert np.allclose(fs0.cov[0], empirical, rtol=0.05, atol=1e-6)
    # The off-diagonal is not small: assuming independence would be wrong.
    assert abs(fs0.cov[0, 0, 1]) > 0.1 * np.sqrt(fs0.cov[0, 0, 0] * fs0.cov[0, 1, 1])


def test_relative_flux_values_and_labels():
    tr = RelativeFluxTransform(reference_band="r")
    fs = tr(np.array([[2.0, 4.0, 6.0, 8.0]]), np.full((1, 4), 0.01), BANDS)
    assert fs.labels == ("g/r", "i/r", "z/r")
    assert np.allclose(fs.x[0], [0.5, 1.5, 2.0])
    assert fs.ref_flux[0] == 4.0
    assert fs.ref_mag[0] == pytest.approx(22.5 - 2.5 * np.log10(4.0))


def test_negative_flux_is_preserved_not_clipped():
    """A negative band flux must survive into the features as a negative ratio."""
    tr = RelativeFluxTransform(reference_band="r")
    fs = tr(np.array([[-0.8, 5.0, 6.0, 7.0]]), np.full((1, 4), 0.25), BANDS)
    assert fs.observed[0, 0]
    assert fs.x[0, 0] == pytest.approx(-0.16)


def test_unusable_band_is_masked_not_imputed():
    tr = RelativeFluxTransform(reference_band="r")
    var = np.array([[0.01, 0.01, 0.0, 0.01]])        # i-band has ivar = 0
    fs = tr(np.array([[2.0, 4.0, 6.0, 8.0]]), var, BANDS)
    assert list(fs.observed[0]) == [True, False, True]
    assert np.isnan(fs.x[0, 1])
    assert np.all(fs.cov[0, 1, :] == 0) and np.all(fs.cov[0, :, 1] == 0)


def test_low_reference_snr_is_flagged_not_silently_used():
    tr = RelativeFluxTransform(reference_band="r", min_ref_snr=5.0)
    fs = tr(np.array([[1.0, 0.2, 1.0, 1.0]]), np.full((1, 4), 0.25), BANDS)
    assert fs.flags["low_ref_snr"][0]
    assert not np.isnan(fs.x[0, 0])   # still computed, just flagged


def test_asinh_colour_covariance_matches_monte_carlo():
    rng = np.random.default_rng(1)
    soft = {b: 0.5 for b in BANDS}
    truth = np.array([2.0, 5.0, 7.0, 9.0])
    var = np.array([0.09, 0.09, 0.09, 0.09])
    tr = AsinhColourTransform(softening=soft)

    n = 300000
    draws = truth + rng.normal(0, np.sqrt(var), size=(n, 4))
    fs = tr(draws, np.broadcast_to(var, (n, 4)), BANDS)
    empirical = np.cov(fs.x.T)
    fs0 = tr(truth[None, :], var[None, :], BANDS)
    # atol covers Monte Carlo noise on the entries that are analytically zero;
    # rtol covers the second-order error of the linearisation, which is a few
    # per cent here because asinh is appreciably curved at f ~ 2b.
    assert np.allclose(fs0.cov[0], empirical, rtol=0.06, atol=5e-5)
    # g-r and i-z share no band, so the analytic covariance is exactly zero.
    assert fs0.cov[0, 0, 2] == 0.0


def test_asinh_colours_share_bands_so_are_anticorrelated():
    """g-r and r-i share r, so their covariance must be negative."""
    tr = AsinhColourTransform(softening={b: 0.5 for b in BANDS})
    fs = tr(np.array([[2.0, 5.0, 7.0, 9.0]]), np.full((1, 4), 0.09), BANDS)
    assert fs.cov[0, 0, 1] < 0
    assert fs.labels == ("g-r", "r-i", "i-z")


def test_asinh_stays_finite_for_negative_flux():
    tr = AsinhColourTransform(softening={b: 0.5 for b in BANDS})
    fs = tr(np.array([[-2.0, 5.0, 7.0, 9.0]]), np.full((1, 4), 0.09), BANDS)
    assert np.isfinite(fs.x[0]).all()


def test_asinh_requires_explicit_softening():
    with pytest.raises(ValueError, match="softening"):
        AsinhColourTransform(softening={"g": 0.5})(
            np.ones((1, 4)), np.ones((1, 4)), BANDS
        )


def test_deredden_propagates_into_the_variance():
    flux = np.array([[10.0, 10.0]])
    ivar = np.array([[4.0, 4.0]])
    t = np.array([[1.0, 0.5]])
    f, v = deredden(flux, ivar, t)
    assert np.allclose(f[0], [10.0, 20.0])
    # var scales as 1/T^2: halving the transmission quadruples the variance.
    assert np.allclose(v[0], [0.25, 1.0])


def test_deredden_marks_bad_transmission_unusable():
    f, v = deredden(np.array([[1.0]]), np.array([[1.0]]), np.array([[0.0]]))
    assert np.isnan(f[0, 0]) and np.isinf(v[0, 0])


def test_flux_to_mag_is_nan_for_nonpositive_flux():
    m = flux_to_mag(np.array([1.0, 0.0, -3.0]))
    assert m[0] == pytest.approx(22.5)
    assert np.isnan(m[1]) and np.isnan(m[2])
