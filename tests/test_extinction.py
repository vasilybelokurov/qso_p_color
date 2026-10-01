import json
from pathlib import Path

import numpy as np
import pytest

from qso_pcolor.extinction import (correction_factors, deredden, deredden_arrays,
                                   latent_coefficients, native_coefficients, sfd_ebv)
from qso_pcolor.multisurvey_data import Photometry, band_labels

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/extinction_coefficients.json').read_text())


def test_every_native_band_has_a_positive_coefficient():
    assert set(CONFIG['coefficients']) == set(band_labels())
    assert all(v > 0 for v in CONFIG['coefficients'].values())


def test_correction_is_identity_at_zero_reddening_and_inverts_exactly():
    rng = np.random.default_rng(3)
    flux = rng.normal(size=(5, 3)); var = rng.uniform(.1, 1, size=(5, 3)); r = np.array([3.2, 2.2, 1.2])
    f0, v0 = deredden_arrays(flux, var, np.zeros(5), r)
    np.testing.assert_array_equal(f0, flux); np.testing.assert_array_equal(v0, var)
    ebv = rng.uniform(0, .2, 5)
    f1, v1 = deredden_arrays(flux, var, ebv, r)
    np.testing.assert_allclose(f1/np.sqrt(v1), flux/np.sqrt(var), rtol=1e-12)    # S/N preserved
    np.testing.assert_allclose(f1/correction_factors(ebv, r), flux, rtol=1e-12)
    assert ((f1 < 0) == (flux < 0)).all()                                         # negative flux kept
    # A magnitude brightens by exactly R E.
    mag = lambda x: -2.5*np.log10(x)
    pos = np.abs(flux)+1
    np.testing.assert_allclose(mag(pos)-mag(deredden_arrays(pos, var, ebv, r)[0]), np.outer(ebv, r), rtol=1e-12)


def test_missing_entries_stay_missing():
    f, v = deredden_arrays(np.array([[np.nan, 1.]]), np.array([[np.inf, 1.]]), np.array([.1]), np.array([3., 2.]))
    assert np.isnan(f[0, 0]) and np.isinf(v[0, 0])


def test_native_coefficients_follow_the_observation_matrix():
    layout = dict(latent_labels=['legacy:g', 'legacy:r', 'sdss:u'],
                  native_labels=['decals_dr9_south:g', 'decals_dr9_south:r', 'decals_dr9_north:g', 'sdss:u'],
                  operators=dict(qso=dict(matrix=[[1, 0, 0], [0, 1, 0], [1.06, -.06, 0], [0, 0, 1]])))
    c = dict(CONFIG['coefficients'])
    r = native_coefficients(layout, c)
    assert r[0] == c['decals_dr9_south:g'] and r[3] == c['sdss:u']
    assert r[2] == pytest.approx(1.06*c['decals_dr9_south:g'] - .06*c['decals_dr9_south:r'])
    np.testing.assert_allclose(latent_coefficients(layout['latent_labels'], c)[:2],
                               [c['decals_dr9_south:g'], c['decals_dr9_south:r']])


def test_full_covariance_scales_as_outer_product():
    rec = dict(labels=['sdss:g', 'sdss:r'], coefficients=[3.3, 2.3])
    p = Photometry(np.array([[1., 2.]]), np.array([[.1, .2]]), ('sdss:g', 'sdss:r'))
    cov = np.array([[[.1, .05], [.05, .2]]])
    out, c, ebv = deredden(p, 30., 60., rec, flux_covariance=cov)
    f = 10**(0.4*np.array([3.3, 2.3])*ebv[0])
    np.testing.assert_allclose(c[0], cov[0]*np.outer(f, f)); np.testing.assert_allclose(out.flux[0], [1*f[0], 2*f[1]])


def test_sfd_reddening_is_small_at_the_poles_and_large_in_the_plane():
    ebv = sfd_ebv([0., 0., 10.], [90., -90., 0.])
    assert (ebv[:2] < .05).all() and ebv[2] > 1.
