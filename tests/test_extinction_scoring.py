"""A bundle declaring extinction corrects raw catalogue input exactly once."""
import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

from qso_pcolor import PSFMultiSurveyBaseline, Photometry, RedshiftMatch, BlendPolicy
from qso_pcolor.extinction import deredden, sfd_ebv
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.unified import qso_support

ROOT = Path(__file__).resolve().parents[1]
ARGS = dict(morphology=["PSF", "PSF"], z_primary=1.8, l_deg=np.array([276.337, 150.]), b_deg=np.array([60.189, 30.]),
            match=RedshiftMatch(half_width_kms=2000.), blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
            separation_arcsec=6., fracflux=.05, ood_flag_sigma=4.)
BANDS = ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z", "sdss:i")


def _bundles():
    plain = PSFMultiSurveyBaseline.load(ROOT/"models/multisurvey_psf/previous")
    declared = deepcopy(plain)
    labels = list(plain.model.transform.bands)
    coeffs = json.loads((ROOT/"configs/extinction_coefficients.json").read_text())["coefficients"]
    declared.model.meta["extinction"] = dict(map="SFD98", labels=labels, coefficients=[coeffs[b] for b in labels])
    return plain, declared


def test_declared_bundle_on_raw_input_equals_plain_bundle_on_corrected_input():
    plain, declared = _bundles()
    raw = Photometry([[1.9, 2.6, 3.1, 2.9], [5., 6., 7., 6.5]], [[1/120, 1/150, 1/60, 1/40]]*2, BANDS)
    corrected, _, ebv = deredden(raw, ARGS["l_deg"], ARGS["b_deg"], declared.model.meta["extinction"])
    assert ebv[1] > .03                                   # a position with noticeable reddening
    a, _ = declared.score(raw, **ARGS)
    b, _ = plain.score(corrected, **ARGS)
    c, _ = plain.score(raw, **ARGS)
    for x, y, z in zip(a, b, c):
        assert x.log_r_per_unit_z == pytest.approx(y.log_r_per_unit_z, abs=1e-10)
        assert x.loglike_qso_zprimary == pytest.approx(y.loglike_qso_zprimary, abs=1e-10)
    assert abs(a[1].loglike_qso_zprimary - c[1].loglike_qso_zprimary) > 1e-3   # the correction matters
    assert '"extinction": "SFD98"' in a[0].scoring_config_json


def test_support_refuses_an_extinction_corrected_model_without_positions():
    _, declared = _bundles()
    raw = Photometry([[1.9, 2.6, 3.1, 2.9]], [[1/120, 1/150, 1/60, 1/40]], BANDS)
    with pytest.raises(ValueError, match="l_deg and b_deg"):
        qso_support(declared.model, raw, 1.8, draws=7, seed=1)


def test_magnitude_colour_coordinates_leave_the_native_model_unchanged(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    import run_unified_pilot as runner
    labels = ["legacy:g", "legacy:r", "sdss:u"]
    t = runner.magnitude_colour_matrix(labels, "legacy:r")
    rng = np.random.default_rng(5)
    h = np.array([[1, 0, 0], [0, 1, 0], [1.06, -.06, 0], [0, 0, 1.]])
    a = rng.normal(size=(2, 3, 3)); x_covs = a @ a.swapaxes(1, 2) + np.eye(3)
    x_mix = GaussianMixture(np.array([.3, .7]), rng.normal(size=(2, 3)), x_covs)
    u_means, u_covs = x_mix.means @ t.T, t @ x_mix.covs @ t.T       # no constraint: pure change of variables
    np.testing.assert_allclose(u_means @ (h @ np.linalg.inv(t)).T, x_mix.means @ h.T, atol=1e-12)
    np.testing.assert_allclose((h @ np.linalg.inv(t)) @ u_covs @ (h @ np.linalg.inv(t)).T, h @ x_covs @ h.T, atol=1e-12)
    # u = (g - r, r, u - r): every colour minus the magnitude coordinate.
    np.testing.assert_allclose(t @ np.array([3., 2., 5.]), [1., 2., 3.])
    means, covs = runner.to_magnitude_colour(x_mix, t, 1, dict(mean=20., variance=100.))
    assert (means[:, 1] == 20).all() and (covs[:, 1, 1] == 100).all() and (covs[:, 1, [0, 2]] == 0).all()
