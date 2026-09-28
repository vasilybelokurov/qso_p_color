"""Independent checks of the observable-field simulator and release verdict."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qso_pcolor.features import FeatureSet
from qso_pcolor.gaussmix import GaussianMixture


spec = importlib.util.spec_from_file_location("xdqso_recovery_checks",
    Path(__file__).resolve().parents[1] / "scripts/xdqso_recovery_checks.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_draw_includes_correlated_noise_and_missing_band():
    n, d = 40000, 2
    intrinsic = np.array([[.7, .2], [.2, 1.2]])
    noise = np.array([[.4, -.1], [-.1, .3]])
    mu = np.array([1., -2.])
    mix = GaussianMixture(np.ones(1), mu[None], intrinsic[None])
    obs = np.ones((n, d), bool); obs[n // 2:, 1] = False
    cov = np.broadcast_to(noise, (n, d, d)).copy()
    cov = np.where(obs[:, :, None] & obs[:, None, :], cov, 0.)
    f = FeatureSet(np.zeros((n, d)), cov, obs, np.ones(n), np.full(n, 20.),
                   np.full(n, 10.), ("g/r", "z/r"))
    parts = dict(qso_prior=SimpleNamespace(mag_centres=np.array([19., 21.]),
                     sigma=np.zeros((1, 2)), z_edges=np.array([0., 1.])),
                 background=SimpleNamespace(global_=[mix], mag_bin=lambda m: np.zeros(len(m), int)),
                 outlier=SimpleNamespace(mean=mu, cov=intrinsic, fraction_at=lambda m: np.zeros(len(m))),
                 background_density=lambda m, l, b: np.ones(len(m)))
    x, pq = checks.draw_population(parts, f, np.ones(n), np.random.default_rng(43))
    np.testing.assert_allclose(x[:n // 2].mean(0), mu, atol=.025)
    np.testing.assert_allclose(np.cov(x[:n // 2].T), intrinsic + noise, atol=.035)
    assert np.isnan(x[n // 2:, 1]).all() and not pq.any()
    assert abs(x[n // 2:, 0].var() - 1.1) < .035


def test_slice_weights_integrate_redshift_once():
    prior = SimpleNamespace(mag_centres=np.array([19., 21.]),
                            sigma=np.array([[2., 2.], [3., 3.]]), z_edges=np.array([0., .1, .4]))
    np.testing.assert_allclose(checks.quasar_weights(prior, np.array([20.])), [[.2, .9]])


def test_tail_gate_requires_discrepancy_and_statistical_power():
    gates = dict(tails_min_expected=20., tails_factor=2., tails_sigma=3.)
    rows = checks.tail_bins(np.array([300, 21, 0]), np.tile([100., 10., 100.], (5, 1)),
                            np.array([0., 1., 2., np.inf]), gates)
    assert [r["fail"] for r in rows] == [True, False, True]
    assert [r["tested"] for r in rows] == [True, False, True]


def test_future_psf_training_uses_candidate_release_and_band_selection():
    spec = importlib.util.spec_from_file_location("train_qso_model",
        Path(__file__).resolve().parents[1] / "scripts/train_qso_model.py")
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    n = 5
    rows = dict(ra=np.full(n, 180.), dec=np.zeros(n), release=np.array([9010, 9012, 9011, 9010, 9010]),
                type=np.array(["PSF", "PSF", "PSF", "REX", ""]), maskbits=np.zeros(n, int))
    for band in ("g", "r", "z", "w1", "w2"):
        rows[f"flux_{band}"] = np.full(n, 10.)
        rows[f"flux_ivar_{band}"] = np.full(n, 100.)
        rows[f"nobs_{band}"] = np.ones(n, int)
    rows["flux_g"][0] = -2.
    rows["nobs_w2"][1] = 0
    config = dict(selection=dict(morphology="point", maskbits_zero=True, reference="r"), min_abs_b_deg=25.)
    use, flux, ivar = trainer.point_training_photometry(rows, "south", config)
    assert use.tolist() == [True, True, False, False, False]
    assert flux[0, 0] == -2. and ivar[0, 0] > 0
    assert np.isnan(flux[1, -1]) and ivar[1, -1] == 0
