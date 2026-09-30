import itertools
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.projected_xd import fit_projected, native_mixture, projected_moments
from qso_pcolor.xd import fit_xd


def test_identity_projected_training_matches_existing_xd_with_missing_bands():
    rng = np.random.default_rng(311)
    x = rng.normal(size=(81, 3)); obs = rng.uniform(size=x.shape) > .25
    obs[:, 0] = True; x[~obs] = np.nan
    a = rng.normal(size=(81, 3, 3)); cov = a @ a.swapaxes(1, 2) * .01
    init = GaussianMixture(np.array([.4, .6]), np.array([[0., 0., 0.], [1., 1., 1.]]), np.tile(np.eye(3), (2, 1, 1)))
    def source():
        for i in range(0, len(x), 13):
            yield x[i:i+13], cov[i:i+13], obs[i:i+13], np.zeros(len(x[i:i+13]), int)
    options = dict(init=init, max_iter=5, tol=0., regularization=.001)
    direct = fit_xd(x, cov, observed=obs, **options)
    projected = fit_projected(source, operators={0: (np.eye(3), np.zeros(3), np.zeros((3, 3)))}, expected_rows=len(x), **options)
    np.testing.assert_allclose(projected.mixture.means, direct.mixture.means, atol=1e-12)
    np.testing.assert_allclose(projected.mixture.covs, direct.mixture.covs, atol=1e-12)
    assert projected.mean_loglike == pytest.approx(direct.mean_loglike, abs=1e-12)


def test_rank_deficient_measurement_conditions_unobserved_latent_coordinate():
    mean = np.array([1., 2.]); v = np.array([[2., .4], [.4, 1.]])
    mix = GaussianMixture(np.ones(1), mean[None], v[None])
    h = np.array([[1., .5]]); noise = np.array([[[.2]]]); y = np.array([[2.7]])
    lp, shift, post = projected_moments(mix, y, noise, h, np.array([.1]))
    variance = float((h @ v @ h.T)[0, 0] + .2)
    gain = (v @ h.T)[:, 0] / variance
    np.testing.assert_allclose(shift[0, 0], gain*(2.7-2.1))
    np.testing.assert_allclose(post[0, 0], v-np.outer(gain, gain)*variance)
    assert lp[0, 0] == pytest.approx(multivariate_normal.logpdf(y[0], h@mean+.1, [[variance]]))


def test_all_nonempty_subsets_match_independent_native_gaussian():
    mix = GaussianMixture(np.ones(1), np.array([[20., 19., 18.]]), np.array([[[1., .3, .2], [.3, 2., .1], [.2, .1, 1.]]]))
    h = np.array([[1.05, -.05, 0], [0, 1.02, -.02], [0, -.01, 1.01]])
    b = np.array([.01, -.02, .03]); t = np.diag([.01, .02, .03]); c = np.eye(3)*.04
    native = native_mixture(mix, h, b, t)
    y = np.array([[21., 20., 19.]])
    for bits in itertools.product((False, True), repeat=3):
        obs = np.array(bits)
        if not obs.any(): continue
        expected = multivariate_normal.logpdf(y[0, obs], (h@mix.means[0]+b)[obs], (h@mix.covs[0]@h.T+t+c)[np.ix_(obs, obs)])
        assert native.log_prob(y, c, observed=obs)[0] == pytest.approx(expected, abs=1e-12)


def test_pooled_em_recovers_latent_distribution_from_two_instruments():
    rng = np.random.default_rng(872)
    mu = np.array([1., -1.]); v = np.array([[.7, .2], [.2, .4]])
    x = rng.multivariate_normal(mu, v, 12000); system = np.arange(len(x)) % 2
    ops = {0: (np.eye(2), np.zeros(2), np.zeros((2, 2))), 1: (np.array([[1., .4], [-.2, 1.]]), np.array([.3, -.4]), np.eye(2)*.02)}
    cov = np.tile(np.eye(2)*.03, (len(x), 1, 1)); y = np.empty_like(x)
    for s, (h, b, t) in ops.items():
        rows = system == s
        y[rows] = x[rows]@h.T+b+rng.multivariate_normal(np.zeros(2), t+cov[0], rows.sum())
    obs = np.ones_like(y, bool); obs[::5, 1] = False; y[~obs] = np.nan
    def source():
        for i in range(0, len(y), 1024): yield y[i:i+1024], cov[i:i+1024], obs[i:i+1024], system[i:i+1024]
    result = fit_projected(source, init=GaussianMixture(np.ones(1), np.zeros((1, 2)), np.eye(2)[None]), operators=ops, expected_rows=len(y), max_iter=30, tol=1e-8, regularization=0.)
    np.testing.assert_allclose(result.mixture.means[0], mu, atol=.03)
    np.testing.assert_allclose(result.mixture.covs[0], v, atol=.03)
    assert np.min(np.diff(result.history)) > -1e-10


def test_projected_fit_refuses_lost_rows():
    mix = GaussianMixture(np.ones(1), np.zeros((1, 1)), np.ones((1, 1, 1)))
    def source(): yield np.zeros((2, 1)), np.zeros((2, 1, 1)), np.ones((2, 1), bool), np.zeros(2, int)
    with pytest.raises(ValueError, match='accounting'):
        fit_projected(source, init=mix, operators={0: (np.eye(1), np.zeros(1), np.zeros((1, 1)))}, expected_rows=3, max_iter=1, tol=0., regularization=0.)
