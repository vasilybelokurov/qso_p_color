"""QSO colours independent of magnitude: a fixed, uncorrelated magnitude coordinate."""
import numpy as np
import pytest

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import conditional_log_prob
from qso_pcolor.projected_xd import fit_projected

FIX = dict(index=0, mean=20., variance=100.)


def _data(seed=4, n=1500):
    """u = (magnitude, two colours); magnitude drawn independently of colour."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(17, 23, n)
    c = np.where(rng.uniform(size=n)[:, None] < .4, rng.normal([.3, .1], [.1, .05], (n, 2)),
                 rng.normal([-.2, .4], [.08, .12], (n, 2)))
    u = np.column_stack([m, c]); cov = np.tile(np.diag([.01, .002, .002]), (n, 1, 1))
    y = u + np.einsum('nij,nj->ni', np.linalg.cholesky(cov), rng.normal(size=u.shape))
    obs = np.ones_like(y, bool); obs[::7, 2] = False; y[~obs] = np.nan
    return y, cov, obs


def _init(d=3):
    means = np.array([[20., .2, .2], [20., -.1, .3]]); covs = np.tile(np.eye(d)*.05, (2, 1, 1))
    covs[:, 0, 0] = 100.
    return GaussianMixture(np.array([.5, .5]), means, covs)


def _source(y, cov, obs):
    def source():
        for i in range(0, len(y), 211):
            yield y[i:i+211], cov[i:i+211], obs[i:i+211], np.zeros(len(y[i:i+211]), int)
    return source


@pytest.mark.parametrize('update', ['additive', 'map'])
def test_constraint_holds_and_map_objective_never_decreases(update):
    y, cov, obs = _data(); ops = {0: (np.eye(3), np.zeros(3), np.zeros((3, 3)))}
    fit = fit_projected(_source(y, cov, obs), init=_init(), operators=ops, expected_rows=len(y), max_iter=60, tol=0.,
                        regularization=1e-3, covariance_update=update, fixed_coordinate=FIX)
    mix = fit.mixture
    np.testing.assert_allclose(mix.means[:, 0], 20.); np.testing.assert_allclose(mix.covs[:, 0, 0], 100.)
    np.testing.assert_array_equal(mix.covs[:, 0, 1:], 0.); np.testing.assert_array_equal(mix.covs[:, 1:, 0], 0.)
    if update == 'map':
        assert np.diff(fit.history).min() > -1e-12 * max(1., abs(fit.history[-1]))


def test_constrained_colour_block_equals_a_colour_only_fit():
    """Independent route: drop the magnitude entirely and fit the colours alone."""
    y, cov, obs = _data(seed=8)
    full = fit_projected(_source(y, cov, obs), init=_init(), operators={0: (np.eye(3), np.zeros(3), np.zeros((3, 3)))},
                         expected_rows=len(y), max_iter=15, tol=0., regularization=1e-3, covariance_update='map',
                         fixed_coordinate=FIX)
    init = _init(); colour_init = GaussianMixture(init.weights, init.means[:, 1:], init.covs[:, 1:, 1:])
    # Each row's magnitude likelihood is identical for every component, so it cancels in the responsibilities.
    colour = fit_projected(_source(y[:, 1:], cov[:, 1:, 1:], obs[:, 1:]), init=colour_init,
                           operators={0: (np.eye(2), np.zeros(2), np.zeros((2, 2)))}, expected_rows=len(y),
                           max_iter=15, tol=0., regularization=1e-3, covariance_update='map')
    np.testing.assert_allclose(full.mixture.means[:, 1:], colour.mixture.means, atol=1e-10)
    np.testing.assert_allclose(full.mixture.covs[:, 1:, 1:], colour.mixture.covs, atol=1e-10)
    np.testing.assert_allclose(full.mixture.weights, colour.mixture.weights, atol=1e-10)


def test_native_conditional_colour_density_does_not_depend_on_brightness():
    """Magnitude + colours mapped to three native bands, conditioned on band 0."""
    y, cov, obs = _data(seed=2)
    fit = fit_projected(_source(y, cov, obs), init=_init(), operators={0: (np.eye(3), np.zeros(3), np.zeros((3, 3)))},
                        expected_rows=len(y), max_iter=20, tol=0., regularization=1e-3, covariance_update='map',
                        fixed_coordinate=FIX).mixture
    t_inv = np.array([[1., 0, 0], [1, 1, 0], [1, 0, 1]])          # native bands: r, r+c1, r+c2
    native = GaussianMixture(fit.weights, fit.means @ t_inv.T, t_inv @ fit.covs @ t_inv.T)
    x = np.array([[19., 19.3, 19.1]]); noise = np.diag([.01, .01, .01])[None]; obs1 = np.ones((1, 3), bool)
    values = [conditional_log_prob(native, x + d, noise, obs1, 0)[0] for d in (-3., 0., 3.)]
    assert max(values) - min(values) < 1e-3


def test_decoupled_magnitude_keeps_colour_shapes_and_map_objective_monotone():
    """Free per-component magnitude Gaussian, zero magnitude-colour covariance."""
    rng = np.random.default_rng(11); n = 2000
    k = rng.uniform(size=n) < .5
    m = np.where(k, rng.normal(18, .8, n), rng.normal(21, 1., n))         # bright and faint populations
    c = np.where(k[:, None], rng.normal([.4, .2], [.05, .04], (n, 2)), rng.normal([1.2, .9], [.08, .1], (n, 2)))
    y = np.column_stack([m, c]); cov = np.tile(np.diag([.01, .003, .003]), (n, 1, 1))
    y = y + np.einsum('nij,nj->ni', np.linalg.cholesky(cov), rng.normal(size=y.shape)); obs = np.ones_like(y, bool)
    init = GaussianMixture(np.array([.5, .5]), np.array([[19., .5, .3], [20., 1., .8]]), np.tile(np.diag([4., .1, .1]), (2, 1, 1)))
    fit = fit_projected(_source(y, cov, obs), init=init, operators={0: (np.eye(3), np.zeros(3), np.zeros((3, 3)))},
                        expected_rows=n, max_iter=80, tol=0., regularization=1e-4, covariance_update='map', decoupled_coordinate=0)
    mix = fit.mixture
    assert np.abs(mix.covs[:, 0, 1:]).max() == 0 and np.abs(mix.covs[:, 1:, 0]).max() == 0
    assert np.diff(fit.history).min() > -1e-12*max(1., abs(fit.history[-1]))
    o = np.argsort(mix.means[:, 0])                                          # bright component first
    np.testing.assert_allclose(mix.means[o, 0], [18, 21], atol=.1)
    np.testing.assert_allclose(mix.means[o][:, 1:], [[.4, .2], [1.2, .9]], atol=.02)
    # Conditional colour mean of each component does not move with magnitude: shapes fixed, only weights change.
    from qso_pcolor.gaussmix import condition_joint
    for r0 in (17., 22.):
        w, cm = condition_joint(mix, np.array([r0]), np.array([0]))
        np.testing.assert_allclose(cm.means, mix.means[:, 1:], atol=1e-12)
    with pytest.raises(ValueError, match='decoupled'):
        bad = GaussianMixture(init.weights, init.means, init.covs + .01)
        fit_projected(_source(y, cov, obs), init=bad, operators={0: (np.eye(3), np.zeros(3), np.zeros((3, 3)))},
                      expected_rows=n, max_iter=1, tol=0., regularization=1e-4, covariance_update='map', decoupled_coordinate=0)
