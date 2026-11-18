"""Frozen extension of a latent mixture by new bands (docs/euclid/PLAN.md, T-U2, T-U3)."""
import numpy as np
import pytest

from qso_pcolor.euclid_extension import Extension, extend_mixture, fit_extension, regressor_matrix
from qso_pcolor.gaussmix import GaussianMixture

LABELS = ('g', 'r', 'z', 'w1')


def _parent(rng):
    k, d = 3, len(LABELS)
    means = np.array([[20.5, 20., 19.8, 19.5], [21.5, 20.5, 20.0, 18.0], [19.0, 18.2, 17.8, 17.9]])
    covs = []
    for _ in range(k):
        a = rng.normal(size=(d, d))*0.15
        covs.append(a @ a.T + 0.02*np.eye(d) + 0.5*np.ones((d, d)))     # strong magnitude correlation
    return GaussianMixture(np.array([0.5, 0.3, 0.2]), means, np.array(covs), LABELS)


def _truth(t_d):
    k, m, p = 3, 2, t_d.shape[0]
    a = np.array([[0.3, -0.2], [1.0, 0.4], [-0.5, -0.6]])
    b = np.tile(np.array([[0.8, 0.1], [0.3, 0.9]]), (k, 1, 1))[:, :m, :p]
    w = np.tile(np.array([[0.02, 0.005], [0.005, 0.03]]), (k, 1, 1))
    return Extension(a, b, w)


def test_old_block_is_copied_exactly():
    rng = np.random.default_rng(1)
    mix = _parent(rng)
    t_d = regressor_matrix(LABELS, ('g', 'z'), 'r', include_magnitude=False)
    ext = _truth(t_d)
    big = extend_mixture(mix, ext, t_d, LABELS.index('r'), ('e1', 'e2'))
    assert np.array_equal(big.means[:, :4], mix.means)
    assert np.array_equal(big.covs[:, :4, :4], mix.covs)
    assert np.array_equal(big.weights, mix.weights)
    np.linalg.cholesky(big.covs)                                            # positive definite


def test_new_block_has_the_declared_conditional():
    rng = np.random.default_rng(2)
    mix = _parent(rng)
    t_d = regressor_matrix(LABELS, ('g', 'z'), 'r', include_magnitude=False)
    ext = _truth(t_d); r = LABELS.index('r')
    big = extend_mixture(mix, ext, t_d, r, ('e1', 'e2'))
    x = big.sample(400_000, rng)
    comp = 0
    # colour c = x_E - x_r regressed on d within component 0 (sample that component alone)
    one = GaussianMixture(np.array([1.]), big.means[[comp]], big.covs[[comp]])
    s = one.sample(400_000, rng)
    c = s[:, 4:] - s[:, [r]]; dd = s[:, :4] @ t_d.T
    u = np.column_stack([np.ones(len(dd)), dd - mix.means[comp] @ t_d.T])
    coef, *_ = np.linalg.lstsq(u, c, rcond=None)
    assert np.allclose(coef[0], ext.a[comp], atol=0.01)
    assert np.allclose(coef[1:].T, ext.b[comp], atol=0.01)
    res = c - u @ coef
    assert np.allclose(np.cov(res.T), ext.w[comp], atol=2e-3)


@pytest.mark.parametrize('kappa_b', [np.inf, 50.])
def test_em_recovers_truth_with_noise_and_missing_bands(kappa_b):
    rng = np.random.default_rng(3)
    mix = _parent(rng)
    t_d = regressor_matrix(LABELS, ('g', 'z'), 'r', include_magnitude=False)
    truth = _truth(t_d); r = LABELS.index('r')
    big = extend_mixture(mix, truth, t_d, r, ('e1', 'e2'))
    n = 30_000
    x = big.sample(n, rng)
    noise = rng.uniform(0.0005, 0.02, size=(n, 6))                          # variances, mag^2
    y = x + rng.normal(size=x.shape)*np.sqrt(noise)
    observed = rng.random((n, 6)) > 0.15
    observed[:, 4:] |= rng.random((n, 1)) > 0.3                              # new bands mostly present
    observed[~observed.any(1), 1] = True
    y[~observed] = np.nan; noise[~observed] = np.inf
    h = np.eye(6); off = np.zeros(6); extra = np.zeros((6, 6))
    groups = [np.arange(3)]
    ext, hist = fit_extension(mix, y, noise, observed, h=h, offset=off, extra=extra, t_d=t_d, r_index=r,
                              new_labels=('e1', 'e2'), groups=groups, kappa_a=0., kappa_b=kappa_b, kappa_w=0.,
                              w_floor=1e-6, n_iter=200, tol=1e-7)
    if np.isinf(kappa_b):                                                     # exact EM: monotone
        assert all(b >= a - 1e-7 for a, b in zip(hist, hist[1:]))
    assert np.allclose(ext.a, truth.a, atol=0.02)
    assert np.allclose(ext.b, truth.b, atol=0.06)
    assert np.allclose(ext.w, truth.w, atol=0.006)
