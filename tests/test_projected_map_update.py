"""MAP covariance update for projected XD: monotone objective and correct M step.

The historical additive update adds w I after every M step and maximises no
fixed objective; the conjugate-prior update of Bovy, Hogg & Roweis (2011,
eqs. 19-20) does, so EM must never decrease the reported log posterior.
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.projected_xd import (accumulate_projected, fit_projected,
                                     log_covariance_prior)


def _problem(seed=5, n=900):
    """Two latent populations, one tight direction, missing bands, correlated errors."""
    rng = np.random.default_rng(seed)
    a = rng.multivariate_normal([0., 0., 0.], np.diag([1., .5, .002]), n//3)
    b = rng.multivariate_normal([2., 1., -1.], [[.3, .1, 0], [.1, .2, 0], [0, 0, .004]], n - n//3)
    x = np.vstack([a, b])
    h = np.array([[1., .1, 0.], [0., 1., .05], [0., 0., 1.], [.5, .5, .5]])
    noise_root = rng.normal(size=(n, 4, 4)) * .02
    cov = noise_root @ noise_root.swapaxes(1, 2) + np.eye(4) * 1e-3
    y = x @ h.T + np.einsum('nij,nj->ni', np.linalg.cholesky(cov), rng.normal(size=(n, 4)))
    obs = rng.uniform(size=y.shape) > .3
    obs[:, 2] = True
    y[~obs] = np.nan
    ops = {0: (h, np.zeros(4), np.zeros((4, 4)))}

    def source():
        for i in range(0, n, 97):
            yield y[i:i+97], cov[i:i+97], obs[i:i+97], np.zeros(len(y[i:i+97]), int)

    init = GaussianMixture(np.array([.5, .5]), np.array([[.1, 0, 0], [1.5, .8, -.8]]),
                           np.tile(np.eye(3), (2, 1, 1)))
    return source, init, ops, n


def test_map_update_never_decreases_log_posterior():
    source, init, ops, n = _problem()
    fit = fit_projected(source, init=init, operators=ops, expected_rows=n, max_iter=150,
                        tol=0., regularization=.05, covariance_update='map', prior_strength=3.)
    steps = np.diff(fit.history)
    assert steps.min() > -1e-12 * max(1., abs(fit.history[-1]))
    assert fit.history[-1] > fit.history[0]


def test_additive_update_declines_where_map_does_not():
    """Start at a good fit: adding w I each step must lower the likelihood."""
    source, init, ops, n = _problem()
    warm = fit_projected(source, init=init, operators=ops, expected_rows=n, max_iter=60,
                         tol=0., regularization=1e-6, covariance_update='map').mixture
    options = dict(init=warm, operators=ops, expected_rows=n, max_iter=20, tol=0., regularization=.05)
    additive = fit_projected(source, **options)
    posterior = fit_projected(source, covariance_update='map', **options)
    assert np.diff(additive.history).min() < -1e-3
    assert np.diff(posterior.history).min() > -1e-12


def test_map_m_step_maximises_penalised_surrogate_numerically():
    """Closed form versus direct numerical maximisation over Cholesky factors."""
    source, init, ops, n = _problem(seed=9, n=300)
    count, first, second, _, _ = accumulate_projected(source, init, ops)
    w, nu = .05, 2.
    fit = fit_projected(source, init=init, operators=ops, expected_rows=n, max_iter=1, tol=0.,
                        regularization=w, covariance_update='map', prior_strength=nu)
    d = init.n_dim
    tril = np.tril_indices(d)
    for j in range(init.n_components):
        mean = init.means[j] + first[j]/count[j]
        centred = second[j] - np.outer(first[j], first[j])/count[j]   # S = q c about the new mean

        def negative(theta):
            chol = np.zeros((d, d)); chol[tril] = theta
            v = chol @ chol.T + 1e-12*np.eye(d)
            sign, logdet = np.linalg.slogdet(v)
            inv = np.linalg.inv(v)
            return .5*(count[j]+nu)*logdet + .5*np.trace(inv @ (centred + nu*w*np.eye(d)))

        start = np.linalg.cholesky(np.eye(d))[tril]
        best = minimize(negative, start, method='BFGS', options=dict(gtol=1e-10, maxiter=10000))
        chol = np.zeros((d, d)); chol[tril] = best.x
        np.testing.assert_allclose(fit.mixture.covs[j], chol @ chol.T, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(fit.mixture.means[j], mean, atol=1e-12)


def test_map_reduces_to_maximum_likelihood_as_prior_vanishes():
    source, init, ops, n = _problem(seed=13, n=400)
    options = dict(init=init, operators=ops, expected_rows=n, max_iter=5, tol=0.)
    ml = fit_projected(source, regularization=0., **options)
    tiny = fit_projected(source, regularization=1e-12, prior_strength=1e-9, covariance_update='map', **options)
    np.testing.assert_allclose(tiny.mixture.covs, ml.mixture.covs, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(tiny.mixture.means, ml.mixture.means, atol=1e-10)


def test_log_covariance_prior_matches_direct_formula():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 4, 4)); covs = a @ a.swapaxes(1, 2) + np.eye(4)*.1
    expected = sum(-.5*1.7*(np.linalg.slogdet(v)[1] + .03*np.trace(np.linalg.inv(v))) for v in covs)
    assert log_covariance_prior(covs, .03, 1.7) == pytest.approx(expected, rel=1e-12)


def test_map_stops_on_small_change_of_either_sign_and_restarts_consistently():
    source, init, ops, n = _problem(seed=21, n=300)
    options = dict(init=init, operators=ops, expected_rows=n, max_iter=400, tol=1e-6,
                   regularization=.05, covariance_update='map')
    fit = fit_projected(source, **options)
    assert fit.converged and fit.n_iter < 400
    # A restart whose last recorded change is a roundoff-sized decline is stationary.
    h = (fit.history[-1] + 1e-13, fit.history[-1])
    again = fit_projected(source, initial_history=h, **{**options, 'init': fit.mixture})
    assert again.converged and again.n_iter == 2
    # The additive rule keeps its historical behaviour: a decline is never convergence.
    legacy = {k: v for k, v in options.items() if k not in ('covariance_update', 'max_iter', 'init')}
    assert not fit_projected(source, initial_history=h, max_iter=2, init=fit.mixture, **legacy).converged


def test_map_update_rejects_unusable_settings():
    source, init, ops, n = _problem(n=50)
    with pytest.raises(ValueError, match='prior'):
        fit_projected(source, init=init, operators=ops, expected_rows=n, max_iter=1, tol=0.,
                      regularization=0., covariance_update='map')
    with pytest.raises(ValueError, match='additive or map'):
        fit_projected(source, init=init, operators=ops, expected_rows=n, max_iter=1, tol=0.,
                      regularization=.1, covariance_update='floor')


@pytest.mark.parametrize('tol', [0., 1e-4])
def test_map_checkpoint_resume_reproduces_uninterrupted_fit(tol):
    """Resume exactly as the production task does: saved history plus updated mixture."""
    source, init, ops, n = _problem(seed=31, n=400)
    options = dict(operators=ops, expected_rows=n, tol=tol, regularization=.05, covariance_update='map')
    whole = fit_projected(source, init=init, max_iter=40, **options)
    history, mixture, saved = [], init, []

    def progress(iteration, mix, value, rows):
        history.append(value); saved.append((iteration, mix, list(history)))

    it = 0
    while True:
        fit = fit_projected(source, init=mixture, max_iter=min(it+7, 40), initial_history=tuple(history),
                            progress=progress, **options)
        if saved:
            it, mixture, history = saved[-1][0], saved[-1][1], list(saved[-1][2])
        if fit.converged or it >= 40:
            break
    assert fit.n_iter == whole.n_iter and fit.converged == whole.converged
    np.testing.assert_allclose(fit.history, whole.history, rtol=0, atol=1e-13)
    np.testing.assert_allclose(fit.mixture.covs, whole.mixture.covs, rtol=0, atol=1e-13)
    np.testing.assert_allclose(fit.mixture.means, whole.mixture.means, rtol=0, atol=1e-13)


def test_map_resume_from_last_checkpoint_matches_uninterrupted_stop():
    """A crash after the final checkpoint write resumes to the same stopped model."""
    source, init, ops, n = _problem(seed=31, n=400)
    options = dict(operators=ops, expected_rows=n, tol=1e-4, regularization=.05, covariance_update='map')
    saved = []
    whole = fit_projected(source, init=init, max_iter=40,
                          progress=lambda it, mix, value, rows: saved.append((it, mix)), **options)
    assert whole.converged and whole.n_iter < 40
    it, mix = saved[-1]
    resumed = fit_projected(source, init=mix, max_iter=40, initial_history=tuple(whole.history), **options)
    assert resumed.converged and resumed.n_iter == whole.n_iter == it
    np.testing.assert_array_equal(resumed.mixture.covs, whole.mixture.covs)
    assert resumed.history == whole.history
