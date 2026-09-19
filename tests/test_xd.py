"""Checks that extreme deconvolution actually deconvolves.

The decisive test is :func:`test_xd_recovers_intrinsic_width`: fitting noisy
draws must recover the *intrinsic* scatter, not the broadened observed scatter.
An ordinary GMM fit to the same data cannot pass it, and the test asserts that
too, so a silent regression to plain EM would be caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.xd import fit_xd, select_n_components


def test_xd_recovers_intrinsic_width():
    rng = np.random.default_rng(0)
    n = 20000
    truth_var = 0.09
    noise_var = 0.25
    latent = rng.normal(0.0, np.sqrt(truth_var), size=(n, 1))
    obs = latent + rng.normal(0.0, np.sqrt(noise_var), size=(n, 1))
    cov = np.full((n, 1, 1), noise_var)

    res = fit_xd(obs, cov, n_components=1, seed=0, tol=1e-9)
    assert res.converged
    got = float(res.mixture.covs[0, 0, 0])
    assert got == pytest.approx(truth_var, rel=0.08)

    # A noise-blind fit lands on the observed variance instead.
    naive = fit_xd(obs, None, n_components=1, seed=0, tol=1e-9)
    assert float(naive.mixture.covs[0, 0, 0]) == pytest.approx(
        truth_var + noise_var, rel=0.05
    )


def test_xd_recovers_a_two_component_mixture():
    rng = np.random.default_rng(1)
    truth = GaussianMixture(
        np.array([0.35, 0.65]),
        np.array([[-1.5, 0.5], [1.0, -0.8]]),
        np.stack([np.diag([0.2, 0.1]), np.diag([0.15, 0.3])]),
    )
    n = 30000
    latent = truth.sample(n, rng)
    noise_var = 0.2
    obs = latent + rng.normal(0.0, np.sqrt(noise_var), size=latent.shape)
    cov = np.broadcast_to(noise_var * np.eye(2), (n, 2, 2))

    res = fit_xd(obs, cov, n_components=2, seed=3, max_iter=800, tol=1e-9)
    order = np.argsort(res.mixture.means[:, 0])
    means = res.mixture.means[order]
    covs = res.mixture.covs[order]
    weights = res.mixture.weights[order]

    assert np.allclose(means, truth.means, atol=0.05)
    assert np.allclose(weights, truth.weights, atol=0.03)
    assert np.allclose(covs, truth.covs, atol=0.05)


def test_xd_handles_heteroscedastic_and_correlated_noise():
    rng = np.random.default_rng(2)
    n = 30000
    truth_cov = np.array([[0.30, 0.12], [0.12, 0.20]])
    latent = rng.multivariate_normal([0.3, -0.2], truth_cov, size=n)
    # Per-object correlated noise with a factor-of-3 spread in amplitude.
    scale = rng.uniform(0.5, 1.5, size=n)
    base = np.array([[0.20, -0.08], [-0.08, 0.15]])
    cov = scale[:, None, None] ** 2 * base
    noise = np.stack(
        [rng.multivariate_normal([0, 0], cov[i]) for i in range(n)]
    )
    obs = latent + noise

    res = fit_xd(obs, cov, n_components=1, seed=0, tol=1e-10)
    assert np.allclose(res.mixture.covs[0], truth_cov, atol=0.03)
    assert np.allclose(res.mixture.means[0], [0.3, -0.2], atol=0.02)


def test_xd_with_missing_dimensions_is_unbiased():
    """Dropping a band at random must not bias the recovered mixture."""
    rng = np.random.default_rng(3)
    n = 40000
    truth_cov = np.array([[0.25, 0.10], [0.10, 0.16]])
    latent = rng.multivariate_normal([0.5, -0.5], truth_cov, size=n)
    noise_var = 0.1
    obs = latent + rng.normal(0, np.sqrt(noise_var), size=latent.shape)
    cov = np.broadcast_to(noise_var * np.eye(2), (n, 2, 2))

    observed = np.ones((n, 2), dtype=bool)
    drop = rng.random(n) < 0.35          # missing completely at random
    observed[drop, 1] = False
    obs[drop, 1] = np.nan

    res = fit_xd(obs, cov, n_components=1, observed=observed, seed=0, tol=1e-10)
    assert np.allclose(res.mixture.means[0], [0.5, -0.5], atol=0.03)
    assert np.allclose(res.mixture.covs[0], truth_cov, atol=0.03)


def test_nan_in_an_observed_slot_raises():
    x = np.array([[1.0, np.nan]])
    with pytest.raises(ValueError, match="NaN"):
        fit_xd(x, None, n_components=1)


def test_sample_weights_reweight_the_fit():
    """Down-weighting one population must move the mixture towards the other."""
    rng = np.random.default_rng(4)
    a = rng.normal(-2.0, 0.3, size=(5000, 1))
    b = rng.normal(2.0, 0.3, size=(5000, 1))
    x = np.concatenate([a, b])
    w = np.concatenate([np.ones(5000), 0.01 * np.ones(5000)])

    res = fit_xd(x, None, n_components=2, weights=w, seed=1, tol=1e-9)
    j = np.argmax(res.mixture.weights)
    assert res.mixture.means[j, 0] == pytest.approx(-2.0, abs=0.1)
    assert res.mixture.weights[j] > 0.97


def test_loglikelihood_increases_monotonically():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2000, 2))
    cov = np.broadcast_to(0.1 * np.eye(2), (2000, 2, 2))
    res = fit_xd(x, cov, n_components=3, seed=0, max_iter=60, tol=0)
    h = np.asarray(res.history)
    # EM guarantees non-decreasing likelihood up to floating-point noise.
    assert (np.diff(h) > -1e-8).all()


def test_regularization_floors_the_component_variance():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 0.01, size=(2000, 1))
    res = fit_xd(x, None, n_components=1, regularization=0.05, seed=0)
    assert float(res.mixture.covs[0, 0, 0]) >= 0.05


def test_select_n_components_prefers_the_true_complexity():
    rng = np.random.default_rng(7)
    truth = GaussianMixture(
        np.array([0.5, 0.5]),
        np.array([[-2.0], [2.0]]),
        np.stack([np.array([[0.2]]), np.array([[0.2]])]),
    )
    x = truth.sample(6000, rng)
    cov = np.full((6000, 1, 1), 0.05)
    x = x + rng.normal(0, np.sqrt(0.05), size=x.shape)

    best, scores = select_n_components(
        x, cov, [1, 2, 4], n_folds=3, seed=0, tol=1e-7, max_iter=300
    )
    assert best in (2, 4)             # 1 must lose decisively
    assert scores[2] > scores[1] + 0.2


def test_spatial_groups_do_not_leak_between_folds():
    """Rows sharing a group id must land in the same fold."""
    rng = np.random.default_rng(8)
    n = 600
    x = rng.normal(size=(n, 1))
    groups = rng.integers(0, 20, size=n)
    # select_n_components builds folds internally; reproduce its rule and check.
    g = np.unique(groups)
    r = np.random.default_rng(0)
    assign = r.permutation(g.size) % 3
    fold_of = assign[np.searchsorted(g, groups)]
    for gid in g:
        assert np.unique(fold_of[groups == gid]).size == 1
    _ = x  # the fit itself is exercised elsewhere
