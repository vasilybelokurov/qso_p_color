"""Checks of the Gaussian-mixture algebra against brute-force references.

Every test here compares an analytic result against an independent numerical
route (scipy's multivariate normal, quadrature, or Monte Carlo), so a sign error
or a transposed matrix cannot pass.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import multivariate_normal

from qso_pcolor.gaussmix import (
    GaussianMixture,
    assert_positive_definite,
    condition_joint,
    log_gauss_batch,
)


def random_mixture(rng, k=3, d=2):
    weights = rng.dirichlet(np.ones(k))
    means = rng.normal(scale=2.0, size=(k, d))
    covs = np.empty((k, d, d))
    for j in range(k):
        a = rng.normal(size=(d, d))
        covs[j] = a @ a.T + 0.5 * np.eye(d)
    return GaussianMixture(weights, means, covs)


def test_log_gauss_batch_matches_scipy():
    rng = np.random.default_rng(1)
    d = 4
    a = rng.normal(size=(d, d))
    cov = a @ a.T + np.eye(d)
    mean = rng.normal(size=d)
    x = rng.normal(size=(7, d))
    got = log_gauss_batch(x, mean, cov)
    want = multivariate_normal(mean, cov).logpdf(x)
    assert np.allclose(got, want, atol=1e-12)


def test_log_gauss_batch_broadcasts_over_components():
    rng = np.random.default_rng(2)
    n, k, d = 5, 3, 3
    x = rng.normal(size=(n, d))
    means = rng.normal(size=(k, d))
    covs = np.stack([np.eye(d) * (j + 1.0) for j in range(k)])
    got = log_gauss_batch(x[:, None, :], means[None, :, :], covs[None, :, :, :])
    assert got.shape == (n, k)
    for i in range(n):
        for j in range(k):
            want = multivariate_normal(means[j], covs[j]).logpdf(x[i])
            assert got[i, j] == pytest.approx(want, abs=1e-12)


def test_mixture_log_prob_equals_explicit_sum():
    rng = np.random.default_rng(3)
    mix = random_mixture(rng, k=4, d=3)
    x = rng.normal(size=(6, 3))
    got = mix.log_prob(x)
    want = np.log(
        sum(
            mix.weights[j] * multivariate_normal(mix.means[j], mix.covs[j]).pdf(x)
            for j in range(mix.n_components)
        )
    )
    assert np.allclose(got, want, atol=1e-12)


def test_mixture_integrates_to_one():
    rng = np.random.default_rng(4)
    mix = random_mixture(rng, k=3, d=1)
    val, _ = integrate.quad(
        lambda t: float(np.exp(mix.log_prob(np.array([[t]])))[0]), -60, 60, limit=400
    )
    assert val == pytest.approx(1.0, abs=1e-6)


def test_error_convolution_matches_numerical_convolution():
    """Evaluating with V_k + S must equal convolving the density with the noise."""
    rng = np.random.default_rng(5)
    mix = random_mixture(rng, k=3, d=1)
    s2 = 0.7
    x0 = 0.35
    analytic = float(np.exp(mix.log_prob(np.array([[x0]]), np.array([[[s2]]])))[0])
    numeric, _ = integrate.quad(
        lambda t: float(np.exp(mix.log_prob(np.array([[t]])))[0])
        * np.exp(-0.5 * (x0 - t) ** 2 / s2)
        / np.sqrt(2 * np.pi * s2),
        -60,
        60,
        limit=400,
    )
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_missing_dimension_marginalises_exactly():
    """Masking a dimension must equal integrating it out numerically."""
    rng = np.random.default_rng(6)
    mix = random_mixture(rng, k=3, d=2)
    x = np.array([[1.1, np.nan]])
    observed = np.array([[True, False]])
    masked = float(np.exp(mix.log_prob(x, observed=observed))[0])

    numeric, _ = integrate.quad(
        lambda t: float(np.exp(mix.log_prob(np.array([[1.1, t]])))[0]), -60, 60, limit=400
    )
    assert masked == pytest.approx(numeric, rel=1e-8)

    # And it must equal the explicit marginal mixture.
    assert masked == pytest.approx(
        float(np.exp(mix.marginal(np.array([0])).log_prob(np.array([[1.1]])))[0]),
        rel=1e-12,
    )


def test_missing_dimension_with_noise():
    """Marginalisation and error convolution must compose correctly."""
    rng = np.random.default_rng(7)
    mix = random_mixture(rng, k=3, d=3)
    s = np.diag([0.3, 5.0, 0.2])  # the huge middle entry is for the masked band
    x = np.array([[0.4, np.nan, -0.9]])
    observed = np.array([[True, False, True]])
    got = mix.log_prob(x, s[None], observed=observed)

    sub = np.array([0, 2])
    want = mix.marginal(sub).log_prob(x[:, sub], s[np.ix_(sub, sub)][None])
    assert got == pytest.approx(want, abs=1e-12)


def test_log_prob_ignores_values_in_unobserved_slots():
    rng = np.random.default_rng(8)
    mix = random_mixture(rng, k=2, d=2)
    observed = np.array([[True, False]])
    a = mix.log_prob(np.array([[0.5, np.nan]]), observed=observed)
    b = mix.log_prob(np.array([[0.5, 1e6]]), observed=observed)
    assert a == pytest.approx(b, abs=1e-12)


def test_mixed_observation_patterns_in_one_call():
    """Grouping by mask pattern must not scramble the row order."""
    rng = np.random.default_rng(9)
    mix = random_mixture(rng, k=3, d=3)
    x = rng.normal(size=(20, 3))
    observed = rng.random((20, 3)) > 0.3
    observed[:, 0] = True  # keep at least one dimension everywhere
    got = mix.log_prob(x, observed=observed)
    want = np.array(
        [mix.log_prob(x[i : i + 1], observed=observed[i : i + 1])[0] for i in range(20)]
    )
    assert np.allclose(got, want, atol=1e-12)


def test_condition_joint_matches_ratio_of_densities():
    """p(a|b) from condition_joint must equal p(a,b)/p(b) pointwise."""
    rng = np.random.default_rng(10)
    mix = random_mixture(rng, k=4, d=3)  # dims 0,1 = "colors", dim 2 = "redshift"
    y = np.array([0.6])
    _, cond = condition_joint(mix, y, np.array([2]))

    a = np.array([[0.2, -0.4]])
    joint = float(np.exp(mix.log_prob(np.array([[0.2, -0.4, 0.6]])))[0])
    marg_b = float(np.exp(mix.marginal(np.array([2])).log_prob(y[None, :]))[0])
    assert float(np.exp(cond.log_prob(a))[0]) == pytest.approx(joint / marg_b, rel=1e-10)


def test_condition_joint_with_noisy_conditioner():
    """A conditioning uncertainty must act as a convolution in the conditioner."""
    rng = np.random.default_rng(11)
    mix = random_mixture(rng, k=3, d=2)
    y = np.array([0.4])
    sy = np.array([[0.25]])
    _, cond = condition_joint(mix, y, np.array([1]), y_var=sy)

    # Reference: p(a | y) = int dz p(a, z) N(y | z, sy) / int dz p(z) N(y | z, sy)
    def num(t):
        return integrate.quad(
            lambda z: float(np.exp(mix.log_prob(np.array([[t, z]])))[0])
            * np.exp(-0.5 * (y[0] - z) ** 2 / sy[0, 0]) / np.sqrt(2 * np.pi * sy[0, 0]),
            -60, 60, limit=400,
        )[0]

    den = integrate.quad(
        lambda z: float(np.exp(mix.marginal(np.array([1])).log_prob(np.array([[z]])))[0])
        * np.exp(-0.5 * (y[0] - z) ** 2 / sy[0, 0]) / np.sqrt(2 * np.pi * sy[0, 0]),
        -60, 60, limit=400,
    )[0]

    for t in (-0.5, 0.0, 1.3):
        assert float(np.exp(cond.log_prob(np.array([[t]])))[0]) == pytest.approx(
            num(t) / den, rel=1e-6
        )


def test_mixture_with_is_a_normalised_density():
    rng = np.random.default_rng(12)
    a = random_mixture(rng, k=2, d=1)
    b = random_mixture(rng, k=3, d=1)
    c = a.mixture_with(b, 0.3)
    assert c.weights.sum() == pytest.approx(1.0)
    val, _ = integrate.quad(
        lambda t: float(np.exp(c.log_prob(np.array([[t]])))[0]), -80, 80, limit=400
    )
    assert val == pytest.approx(1.0, abs=1e-6)
    # And it really is the stated convex combination.
    x = np.array([[0.7]])
    assert float(np.exp(c.log_prob(x))[0]) == pytest.approx(
        0.3 * float(np.exp(a.log_prob(x))[0]) + 0.7 * float(np.exp(b.log_prob(x))[0]),
        rel=1e-12,
    )


def test_responsibilities_sum_to_one_and_match_bayes():
    rng = np.random.default_rng(13)
    mix = random_mixture(rng, k=3, d=2)
    x = rng.normal(size=(5, 2))
    r = mix.responsibilities(x)
    assert np.allclose(r.sum(axis=1), 1.0)
    j = 1
    num = mix.weights[j] * multivariate_normal(mix.means[j], mix.covs[j]).pdf(x)
    den = np.exp(mix.log_prob(x))
    assert np.allclose(r[:, j], num / den, atol=1e-12)


def test_serialisation_round_trip_is_exact():
    rng = np.random.default_rng(14)
    mix = random_mixture(rng, k=3, d=3)
    back = GaussianMixture.from_dict(mix.to_dict())
    x = rng.normal(size=(10, 3))
    assert np.allclose(mix.log_prob(x), back.log_prob(x), atol=1e-12)
    assert_positive_definite(back.covs)


def test_non_positive_definite_is_rejected():
    bad = np.array([[[1.0, 2.0], [2.0, 1.0]]])  # eigenvalues -1, 3
    with pytest.raises(ValueError):
        assert_positive_definite(bad)


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        GaussianMixture(np.array([0.5, 0.4]), np.zeros((2, 2)), np.stack([np.eye(2)] * 2))
