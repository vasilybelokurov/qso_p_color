"""Independent checks of the colour marginals shown against redshift."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.qso_model import SlicedColourRedshiftModel

spec = importlib.util.spec_from_file_location(
    "colour_redshift_plot", Path(__file__).parents[1]/"scripts/plot_colour_redshift.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_projected_colour_density_matches_independent_brightness_integral():
    weights = np.array([.4, .6])
    means = np.array([[20., 19., 21.], [22., 20., 18.]])
    covariance = np.array([[.8, .65, .1], [.65, .9, .2], [.1, .2, 1.1]])
    covs = np.array([covariance, 1.4*covariance])
    mix = GaussianMixture(weights, means, covs, ("sdss:g", "sdss:r", "allwise:w1"))
    result = plot.project_colour(mix, ("sdss:g", "sdss:r"))
    # Integrate p(g=c+r, r) over reference brightness; the third band is
    # marginalised by the independent scipy bivariate Gaussian.
    for colour in (-.5, 1., 2., 3.5):
        def integrand(reference):
            return sum(w*multivariate_normal.pdf([colour+reference, reference], mu[:2], cov[:2, :2])
                       for w, mu, cov in zip(weights, means, covs))
        expected = quad(integrand, 5., 35., epsabs=1e-11)[0]
        assert np.exp(result.log_prob([[colour]])[0]) == pytest.approx(expected, rel=1e-9)
    assert quad(lambda c: np.exp(result.log_prob([[c]])[0]), -np.inf, np.inf)[0] == pytest.approx(1., abs=1e-9)


def test_interpolated_colour_quantiles_use_density_mixture_not_interpolated_means():
    labels = ("ps1:g", "ps1:r")
    mixtures = [GaussianMixture(np.ones(1), np.array([[20.+offset, 20.]]),
                                np.array([[[.6, .4], [.4, .6]]]), labels)
                for offset in (0., 3.)]
    model = SlicedColourRedshiftModel(np.array([1., 2.]), mixtures, np.array([100, 100]), "test", labels)
    projected = plot.colour_model(model, labels)
    interpolated = plot.mixture_at_z(projected, 1.25)
    sigma = np.sqrt(.4)
    for level in (.16, .5, .84):
        colour = plot.colour_quantile(interpolated, level)
        expected_cdf = .75*norm.cdf(colour, 0., sigma)+.25*norm.cdf(colour, 3., sigma)
        assert expected_cdf == pytest.approx(level, abs=1e-10)
    for colour in (-1., .5, 2., 3.):
        expected = .75*norm.pdf(colour, 0., sigma)+.25*norm.pdf(colour, 3., sigma)
        actual = np.exp(projected.log_p_colour_given_z([[colour]], None, [1.25])[0, 0])
        assert actual == pytest.approx(expected, rel=1e-12)
