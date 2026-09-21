"""Independent checks of the Figure 10 display-only coordinate change."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal

spec = importlib.util.spec_from_file_location(
    "optical_coordinates", Path(__file__).parents[1]/"scripts/redraw_optical_examples.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def test_luptitude_density_jacobian_normalises_against_quadrature():
    ref, softening = 2., np.array([.12, .2, .3])
    mean, covariance = np.array([.3, 1.2]), np.array([[.04, .025], [.025, .09]])
    # Independent ratio-space Gaussian; integrate its transformed density in
    # colour space. Bounds enclose eight marginal standard deviations.
    sigma = np.sqrt(np.diag(covariance))
    edges = plot.ratio_to_colour(np.array([mean-8*sigma, mean+8*sigma]), ref, softening)
    lower, upper = edges.min(axis=0), edges.max(axis=0)

    def density(c2, c1):
        ratios, log_jacobian = plot.colour_to_ratio(np.array([c1, c2]), ref, softening)
        return np.exp(multivariate_normal.logpdf(ratios, mean, covariance)+log_jacobian)

    integral, error = dblquad(density, lower[0], upper[0],
                              lambda _: lower[1], lambda _: upper[1], epsabs=2e-7, epsrel=2e-7)
    assert integral == pytest.approx(1., abs=1e-6)
    assert error < 1e-6


def test_ratio_colour_map_and_jacobian_include_negative_fluxes():
    ref, softening = 1.7, np.array([.12, .2, .3])
    for ratios in (np.array([-.3, 2.1]), np.array([2.3, -.1]), np.zeros(2)):
        colour = plot.ratio_to_colour(ratios, ref, softening)
        recovered, log_jacobian = plot.colour_to_ratio(colour, ref, softening)
        np.testing.assert_allclose(recovered, ratios, atol=1e-14)
        # Differentiate the forward mapping numerically, independently of the
        # closed-form inverse Jacobian used to transform model densities.
        eps = 1e-6
        numerical = np.column_stack([(plot.ratio_to_colour(ratios+step, ref, softening)
                                      -plot.ratio_to_colour(ratios-step, ref, softening))/(2*eps)
                                     for step in eps*np.eye(2)])
        assert log_jacobian == pytest.approx(-np.log(abs(np.linalg.det(numerical))), abs=1e-8)
