"""Scientific checks of the survey-comparison figure projections."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import multivariate_normal, norm

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.qso_model import SlicedColourRedshiftModel

spec = importlib.util.spec_from_file_location(
    "multisurvey_examples", Path(__file__).parents[1]/"scripts/make_multisurvey_examples.py")
examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(examples)


def test_figure_projection_matches_independent_joint_density_ratio_and_slice_interpolation():
    # Different reference distributions in each slice catch conditioning a
    # pooled joint mixture instead of interpolating conditional densities.
    labels = ("sdss:r", "sdss:g", "sdss:z", "sdss:i")
    means = np.array([[19., 20., 18.5, 21.], [21., 21.5, 20., 20.]])
    covariance = np.array([[.7, .2, -.1, .05], [.2, .8, .15, -.1],
                           [-.1, .15, .6, .1], [.05, -.1, .1, .9]])
    mixtures = [GaussianMixture(np.array([.3, .7]), means+offset,
                                np.array([covariance, covariance*1.6]), labels)
                for offset in (0., 1.4)]
    model = SimpleNamespace(qso=SlicedColourRedshiftModel(
        np.array([1., 2.]), mixtures, np.array([100, 100]), "test", labels))
    # Reverse the two plotted bands and include correlated measurement noise.
    plane = (labels[0], labels[2], labels[1])
    noise = np.array([[.1, .025, -.01], [.025, .2, .03], [-.01, .03, .15]])
    anchor, z = 20.2, 1.3
    projection = examples.qso_projection(model, z, plane, anchor, noise)
    colours = np.array([[-1., .8], [.5, 2.], [1.2, -1.3]])
    expected = np.zeros(len(colours))
    indices = [0, 2, 1]
    for mix, fraction in zip(mixtures, (.7, .3)):
        numerator = np.zeros(len(colours)); denominator = 0.
        for weight, mean, cov in zip(mix.weights, mix.means, mix.covs):
            cov3 = cov[np.ix_(indices, indices)] + noise
            values = np.column_stack([np.full(len(colours), anchor), colours+anchor])
            numerator += weight*multivariate_normal.pdf(values, mean[indices], cov3)
            denominator += weight*norm.pdf(anchor, mean[0], np.sqrt(cov3[0, 0]))
        expected += fraction*numerator/denominator
    np.testing.assert_allclose(np.exp(projection.log_prob(colours)), expected, rtol=1e-12)
    assert np.isclose(projection.weights.sum(), 1.)


def test_background_route_uses_unplotted_observed_bands():
    # Identical plotted grz colours must use different field fits when the
    # actual scored input also contains infrared data.
    labels = ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z", "allwise:w1")
    joint = GaussianMixture(np.ones(1), np.zeros((1, 4)), np.eye(4)[None], labels)
    marginal = GaussianMixture(np.ones(1), np.ones((1, 3)), np.eye(3)[None], labels[:3])
    model = SimpleNamespace(background=joint, background_marginals=(marginal,),
                            transform=SimpleNamespace(bands=labels))
    assert examples.background_for_observed(model, np.array([1, 1, 1, 0], bool)) is marginal
    assert examples.background_for_observed(model, np.ones(4, bool)) is joint
