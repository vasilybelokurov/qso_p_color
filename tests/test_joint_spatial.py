"""Spatial weights must survive every observed-band marginal unchanged."""
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from qso_pcolor.background import galactic_healpix
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.joint_spatial import (JointSpatialWeights, fit_joint_spatial_weights,
                                     fit_local_joint_weights)
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.qso_model import RedshiftMatch
from test_multisurvey import synthetic_model


def model_with_sky():
    model = synthetic_model()
    labels = model.transform.bands
    d = len(labels)
    means = np.stack([np.full(d, 21.), np.full(d, 22.)])
    means[1, 1] = 20.
    cov = np.tile((np.eye(d) * .2 + .3)[None], (2, 1, 1))
    model.background = GaussianMixture(np.array([.5, .5]), means, cov, labels)
    pixels = galactic_healpix(np.array([0., 180.]), np.array([45., 45.]), 4)
    model.spatial_background = JointSpatialWeights(4, 2, 10., np.array([.5, .5]),
        cells={int(pixels[0]): [.9, .1], int(pixels[1]): [.1, .9]},
        counts={int(p): 100. for p in pixels})
    return model


def test_public_spatial_scores_are_marginals_of_the_same_position_dependent_mixture(tmp_path):
    model = model_with_sky()
    # Choose u,g: the two populations differ in their conditional g colour.
    phot = Photometry([[3., 4.], [3., 4.]], [[.01, .02], [.01, .02]], ("sdss:u", "sdss:g"))
    l, b = np.array([0., 180.]), np.array([45., 45.])
    scores = model.score(phot, z_primary=1.5, match=RedshiftMatch(half_width_kms=2000.),
                         l_deg=l, b_deg=b)
    f = model.transform(phot)
    weights, _ = model.spatial_background.evaluate(l, b)
    for i, score in enumerate(scores):
        total = model.background.covs[:, :2, :2] + f.cov[i, :2, :2]
        joint = np.array([multivariate_normal.logpdf(f.x[i, :2], mu[:2], v)
                         for mu, v in zip(model.background.means, total)])
        ref = np.array([multivariate_normal.logpdf(f.x[i, 0], mu[0], v[0, 0])
                       for mu, v in zip(model.background.means, total)])
        expected = logsumexp(np.log(weights[i]) + joint) - logsumexp(np.log(weights[i]) + ref)
        assert score.loglike_bkg == pytest.approx(expected)
        assert score.background_nside == 4 and score.background_model_mode == "spatial"
    assert abs(scores[0].loglike_bkg - scores[1].loglike_bkg) > .1
    path = tmp_path / "spatial.json"
    model.save(path)
    again = MultiSurveyModel.load(path).score(phot, z_primary=1.5,
        match=RedshiftMatch(half_width_kms=2000.), l_deg=l, b_deg=b)
    assert [s.loglike_bkg for s in scores] == [s.loglike_bkg for s in again]
    with pytest.raises(ValueError, match="Galactic coordinates"):
        model.background_log_prob(f.x, f.cov, f.observed, 0)
    singles = model.score(phot.keep_bands(("sdss:g",)), z_primary=1.5,
        match=RedshiftMatch(half_width_kms=2000.), l_deg=l, b_deg=b)
    assert [s.loglike_bkg for s in singles] == [0., 0.]


def test_spatial_training_recovers_weights_without_changing_component_shapes():
    rng = np.random.default_rng(4151)
    model = model_with_sky()
    mixture = model.background
    d = mixture.n_dim
    # Widely separated components make this a direct check against known counts.
    mixture = GaussianMixture(mixture.weights, np.stack([np.zeros(d), np.full(d, 8.)]),
                               np.tile(np.eye(d)[None], (2, 1, 1)), mixture.labels)
    assignments = np.r_[np.zeros(450, int), np.ones(50, int),
                        np.zeros(50, int), np.ones(450, int)]
    x = mixture.means[assignments] + rng.normal(size=(1000, d))
    observed = np.zeros_like(x, bool)
    observed[:, [0, 6, 16]] = True
    covariance = np.zeros((1000, d, d))
    sky = fit_joint_spatial_weights(mixture, x, covariance, observed,
        np.repeat([0., 180.], 500), np.full(1000, 45.), np.ones(1000),
        nside=4, nside_parent=2, n0=0., max_iter=100, tol=1e-9, meta={})
    recovered, _ = sky.evaluate([0., 180.], [45., 45.])
    assert np.allclose(recovered, [[.9, .1], [.1, .9]], atol=1e-5)
    for _, fitted, _ in sky.groups(mixture, [0., 180.], [45., 45.]):
        assert np.array_equal(fitted.means, mixture.means)
        assert np.array_equal(fitted.covs, mixture.covs)
    local = fit_local_joint_weights(mixture, sky, x[500:], covariance[500:], observed[500:],
        np.ones(500), l_deg=0., b_deg=45., radius_deg=.5, n0=0., max_iter=100,
        tol=1e-9, meta={})
    assert np.allclose(local.evaluate([0.], [45.])[0], [[.1, .9]], atol=1e-5)
    with pytest.raises(ValueError, match="outside"):
        local.evaluate([180.], [45.])


def test_parent_and_global_fallback_are_distinct():
    sky = JointSpatialWeights(4, 2, 10., np.array([.5, .5]),
        parents={0: [.8, .2]}, parent_counts={0: 10.})
    import healpy as hp
    l, b = hp.pix2ang(4, [0, 8], nest=True, lonlat=True)
    weights, share = sky.evaluate(l, b)
    assert np.allclose(weights, [[.65, .35], [.5, .5]])
    assert np.array_equal(share, [.5, 0.])


def test_softmax_gate_recovers_sky_trend_and_tabulates_for_the_scorer():
    from qso_pcolor.gaussmix import GaussianMixture
    from qso_pcolor.joint_spatial import fit_softmax_gate, gate_weights, gate_to_spatial_weights
    rng = np.random.default_rng(3); n = 6000
    l = rng.uniform(0, 360, n); b = rng.choice([-1, 1], n) * rng.uniform(25, 90, n)
    true = 1 / (1 + np.exp(-(1/np.sin(np.deg2rad(np.abs(b))) - 1.6) * 3))   # share of component 0 rises toward the plane
    comp = (rng.uniform(size=n) > true).astype(int)
    x = np.where(comp == 0, rng.normal(0, 1, n), rng.normal(4, 1, n))
    lp = np.column_stack([-.5*x**2, -.5*(x - 4)**2]) - .5*np.log(2*np.pi)
    gate = fit_softmax_gate(lp, l, b, np.array([.5, .5]), ridge=1e-3, max_iter=500)
    w = gate_weights(gate, l, b)
    assert np.corrcoef(w[:, 0], true)[0, 1] > .95
    s = gate_to_spatial_weights(gate, nside=16, min_abs_b_deg=25., meta={})
    tab = s.evaluate(l, b)[0]
    assert np.abs(tab - w).max() < .1 and np.allclose(tab.sum(axis=1), 1)
    mix = GaussianMixture(np.array([.5, .5]), np.array([[0.], [4.]]), np.ones((2, 1, 1)))
    rows = sum(len(r) for r, _, _ in s.groups(mix, l, b))
    assert rows == n
