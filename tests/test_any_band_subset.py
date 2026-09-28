"""One joint distribution, marginalised identically for every band selection."""
from itertools import combinations

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import MultiSurveyModel, conditional_log_prob
from qso_pcolor.multisurvey_data import Photometry, band_labels
from qso_pcolor.qso_model import RedshiftMatch
from test_multisurvey import synthetic_model


def test_all_820_band_pairs_and_random_subsets_against_scipy_marginals():
    rng = np.random.default_rng(20260928)
    labels = band_labels()
    d = len(labels)
    matrices = rng.normal(size=(2, d, d)) / np.sqrt(d)
    intrinsic = matrices @ matrices.swapaxes(1, 2) + np.eye(d)
    means = rng.normal(size=(2, d))
    weights = np.array([.3, .7])
    mix = GaussianMixture(weights, means, intrinsic, labels)
    subsets = list(combinations(range(d), 2))
    subsets += [tuple(sorted(rng.choice(d, size, replace=False)))
                for size in range(3, d + 1)]
    x = rng.normal(size=(len(subsets), d))
    noise = .2 * np.eye(d) + .03 * np.ones((d, d))
    covariance = np.broadcast_to(noise, (len(subsets), d, d))
    observed = np.zeros_like(x, bool)
    for i, dims in enumerate(subsets):
        observed[i, list(dims)] = True
    # Missing values (and covariance entries) must be irrelevant, even if NaN.
    x[~observed] = np.nan
    masked_cov = np.where(observed[:, :, None] & observed[:, None, :], covariance, np.nan)
    for anchor in range(d):
        rows = np.array([i for i, dims in enumerate(subsets) if dims[0] == anchor])
        if not len(rows):
            continue
        actual = conditional_log_prob(mix, x[rows], masked_cov[rows], observed[rows], anchor)
        for i, got in zip(rows, actual):
            dims = np.array(subsets[i])
            joint = [multivariate_normal.logpdf(x[i, dims], means[k, dims],
                     (intrinsic[k] + noise)[np.ix_(dims, dims)]) for k in range(2)]
            ref = [multivariate_normal.logpdf(x[i, anchor], means[k, anchor],
                   intrinsic[k, anchor, anchor] + noise[anchor, anchor]) for k in range(2)]
            expected = logsumexp(np.log(weights) + joint) - logsumexp(np.log(weights) + ref)
            assert got == pytest.approx(expected, abs=2e-12)


def test_all_41_single_bands_are_prior_only_not_rejected():
    model = synthetic_model()
    d = len(model.transform.bands)
    flux = np.full((d + 1, d), np.nan)
    variance = np.full_like(flux, np.inf)
    flux[np.arange(d), np.arange(d)] = 3.
    variance[np.arange(d), np.arange(d)] = .01
    scores = model.score(Photometry(flux, variance, model.transform.bands),
                         z_primary=1.5, l_deg=180., b_deg=45.,
                         match=RedshiftMatch(half_width_kms=2000.))
    for label, score in zip(model.transform.bands, scores[:-1]):
        assert score.reference_band == label
        assert score.bands_used == (label,)
        assert score.status == "no_prior_posterior_unavailable"
        assert score.loglike_qso_zprimary == pytest.approx(0., abs=1e-12)
        assert score.loglike_bkg == 0.
        assert score.log_bayes_factor_qz_bkg == pytest.approx(0., abs=1e-12)
        assert score.qso_ood_sigma_any_z == score.bkg_ood_sigma == 0.
        assert "no_colour_information" in score.quality_flags
        assert np.isnan(score.p_sameq)
    assert scores[-1].status == "insufficient_photometry"
    assert scores[-1].reference_band == ""


def test_default_background_is_joint_even_when_old_subset_fits_are_saved(tmp_path):
    model = synthetic_model()
    labels = ("sdss:g", "sdss:r")
    model.background_marginals = (GaussianMixture(np.ones(1), np.zeros((1, 2)),
                                  np.eye(2)[None], labels),)
    path = tmp_path / "joint.json"
    model.save(path)
    model = MultiSurveyModel.load(path)
    phot = Photometry([[2., 3.]], [[.1, .1]], labels)
    features = model.transform(phot)
    anchor = features.labels.index(labels[0])
    expected = conditional_log_prob(model.background, features.x, features.cov,
                                    features.observed, anchor)
    assert np.array_equal(model.background_log_prob(features.x, features.cov,
                          features.observed, anchor), expected)
    score = model.score(phot, z_primary=1.5, l_deg=180., b_deg=45.,
                        match=RedshiftMatch(half_width_kms=2000.))[0]
    assert score.loglike_bkg == expected[0]


def test_explicit_selection_is_identical_to_missing_measurements_and_order_invariant():
    model = synthetic_model()
    full = Photometry(np.full((1, 41), 3.), np.full((1, 41), .01), band_labels())
    selected = ("vhs:ks", "allwise:w2", "sdss:u")  # no r, no mandatory W1
    short = full.keep_bands(selected)
    kwargs = dict(z_primary=1.5, l_deg=180., b_deg=45.,
                  match=RedshiftMatch(half_width_kms=2000.))
    scores = [model.score(p, **kwargs)[0] for p in
              (short, short.align(full.bands), short.keep_bands(selected[::-1]))]
    for score in scores[1:]:
        assert score.log_bayes_factor_qz_bkg == scores[0].log_bayes_factor_qz_bkg
        assert score.p_zmatch_given_qso == scores[0].p_zmatch_given_qso
        assert score.reference_band == scores[0].reference_band
    empty = model.score(full.keep_bands(()), **kwargs)[0]
    assert empty.status == "insufficient_photometry"
    with pytest.raises(ValueError, match="duplicate"):
        full.keep_bands((selected[0], selected[0]))
