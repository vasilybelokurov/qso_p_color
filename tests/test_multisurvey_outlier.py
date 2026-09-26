"""The unmodelled hypothesis in the conditional multi-survey coordinates.

The joint Gaussian U is conditioned on the reference band exactly as the
quasar and background mixtures are.  Checked against closed-form conditional
Gaussians (scipy), the Schur-complement dominance argument, the three-
hypothesis scorer at eta = 0, and a far colour outlier.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import (BandLuptitudeTransform, MultiSurveyModel,
                                    MultiSurveyOutlier)
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.outlier import envelope_covariance, min_dominant_kappa, mixture_moments
from qso_pcolor.priors import GridQSOPrior
from qso_pcolor.qso_model import RedshiftMatch, SlicedColourRedshiftModel

LABELS = ("allwise:w1", "allwise:w2", "allwise:w3")


def toy():
    z = np.linspace(0.5, 2.5, 11)
    mixtures = [GaussianMixture(np.ones(1), np.array([[18.0, 17.5 + 0.3 * zz, 16.0]]),
                                np.diag([0.4, 0.05, 0.1])[None], LABELS) for zz in z]
    qso = SlicedColourRedshiftModel(z, mixtures, np.full(z.size, 1000), "toy_ms", LABELS)
    bkg = GaussianMixture(np.ones(1), np.array([[18.0, 18.2, 17.9]]),
                          np.array([[[0.5, 0.2, 0.1], [0.2, 0.3, 0.1], [0.1, 0.1, 0.4]]]), LABELS)
    model = MultiSurveyModel(qso, bkg, BandLuptitudeTransform(LABELS, np.ones(3)), LABELS,
                             np.tile([10.0, 30.0], (3, 1)))
    meta = dict(reference_band=LABELS[0], transform_id=model.transform_id)
    prior = GridQSOPrior(z, np.array([10.0, 30.0]), np.ones((z.size, 2)), meta=dict(meta))

    class Density:
        def __init__(self):
            self.meta = dict(meta)

        def __call__(self, m, l, b):
            return np.full(len(m), 50.0)

        def level(self, m, l, b):
            return np.full(len(m), 2)

    return model, {LABELS[0]: (prior, Density())}


def outlier_for(model, eta, kappa_factor=1.05):
    check = list(model.qso.mixtures) + [model.background]
    mean, base = mixture_moments([model.background])
    env = envelope_covariance(base, check)
    kmin = min_dominant_kappa(env, check)
    k = kappa_factor * kmin
    return MultiSurveyOutlier(mean, k**2 * env, k, kmin, LABELS, model.transform_id,
                              {LABELS[0]: (np.array([10.0, 30.0]), np.array([eta]))})


def lupt_to_flux(u):
    return 2 * np.sinh((22.5 - np.asarray(u, float)) / (2.5 / np.log(10.0)))


def run(model, priors, u, z0=1.5, **kw):
    phot = Photometry(lupt_to_flux(np.atleast_2d(u)), np.full(np.atleast_2d(u).shape, 1e-4),
                      LABELS)
    n = phot.flux.shape[0]
    return model.score(phot, z_primary=np.full(n, z0), match=RedshiftMatch(half_width_kms=2000.0),
                       min_bands=2, l_deg=np.full(n, 180.0), b_deg=np.full(n, 45.0),
                       priors=priors, **kw)


def test_conditional_outlier_density_matches_a_closed_form_conditional_gaussian():
    model, _ = toy()
    um = outlier_for(model, 1e-3)
    c = um.conditional(0, LABELS[0], model.qso.system)
    rng = np.random.default_rng(0)
    x = rng.normal(18, 2, (4, 3))
    s = np.array([np.diag(rng.uniform(0.01, 0.2, 3)) for _ in range(4)])
    got = c.log_prob(x, s, observed=np.ones((4, 3), bool))
    for i in range(4):
        t = um.cov + s[i]
        cond_mean = um.mean[1:] + t[1:, 0] / t[0, 0] * (x[i, 0] - um.mean[0])
        cond_cov = t[1:, 1:] - np.outer(t[1:, 0], t[1:, 0]) / t[0, 0]
        ref = multivariate_normal(cond_mean, cond_cov).logpdf(x[i, 1:])
        assert got[i] == pytest.approx(ref, abs=1e-9)


def test_conditioning_preserves_dominance():
    """Schur complements are monotone: E >= V implies E/E_aa >= V/V_aa."""
    model, _ = toy()
    um = outlier_for(model, 1e-3)
    for v in [c for m in model.qso.mixtures for c in m.covs] + list(model.background.covs):
        e_s = um.cov[1:, 1:] - np.outer(um.cov[1:, 0], um.cov[1:, 0]) / um.cov[0, 0]
        v_s = v[1:, 1:] - np.outer(v[1:, 0], v[1:, 0]) / v[0, 0]
        assert np.linalg.eigvalsh(e_s - v_s).min() > -1e-12


def test_zero_fraction_reproduces_the_scorer_without_it():
    model, priors = toy()
    u = np.array([[18.0, 17.9, 16.1], [18.3, 18.2, 17.8]])
    a = run(model, priors, u)
    b = run(model, priors, u, outlier=outlier_for(model, 0.0))
    for ra, rb in zip(a, b):
        for k in ("log_r_per_unit_z", "log_bayes_factor_qz_bkg", "p_sameq", "loglike_bkg"):
            assert getattr(rb, k) == pytest.approx(getattr(ra, k), rel=1e-12)
        assert rb.p_outlier == 0.0


def test_posteriors_sum_to_one_and_a_far_outlier_goes_to_unmodelled():
    model, priors = toy()
    um = outlier_for(model, 1e-3)
    near, far = run(model, priors, [[18.0, 17.9, 16.1], [18.0, 28.0, 8.0]], outlier=um)
    for r in (near, far):
        lam = [r.log_lambda_sameq, r.log_lambda_fieldq, r.log_lambda_bkg, r.log_lambda_out]
        assert r.p_outlier == pytest.approx(np.exp(lam[3] - logsumexp(lam)), rel=1e-10)
        assert r.p_sameq == pytest.approx(np.exp(r.log_r_per_unit_z) * r.dz_match_eff, rel=1e-10)
    assert near.p_outlier < 1e-3
    assert far.p_outlier > 0.999
    assert far.qso_ood_sigma_any_z > 10 and far.bkg_ood_sigma > 10


def test_other_reference_bands_fall_back_to_the_pooled_fraction():
    model, _ = toy()
    um = outlier_for(model, 1e-3)
    um.fractions = {"*": (np.array([-99.0, 99.0]), np.array([2e-3]))}
    c = um.conditional(1, LABELS[1], model.qso.system)
    assert c.fraction_at(np.array([17.0]))[0] == 2e-3


def test_round_trip_and_transform_check(tmp_path):
    model, priors = toy()
    um = outlier_for(model, 1e-3)
    um.save(tmp_path / "u.json")
    back = MultiSurveyOutlier.load(tmp_path / "u.json")
    assert np.array_equal(back.cov, um.cov) and back.kappa == um.kappa
    back.transform_id = "0" * 64
    with pytest.raises(ValueError, match="transform"):
        run(model, priors, [[18.0, 17.9, 16.1]], outlier=back)


def test_student_t_conditional_is_the_joint_over_the_reference_marginal():
    from scipy.stats import multivariate_t

    model, priors = toy()
    g = outlier_for(model, 1e-3)
    um = MultiSurveyOutlier(g.mean, np.diag([0.5, 0.3, 0.4]), 1.0, 0.0, LABELS,
                            model.transform_id, g.fractions, family="student_t", nu=2.0)
    c = um.conditional(0, LABELS[0], model.qso.system)
    x = np.array([[18.0, 17.0, 16.0], [25.0, 10.0, 30.0]])
    got = c.log_prob(x, np.zeros((2, 3, 3)), observed=np.ones((2, 3), bool))
    for i in range(2):
        joint = multivariate_t(um.mean, um.cov, df=2.0).logpdf(x[i])
        marg = multivariate_t(um.mean[:1], um.cov[:1, :1], df=2.0).logpdf(x[i, :1])
        assert got[i] == pytest.approx(joint - marg, abs=1e-10)
    # a far object goes to U through the scorer
    far = run(model, priors, [[18.0, 40.0, -10.0]], outlier=um)[0]
    assert far.p_outlier > 0.99


def test_a_student_t_outlier_needs_positive_nu():
    model, _ = toy()
    g = outlier_for(model, 1e-3)
    with pytest.raises(ValueError, match="nu"):
        MultiSurveyOutlier(g.mean, g.cov, 1.0, 0.0, LABELS, model.transform_id, g.fractions,
                           family="student_t", nu=None)
