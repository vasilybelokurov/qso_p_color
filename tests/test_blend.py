"""Blend mode: conditioning, linearised flux addition, limits and an end-to-end smoke test."""
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from qso_pcolor.blend import BlendModel, blend_colour_loglike, condition_mixture, flux_of, luptitude
from qso_pcolor.gaussmix import GaussianMixture, condition_joint
from qso_pcolor.multisurvey_data import Photometry

ROOT = Path(__file__).resolve().parents[1]


def random_mixture(rng, k, d, scale=.05):
    means = rng.normal(20., 1., (k, d))
    a = rng.normal(0, scale, (k, d, d)); covs = a @ a.swapaxes(1, 2) + np.eye(d)*scale**2
    return np.log(rng.dirichlet(np.ones(k))), means, covs


def test_condition_mixture_matches_condition_joint():
    rng = np.random.default_rng(1); lw, mu, cv = random_mixture(rng, 4, 4, .3)
    mix = GaussianMixture(np.exp(lw), mu, cv)
    w, cm = condition_joint(mix, np.array([20.3]), np.array([1]))
    lw2, mu2, c2 = condition_mixture(lw, mu, cv, 1, 20.3, np.array([0, 2, 3]), prune=1e9)
    np.testing.assert_allclose(np.exp(lw2), w, rtol=1e-9)
    np.testing.assert_allclose(mu2, cm.means, rtol=1e-9); np.testing.assert_allclose(c2, cm.covs, rtol=1e-8, atol=1e-12)


def monte_carlo(y, noise, s, comp1, comp2, n, rng):
    """Brute force: draw both components, add fluxes, Gaussian measurement noise in luptitude."""
    def draw(comp):
        lw, mu, c = comp; k = rng.choice(len(lw), n, p=np.exp(lw))
        return mu[k] + np.einsum('nij,nj->ni', np.linalg.cholesky(c)[k], rng.standard_normal((n, mu.shape[1])))
    l1, l2 = draw(comp1), draw(comp2)
    lb = luptitude(flux_of(l1, s) + flux_of(l2, s), s)
    inv = np.linalg.inv(noise); r = y - lb
    ln = -.5*np.einsum('ni,ij,nj->n', r, inv, r) - .5*np.linalg.slogdet(2*np.pi*noise)[1]
    return float(logsumexp(ln) - np.log(n))


@pytest.mark.parametrize('ratio', [.1, .5, .9])
def test_linearised_blend_likelihood_agrees_with_monte_carlo(ratio):
    rng = np.random.default_rng(2); d = 3; s = np.full(d, .3)
    lw1, mu1, c1 = random_mixture(rng, 2, d, .06); lw2, mu2, c2 = random_mixture(rng, 2, d, .06)
    mu2 = mu2 - 2.5*np.log10(ratio/(1 - ratio)) + (mu1.mean() - mu2.mean())   # companion flux share ~ ratio
    comp1, comp2 = (lw1, mu1, c1), (lw2, mu2, c2)
    noise = np.eye(d)*.04**2
    k1, k2 = rng.choice(2, p=np.exp(lw1)), rng.choice(2, p=np.exp(lw2))
    y = luptitude(flux_of(mu1[k1], s) + flux_of(mu2[k2], s), s) + rng.normal(0, .04, d)
    analytic = blend_colour_loglike(y, noise, s, comp1, comp2)
    mc = monte_carlo(y, noise, s, comp1, comp2, 400000, rng)
    assert abs(analytic - mc) < .1, (analytic, mc)


def test_vanishing_companion_recovers_single_component_density():
    rng = np.random.default_rng(3); d = 3; s = np.full(d, .3)
    lw1, mu1, c1 = random_mixture(rng, 3, d, .08)
    comp2 = (np.zeros(1), luptitude(np.zeros((1, d)), s), np.eye(d)[None]*1e-10)   # exactly zero flux
    noise = np.eye(d)*.03**2; y = mu1[0] + .02
    single = logsumexp([lw + -.5*(y - m) @ np.linalg.solve(c + noise, y - m) - .5*np.linalg.slogdet(2*np.pi*(c + noise))[1]
                        for lw, m, c in zip(lw1, mu1, c1)])
    assert abs(blend_colour_loglike(y, noise, s, (lw1, mu1, c1), comp2) - single) < 1e-6


@pytest.mark.skipif(not (ROOT/'models/multisurvey_psf/current').exists(), reason='shipped bundle required')
def test_blend_model_end_to_end_qq_and_qs_cases():
    model = BlendModel.load(ROOT/'models/multisurvey_psf/current', n_alpha=24)
    bands = ('decals_dr9_south:g', 'decals_dr9_south:r', 'decals_dr9_south:z', 'decals_dr9_south:w1', 'decals_dr9_south:w2')
    soft = model.soft[[model.bands.index(b) for b in bands]]
    # quasar-like (blue g-r, red z-W1) and star-like (stellar locus, blue z-W1) companions of a quasar at z = 1.5
    q = np.array([20.6, 20.4, 20.2, 19.2, 18.6]); star = np.array([21.2, 20.4, 20.0, 20.7, 21.3])
    def blend(a, b):
        f = flux_of(a, soft) + flux_of(b, soft); return f, (f/40.)**2
    rows = [blend(q, q + .3), blend(q, star)]
    phot = Photometry(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), bands)
    out = model.score(phot, ra_deg=180., dec_deg=30., z_qso=1.5, dereddened=True, l_deg=180., b_deg=45.)
    assert all(o.status == 'ok' for o in out)
    assert out[0].log_bf_qq_qs > out[1].log_bf_qq_qs
    assert 0 < out[0].alpha_mean_qq < 1 and out[0].alpha_range[0] > 0
    faint = Photometry(np.array([rows[0][0]*1e-3]), np.array([rows[0][1]]), bands)
    assert model.score(faint, ra_deg=180., dec_deg=30., z_qso=1.5, dereddened=True, l_deg=180., b_deg=45.)[0].status \
        == 'too_faint_for_two_components'
    odds = [model.score(phot.subset([1]), ra_deg=180., dec_deg=30., z_qso=1.5, prior_odds_qq=k, dereddened=True,
                        l_deg=180., b_deg=45.)[0].p_qq for k in (.1, 1., 10.)]
    assert odds[0] < odds[1] < odds[2]
