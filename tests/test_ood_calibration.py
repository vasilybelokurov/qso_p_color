"""Calibrated outside-the-model test: p-values are uniform for members, small for outliers, any dimension."""
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kstest

from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.ood_calibration import _components, tail_pvalue

ROOT = Path(__file__).resolve().parents[1]


def mixture(rng, k, d):
    a = rng.normal(0, .3, (k, d, d))
    return GaussianMixture(rng.dirichlet(np.ones(k)), rng.normal(0, 2, (k, d)), a @ a.swapaxes(1, 2) + .05*np.eye(d))


@pytest.mark.parametrize('d', [4, 12, 25])
def test_members_have_uniform_p_in_any_dimension(d):
    """The fixed-sigma cut fails exactly here: the distance grows with d, the calibrated p does not."""
    rng = np.random.default_rng(d); mix = mixture(rng, 6, d); comp = _components([mix])
    s = np.eye(d)*.02**2; obs = np.ones(d, bool)
    ps, dist = [], []
    for i in range(300):
        k = rng.choice(6, p=mix.weights)
        x = rng.multivariate_normal(mix.means[k], mix.covs[k] + s)
        p, t = tail_pvalue(comp, x, s, obs, 0, draws=200, seed=i); ps.append(p); dist.append(np.sqrt(t))
    assert kstest(ps, 'uniform').pvalue > 1e-3
    assert np.mean(np.array(ps) < .02) < .05
    if d == 25:
        assert np.median(dist) > 3          # a typical member is far in sigma ...
        assert np.mean(np.array(ps) < 2/256) < .03    # ... but not flagged


def test_outlier_gets_small_p_and_missing_bands_are_ignored():
    rng = np.random.default_rng(1); d = 8; mix = mixture(rng, 4, d); comp = _components([mix])
    s = np.eye(d)*.02**2; obs = np.ones(d, bool)
    x = mix.means[0] + 6.                      # far from everything
    assert tail_pvalue(comp, x, s, obs, 0, draws=256, seed=0)[0] < 2/256
    obs2 = obs.copy(); obs2[3:] = False; x2 = x.copy(); x2[3:] = np.nan
    p, t = tail_pvalue(comp, x2, s, obs2, 0, draws=256, seed=0)
    assert np.isfinite(p) and np.isfinite(t)
    assert np.isnan(tail_pvalue(comp, x2, s, np.eye(d, dtype=bool)[0], 0, draws=16, seed=0)[0])   # reference only


@pytest.mark.skipif(not (ROOT/'models/multisurvey_psf/current').exists(), reason='shipped bundle required')
def test_scorer_reports_calibrated_p_and_uses_it_for_the_flag():
    from qso_pcolor import BlendPolicy, Photometry, RedshiftMatch
    from qso_pcolor.unified import UnifiedPSFModel
    m = UnifiedPSFModel.load(ROOT/'models/multisurvey_psf/current')
    bands = ("decals_dr9_south:g", "decals_dr9_south:r", "decals_dr9_south:z", "decals_dr9_south:w1", "decals_dr9_south:w2")
    flux = np.array([[2.6, 3.0, 3.6, 9.0, 11.0], [30., 3.0, .2, 40., .1]])     # quasar-like; nonsense colours
    phot = Photometry(flux, 1/np.array([[300., 200., 60., 3., .6]]*2), bands)
    kw = dict(ra_deg=180., dec_deg=30., z_primary=1.8, morphology=['PSF']*2, match=RedshiftMatch(half_width_kms=2000.),
              blend_policy=BlendPolicy(3., .2), separation_arcsec=6., fracflux=.05, ood_flag_sigma=4.)
    old, _ = m.score(phot, **kw)
    new, _ = m.score(phot, ood_calibration=dict(alpha=2/256, draws=256, seed=7), **kw)
    assert np.isnan(old[0].qso_ood_p) and 0 < new[0].qso_ood_p <= 1 and 0 < new[0].bkg_ood_p <= 1
    assert new[0].qso_ood_sigma_any_z == pytest.approx(old[0].qso_ood_sigma_any_z)
    assert 'outside_both_models' not in new[0].quality_flags
    assert 'outside_both_models' in new[1].quality_flags
    again, _ = m.score(phot.subset([1, 0]), ood_calibration=dict(alpha=2/256, draws=256, seed=7), **kw)
    assert again[1].qso_ood_p == new[0].qso_ood_p           # per-object seeds: independent of batch order
