"""Extended (Euclid) bundles: T-U1 invariance of parent-band scores, T-U4 conversion/cleaning, T-U8 thinning."""
from pathlib import Path
import warnings

import numpy as np
import pytest

from qso_pcolor.euclid_bundle import (EUCLID_BANDS, build_bundle, clean, extend_layout, luptitudes, photometry,
                                      point_like)
from qso_pcolor.euclid_data import UJY_TO_NATIVE
from qso_pcolor.euclid_extension import Extension, regressor_matrix
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.multisurvey import thinned_conditional_log_prob, conditional_log_prob, thinning_log_factor
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.qso_model import RedshiftMatch
from qso_pcolor.score import BlendPolicy

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'models/multisurvey_psf/work/unified_full/20261001/5f4002492dbb849c'
FIELDS = ('log_lambda_sameq', 'log_lambda_fieldq', 'log_lambda_bkg', 'log_lambda_out', 'log_r_per_unit_z',
          'p_sameq', 'p_outlier', 'loglike_qso_zprimary', 'qso_ood_sigma_any_z', 'bkg_ood_sigma')


def _dummy(n, p, rng):
    return Extension(rng.normal(0, .3, (n, 4)), rng.normal(0, .2, (n, 4, p)),
                     np.tile(np.eye(4)*.05, (n, 1, 1)))


def _rows(kind, n, rng):
    d = {k: np.load(RUN/kind/f'{k}.npy', mmap_mode='r') for k in ('flux', 'variance', 'ra', 'dec', 'zspec', 'role')}
    pick = np.sort(rng.choice(np.flatnonzero(np.asarray(d['role']) == 3), n, replace=False))
    return {k: np.asarray(v[pick]) for k, v in d.items()}


@pytest.mark.skipif(not RUN.exists(), reason='local training arrays not available')
@pytest.mark.parametrize('bundle', ['20261005_magdep_qlf', '20261005_magindep_qlf'])
def test_extended_bundle_reproduces_parent_scores_on_parent_bands(tmp_path, bundle):
    import json
    from qso_pcolor.unified import UnifiedPSFModel
    rng = np.random.default_rng(7)
    parent_root = ROOT/'models/multisurvey_psf'/bundle
    lat = json.loads((parent_root/'latent.json').read_text())
    labels = lat['layout']['latent_labels']
    t_d = regressor_matrix(labels, ('legacy:g', 'legacy:z', 'legacy:w1', 'legacy:w2'), 'legacy:r',
                           include_magnitude=('magdep' in bundle))
    q = [_dummy(len(m['weights']), t_d.shape[0], rng) for m in lat['qso']]
    b = _dummy(len(lat['background']['weights']), t_d.shape[0], rng)
    out = build_bundle(parent_root, tmp_path/'ext', qso_ext=q, bkg_ext=b, t_d_qso=t_d, t_d_bkg=t_d,
                       softening_euclid=[.1, .3, .3, .3], bounds_euclid=[[14, 27]]*4,
                       euclid_band_counts=np.full((len(q), 4), 500))
    parent = UnifiedPSFModel.load(parent_root); ext = UnifiedPSFModel.load(out)
    for kind in ('qso', 'stars'):
        r = _rows(kind, 60, rng)
        phot = Photometry(r['flux'], r['variance'], parent.base.model.transform.bands)
        kw = dict(ra_deg=r['ra'], dec_deg=r['dec'], z_primary=np.where(r['zspec'] > 0.2, r['zspec'], 1.5),
                  morphology=['PSF']*len(r['ra']), match=RedshiftMatch(half_width_kms=2000.),
                  blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
                  separation_arcsec=np.full(len(r['ra']), 6.), fracflux=np.zeros(len(r['ra'])), ood_flag_sigma=4.)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sp, dp = parent.score(phot, **kw); se, de = ext.score(phot, **kw)
        # Support and outside-both p-values are Monte Carlo with seeds hashed from the full input vectors (parent
        # behaviour kept on purpose), so 4 extra unobserved bands change the draws: a flip is allowed only at the
        # threshold (within binomial noise) and only for those two tests (docs/euclid/PLAN.md T-U1).
        flip = np.flatnonzero((dp['reason'] != de['reason']) | (dp['eligible'] != de['eligible']))
        thr = parent.support['threshold']
        for i in flip:
            assert {dp['reason'][i], de['reason'][i]} <= {'', 'qso_support_rejected', 'outside_both_models'}
            pa_, pe_ = dp['qso_support']['percentile'][i], de['qso_support']['percentile'][i]
            # two independent Monte Carlo ranks of the same object (n draws): difference within 3 sigma + 1 draw
            n = parent.support['draws'] + 1; pm = (pa_ + pe_)/2
            assert abs(pa_ - pe_)*n <= 3*np.sqrt(2*n*pm*(1 - pm)) + 1 and min(pa_, pe_) < thr <= max(pa_, pe_)
        for i, (a, c) in enumerate(zip(sp, se)):
            if a is None:
                assert c is None; continue
            fields = FIELDS if i not in flip else ('log_lambda_sameq', 'log_lambda_fieldq', 'log_lambda_bkg',
                                                   'log_lambda_out', 'loglike_qso_zprimary', 'qso_ood_sigma_any_z',
                                                   'bkg_ood_sigma')
            for f in fields:
                x, y = getattr(a, f), getattr(c, f)
                assert (np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-9, (kind, f, x, y)
        ok = np.setdiff1d(np.arange(len(sp)), flip)
        pa, pe = parent.quasar_probability(sp), ext.quasar_probability(se)
        assert np.allclose(pa[ok], pe[ok], atol=1e-9, equal_nan=True)


def test_catalogue_conversion_cleaning_and_sentinels():
    m = dict(matched=np.array([True, True, True, False, True]),
             flux_vis_to_y_templfit=np.array([10., -2., np.nan, np.nan, 5.]),
             fluxerr_vis_to_y_templfit=np.array([.1, .3, .1, np.nan, 0.]),
             vis_det=np.array([1, 1, 1, -9999, 0]), spurious_flag=np.array([0, 0, 0, -9999, 0]),
             det_quality_flag=np.array([0, 2, 386, -9999, 0]),
             point_like_prob=np.array([.9, .05, np.nan, np.nan, .5]), point_like_flag=np.array([32767]*5))
    for b in ('y', 'j', 'h'):
        m[f'flux_{b}_templfit'] = np.array([20., 1., 1., np.nan, 1.]); m[f'fluxerr_{b}_templfit'] = np.full(5, .2)
    p = photometry(m)
    assert p.bands == EUCLID_BANDS
    assert np.isclose(p.flux[0, 0], 10*UJY_TO_NATIVE) and np.isclose(p.variance[0, 0], (.1*UJY_TO_NATIVE)**2)
    assert p.flux[1, 0] < 0 and np.isfinite(p.variance[1, 0])                # negative flux is a measurement
    assert np.isnan(p.flux[2, 0]) and np.isinf(p.variance[4, 0])             # NaN flux, zero error -> unusable
    assert not p.observed[3].any()                                           # unmatched row
    assert clean(m).tolist() == [True, True, True, False, False]
    assert clean(m, bad_bits=256).tolist() == [True, True, False, False, False]
    assert point_like(m, .1).tolist() == [True, False, False, False, True]
    y, n = luptitudes(np.array([1., np.nan]), np.array([.01, np.inf]), np.array([.1, .1]))
    assert np.isfinite(y[0]) and np.isnan(y[1]) and n[1] == 0


def test_thinning_is_an_unnormalised_component_reweighting():
    rng = np.random.default_rng(3)
    k, d = 4, 3
    means = rng.normal(size=(k, d)); covs = np.array([np.eye(d)*(.5 + i*.1) for i in range(k)])
    mix = GaussianMixture(np.full(k, .25), means, covs)
    x = rng.normal(size=(6, d)); obs = np.ones((6, d), bool)
    one = thinned_conditional_log_prob(mix, x, None, obs, 0, np.zeros((6, k)))
    assert np.allclose(one, conditional_log_prob(mix, x, None, obs, 0))       # f = 1: parent density
    half = thinned_conditional_log_prob(mix, x, None, obs, 0, np.full((6, k), np.log(.5)))
    assert np.allclose(half, one + np.log(.5))                               # uniform f: scales, not renormalised
    t = dict(alpha=[0., 10., -10., 0.], beta=[0., 0., 0., -2.], r0=20.)
    lf = thinning_log_factor(t, np.array([20., 21.]))
    assert np.allclose(np.exp(lf[0]), [.5, 1/(1 + np.exp(-10)), 1/(1 + np.exp(10)), .5])
    assert np.isclose(np.exp(lf[1, 3]), 1/(1 + np.exp(2.)))


def test_layout_extension_keeps_parent_operator():
    import json
    lay = json.loads((ROOT/'models/multisurvey_psf/20261005_magindep_qlf/latent.json').read_text())['layout']
    ext = extend_layout(lay, [.1, .2, .3, .4])
    for kind in lay['operators']:
        h = np.array(ext['operators'][kind]['matrix'])
        assert np.array_equal(h[:41, :36], lay['operators'][kind]['matrix'])
        assert np.array_equal(h[41:, 36:], np.eye(4)) and not h[41:, :36].any() and not h[:41, 36:].any()
    assert ext['native_labels'][-4:] == list(EUCLID_BANDS)
