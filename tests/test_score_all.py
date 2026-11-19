"""Flags instead of refusals, redshift-free scoring, and the weak-Legacy-r fallback (docs/euclid/PLAN_score_all.md,
Phase D-main, PI decision D5 of 19 Nov 2026).

The unit test runs anywhere; the others need the shipped bundle and the training-run arrays (gitignored) and use
real role-3 rows, because synthetic colours land outside both trained models.
"""
from pathlib import Path
import warnings

import numpy as np
import pytest

from qso_pcolor.fallback_reference import fallback_status
from qso_pcolor.multisurvey_data import Photometry, band_labels

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'models/multisurvey_psf/current'
RUN = ROOT/'models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059'
needs_data = pytest.mark.skipif(not (BUNDLE.exists() and (RUN/'qso').exists()), reason='bundle and run arrays required')
CONFIG_FIELDS = ('scoring_config_json', 'config_hash')


def test_fallback_band_is_the_first_that_passes_snr():
    bands = band_labels()
    f = np.full((3, len(bands)), np.nan); v = np.full_like(f, np.inf)
    p1, nsc = bands.index('ps1:r'), bands.index('nsc:r')
    f[0, [p1, nsc]] = 10.; v[0, p1] = 4.; v[0, nsc] = .25        # PS1 S/N 5, NSC S/N 20: NSC is used
    f[1, [p1, nsc]] = 10.; v[1, [p1, nsc]] = 4.                    # both S/N 5: PS1 named, not eligible
    f[2, p1] = 10.; v[2, p1] = .25                                 # PS1 S/N 20
    need, ok, band, snr = fallback_status(Photometry(f, v, bands), bands, 10.)
    assert [bands[k] for k in band] == ['nsc:r', 'ps1:r', 'ps1:r']
    assert list(ok) == [True, False, True] and np.allclose(snr, [20., 5., 20.])


def _rows(kind, n, seed):
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from run_unified_pilot import arrays
    d = arrays(RUN, kind)
    from qso_pcolor.unified import UnifiedPSFModel
    model = UnifiedPSFModel.load(BUNDLE); bands = model.base.model.transform.bands
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(np.flatnonzero(np.asarray(d['role']) == 3), n, replace=False))
    phot = Photometry(np.array(d['flux'][pick]), np.array(d['variance'][pick]), bands)
    return model, d, pick, phot


def _kw(n):
    from qso_pcolor.multisurvey_baseline import BlendPolicy
    from qso_pcolor.score import RedshiftMatch
    return dict(morphology=['PSF']*n, match=RedshiftMatch(half_width_kms=2000.),
                blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2), ood_flag_sigma=4.,
                separation_arcsec=6., fracflux=.05)


def _same(a, b, skip=CONFIG_FIELDS):
    for k in a.__dataclass_fields__:
        if k in skip:
            continue
        x, y = getattr(a, k), getattr(b, k)
        if not (x == y or (isinstance(x, float) and np.isnan(x) and np.isnan(y))):
            return False
    return True


@needs_data
def test_flags_replace_refusals_and_leave_eligible_rows_unchanged():
    warnings.simplefilter('ignore')
    model, d, pick, phot = _rows('stars', 160, 7)
    z = np.full(len(pick), 1.5); kw = _kw(len(pick))
    old, do = model.score(phot, ra_deg=d['ra'][pick], dec_deg=d['dec'][pick], z_primary=z,
                          refuse_flagged=True, weak_reference_fallback=False, **kw)
    new, dn = model.score(phot, ra_deg=d['ra'][pick], dec_deg=d['dec'][pick], z_primary=z,
                          refuse_flagged=False, weak_reference_fallback=True, **kw)
    flagged = np.isin(do['reason'], ['outside_both_models', 'qso_support_rejected'])
    assert do['eligible'].sum() >= 10 and flagged.sum() > 0          # both cases are exercised
    for i in range(len(pick)):
        if do['eligible'][i]:                                          # M1: previously eligible rows unchanged
            assert dn['eligible'][i] and _same(old[i], new[i])
        elif flagged[i]:                                               # refusal -> flag; intensities unchanged
            assert dn['eligible'][i] and do['reason'][i] in new[i].quality_flags
            assert np.isfinite(new[i].log_r_per_unit_z)
            for k in ('log_lambda_sameq', 'log_lambda_fieldq', 'log_lambda_bkg', 'log_lambda_out'):
                assert getattr(old[i], k) == getattr(new[i], k) or (np.isnan(getattr(old[i], k)) and np.isnan(getattr(new[i], k)))
        elif do['reason'][i] != 'too_faint':
            assert dn['reason'][i] == do['reason'][i]


@needs_data
def test_redshift_free_probability_matches_pair_scoring_at_any_z():
    warnings.simplefilter('ignore')
    model, d, pick, phot = _rows('qso', 40, 11)
    kw = _kw(len(pick)); pos = dict(ra_deg=d['ra'][pick], dec_deg=d['dec'][pick])
    s, dec = model.score_sources(phot, **pos, **kw, refuse_flagged=False)
    assert dec['eligible'].mean() > .8
    assert all(np.isnan(r.log_r_per_unit_z) for r in s if r is not None)
    assert not np.isin(dec['reason'], ['window_outside_model_support', 'primary_z_outside_model_support']).any()
    pfree = dec['p_quasar']
    for z0 in (0.6, 1.5, 3.0, 4.2):
        sp, dp = model.score(phot, z_primary=np.full(len(pick), z0), **pos, **kw, refuse_flagged=False)
        both = dec['eligible'] & dp['eligible']
        assert both.sum() > 10
        # the pair scorer adds window nodes around z0 to the same quadrature: agreement to quadrature accuracy
        assert np.max(np.abs(model.quasar_probability(sp)[both] - pfree[both])) < 1e-4


@needs_data
def test_weak_legacy_r_is_scored_through_the_fallback_and_kept_in_the_likelihood():
    warnings.simplefilter('ignore')
    model, d, pick, phot = _rows('qso', 400, 5)
    bands = phot.bands; rs, rn, p1 = (bands.index(b) for b in ('decals_dr9_south:r', 'decals_dr9_north:r', 'ps1:r'))
    with np.errstate(invalid='ignore', divide='ignore'):
        sn = phot.flux/np.sqrt(phot.variance)
    use = np.flatnonzero((sn[:, rs] > 30) & (sn[:, p1] > 15) & ~phot.observed[:, rn])[:4]
    assert len(use) >= 2
    f, v = phot.flux[use].copy(), phot.variance[use].copy()
    weak = v.copy(); weak[:, rs] = (f[:, rs]/5.)**2                     # Legacy r at S/N 5
    flat = v.copy(); flat[:, rs] = (f[:, rs]*1e4)**2                    # S/N 1e-4: no information
    gone = v.copy(); gone[:, rs] = np.inf                               # no Legacy r
    kw = _kw(len(use)); pos = dict(ra_deg=d['ra'][pick][use], dec_deg=d['dec'][pick][use], z_primary=d['zspec'][pick][use])
    score = lambda var, **o: model.score(Photometry(f, var, bands), **pos, **kw,
                                         **{'weak_reference_fallback': True, **o})
    _, d_old = score(weak, weak_reference_fallback=False)
    assert (d_old['reason'] == 'too_faint').all()
    s_w, d_w = score(weak)
    assert d_w['eligible'].all()
    assert all('weak_legacy_r_in_likelihood' in r.quality_flags for r in s_w)
    s_f, _ = score(flat); s_g, _ = score(gone)
    # an uninformative weak r reduces exactly to the no-Legacy-r fallback
    assert np.allclose(model.quasar_probability(s_f), model.quasar_probability(s_g), rtol=0, atol=1e-9)
    # an informative one moves the latent r towards the measurement
    true_r = np.array([r.ref_mag for r in model.score(Photometry(f, v, bands), **pos, **kw)[0]])
    err_w = np.abs(np.array([r.ref_mag for r in s_w]) - true_r); err_g = np.abs(np.array([r.ref_mag for r in s_g]) - true_r)
    assert np.median(err_w) <= np.median(err_g) + 1e-6
