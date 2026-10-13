"""Scoring without Legacy r: the Legacy r integral, its grid, eligibility, and that Legacy-r rows are untouched."""
from pathlib import Path

import numpy as np
import pytest

from qso_pcolor.fallback_reference import (FALLBACK_BANDS, _integrate, fallback_status, flux_of,
                                           legacy_r_grid, luptitude)
from qso_pcolor.multisurvey_data import Photometry, band_labels

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'models/multisurvey_psf/current'


def same_record(a, b):
    """Field-by-field equality with NaN equal to NaN."""
    return all(x == y or (isinstance(x, float) and np.isnan(x) and np.isnan(y))
               for x, y in ((getattr(a, k), getattr(b, k)) for k in a.__dataclass_fields__))


def test_trapezoid_integral_of_a_gaussian_intensity():
    m = legacy_r_grid(20., .02, 41)
    for sigma in (.05, .1, .2):
        lam = np.log(3.) - .5*((m - 20.1)/sigma)**2 - .5*np.log(2*np.pi*sigma**2)
        assert abs(np.exp(_integrate(lam, m[1] - m[0])) - 3.) < 3e-4      # 1.6e-5 relative at sigma = 0.9 step


def test_grid_covers_model_scatter_and_noise():
    g = legacy_r_grid(19., .05, 41)
    assert len(g) == 41 and np.isclose(g[0], 19 - 1.2 - .25) and np.isclose(g[-1], 19 + 1.2 + .25)


def test_luptitude_round_trip():
    s = np.array([.5, 2.])
    assert np.allclose(luptitude(flux_of(np.array([18., 24.]), s), s), [18., 24.])


def test_fallback_status_priorities_and_limits():
    bands = band_labels()
    f = np.full((5, len(bands)), np.nan); v = np.full_like(f, np.inf)
    lr, p1, sm = bands.index('decals_dr9_south:r'), bands.index('ps1:r'), bands.index('skymapper:r')
    f[0, lr], v[0, lr] = 10., 1.                        # has Legacy r: no fallback needed
    f[1, [p1, sm]], v[1, [p1, sm]] = 10., .01          # PS1 r preferred over SkyMapper r
    f[2, sm], v[2, sm] = 10., .01                       # SkyMapper only
    f[3, p1], v[3, p1] = 10., 4.                        # PS1 S/N 5: too faint
    need, ok, band, snr = fallback_status(Photometry(f, v, bands), bands, 10.)
    assert list(need) == [False, True, True, True, True]
    assert list(ok) == [False, True, True, False, False]
    assert bands[band[1]] == 'ps1:r' and bands[band[2]] == 'skymapper:r' and band[4] == -1
    assert FALLBACK_BANDS == ('ps1:r', 'nsc:r', 'skymapper:r')


@pytest.mark.skipif(not BUNDLE.exists(), reason='shipped bundle required')
def test_unified_fallback_leaves_legacy_rows_unchanged_and_scores_ps1_rows():
    from qso_pcolor.multisurvey_baseline import BlendPolicy
    from qso_pcolor.score import RedshiftMatch
    from qso_pcolor.unified import UnifiedPSFModel
    model = UnifiedPSFModel.load(BUNDLE)
    bands = ('decals_dr9_south:g', 'decals_dr9_south:r', 'decals_dr9_south:z', 'ps1:g', 'ps1:r', 'ps1:i', 'allwise:w1', 'allwise:w2')
    soft = np.asarray(model.base.model.transform.softening)[[model.base.model.transform.bands.index(b) for b in bands]]
    mags = np.array([20.6, 20.4, 20.2, 20.65, 20.45, 20.3, 19.6, 19.0])   # a blue, WISE-red quasar-like source
    f = flux_of(mags, soft); v = (f/50.)**2
    F = np.stack([f, f, f]); V = np.stack([v, v, v])
    V[1, :3] = np.inf                                   # no Legacy: PS1 r fallback
    V[2, :3] = np.inf; V[2, 3:6] = np.inf               # no Legacy, no PS1: nothing to fall back on
    kw = dict(ra_deg=30., dec_deg=-50., z_primary=1.5, morphology=['PSF']*3, match=RedshiftMatch(half_width_kms=2000.),
              blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2), ood_flag_sigma=4.,
              separation_arcsec=6., fracflux=.05)
    from qso_pcolor.fallback_reference import LegacyRFallbackWarning
    s0, d0 = model.score(Photometry(F, V, bands), legacy_r_fallback=False, **kw)
    with pytest.warns(LegacyRFallbackWarning, match='1 source'):
        s1, d1 = model.score(Photometry(F, V, bands), **kw)            # on by default, and announced
    assert same_record(s1[0], s0[0]) and d1['reason'][0] == d0['reason'][0]
    assert d0['reason'][1] == 'no_faint_limit_band'
    assert d1['fallback_reference'][1] == 'ps1:r'
    assert any(f.startswith('legacy_r_marginalised:ps1:r') for f in s1[1].quality_flags)
    assert np.isfinite(s1[1].log_lambda_sameq) and abs(s1[1].ref_mag - 20.4) < 0.5
    assert d1['reason'][2] == 'no_fallback_band'


@pytest.mark.skipif(not BUNDLE.exists(), reason='shipped bundle required')
def test_blend_fallback_is_default_and_announced():
    from qso_pcolor.blend import BlendModel
    from qso_pcolor.fallback_reference import LegacyRFallbackWarning
    bands = ('ps1:g', 'ps1:r', 'ps1:i', 'ps1:z', 'allwise:w1', 'allwise:w2')
    on = BlendModel.load(BUNDLE, n_alpha=12, n_fallback_grid=9)
    off = BlendModel.load(BUNDLE, n_alpha=12, legacy_r_fallback=False)
    soft = on.soft[[on.bands.index(b) for b in bands]]
    f = flux_of(np.array([19.9, 19.7, 19.6, 19.5, 18.7, 18.1]), soft); phot = Photometry(f[None], ((f/80.)**2)[None], bands)
    args = dict(ra_deg=150., dec_deg=-35., z_qso=1.5, dereddened=True, l_deg=270., b_deg=30.)
    assert off.score(phot, **args)[0].status == 'no_reference_band'
    with pytest.warns(LegacyRFallbackWarning, match='blend mode'):
        r = on.score(phot, **args)[0]
    assert r.status == 'ok' and 'legacy_r_marginalised:ps1:r' in r.notes and np.isfinite(r.log_bf_qq_qs)


@pytest.mark.skipif(not BUNDLE.exists(), reason='shipped bundle required')
def test_bright_source_near_prior_edge_is_scored_or_refused_with_a_reason():
    """Grid points beyond the tabulated priors (r < ~16.4) must not poison the integral with NaN."""
    from qso_pcolor.multisurvey_baseline import BlendPolicy
    from qso_pcolor.score import RedshiftMatch
    from qso_pcolor.unified import UnifiedPSFModel
    model = UnifiedPSFModel.load(BUNDLE)
    bands = ('ps1:g', 'ps1:r', 'ps1:i', 'ps1:z', 'allwise:w1', 'allwise:w2')
    soft = np.asarray(model.base.model.transform.softening)[[model.base.model.transform.bands.index(b) for b in bands]]
    for r in (16.8, 17.3):
        f = flux_of(np.array([r + .5, r, r - .15, r - .2, r - .6, r - .4]), soft)
        s, d = model.score(Photometry(f[None], ((f/100.)**2)[None], bands), ra_deg=30., dec_deg=-50., z_primary=1.5,
                           morphology=['PSF'], match=RedshiftMatch(half_width_kms=2000.),
                           blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2), ood_flag_sigma=4.,
                           separation_arcsec=6., fracflux=.05)
        assert d['eligible'][0] or d['reason'][0] != ''
        if d['eligible'][0]:
            assert np.isfinite(model.quasar_probability(s)[0])
