#!/usr/bin/env python
"""Outlier stress tests of 28-30 September, repeated on the models promoted 4 October.

Models: `previous` (28 September, active when the outlier problem was found; guarded base scorer, no
support cut), `current` (magnitude-dependent) and `current_magindep` (magnitude-independent), both
through the full scorer (support cut) and, for reference, without the support cut.
Tests (identical synthetic objects for every model; Galactic (180, 45) deg, 0.03 mag errors, z0 = 1.8):
  release   15 x 15 Legacy (g-r, r-z) grids from -6 to 8 mag at r = 18.5, 21 (North, South) - the 30 September
            release audit. Low-density mask: points where the 28 September model AND the 30 September
            candidate both put total-QSO and stellar intensity below 1% (and 0.01%) of their grid peak,
            exactly as in docs/FULL_SAMPLE_RELEASE_2026-09-30.md.
  stress    51 x 51 grids at r = 18.5, 21, 24.3 (North, South) - the 28 September catch-all comparison.
            Low-density probe per model: that model's own total-QSO and stellar intensities below 1% (0.01%)
            of their peaks on the plane; also the 28 September model's probe applied to all (same points).
  example   northern (g, r, z) = (21.5, 18.5, 22.5), the failing point of the 30 September audit.
Counts are points with p_quasar > 0.5 among scorable (eligible) points.
Writes docs/OUTLIER_TESTS_2026-10-04.json; caches scores under models/multisurvey_psf/work/outlier_tests/.
Usage: python scripts/compare_outlier_tests.py
"""
import json
from pathlib import Path
import sys

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qso_pcolor import PSFMultiSurveyBaseline, Photometry
from qso_pcolor.unified import UnifiedPSFModel
from validate_full_sample_release import run_scores
from validate_psf_catchalls import grid_data
from validate_unified_pilot import run_unified

OUT = Path('models/multisurvey_psf/work/outlier_tests')
OLD = Path(json.loads(Path('docs/FULL_SAMPLE_RELEASE_2026-09-30.json').read_text())['cache'])
MODELS = {'previous_28sep': 'models/multisurvey_psf/previous', 'current_magdep': 'models/multisurvey_psf/current',
          'current_magindep': 'models/multisurvey_psf/current_magindep'}


def intensities(r):
    q = np.logaddexp(r['log_lambda_sameq'], r['log_lambda_fieldq']); return q, r['log_lambda_bkg']


def low(q, b, frac):
    with np.errstate(invalid='ignore'):
        return (q < np.nanmax(q) + np.log(frac)) & (b < np.nanmax(b) + np.log(frac))


def score_all(name, path, g, rcfg, tag):
    base = PSFMultiSurveyBaseline.load(path); out = {}
    out['raw'] = run_scores(base, g, rcfg, OUT/f'{tag}_{name}_raw.npz')
    if name != 'previous_28sep':
        phot = Photometry(g['flux'], g['variance'], g['bands']); sky = SkyCoord(l=g['l']*u.deg, b=g['b']*u.deg, frame='galactic').icrs
        out['full'] = run_unified(UnifiedPSFModel.load(path), phot, sky.ra.deg, sky.dec.deg, g['zprimary'], rcfg, OUT/f'{tag}_{name}_full.npz')
    else:
        out['full'] = out['raw']                                  # the 28 September scorer had no support cut
    return out


def high(r):
    return r['eligible'] & (np.nan_to_num(r['p_quasar']) > .5)


def main():
    OUT.mkdir(parents=True, exist_ok=True); report = dict(definition=__doc__, release={}, stress={}, example={})
    rcfg = json.loads(Path('configs/full_sample_release.json').read_text()); rcfg['batch_size'] = 64
    scfg = json.loads(Path('configs/psf_catchall_comparison.json').read_text())
    ref_model = PSFMultiSurveyBaseline.load(MODELS['previous_28sep']).model
    # --- release grids (15 x 15), original mask
    for h in ('north', 'south'):
        for mag in (18.5, 21.0):
            g = grid_data(ref_model, h, mag, {'validation': rcfg['grid']}); key = f'{h}_{mag}'
            masks = {}
            for frac in (.01, 1e-4):
                m = None
                for lab in ('baseline', 'candidate'):
                    m0 = low(*intensities(np.load(OLD/f'grid_{h}_{mag}_{lab}.npz')), frac); m = m0 if m is None else m & m0
                masks[frac] = m
            entry = {}
            for name, path in MODELS.items():
                s = score_all(name, path, g, rcfg, f'release_{key}')
                entry[name] = {f'{v}_{f}': int((high(s[v]) & masks[f]).sum()) for v in ('full', 'raw') for f in (.01, 1e-4)}
            entry['candidate_30sep'] = {f'raw_{f}': int((high(np.load(OLD/f'grid_{h}_{mag}_candidate.npz')) & masks[f]).sum()) for f in (.01, 1e-4)}
            entry['mask_points'] = {str(f): int(masks[f].sum()) for f in masks}
            report['release'][key] = entry; print('release', key, json.dumps(entry), flush=True)
    # --- stress planes (51 x 51)
    v = scfg['validation']
    for h in ('north', 'south'):
        for mag in v['reference_magnitudes']:
            g = grid_data(ref_model, h, mag, {'validation': v}); key = f'{h}_{mag}'
            res = {name: score_all(name, path, g, rcfg, f'stress_{key}') for name, path in MODELS.items()}
            ref_masks = {f: low(*intensities(res['previous_28sep']['raw']), f) for f in v['low_density_peak_fractions']}
            entry = {}
            for name, s in res.items():
                own = {f: low(*intensities(s['raw']), f) for f in v['low_density_peak_fractions']}
                entry[name] = dict(scorable=int(s['full']['eligible'].sum()), high_anywhere=int(high(s['full']).sum()),
                                   **{f'own_{f}': int((high(s['full']) & own[f]).sum()) for f in own},
                                   **{f'ref28sep_{f}': int((high(s['full']) & ref_masks[f]).sum()) for f in ref_masks},
                                   **{f'nocut_own_{f}': int((high(s['raw']) & own[f]).sum()) for f in own})
            report['stress'][key] = entry; print('stress', key, json.dumps(entry), flush=True)
    # --- the failing example
    soft = ref_model.transform.softening; bands = tuple(f'decals_dr9_north:{b}' for b in 'grz')
    x = np.array([[21.5, 18.5, 22.5]]); sv = soft[[ref_model.transform.bands.index(b) for b in bands]]; a = 2.5/np.log(10)
    flux = 2*sv*np.sinh((22.5 - x)/a - np.log(sv)); var = .03**2*(flux**2 + (2*sv)**2)/a**2
    g = dict(flux=flux, variance=var, bands=bands, l=np.array([180.]), b=np.array([45.]), zprimary=np.array([1.8]))
    for name, path in MODELS.items():
        s = score_all(name, path, g, rcfg, 'example')['full']
        report['example'][name] = dict(eligible=bool(s['eligible'][0]), status=str(s['status'][0]), p_quasar=float(np.nan_to_num(s['p_quasar'][0])))
    print('example', json.dumps(report['example']), flush=True)
    Path('docs/OUTLIER_TESTS_2026-10-04.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
