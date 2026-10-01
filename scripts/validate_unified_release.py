#!/usr/bin/env python
"""Release checks for a completed full unified bundle; never changes the active model.

Reuses the tested pilot validation pieces, with four release changes:

* the run is given explicitly (``--root``), never found through the pilot config;
* test rows are role 3 *and* outside every recorded development exclusion
  (the stopped covariance trial, the MAP-update test and the refit's own
  stopping panels), so no row that chose the model judges it;
* complete scores are compared with the ACTIVE model as well as the parent
  candidate, and the report is written to its own file;
* sample sizes come from the command line.

The QSO-support cutoff is calibrated on role 2 only, as in the pilot.

Usage::

    python scripts/validate_unified_release.py --root models/multisurvey_psf/work/unified_full/20260930/<tag> \
        --rows 500 --calibration 400 --out docs/UNIFIED_RELEASE_2026-10-01.json
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from qso_pcolor import PSFMultiSurveyBaseline, Photometry
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.legacy import is_north
from qso_pcolor.unified import UnifiedPSFModel
from run_unified_pilot import arrays
from validate_full_sample_release import run_scores, metrics, auc
from validate_psf_catchalls import grid_data
from validate_unified_pilot import mask_phot, calibration, run_unified

EXCLUSION_FILES = (
    'models/multisurvey_psf/work/unified_covariance_trial/20260930/8176eab7484bf5e1/final_assessment_exclusions.npz',
    'models/multisurvey_psf/work/map_update_test/20260930/c108415a6a7a695f/final_assessment_exclusions.npz',
)


def excluded_rows(root):
    """Source-row IDs (full_training_inputs indices) that must not judge the model."""
    out = {'qso': [], 'stars': []}
    for path in list(EXCLUSION_FILES) + [str(root/'stopping'/'final_assessment_exclusions.npz')]:
        with np.load(path) as saved:
            for kind in out:
                if kind in saved.files:
                    out[kind].append(saved[kind])
    return {k: np.unique(np.concatenate(v)) if v else np.array([], np.int64) for k, v in out.items()}


def choose_clean(data, n, bands, seed, excluded, z_range=None):
    """Role-3, >=5-sigma, >=2-band rows outside every recorded exclusion, per hemisphere."""
    rng = np.random.default_rng(seed); p = Photometry(data['flux'], data['variance'], bands)
    with np.errstate(invalid='ignore', divide='ignore'):
        detected = (p.observed & (p.flux/np.sqrt(p.variance) >= 5)).any(axis=1)
    keep = (data['role'] == 3) & detected & (p.observed.sum(axis=1) >= 2) & ~np.isin(data['source_row'], excluded)
    if z_range is not None:
        keep &= (data['zspec'] >= z_range[0]) & (data['zspec'] <= z_range[1])
    north = is_north(data['ra'], data['dec'], data['b'])
    return {h: rng.choice(np.flatnonzero(keep & (north == v)), min(n, int((keep & (north == v)).sum())), replace=False)
            for h, v in (('south', False), ('north', True))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=500, help='test rows per class per hemisphere')
    parser.add_argument('--calibration', type=int, default=400, help='role-2 calibration QSOs per hemisphere')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); root = args.root; started = time.monotonic()
    cfg = json.loads((root/'config.json').read_text())
    if not (root/'bundle'/'manifest.json').exists():
        raise RuntimeError('bundle completion has not finished')
    cfg['validation'] = dict(cfg['validation'], calibration_qsos_per_hemisphere=args.calibration)
    vcfg = cfg['validation']; work = root/'release'; work.mkdir(exist_ok=True)
    base = PSFMultiSurveyBaseline.load(root/'bundle')
    pointer = json.loads(Path(cfg['active_pointer']).read_text())
    alternatives = {'active': PSFMultiSurveyBaseline.load(Path(cfg['active_pointer']).parent/pointer['bundle']),
                    'parent': PSFMultiSurveyBaseline.load(cfg['parent'])}
    for model in alternatives.values():          # identical >=5-sigma reference preference
        model.model.meta['reference_min_snr'] = cfg['reference_min_snr']
    rcfg = json.loads(Path('configs/full_sample_release.json').read_text()); rcfg['batch_size'] = vcfg['score_batch_size']
    bands = base.model.transform.bands
    masks = dict(all=np.ones(len(bands), bool),
                 legacy_optical=np.array([b.startswith('decals_') and b.split(':')[1] in ('g', 'r', 'z') for b in bands]),
                 sdss=np.array([b.startswith('sdss:') for b in bands]), ps1=np.array([b.startswith('ps1:') for b in bands]))
    data = {kind: arrays(root, kind) for kind in ('qso', 'stars')}
    policy = calibration(root, base, data['qso'], cfg, masks)      # role 2 only; cached in bundle/support.json
    manifest = json.loads((root/'bundle'/'manifest.json').read_text())
    manifest['unified_files'] = {name: file_hash(root/'bundle'/name) for name in ('latent.json', 'support.json')}
    manifest['status'] = 'full unified MAP refit candidate; not active; not full probability calibration'
    write_json(root/'bundle'/'manifest.json', manifest)
    unified = UnifiedPSFModel.load(root/'bundle')
    excluded = excluded_rows(root)
    chosen = {kind: choose_clean(d, args.rows, bands, cfg['seed']+101+j, excluded[kind],
                                 base.model.qso.support if kind == 'qso' else None) for j, (kind, d) in enumerate(data.items())}
    for kind, halves in chosen.items():
        for rows in halves.values():
            assert not np.isin(data[kind]['source_row'][rows], excluded[kind]).any()
    np.savez(work/'test_rows.npz', **{kind+'_'+h: r for kind, halves in chosen.items() for h, r in halves.items()})
    report = dict(root=str(root), support_policy=policy, real={}, grids={}, checks={},
                  excluded_rows={k: len(v) for k, v in excluded.items()},
                  test_rows={kind: {h: len(r) for h, r in halves.items()} for kind, halves in chosen.items()},
                  probability_calibration=False, active_model_changed=False,
                  limits=['Role-3 rows outside recorded exclusions; earlier pilot/release samples were drawn from role 3 '
                          'without row records and may overlap a few test rows',
                          'QSO abundance priors inherited; external survey areas approximate',
                          'Random role-3 PSF background objects measure high-score incidence, not labelled contamination'])
    for mode in masks:
        for hemi in ('south', 'north'):
            out = {}
            for kind in ('qso', 'stars'):
                d = data[kind]; rr = chosen[kind][hemi]; phot = mask_phot(d, rr, bands, masks[mode]); keep = phot.observed.sum(axis=1) >= 2
                rr = rr[keep]; phot = phot.subset(keep)
                z = d['zspec'][rr] if kind == 'qso' else np.resize(data['qso']['zspec'][chosen['qso'][hemi]], len(rr))
                values = dict(flux=phot.flux, variance=phot.variance, bands=bands, l=d['l'][rr], b=d['b'][rr], zprimary=z)
                res = {name: run_scores(m, values, rcfg, work/f'{mode}_{hemi}_{kind}_{name}.npz') for name, m in alternatives.items()}
                res['raw'] = run_scores(base, values, rcfg, work/f'{mode}_{hemi}_{kind}_raw.npz')
                res['unified'] = run_unified(unified, phot, d['ra'][rr], d['dec'][rr], z, rcfg, work/f'{mode}_{hemi}_{kind}_unified.npz')
                out[kind] = res
                report['real'].setdefault(mode+'_'+hemi, {})[kind] = dict(rows=len(rr),
                    **{name: metrics(r, rcfg) for name, r in res.items()},
                    support_retention=float(np.mean(~res['unified']['support_rejected'])) if len(rr) else None)
            entry = report['real'][mode+'_'+hemi]
            entry['auc'] = {name: auc(out['qso'][name], out['stars'][name]) for name in out['qso']}
            entry['high_qso_background'] = {name: int((out['stars'][name]['eligible'] & (out['stars'][name]['p_quasar'] > rcfg['high_qso_probability'])).sum())
                                            for name in out['stars']}
            report['checks'][f'ranking_vs_active_{mode}_{hemi}'] = entry['auc']['unified'] >= entry['auc']['active'] - 0.005
            report['checks'][f'support_retention_{mode}_{hemi}'] = entry['qso']['support_retention'] >= vcfg['minimum_test_support_retention']
            print('REAL', mode, hemi, json.dumps(dict(auc=entry['auc'], high_qso_background=entry['high_qso_background'],
                  retention=entry['qso']['support_retention'])), flush=True)
            write_json(work/'progress.json', report)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    old_cache = Path(json.loads(Path('docs/FULL_SAMPLE_RELEASE_2026-09-30.json').read_text())['cache'])
    for hemi in ('south', 'north'):
        for mag in rcfg['grid']['reference_magnitudes'][:2]:
            g = grid_data(base.model, hemi, mag, {'validation': rcfg['grid']}); phot = Photometry(g['flux'], g['variance'], g['bands'])
            sky = SkyCoord(l=g['l']*u.deg, b=g['b']*u.deg, frame='galactic').icrs
            low = np.ones(len(phot.flux), bool)
            for label in ('baseline', 'candidate'):
                saved = np.load(old_cache/f'grid_{hemi}_{mag}_{label}.npz')
                q = np.logaddexp(saved['log_lambda_sameq'], saved['log_lambda_fieldq']); b = saved['log_lambda_bkg']
                low &= (q < np.nanmax(q)+np.log(.01)) & (b < np.nanmax(b)+np.log(.01))
            res = {name: run_scores(m, g, rcfg, work/f'grid_{hemi}_{mag}_{name}.npz') for name, m in alternatives.items()}
            res['unified'] = run_unified(unified, phot, sky.ra.deg, sky.dec.deg, g['zprimary'], rcfg, work/f'grid_{hemi}_{mag}_unified.npz')
            report['grids'][f'{hemi}_{mag}'] = {name: int((low & r['eligible'] & (r['p_quasar'] > .5)).sum()) for name, r in res.items()}
            print('GRID', hemi, mag, report['grids'][f'{hemi}_{mag}'], flush=True)
    report['checks']['northern_original_tail'] = all(report['grids'][f'north_{m}']['unified'] == 0 for m in rcfg['grid']['reference_magnitudes'][:2])
    report['checks']['southern_original_tail'] = all(report['grids'][f'south_{m}']['unified'] <= report['grids'][f'south_{m}']['active']
                                                     for m in rcfg['grid']['reference_magnitudes'][:2])
    report['checks']['support_has_rejection_power'] = policy['informative_threshold']
    report['checks']['active_pointer_unchanged'] = file_hash(Path(cfg['active_pointer'])) == json.loads((root/'prepared.json').read_text())['active_pointer_hash']
    report['checks_pass'] = all(report['checks'].values()); report['elapsed_seconds'] = time.monotonic()-started
    write_json(work/'report.json', report); write_json(args.out, report)
    print('DONE', json.dumps(report['checks']), flush=True)


if __name__ == '__main__':
    main()
