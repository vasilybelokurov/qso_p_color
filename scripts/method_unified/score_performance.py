#!/usr/bin/env python
"""Score large clean test panels with both promoted models for the method write-up (F12-F14).

Rows are role 3, detected at >=5 sigma in some band with >=2 observed bands, and
outside every recorded development exclusion AND the release-check test rows
(which chose the support threshold), so these panels also confirm that threshold
independently. Never changes a model.

Panels (per hemisphere):
  main   QSOs and background objects with their actual band coverage; background
         objects are scored at redshifts drawn from the QSO test redshifts.
  zgrid  background objects scored at fixed redshifts (for performance versus z).
  masks  QSOs and background objects with artificial band masks.

Usage::

    python scripts/method_unified/score_performance.py --out models/multisurvey_psf/work/method_unified/performance

``--faint-limit S`` applies the PI faint limit at run time (model meta ``faint_limit``: Legacy r,
South then North, S/N >= S; that band also becomes the reference). Objects below it are not scored,
as in the scorer, and are left out of the output; the saved bundles are not changed.
``--panels main`` restricts the run to the listed panels; row selection is unchanged.
``--bundles independent=DIR dependent=DIR`` scores other bundles (e.g. test bundles) on the same rows.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor import Photometry
from qso_pcolor.legacy import is_north
from run_unified_pilot import arrays
from validate_unified_release import excluded_rows

ROOTS = {'independent': 'models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059',
         'dependent': 'models/multisurvey_psf/work/unified_full/20261001/5f4002492dbb849c'}
POINTERS = {'independent': 'models/multisurvey_psf/current', 'dependent': 'models/multisurvey_psf/current_magdep'}
Z_GRID = [0.55, 0.95, 1.45, 1.95, 2.45, 2.95, 3.45, 3.95]
MASKS = {'legacy_grz': lambda b: b.startswith('decals_') and b.split(':')[1] in ('g', 'r', 'z'),
         'legacy_all': lambda b: b.startswith('decals_'),
         'sdss': lambda b: b.startswith('sdss:'),
         'ps1': lambda b: b.startswith('ps1:'),
         'optical_all': lambda b: not b.startswith(('allwise:', 'vhs:')) and not b.endswith((':w1', ':w2'))}
SIZES = dict(main=20000, zgrid=2000, masks=3000)
FAINT_BANDS = ('decals_dr9_south:r', 'decals_dr9_north:r')
CHUNK = 2000
_model = {}


def select(data, n, seed, excluded, z_range=None):
    """Clean role-3 rows per hemisphere, outside every exclusion."""
    bands = tuple(json.loads((Path(ROOTS['independent'])/'layout.json').read_text())['native_labels'])
    p = Photometry(data['flux'], data['variance'], bands); rng = np.random.default_rng(seed)
    with np.errstate(invalid='ignore', divide='ignore'):
        detected = (p.observed & (p.flux/np.sqrt(p.variance) >= 5)).any(axis=1)
    keep = (data['role'] == 3) & detected & (p.observed.sum(axis=1) >= 2) & ~np.isin(data['source_row'], excluded)
    if z_range is not None:
        keep &= (data['zspec'] >= z_range[0]) & (data['zspec'] <= z_range[1])
    north = is_north(data['ra'], data['dec'], data['b'])
    return {h: np.sort(rng.choice(np.flatnonzero(keep & (north == v)), min(n, int((keep & (north == v)).sum())), replace=False))
            for h, v in (('south', False), ('north', True))}


def task(args):
    model_name, kind, rows, z, mask_name, path, faint, pointers = args
    from validate_unified_pilot import run_unified
    from qso_pcolor.unified import UnifiedPSFModel
    if model_name not in _model:
        _model[model_name] = UnifiedPSFModel.load(pointers[model_name])
        if faint is not None:
            _model[model_name].base.model.meta['faint_limit'] = dict(bands=list(FAINT_BANDS), min_snr=float(faint))
    model = _model[model_name]; data = arrays(ROOTS['independent'], kind); bands = model.base.model.transform.bands
    variance = np.array(data['variance'][rows])
    if mask_name is not None:
        keep = np.array([MASKS[mask_name](b) for b in bands]); variance[:, ~keep] = np.inf
    phot = Photometry(data['flux'][rows], variance, bands)
    use = phot.observed.sum(axis=1) >= 1
    if faint is not None:                          # the scorer would flag these, not score them
        has, bright, _ = model.base.model.faint_limit_status(phot); use &= has & bright
    cfg = json.loads(Path('configs/full_sample_release.json').read_text()); cfg['batch_size'] = 32
    run_unified(model, phot.subset(use), data['ra'][rows][use], data['dec'][rows][use], np.asarray(z)[use], cfg, Path(path))
    # Prepared-array row index and redshift of every scored object, in output order.
    np.savez(Path(path).with_suffix('.rows.npz'), rows=np.asarray(rows)[use], z=np.asarray(z)[use])
    return path, int(use.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--smoke', action='store_true', help='tiny panels to test the pipeline')
    parser.add_argument('--faint-limit', type=float, help='Legacy r S/N limit applied at run time')
    parser.add_argument('--panels', nargs='*', default=['main', 'zgrid', 'masks'])
    parser.add_argument('--bundles', nargs='*', default=[], help='name=bundle_dir overrides of the promoted pointers')
    args = parser.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)
    global CHUNK
    if args.smoke:
        SIZES.update(main=12, zgrid=4, masks=4); CHUNK = 8
    root = Path(ROOTS['independent']); cfg = json.loads((root/'config.json').read_text())
    pointers = dict(POINTERS, **dict(b.split('=', 1) for b in args.bundles))
    excluded = excluded_rows(root)
    for r in ROOTS.values():                       # release-check rows chose the support threshold
        with np.load(Path(r)/'release'/'test_rows.npz') as t:
            for kind in ('qso', 'stars'):
                d = arrays(root, kind)
                used = np.concatenate([t[k] for k in t.files if k.startswith(kind+'_')])
                excluded[kind] = np.union1d(excluded[kind], d['source_row'][used])
    data = {k: arrays(root, k) for k in ('qso', 'stars')}
    support = json.loads((root/'bundle'/'support.json').read_text())
    lo, hi = 0.15, 4.35
    rows = {('qso', 'main'): select(data['qso'], SIZES['main'], cfg['seed']+501, excluded['qso'], (lo, hi)),
            ('stars', 'main'): select(data['stars'], SIZES['main'], cfg['seed']+502, excluded['stars']),
            ('stars', 'zgrid'): select(data['stars'], SIZES['zgrid'], cfg['seed']+503, excluded['stars']),
            ('qso', 'masks'): select(data['qso'], SIZES['masks'], cfg['seed']+504, excluded['qso'], (lo, hi)),
            ('stars', 'masks'): select(data['stars'], SIZES['masks'], cfg['seed']+505, excluded['stars'])}
    np.savez(out/'rows.npz', **{f'{k}_{p}_{h}': v for (k, p), halves in rows.items() for h, v in halves.items()})
    jobs = []
    for model_name in ROOTS:
        for (kind, panel), halves in rows.items():
            if panel not in args.panels:
                continue
            for h, rr in halves.items():
                qz = data['qso']['zspec'][rows[('qso', panel if panel != 'zgrid' else 'main')][h]]
                zs = Z_GRID if panel == 'zgrid' else [None]
                masks = list(MASKS) if panel == 'masks' else [None]
                for zval in zs:
                    for mask in masks:
                        for c in range(0, len(rr), CHUNK):
                            part = rr[c:c+CHUNK]
                            if kind == 'qso':
                                z = data['qso']['zspec'][part]
                            elif zval is not None:
                                z = np.full(len(part), zval)
                            else:
                                z = np.resize(qz, len(rr))[c:c+CHUNK]
                            name = f'{model_name}_{kind}_{panel}_{h}' + (f'_z{zval}' if zval else '') + (f'_{mask}' if mask else '') + f'_{c:06d}.npz'
                            jobs.append((model_name, kind, part, z, mask, str(out/name), args.faint_limit, pointers))
    np.save(out/'job_count.npy', len(jobs))
    print('JOBS', len(jobs), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        done = 0
        for future in as_completed([pool.submit(task, j) for j in jobs if not Path(j[5]).exists()]):
            path, n = future.result(); done += 1
            print('DONE', done, Path(path).name, n, flush=True)
    (out/'complete.json').write_text(json.dumps(dict(jobs=len(jobs), support_threshold=support['threshold'],
        faint_limit=args.faint_limit, panels=args.panels, bundles=pointers,
        excluded={k: int(len(v)) for k, v in excluded.items()}, z_grid=Z_GRID, masks=list(MASKS))))
    print('ALL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
