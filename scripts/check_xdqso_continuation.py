#!/usr/bin/env python
"""Continue the published quasar slices on their recorded training population.

The historical selection is replayed explicitly, including release 9010-only
south. Counts and every slice membership must match before any EM step runs.
Outputs are diagnostics, never promoted model artifacts. Test rows enter only
the evaluation, not the continuation fit. Saved slices permit resuming a run.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

# This temporary path will become the script's own directory when committed.
sys.path.insert(0, str(Path.cwd() / 'scripts'))


def replay(h, model, output):
    from qso_pcolor.store import load
    from qso_pcolor.features import RelativeFluxTransform, deredden
    from qso_pcolor.legacy import morphology_status
    from qso_pcolor.data import galactic_from_equatorial
    from qso_pcolor.background import galactic_healpix
    from train_qso_model import deduplicate
    bands = ('g', 'r', 'z', 'w1', 'w2')
    parts = []
    for channel, name in enumerate(('desi_dr1_qso', 'dr16q_dr9')):
        keys = ['ra', 'dec', 'zspec', 'release', 'type', 'maskbits']
        keys += [p + b for p in ('flux_', 'flux_ivar_', 'mw_transmission_') for b in bands]
        if channel:
            keys += ['ls_ra', 'ls_dec', 'zwarning']
        r = load(name, tuple(keys))
        z = r['zspec']; lo, hi = model.meta['z_range']
        use = ((z > lo) & (z < hi) & (r['release'] == model.meta['release'])
               & (r['maskbits'] == 0) & (morphology_status(r['type']) == 'point'))
        if channel:
            use &= r['zwarning'] == 0
            r['ra'], r['dec'] = r['ls_ra'], r['ls_dec']
        p = {k: r[k][use] for k in ('ra', 'dec', 'zspec')}
        for key in ('flux_', 'flux_ivar_', 'mw_transmission_'):
            p[key] = np.stack([r[key + b][use] for b in bands], axis=1)
        p['channel'] = np.full(use.sum(), channel, dtype=np.int8)
        parts.append(p)
    r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    keep = deduplicate(r['ra'], r['dec'], r['zspec'], model.meta['dedup_arcsec'])
    r = {k: v[keep] for k, v in r.items()}
    f, v = deredden(r['flux_'], r['flux_ivar_'], r['mw_transmission_'])
    fs = RelativeFluxTransform('r', model.meta['validity']['min_ref_snr'])(f, v, bands)
    ok = fs.usable(model.meta['validity']['min_dims']) & np.isfinite(fs.ref_mag)
    fs = fs.subset(np.flatnonzero(ok))
    l, b = galactic_from_equatorial(r['ra'][ok], r['dec'][ok])
    held = np.isin(galactic_healpix(l, b, model.meta['holdout_nside']), model.meta['holdout_blocks'])
    assert int((~held).sum()) == model.meta['n_train'], (h, int((~held).sum()), model.meta['n_train'])
    assert int(held.sum()) == model.meta['n_holdout']
    data = dict(x=fs.x, cov=fs.cov, observed=fs.observed, z=r['zspec'][ok], held=held,
                channel=r['channel'][ok])
    output.mkdir(parents=True, exist_ok=True)
    for k, value in data.items():
        np.save(output / f'{k}.npy', value)
    print(f'{h}: replay verified, {int((~held).sum()):,} train, {int(held.sum()):,} held', flush=True)


def one_slice(job):
    from qso_pcolor.gaussmix import GaussianMixture
    from qso_pcolor.xd import fit_xd
    root, j, bounds, mixdict, iterations, expected, identity = job
    output = Path(root) / f'slice_{j:02d}.json'
    if output.exists():
        cached = json.loads(output.read_text())
        if cached['identity'] == identity:
            return j, cached
    d = {k: np.load(Path(root) / f'{k}.npy', mmap_mode='r')
         for k in ('x', 'cov', 'observed', 'z', 'held', 'channel')}
    inside = (d['z'] >= bounds[0]) & (d['z'] < bounds[1])
    fit, held = inside & ~d['held'], inside & d['held']
    assert fit.sum() == expected, (j, int(fit.sum()), expected)
    mix = GaussianMixture.from_dict(mixdict)
    t0 = time.monotonic()
    res = fit_xd(d['x'][fit], d['cov'][fit], observed=d['observed'][fit], init=mix,
                 max_iter=iterations, tol=0., regularization=1e-6, labels=mix.labels)
    before = mix.log_prob(d['x'][held], d['cov'][held], observed=d['observed'][held])
    after = res.mixture.log_prob(d['x'][held], d['cov'][held], observed=d['observed'][held])
    delta = after - before
    def stats(values):
        return dict(n=int(values.size), mean_delta=float(values.mean()),
                    median_abs_delta=float(np.median(np.abs(values)))) if values.size else dict(n=0)
    report = dict(identity=identity, slice=j, n_fit=int(fit.sum()), n_held=int(held.sum()),
                  held=stats(delta), by_channel={name: stats(delta[d['channel'][held] == k])
                      for k, name in enumerate(('DESI', 'SDSS'))},
                  iterations=res.n_iter, seconds=time.monotonic() - t0,
                  mixture=res.mixture.to_dict())
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(report)); tmp.replace(output)
    return j, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, default=Path('models/legacy_psf_xdqso/current'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--iterations', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    from qso_pcolor.baseline import XDQSOBaseline
    bl = XDQSOBaseline.load(args.bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(dict(manifest=bl.manifest, iterations=args.iterations,
                                             script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                                        sort_keys=True).encode()).hexdigest()
    jobs = []
    for h, parts in bl.parts.items():
        q = parts['qso']; root = args.output / h
        replay(h, q, root)
        edges = np.linspace(*q.meta['z_range'], len(q.mixtures) + 1)
        for j, mix in enumerate(q.mixtures):
            width = edges[j + 1] - edges[j]
            bounds = [edges[j] - q.meta['overlap'] * width, edges[j + 1] + q.meta['overlap'] * width]
            jobs.append((str(root), j, bounds, mix.to_dict(), args.iterations, int(q.n_train[j]), identity))
    jobs.sort(key=lambda x: -x[5])
    results = {h: {} for h in bl.parts}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(one_slice, job): Path(job[0]).name for job in jobs}
        for future in as_completed(futures):
            h = futures[future]; j, rec = future.result(); results[h][j] = rec
            print(f'{h} slice {j:02d}: n={rec["n_fit"]:,}, held {rec["held"]}, {rec["seconds"]:.0f}s', flush=True)
    for h, parts in bl.parts.items():
        from qso_pcolor.gaussmix import GaussianMixture
        from qso_pcolor.qso_model import SlicedColourRedshiftModel
        q = parts['qso']
        mixes = [GaussianMixture.from_dict(results[h][j]['mixture']) for j in range(len(q.mixtures))]
        continued = SlicedColourRedshiftModel(q.z_centres, mixes, q.n_train, q.system, q.labels,
                                              dict(q.meta, diagnostic_continuation=args.iterations))
        continued.save(args.output / f'{h}_continued_qso.json')
    summary = {h: [dict((k, v) for k, v in results[h][j].items() if k != 'mixture')
                   for j in sorted(results[h])] for h in results}
    (args.output / 'report.json').write_text(json.dumps(dict(bundle=bl.bundle_id, identity=identity,
                                                          hemispheres=summary), indent=2))


if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    main()
