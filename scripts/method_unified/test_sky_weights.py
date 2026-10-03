#!/usr/bin/env python
"""Step 4: how should background component weights vary on the sky?

Background: the cleaned binned mixture (160 native components) of a test bundle
(build_binned_test_bundle.py). Component shapes are fixed; only weights change. Options:
  global      one weight vector (refitted on the training rows).
  healpix     the production hierarchy (joint_spatial.py): nside-4 cells pooled to nside-2 parents
              and to global with count/(count + n0), for n0 in N0S.
  smooth      softmax gate w_k(l, b) = softmax_k(a_k + beta_k . f(l, b)) (joint_spatial.fit_softmax_gate;
              features csc|b|, cos l, sin l, csc|b| cos l, sign b, centred), L2 penalty RIDGE on beta;
              also tabulated at nside-16 pixel centres (gate_to_spatial_weights), the scorer's form.
Training rows: fit/select cones, Legacy r S/N >= 10, QSO flag removed (clean_stellar_qso_contamination.py).
Held-out rows: test cones (role 3, separate sky regions), same cuts, light-test rows excluded.
Score: per-row ln p(other observed bands | Legacy r) with row-specific weights,
  logsumexp_k(ln w_k + ln p_k(all)) - logsumexp_k(ln w_k + ln p_k(Legacy r)), per mag^(N-1).
Reported overall and by Legacy r and |b| bins, as paired differences against global.
Writes docs/method_unified/sky_weights_test.json; caches log densities under --work.

Usage: python scripts/method_unified/test_sky_weights.py --bundle models/multisurvey_psf/work/binned_bundle_clean/independent
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor.joint_spatial import fit_joint_spatial_log_prob, fit_softmax_gate, gate_to_spatial_weights, gate_weights
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.spatial import component_log_prob, fit_component_weights
from complete_unified_pilot import fit_arrays

RUN = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')
LIGHT = [Path('models/multisurvey_psf/work/background_designs')/t/'rows.npz' for t in ('test1', 'test2', 'test3')]
N0S = (100., 1000., 10000.)
RIDGE = 1.
MAG = [15.5, 18, 19, 20, 21, 22, 23, 24.5]
BABS = [25, 35, 45, 60, 90]
_s = {}


def _init(model_path):
    _s['model'] = MultiSurveyModel.load(model_path); _s['data'] = fit_arrays(RUN, 'stars')


def _lp(rows):
    m, d = _s['model'], _s['data']; p = Photometry(d['flux'][rows], d['variance'][rows], m.transform.bands); f = m.transform(p)
    joint = component_log_prob(m.background, f.x, f.cov, f.observed)
    has, _, band = m.faint_limit_status(p); anchor = np.zeros_like(f.observed); anchor[np.arange(len(rows)), band] = True
    return joint, component_log_prob(m.background, f.x, f.cov, anchor & f.observed), f.x[np.arange(len(rows)), band]


def log_densities(model_path, rows, workers, cache):
    if cache.exists():
        z = np.load(cache); return z['joint'], z['anchor'], z['mag']
    chunks = [rows[i:i+1000] for i in range(0, len(rows), 1000)]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init, initargs=(str(model_path),)) as pool:
        parts = list(pool.map(_lp, chunks))
    out = [np.concatenate([q[i] for q in parts]) for i in range(3)]
    np.savez(cache, joint=out[0], anchor=out[1], mag=out[2]); return out


def score(weights, joint, anchor):
    lw = np.log(np.maximum(weights, 1e-300))
    return logsumexp(lw + joint, axis=1) - logsumexp(lw + anchor, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--work', type=Path, default=Path('models/multisurvey_psf/work/sky_weights'))
    p.add_argument('--n-train', type=int, default=300000); p.add_argument('--n-eval', type=int, default=60000)
    p.add_argument('--workers', type=int, default=12); a = p.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
    model = MultiSurveyModel.load(a.bundle/'model.json'); data = fit_arrays(RUN, 'stars'); cfg = json.loads((RUN/'config.json').read_text())
    flag = np.load(RUN/'stars'/'qso_flag.npy'); role = np.asarray(data['role'])
    has, bright, _ = model.faint_limit_status(Photometry(data['flux'], data['variance'], model.transform.bands))
    ok = has & bright & ~flag; rng = np.random.default_rng(20261003)
    light = np.unique(np.concatenate([v for f in LIGHT if f.exists() for v in np.load(f).values()]))
    train = np.flatnonzero(ok & np.isin(role, cfg['fit_roles'])); train = np.sort(rng.choice(train, min(a.n_train, len(train)), replace=False))
    ev = np.setdiff1d(np.flatnonzero(ok & (role == 3)), light); ev = np.sort(rng.choice(ev, min(a.n_eval, len(ev)), replace=False))
    print('train', len(train), 'eval', len(ev), flush=True)
    lt, _, _ = log_densities(a.bundle/'model.json', train, a.workers, a.work/'train_lp.npz')
    ej, ea, emag = log_densities(a.bundle/'model.json', ev, a.workers, a.work/'eval_lp.npz')
    l, b = np.asarray(data['l']), np.asarray(data['b']); w0 = model.background.weights
    wg, _ = fit_component_weights(lt, w0, np.ones(len(train)), max_iter=500, tol=1e-7)
    options = {'bundle (bin shares)': np.tile(w0, (len(ev), 1)), 'global': np.tile(wg, (len(ev), 1))}
    sp = cfg['spatial']
    for n0 in N0S:
        s = fit_joint_spatial_log_prob(model.background, lt, l[train], b[train], np.ones(len(train)), nside=sp['nside'],
                                       nside_parent=sp['nside_parent'], n0=n0, max_iter=sp['max_iter'], tol=sp['tol'], meta={})
        options[f'healpix n0={n0:g}'] = s.evaluate(l[ev], b[ev])[0]
    gate = fit_softmax_gate(lt, l[train], b[train], wg, ridge=RIDGE, max_iter=3000)
    print('smooth fit:', gate['message'], gate['n_iter'], flush=True)
    options['smooth (l,b)'] = gate_weights(gate, l[ev], b[ev])
    options['smooth tabulated nside16'] = gate_to_spatial_weights(gate, nside=16, min_abs_b_deg=25., meta={}).evaluate(l[ev], b[ev])[0]
    sc = {k: score(w, ej, ea) for k, w in options.items()}
    ref = sc['global']; babs = np.abs(b[ev]); report = dict(definition=__doc__, n_train=int(len(train)), n_eval=int(len(ev)), options={})
    for k, v in sc.items():
        d = v - ref
        rep = dict(mean=float(v.mean()), diff=float(d.mean()), diff_se=float(d.std()/np.sqrt(len(d))),
                   by_mag=[dict(bin=[lo, hi], n=int(((emag >= lo) & (emag < hi)).sum()), diff=float(d[(emag >= lo) & (emag < hi)].mean()),
                                se=float(d[(emag >= lo) & (emag < hi)].std()/np.sqrt(max(((emag >= lo) & (emag < hi)).sum(), 1))))
                           for lo, hi in zip(MAG[:-1], MAG[1:])],
                   by_b=[dict(bin=[lo, hi], n=int(((babs >= lo) & (babs < hi)).sum()), diff=float(d[(babs >= lo) & (babs < hi)].mean()),
                              se=float(d[(babs >= lo) & (babs < hi)].std()/np.sqrt(max(((babs >= lo) & (babs < hi)).sum(), 1))))
                         for lo, hi in zip(BABS[:-1], BABS[1:])])
        report['options'][k] = rep
        print(f"{k:22s} mean {rep['mean']:8.4f}  vs global {rep['diff']:+.4f} +- {rep['diff_se']:.4f}")
        print('    by r   ', ' '.join(f"{x['bin'][0]:g}:{x['diff']:+.3f}" for x in rep['by_mag']))
        print('    by |b| ', ' '.join(f"{x['bin'][0]:g}:{x['diff']:+.3f}+-{x['se']:.3f}" for x in rep['by_b']))
    Path('docs/method_unified/sky_weights_test.json').write_text(json.dumps(report, indent=1))
    np.savez(a.work/'smooth_gate.npz', a=gate['a'], beta=gate['beta'], centre=gate['centre'], global_weights=wg)


if __name__ == '__main__':
    main()
