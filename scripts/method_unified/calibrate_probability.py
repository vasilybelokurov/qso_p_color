#!/usr/bin/env python
"""Is p_quasar a calibrated probability on a natural PSF population? (Reliability; log-odds map.)

Outcome on 3 October 2026 (docs/method_unified/probability_calibration.json, F18): raw p_quasar lies
between the two label bounds in 7-8 of 12 bins for the new bundles; outside them it is too low for
p ~ 0 (0.1-2.5% QSOs) and too high above 0.97. The two-parameter log-odds map is driven by the large
p ~ 0 population and pushes mid-range values below both bounds (p 0.5 -> 0.29), so it is reported
but NOT applied to any bundle (no calibration.json).

Natural sample: the 49 held-out test cones (role 3). Every known spectroscopic QSO (DESI DR1 /
DR16Q) inside a cone, plus a random subsample of the cone's stellar-sample objects (known QSOs
were removed from that sample at preparation, so together they are the full PSF population).
All must pass the faint limit (Legacy r S/N >= 10). Stellar objects carry weight
N_pass / n_sampled so the mixture has its natural QSO fraction; rows used by the background light
tests are excluded. Objects are scored at a primary redshift drawn from the test QSO redshifts.

Labels: known QSO = 1; stellar objects = 0, or (upper bound) 1 when flagged as likely QSO
(Quaia or WISE AGN colours, clean_stellar_qso_contamination.py), since the stellar sample still
contains unidentified QSOs. A calibrated p_quasar lies between the two curves.
Recalibration: logit p' = alpha + beta * logit p (weighted logistic regression) fitted on cones with
even index and checked on odd cones (and vice versa). Primary labels count a flagged stellar object
as half a QSO (the flags are not all QSOs, and unflagged faint QSOs remain), between the two bounds.

Writes <out>/<name>.npz scores, docs/method_unified/probability_calibration.json and
plots/method_unified/F18_probability_calibration.png.
Usage: python scripts/method_unified/calibrate_probability.py --bundle name=DIR [--bundle name=DIR ...]
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
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor.multisurvey_data import Photometry
from run_unified_pilot import arrays

RUN = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')
LIGHT = [Path('models/multisurvey_psf/work/background_designs')/t/'rows.npz' for t in ('test1', 'test2', 'test3')]
FAINT = dict(bands=['decals_dr9_south:r', 'decals_dr9_north:r'], min_snr=10.)
EDGES = np.array([0, .01, .03, .1, .2, .35, .5, .65, .8, .9, .97, .99, 1.])
_s = {}


def sample(n_stars, seed=20261005, max_qso=None):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from qso_pcolor.multisurvey import MultiSurveyModel
    cfg = json.loads((RUN/'config.json').read_text())
    cones = [r for r in json.loads((Path(cfg['stellar_root'])/'spatial_roles.json').read_text())['regions'] if r['role'] == 'test']
    m = MultiSurveyModel.load('models/multisurvey_psf/13866e45ef794059/model.json'); m.meta['faint_limit'] = FAINT
    out = {}
    for kind in ('qso', 'stars'):
        d = arrays(RUN, kind); has, bright, _ = m.faint_limit_status(Photometry(d['flux'], d['variance'], m.transform.bands))
        ok = has & bright & (d['role'] == 3)
        cone = np.full(len(ok), -1)
        if kind == 'stars':
            field = np.asarray(d['field']); test = np.array([c['cone'] for c in cones]); cone = np.where(np.isin(field, test), field, -1)
        else:
            cq = SkyCoord(d['ra']*u.deg, d['dec']*u.deg)
            for c in cones:
                cone[cq.separation(SkyCoord(c['ra']*u.deg, c['dec']*u.deg)).deg < .3] = c['cone']
        out[kind] = (ok & (cone >= 0), cone)
    light = np.unique(np.concatenate([v for f in LIGHT if f.exists() for v in np.load(f).values()]))
    qrows = np.flatnonzero(out['qso'][0])[:max_qso]; srows_all = np.setdiff1d(np.flatnonzero(out['stars'][0]), light)
    rng = np.random.default_rng(seed); srows = np.sort(rng.choice(srows_all, min(n_stars, len(srows_all)), replace=False))
    flag = np.load(RUN/'stars'/'qso_flag.npy')
    zq = arrays(RUN, 'qso')['zspec'][qrows]
    return dict(qso=qrows, stars=srows, qcone=out['qso'][1][qrows], scone=out['stars'][1][srows],
                weight_star=len(srows_all)/len(srows), sflag=flag[srows], zq=zq, zs=rng.choice(zq, len(srows)))


def _init(bundle):
    from qso_pcolor.unified import UnifiedPSFModel
    _s['model'] = UnifiedPSFModel.load(bundle)
    if 'faint_limit' not in _s['model'].base.model.meta:
        _s['model'].base.model.meta['faint_limit'] = FAINT


def _score(args):
    from validate_unified_pilot import run_unified
    kind, rows, z, path = args
    d = arrays(RUN, kind); m = _s['model']
    phot = Photometry(d['flux'][rows], d['variance'][rows], m.base.model.transform.bands)
    cfg = json.loads(Path('configs/full_sample_release.json').read_text()); cfg['batch_size'] = 32
    r = run_unified(m, phot, d['ra'][rows], d['dec'][rows], z, cfg, Path(path))
    return path, r['p_quasar'], r['eligible']


def score(bundle, s, work, name, workers):
    jobs = []
    for kind, rows, z in (('qso', s['qso'], s['zq']), ('stars', s['stars'], s['zs'])):
        for i in range(0, len(rows), 1000):
            jobs.append((kind, rows[i:i+1000], z[i:i+1000], str(work/f'{name}_{kind}_{i:06d}.npz')))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init, initargs=(str(bundle),)) as pool:
        res = list(pool.map(_score, jobs))
    p = {k: np.concatenate([r[1] for j, r in zip(jobs, res) if j[0] == k]) for k in ('qso', 'stars')}
    e = {k: np.concatenate([r[2] for j, r in zip(jobs, res) if j[0] == k]) for k in ('qso', 'stars')}
    return p, e


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p/(1 - p))


def platt(x, y, w):
    def nll(t):
        z = t[0] + t[1]*x; pz = 1/(1 + np.exp(-z))
        return -(w*(y*np.log(np.clip(pz, 1e-12, 1)) + (1 - y)*np.log(np.clip(1 - pz, 1e-12, 1)))).sum(), \
            -np.array([(w*(y - pz)).sum(), (w*(y - pz)*x).sum()])
    return minimize(nll, [0., 1.], jac=True, method='L-BFGS-B').x


def reliability(p, y, w):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        out.append(dict(bin=[float(lo), float(hi)], weight=float(w[m].sum()), n=int(m.sum()),
                        mean_p=float(np.average(p[m], weights=w[m])) if m.any() else np.nan,
                        frac=float(np.average(y[m], weights=w[m])) if m.any() else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', action='append', required=True, help='name=bundle_dir')
    ap.add_argument('--n-stars', type=int, default=40000); ap.add_argument('--max-qso', type=int); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--work', type=Path, default=Path('models/multisurvey_psf/work/probability_calibration')); a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True); s = sample(a.n_stars, max_qso=a.max_qso)
    print('sample: QSOs', len(s['qso']), 'stars', len(s['stars']), 'star weight', round(s['weight_star'], 2),
          'flagged stars', int(s['sflag'].sum()), flush=True)
    report = dict(definition=__doc__, n_qso=int(len(s['qso'])), n_stars=int(len(s['stars'])), star_weight=s['weight_star'], bundles={})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(a.bundle), figsize=(4.6*len(a.bundle), 4.2), squeeze=False)
    for ax, item in zip(axes[0], a.bundle):
        name, path = item.split('=', 1)
        p, e = score(Path(path), s, a.work, name, a.workers)
        # Every sampled object counts; objects the scorer does not rank (support-rejected, outside both
        # models) are not quasar candidates, so they enter with p = 0.
        keepq, keeps = np.ones(len(e['qso']), bool), np.ones(len(e['stars']), bool)
        pr = np.concatenate([np.where(e['qso'], np.nan_to_num(p['qso']), 0.), np.where(e['stars'], np.nan_to_num(p['stars']), 0.)])
        w = np.concatenate([np.ones(keepq.sum()), np.full(keeps.sum(), s['weight_star'])])
        y_lo = np.concatenate([np.ones(keepq.sum()), np.zeros(keeps.sum())])
        y_hi = np.concatenate([np.ones(keepq.sum()), s['sflag'][keeps].astype(float)])
        y_mid = .5*(y_lo + y_hi)
        cone = np.concatenate([s['qcone'][keepq], s['scone'][keeps]])
        rep = dict(ranked_qso=int(e['qso'].sum()), ranked_stars=int(e['stars'].sum()), n_qso=int(len(e['qso'])), n_stars=int(len(e['stars'])), natural_qso_fraction=float(np.average(y_lo, weights=w)),
                   mean_p=float(np.average(pr, weights=w)), reliability_known=reliability(pr, y_lo, w), reliability_flagged=reliability(pr, y_hi, w))
        fits = {}
        for half in (0, 1):
            tr = (cone % 2) == half; te = ~tr
            ab = platt(logit(pr[tr]), y_mid[tr], w[tr]); pc = 1/(1 + np.exp(-(ab[0] + ab[1]*logit(pr[te]))))
            ll = lambda q: float(-np.average(y_mid[te]*np.log(np.clip(q, 1e-12, 1)) + (1 - y_mid[te])*np.log(np.clip(1 - q, 1e-12, 1)), weights=w[te]))
            fits[f'train_half_{half}'] = dict(alpha=float(ab[0]), beta=float(ab[1]), test_logloss_raw=ll(pr[te]), test_logloss_calibrated=ll(pc),
                                              test_reliability_calibrated=reliability(pc, y_mid[te], w[te]))
        ab = platt(logit(pr), y_mid, w); rep['alpha_bounds'] = dict(known=platt(logit(pr), y_lo, w).tolist(), flagged=platt(logit(pr), y_hi, w).tolist()); rep['fits'] = fits; rep['alpha'], rep['beta'] = float(ab[0]), float(ab[1])
        report['bundles'][name] = rep
        print(name, json.dumps({k: rep[k] for k in ('ranked_qso', 'ranked_stars', 'natural_qso_fraction', 'mean_p', 'alpha', 'beta')}), flush=True)
        for k, f in fits.items():
            print('  ', k, 'alpha %.3f beta %.3f  test logloss raw %.5f -> calibrated %.5f' % (f['alpha'], f['beta'], f['test_logloss_raw'], f['test_logloss_calibrated']))
        for lo_, hi_ in zip(rep['reliability_known'], rep['reliability_flagged']):
            print(f"   p in [{lo_['bin'][0]:.2f},{lo_['bin'][1]:.2f}) n {lo_['n']:6d} mean p {lo_['mean_p']:.3f}  known-QSO frac {lo_['frac']:.3f}  incl. flagged {hi_['frac']:.3f}")
        for rel, lab, mk in ((rep['reliability_known'], 'known QSOs only', 'o'), (rep['reliability_flagged'], 'incl. flagged likely QSOs', 's')):
            x = [r['mean_p'] for r in rel if r['n'] >= 20]; yv = [r['frac'] for r in rel if r['n'] >= 20]
            ax.plot(x, yv, mk+'-', ms=4, label=lab)
        ax.plot([0, 1], [0, 1], ':', color='0.5'); ax.set(xlabel='predicted $p_Q$', ylabel='observed QSO fraction', title=name)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); Path('plots/method_unified').mkdir(parents=True, exist_ok=True)
    fig.savefig('plots/method_unified/F18_probability_calibration.png', dpi=150)
    Path('docs/method_unified/probability_calibration.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
