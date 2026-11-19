#!/usr/bin/env python
"""M4: does keeping a weak Legacy r in the likelihood help, against dropping it? (docs/euclid/PLAN_score_all.md)

Held-out main-panel rows (models/multisurvey_psf/work/score_all/performance_d5/rows.npz) with Legacy South r at
S/N > 30, no North r, and PS1/NSC/SkyMapper r at S/N >= 10. Each row is scored three ways with the D5 rules:
  good  - the measured Legacy r (normal scoring);
  weak  - Legacy r degraded to S/N 5: f' = f + N(0, (f/5)^2 - var), var' = (f/5)^2, scored through the fallback
          integral with the weak r in the likelihood;
  none  - Legacy r removed, scored through the fallback alone.
QSOs at their own redshift; background objects at QSO redshifts drawn as in score_performance.py.

Pass (fixed before running): lower 95% bound of AUC(weak) - AUC(none) > -0.002 (paired bootstrap by HEALPix
nside-4 cell for QSOs and stellar cone for background, 1000 resamples), and the median |logit p - logit p_good| of
weak is not above that of none. |p_weak - p_good| quantiles are reported.

    python scripts/score_all/weak_r_masking.py --n 3000 --workers 10
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
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(ROOT/'scripts/euclid'))
RUN = ROOT/'models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059'
POINTERS = {'independent': 'models/multisurvey_psf/current_magindep', 'dependent': 'models/multisurvey_psf/current'}
OUT = ROOT/'models/multisurvey_psf/work/score_all/weak_r_masking'
FALLBACK = ('ps1:r', 'nsc:r', 'skymapper:r')


def task(args):
    model_name, kind, rows, z, seed = args
    warnings.simplefilter('ignore')
    from run_unified_pilot import arrays
    from qso_pcolor import Photometry
    from qso_pcolor.multisurvey_baseline import BlendPolicy
    from qso_pcolor.score import RedshiftMatch
    from qso_pcolor.unified import UnifiedPSFModel
    model = UnifiedPSFModel.load(POINTERS[model_name])
    model.base.model.meta['faint_limit'] = dict(bands=['decals_dr9_south:r', 'decals_dr9_north:r'], min_snr=10.)
    d = arrays(RUN, kind); bands = model.base.model.transform.bands; rs = bands.index('decals_dr9_south:r')
    f = np.array(d['flux'][rows]); v = np.array(d['variance'][rows])
    cfg = json.loads(Path('configs/full_sample_release.json').read_text())
    kw = dict(ra_deg=np.asarray(d['ra'][rows]), dec_deg=np.asarray(d['dec'][rows]), z_primary=z,
              morphology=['PSF']*len(rows), match=RedshiftMatch(half_width_kms=cfg['window_kms']),
              blend_policy=BlendPolicy(**cfg['blend_policy']), ood_flag_sigma=cfg['ood_flag_sigma'],
              separation_arcsec=cfg['fixture_separation_arcsec'], fracflux=cfg['fixture_fracflux'],
              refuse_flagged=False, weak_reference_fallback=True)
    rng = np.random.default_rng(seed)
    vw = (f[:, rs]/5.)**2
    fw = f.copy(); fw[:, rs] = f[:, rs] + rng.normal(0, np.sqrt(np.maximum(vw - v[:, rs], 0)))
    Vw = v.copy(); Vw[:, rs] = vw
    Vn = v.copy(); Vn[:, rs] = np.inf
    out = {}
    for name, (F, V) in dict(good=(f, v), weak=(fw, Vw), none=(f, Vn)).items():
        s, dec = model.score(Photometry(F, V, bands), **kw)
        p = model.quasar_probability(s, calibrated=False)
        out[name] = dict(p=np.where(dec['eligible'], p, np.nan), eligible=np.asarray(dec['eligible']),
                         logr=np.array([np.nan if r is None else r.log_r_per_unit_z for r in s]))
    return model_name, kind, rows, out


def auc(sq, sb):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([sq, sb])); n1 = len(sq)
    return (r[:n1].sum() - n1*(n1 + 1)/2)/(n1*len(sb))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=3000, help='rows per class per model')
    ap.add_argument('--workers', type=int, default=10); ap.add_argument('--chunk', type=int, default=150)
    ap.add_argument('--n-boot', type=int, default=1000)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    from run_unified_pilot import arrays
    from qso_pcolor import Photometry
    import healpy as hp
    panel = np.load(ROOT/'models/multisurvey_psf/work/score_all/performance_d5/rows.npz')
    bands = tuple(json.loads((RUN/'layout.json').read_text())['native_labels'])
    rs, rn = bands.index('decals_dr9_south:r'), bands.index('decals_dr9_north:r'); fb = [bands.index(b) for b in FALLBACK]
    rng = np.random.default_rng(29); chosen = {}
    for kind in ('qso', 'stars'):
        d = arrays(RUN, kind); rows = np.concatenate([panel[k] for k in panel.files if k.startswith(kind + '_main_')])
        F = np.array(d['flux'][rows]); V = np.array(d['variance'][rows])
        with np.errstate(invalid='ignore', divide='ignore'):
            sn = F/np.sqrt(V)
        ok = (sn[:, rs] > 30) & ~(np.isfinite(V[:, rn]) & (V[:, rn] > 0)) & (sn[:, fb] >= 10).any(axis=1)
        chosen[kind] = np.sort(rng.choice(rows[ok], min(a.n, int(ok.sum())), replace=False))
    dq = arrays(RUN, 'qso'); zq = np.asarray(dq['zspec'][chosen['qso']])
    jobs = []
    for model_name in POINTERS:
        for kind in ('qso', 'stars'):
            rr = chosen[kind]
            z = zq if kind == 'qso' else np.resize(zq, len(rr))
            for c in range(0, len(rr), a.chunk):
                jobs.append((model_name, kind, rr[c:c+a.chunk], z[c:c+a.chunk], 1000 + c))
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, (m, k, rows, out) in enumerate(pool.map(task, jobs)):
            print('DONE', i + 1, len(jobs), m, k, flush=True)
            res.setdefault((m, k), []).append((rows, out))
    l = np.load(RUN/'qso/l.npy', mmap_mode='r'); b = np.load(RUN/'qso/b.npy', mmap_mode='r')
    field = np.load(RUN/'stars/field.npy', mmap_mode='r'); report = {}
    boot_rng = np.random.default_rng(31)
    for m in POINTERS:
        cat = {}
        for k in ('qso', 'stars'):
            parts = res[(m, k)]
            cat[k] = dict(rows=np.concatenate([r for r, _ in parts]),
                          **{w: {f: np.concatenate([o[w][f] for _, o in parts]) for f in ('p', 'eligible', 'logr')}
                             for w in ('good', 'weak', 'none')})
        q, s = cat['qso'], cat['stars']
        score = lambda d, w: np.where(d[w]['eligible'], d[w]['logr'], -np.inf)
        qcell = hp.ang2pix(4, np.asarray(l[q['rows']]), np.asarray(b[q['rows']]), lonlat=True); bcell = np.asarray(field[s['rows']])
        base = {w: auc(score(q, w), score(s, w)) for w in ('good', 'weak', 'none')}
        uq, ub = np.unique(qcell), np.unique(bcell); iq = {c: np.flatnonzero(qcell == c) for c in uq}
        ib = {c: np.flatnonzero(bcell == c) for c in ub}; diffs = []
        for _ in range(a.n_boot):
            i = np.concatenate([iq[c] for c in boot_rng.choice(uq, len(uq))]); j = np.concatenate([ib[c] for c in boot_rng.choice(ub, len(ub))])
            diffs.append(auc(score(q, 'weak')[i], score(s, 'weak')[j]) - auc(score(q, 'none')[i], score(s, 'none')[j]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        logit = lambda p: np.log(np.clip(p, 1e-300, 1)) - np.log(np.clip(1 - p, 1e-300, 1))
        dev = {}
        for w in ('weak', 'none'):
            allp = np.concatenate([q[w]['p'], s[w]['p']]); good = np.concatenate([q['good']['p'], s['good']['p']])
            both = np.isfinite(allp) & np.isfinite(good)
            dev[w] = dict(n=int(both.sum()), median_abs_dlogit=float(np.median(np.abs(logit(allp[both]) - logit(good[both])))),
                          abs_dp_q50_q95_q99=[float(x) for x in np.percentile(np.abs(allp[both] - good[both]), [50, 95, 99])])
        report[m] = dict(n_qso=len(q['rows']), n_bkg=len(s['rows']), auc=base,
                         eligible=dict(**{w: dict(qso=float(q[w]['eligible'].mean()), bkg=float(s[w]['eligible'].mean()))
                                          for w in ('good', 'weak', 'none')}),
                         recall_p_gt_half={w: float(np.mean(np.nan_to_num(q[w]['p']) > .5)) for w in ('good', 'weak', 'none')},
                         bkg_p_gt_half={w: float(np.mean(np.nan_to_num(s[w]['p']) > .5)) for w in ('good', 'weak', 'none')},
                         auc_weak_minus_none=dict(difference=base['weak'] - base['none'], ci95=[float(lo), float(hi)],
                                                  se=float(np.std(diffs))),
                         deviation_from_good=dev,
                         pass_=bool(lo > -0.002 and dev['weak']['median_abs_dlogit'] <= dev['none']['median_abs_dlogit']))
    (OUT/'m4.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
