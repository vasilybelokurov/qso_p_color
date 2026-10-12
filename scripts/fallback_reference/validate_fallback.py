#!/usr/bin/env python
"""Validate scoring without Legacy r (qso_pcolor.fallback_reference) on held-out in-footprint objects.

Objects are role-3 test rows outside every development exclusion and the release-check rows (as
scripts/method_unified/score_performance.py), with Legacy r S/N >= 10 AND the fallback band at S/N >= 10.
Each is scored three ways with the same model:
  full      all observed bands, Legacy r reference (the promoted scorer);
  legacy_r  non-Legacy bands + Legacy r only: the baseline with the true Legacy r and the fallback mode's bands;
  fallback  non-Legacy bands only (Legacy r unobserved), Legacy r integrated out (``legacy_r_fallback=True``).
The fallback-vs-legacy_r difference is the cost of not knowing Legacy r. Modes:
  ps1        drop all Legacy bands; PS1 r is the fallback band;
  skymapper  drop Legacy, PS1, SDSS and NSC; SkyMapper r is the fallback band (deep-south case).
Single-object part: QSOs at their own z, background objects at z drawn from the QSO test redshifts.
Blend part: synthetic QQ/QS blends of held-out objects (scripts/validate_blend_mode.py construction) whose
components have the fallback band at S/N >= 10, scored with the Legacy r and the fallback blend scorers.

Acceptance (declared before running, 13 October 2026): single objects - AUC(fallback) >= AUC(legacy_r) - 0.01
with the 95% paired sky-bootstrap upper bound on the loss < 0.02, background p>0.5 rate up by < 0.5 pp, QSO
recall at p>0.5 down by < 5 pp; blends - AUC loss <= 0.02. Log score and unranked fractions are reported.

Usage: python scripts/fallback_reference/validate_fallback.py --part single --mode ps1 --model current --n 3000
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(ROOT/'scripts'/'method_unified')); sys.path.insert(0, str(ROOT/'src'))
from run_unified_pilot import arrays
from validate_unified_release import excluded_rows
from validate_blend_mode import auc, snr_r

RUN = ROOT/'models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059'
OUT = ROOT/'models/multisurvey_psf/work/fallback_reference/validation'
DOC = ROOT/'docs/fallback_reference'
MODES = {'ps1': dict(band='ps1:r', drop=('decals_',)),
         'skymapper': dict(band='skymapper:r', drop=('decals_', 'ps1:', 'sdss:', 'nsc:'))}
SCHEMES = ('full', 'legacy_r', 'fallback')
_s = {}


def band_snr(d, band, lab):
    j = lab.index(band); f = np.asarray(d['flux'][:, j]); v = np.asarray(d['variance'][:, j])
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.asarray(d['observed'][:, j]) & (v > 0), f/np.sqrt(v), 0.)


def masked_variance(V, lab, scheme, mode):
    drop = np.array([b.startswith(MODES[mode]['drop']) for b in lab])
    V = V.copy()
    if scheme == 'legacy_r':
        V[:, drop & ~np.isin(lab, ('decals_dr9_south:r', 'decals_dr9_north:r'))] = np.inf
    elif scheme == 'fallback':
        V[:, drop] = np.inf
    return V


def excluded():
    ex = excluded_rows(RUN)
    for r in (RUN, ROOT/'models/multisurvey_psf/work/unified_full/20261001/5f4002492dbb849c'):
        with np.load(Path(r)/'release'/'test_rows.npz') as t:
            for kind in ('qso', 'stars'):
                d = arrays(RUN, kind); used = np.concatenate([t[k] for k in t.files if k.startswith(kind+'_')])
                ex[kind] = np.union1d(ex[kind], d['source_row'][used])
    return ex


def select(kind, mode, n, seed, ex):
    lab = json.loads((RUN/'layout.json').read_text())['native_labels']; d = arrays(RUN, kind)
    sr, _ = snr_r(d, np.arange(len(d['role'])), lab); sf = band_snr(d, MODES[mode]['band'], lab)
    keep = (np.asarray(d['role']) == 3) & (sr >= 10) & (sf >= 10) & ~np.isin(d['source_row'], ex[kind])
    if kind == 'qso':
        z = np.asarray(d['zspec']); keep &= (z > .15) & (z < 4.35)
    else:
        keep &= ~np.load(RUN/'stars'/'qso_flag.npy')
    idx = np.flatnonzero(keep); rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, min(n, len(idx)), replace=False)), int(len(idx))


# ---- single objects ---------------------------------------------------------------------------------
def _init(path):
    from qso_pcolor.unified import UnifiedPSFModel
    _s['m'] = UnifiedPSFModel.load(path)


def _score_single(a):
    from qso_pcolor.multisurvey_data import Photometry
    from qso_pcolor.score import RedshiftMatch
    from qso_pcolor.multisurvey_baseline import BlendPolicy
    F, V, ra, dec, z, fallback = a; m = _s['m']; n = len(F)
    s, d = m.score(Photometry(F, V, tuple(m.base.model.transform.bands)), ra_deg=ra, dec_deg=dec, z_primary=z,
                   morphology=['PSF']*n, match=RedshiftMatch(half_width_kms=2000.),
                   blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2), ood_flag_sigma=4.,
                   separation_arcsec=6., fracflux=.05, legacy_r_fallback=fallback)
    p = m.quasar_probability(s)
    return [(float(p[i]), bool(d['eligible'][i]), str(d['reason'][i]),
             float('nan') if s[i] is None else float(s[i].ref_mag),
             '' if s[i] is None else ','.join(f for f in s[i].quality_flags if f.startswith('legacy_r'))) for i in range(n)]


def run_single(args):
    lab = json.loads((RUN/'layout.json').read_text())['native_labels']; ex = excluded()
    rows = {k: select(k, args.mode, args.n, 20261013 + i, ex) for i, k in enumerate(('qso', 'stars'))}
    q = arrays(RUN, 'qso'); zq = np.asarray(q['zspec'])[rows['qso'][0]]
    rng = np.random.default_rng(20261014)
    zs = {'qso': zq, 'stars': rng.choice(zq, len(rows['stars'][0]))}
    path = ROOT/'models/multisurvey_psf'/args.model
    res = {}
    for kind in ('qso', 'stars'):
        d = arrays(RUN, kind); r = rows[kind][0]
        F = np.asarray(d['flux'])[r]; V0 = np.asarray(d['variance'])[r]
        ra, dec = np.asarray(d['ra'])[r], np.asarray(d['dec'])[r]
        for scheme in SCHEMES:
            V = masked_variance(V0, lab, scheme, args.mode); B = 16
            jobs = [(F[i:i+B], V[i:i+B], ra[i:i+B], dec[i:i+B], zs[kind][i:i+B], scheme == 'fallback') for i in range(0, len(r), B)]
            with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init, initargs=(str(path),)) as pool:
                out = [x for chunk in pool.map(_score_single, jobs) for x in chunk]
            res[(kind, scheme)] = dict(p=np.array([o[0] for o in out]), eligible=np.array([o[1] for o in out]),
                                       reason=np.array([o[2] for o in out]), ref_mag=np.array([o[3] for o in out]),
                                       flags=np.array([o[4] for o in out]))
            print(kind, scheme, 'eligible %.4f' % res[(kind, scheme)]['eligible'].mean(), flush=True)
    cells = {k: np.asarray(arrays(RUN, k)['cell'])[rows[k][0]] for k in ('qso', 'stars')}
    tag = f'single_{args.mode}_{args.model}'
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT/f'{tag}.npz', **{f'{k}_{s}_{f}': v for (k, s), dd in res.items() for f, v in dd.items()},
             qso_rows=rows['qso'][0], stars_rows=rows['stars'][0], qso_cell=cells['qso'], stars_cell=cells['stars'],
             qso_z=zs['qso'], stars_z=zs['stars'])
    rep = dict(mode=args.mode, model=args.model, available=dict(qso=rows['qso'][1], stars=rows['stars'][1]),
               n=dict(qso=len(rows['qso'][0]), stars=len(rows['stars'][0])), schemes={})
    both = res[('qso', 'legacy_r')]['eligible'] & res[('qso', 'fallback')]['eligible']
    bothb = res[('stars', 'legacy_r')]['eligible'] & res[('stars', 'fallback')]['eligible']
    for s in SCHEMES:
        pq, pb = res[('qso', s)]['p'], res[('stars', s)]['p']; eq, eb = res[('qso', s)]['eligible'], res[('stars', s)]['eligible']
        rep['schemes'][s] = dict(
            unranked_qso=float(1 - eq.mean()), unranked_bkg=float(1 - eb.mean()),
            auc_own=float(auc(pq[eq], pb[eb])), auc_common=float(auc(pq[both], pb[bothb])),
            recall_p05=float((pq[both] > .5).mean()), bkg_p05=float((pb[bothb] > .5).mean()),
            log_score_qso=float(np.mean(np.log(np.clip(pq[both], 1e-6, 1)))),
            log_score_bkg=float(np.mean(np.log(np.clip(1 - pb[bothb], 1e-6, 1)))),
            reasons_qso={k: int(v) for k, v in zip(*np.unique(res[('qso', s)]['reason'][~eq], return_counts=True))},
            reasons_bkg={k: int(v) for k, v in zip(*np.unique(res[('stars', s)]['reason'][~eb], return_counts=True))})
    # paired sky bootstrap over HEALPix cells of the AUC loss (legacy_r - fallback), on the common rows
    rng = np.random.default_rng(7); uq, ub = np.unique(cells['qso'][both]), np.unique(cells['stars'][bothb])
    cq, cb = cells['qso'][both], cells['stars'][bothb]
    pa, pf = res[('qso', 'legacy_r')]['p'][both], res[('qso', 'fallback')]['p'][both]
    ba, bf = res[('stars', 'legacy_r')]['p'][bothb], res[('stars', 'fallback')]['p'][bothb]
    iq = {c: np.flatnonzero(cq == c) for c in uq}; ib = {c: np.flatnonzero(cb == c) for c in ub}
    loss, dbk, drec = [], [], []
    for _ in range(200):
        sq = np.concatenate([iq[c] for c in rng.choice(uq, len(uq))]); sb = np.concatenate([ib[c] for c in rng.choice(ub, len(ub))])
        loss.append(auc(pa[sq], ba[sb]) - auc(pf[sq], bf[sb]))
        dbk.append((bf[sb] > .5).mean() - (ba[sb] > .5).mean()); drec.append((pa[sq] > .5).mean() - (pf[sq] > .5).mean())
    L = rep['schemes']['legacy_r']; Fb = rep['schemes']['fallback']
    rep['auc_loss'] = dict(value=L['auc_common'] - Fb['auc_common'], ci95=[float(np.nanpercentile(loss, 2.5)), float(np.nanpercentile(loss, 97.5))])
    rep['bkg_p05_increase'] = dict(value=Fb['bkg_p05'] - L['bkg_p05'], ci95=[float(np.nanpercentile(dbk, 2.5)), float(np.nanpercentile(dbk, 97.5))])
    rep['recall_loss'] = dict(value=L['recall_p05'] - Fb['recall_p05'], ci95=[float(np.nanpercentile(drec, 2.5)), float(np.nanpercentile(drec, 97.5))])
    rep['grid_truncated_fraction'] = float(np.mean([',legacy_r_grid_truncated' in ','+f for f in
                                                    np.concatenate([res[('qso', 'fallback')]['flags'], res[('stars', 'fallback')]['flags']])]))
    rep['legacy_r_recovery'] = dict(  # posterior-mean Legacy r minus the scorer's Legacy r reference, both eligible
        qso_median=float(np.nanmedian(res[('qso', 'fallback')]['ref_mag'][both] - res[('qso', 'full')]['ref_mag'][both])),
        qso_nmad=float(1.4826*np.nanmedian(np.abs(res[('qso', 'fallback')]['ref_mag'][both] - res[('qso', 'full')]['ref_mag'][both]))))
    rep['accept'] = bool(rep['auc_loss']['value'] <= .01 and rep['auc_loss']['ci95'][1] < .02 and
                         rep['bkg_p05_increase']['value'] < .005 and rep['recall_loss']['value'] < .05)
    DOC.mkdir(parents=True, exist_ok=True); (DOC/f'validation_{tag}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ---- blends -----------------------------------------------------------------------------------------
def _init_blend(path):
    from qso_pcolor.blend import BlendModel
    _s['b'] = BlendModel.load(path, legacy_r_fallback=True)


def _score_blend(a):
    from qso_pcolor.multisurvey_data import Photometry
    F, V, bands, z, l, b = a
    o = _s['b'].score(Photometry(F, V, bands), ra_deg=0., dec_deg=0., z_qso=z, dereddened=True, l_deg=l, b_deg=b)[0]
    return o.status, o.log_bf_qq_qs, o.p_qq, ','.join(o.notes)


def run_blend(args):
    from validate_blend_mode import blend_photometry
    lab = json.loads((RUN/'layout.json').read_text())['native_labels']; ex = excluded()
    qrows, _ = select('qso', args.mode, 10**9, 20261015, ex); srows, _ = select('stars', args.mode, 10**9, 20261016, ex)
    q, st = arrays(RUN, 'qso'), arrays(RUN, 'stars'); z = np.asarray(q['zspec'])
    _, qh = snr_r(q, np.arange(len(q['role'])), lab); _, sh = snr_r(st, np.arange(len(st['role'])), lab)
    rng = np.random.default_rng(20261017); pairs = []
    for kind in ('qq', 'qs'):
        got = 0
        while got < args.n:
            a = rng.choice(qrows); h = qh[a]
            cand = qrows[(np.abs(z[qrows] - z[a]) < .02) & (qh[qrows] == h) & (qrows != a)] if kind == 'qq' else srows[sh[srows] == h]
            if not len(cand):
                continue
            pairs.append((kind, int(a), int(rng.choice(cand)), int(rng.choice(srows)))); got += 1
    F, V, info = blend_photometry(pairs, lab)
    isqq = np.array([i['kind'] == 'qq' for i in info]); path = str(ROOT/'models/multisurvey_psf'/args.model)
    res = {}
    for scheme in ('legacy_r', 'fallback'):
        Vs = masked_variance(V, lab, scheme, args.mode); Fs = np.where(np.isfinite(Vs), F, 0.)
        jobs = [(Fs[i:i+1], Vs[i:i+1], tuple(lab), info[i]['z'], info[i]['l'], info[i]['b']) for i in range(len(info))]
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init_blend, initargs=(path,)) as pool:
            out = list(pool.map(_score_blend, jobs, chunksize=2))
        res[scheme] = dict(status=np.array([o[0] for o in out]), bf=np.array([o[1] for o in out], float),
                           p=np.array([o[2] for o in out], float), notes=np.array([o[3] for o in out]))
        print(scheme, dict(zip(*np.unique(res[scheme]['status'], return_counts=True))), flush=True)
    tag = f'blend_{args.mode}_{args.model}'; OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT/f'{tag}.npz', isqq=isqq, alpha=np.array([i['alpha_true'] for i in info]),
             **{f'{s}_{k}': v for s, dd in res.items() for k, v in dd.items()})
    ok = (res['legacy_r']['status'] == 'ok') & (res['fallback']['status'] == 'ok')
    rep = dict(mode=args.mode, model=args.model, n_qq=int(isqq.sum()), n_qs=int((~isqq).sum()), n_common=int(ok.sum()), schemes={})
    edges = np.linspace(0, 1, 6)
    for s in res:
        bf, p = res[s]['bf'][ok], res[s]['p'][ok]; t = isqq[ok]
        rel = [dict(bin=[float(lo), float(hi)], n=int(((p >= lo) & (p < hi)).sum()),
                    frac_qq=float(t[(p >= lo) & (p < hi)].mean()) if ((p >= lo) & (p < hi)).any() else None)
               for lo, hi in zip(edges[:-1], edges[1:])]
        rep['schemes'][s] = dict(scored=float((res[s]['status'] == 'ok').mean()), auc=float(auc(bf[t], bf[~t])),
                                 qq_p_gt_0p9=float((p[t] > .9).mean()), qs_p_gt_0p9=float((p[~t] > .9).mean()),
                                 log_score=float(np.mean(np.log(np.clip(np.where(t, p, 1 - p), 1e-6, 1)))), reliability=rel)
    d = res['fallback']['bf'][ok] - res['legacy_r']['bf'][ok]
    rep['dlogbf'] = dict(median=float(np.median(d)), nmad=float(1.4826*np.median(np.abs(d - np.median(d)))),
                         sign_agreement=float(np.mean(np.sign(res['fallback']['bf'][ok]) == np.sign(res['legacy_r']['bf'][ok]))))
    rep['grid_truncated_fraction'] = float(np.mean(['legacy_r_grid_truncated' in n for n in res['fallback']['notes'][ok]]))
    rep['auc_loss'] = rep['schemes']['legacy_r']['auc'] - rep['schemes']['fallback']['auc']
    rep['accept'] = bool(rep['auc_loss'] <= .02)
    DOC.mkdir(parents=True, exist_ok=True); (DOC/f'validation_{tag}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--part', choices=('single', 'blend'), required=True)
    ap.add_argument('--mode', choices=tuple(MODES), required=True)
    ap.add_argument('--model', default='current', choices=('current', 'current_magindep'))
    ap.add_argument('--n', type=int, default=3000, help='objects per class (single) or blends per class (blend)')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    (run_single if args.part == 'single' else run_blend)(args)


if __name__ == '__main__':
    main()
