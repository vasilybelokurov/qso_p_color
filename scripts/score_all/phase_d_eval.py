#!/usr/bin/env python
"""Phase D evaluation on untouched sky (PLAN_score_all.md T1b, T5, T6; PI decision D3, 19 Nov 2026).

Background: deployment rows (scripts/euclid/cones.py) of the 40 new eval cones (docs/score_all/new_cones.json)
without a known-QSO match, <= --per-cone per cone (seeded), scored at redshifts drawn from the QSOs.
QSOs: DESI DR1 QSOs in Euclid Deep Field North (scripts/score_all/fetch_edfn_qsos.py) passing the cleaning cuts
and the point-like cut of docs/euclid/settings.json, 0.15 <= z <= 4.35 (model range), with isolation NOT applied (it needs a deep-table neighbour
query; stated in the output). Deep fluxes are degraded to wide depth: per band the wide variance model
var_w(f) = a + c f is fitted to the eval-cone rows (median variance in flux bins), and
f' = f + N(0, var_w(f) - var_deep), var' = max(var_w(f), var_deep).
Scoring: stage-1 bundles (A = magdep, B = magindep), modes euclid (VIS, Y, J, H) and combined (Legacy + Euclid),
with the D5 rules (the defaults). Old-rule results are derived from the same scores: rows flagged
qso_support_rejected, outside_both_models or weak_legacy_r_in_likelihood were refused before D5 (test M1 showed
every other row is unchanged), so they are set to unranked.
Tests (thresholds fixed before running; plan section 4 with D3b):
  T1b  finite p for >= 99.5% of target rows at VIS < 23 (background sample)
  T5   VIS < 23, new vs old, identical cohort: AUC lower 95% bound > -0.002; recall p > 0.5 lower bound > -2 pp
  T6   unflagged background p > 0.5 (known/likely QSOs removed): upper 95% bound of the change < +0.1 pp
Paired bootstrap, 1000 resamples: background by cone, QSOs by HEALPix nside-64 cell (EDF-N is one field).

    python scripts/score_all/phase_d_eval.py --workers 6            # full run
    python scripts/score_all/phase_d_eval.py --smoke --workers 1    # timing check
"""
import argparse
import glob
import json
import os
from pathlib import Path
import sys
import warnings

for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/euclid'))
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
MODELS = ROOT/'models/multisurvey_psf'
OUT = ROOT/'models/multisurvey_psf/work/score_all/phase_d_eval'
BUNDLES = dict(A='20261005_magdep_qlf', B='20261005_magindep_qlf')
EUC = ('euclid:vis', 'euclid:y', 'euclid:j', 'euclid:h')
LEG = [f'decals_dr9_{h}:{b}' for h in ('south', 'north') for b in ('g', 'r', 'z', 'w1', 'w2')]
ECOL = ('vis_to_y_templfit', 'y_templfit', 'j_templfit', 'h_templfit')
UJY = 10**((22.5 - 23.9)/2.5)
OLD_REFUSED = ('qso_support_rejected', 'outside_both_models', 'weak_legacy_r_in_likelihood')
_M = {}


def _model(path):
    if path not in _M:
        from qso_pcolor.unified import UnifiedPSFModel
        _M[path] = UnifiedPSFModel.load(path)
    return _M[path]


def score_chunk(args):
    path, flux, var, bands, ra, dec, z = args
    from qso_pcolor.multisurvey_data import Photometry
    from qso_pcolor.qso_model import RedshiftMatch
    from qso_pcolor.score import BlendPolicy
    m = _model(path); n = len(ra)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s, d = m.score(Photometry(flux, var, tuple(bands)), ra_deg=ra, dec_deg=dec, z_primary=z, morphology=['PSF']*n,
                       match=RedshiftMatch(half_width_kms=2000.), blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
                       separation_arcsec=np.full(n, 6.), fracflux=np.zeros(n), ood_flag_sigma=4.)
    get = lambda k: np.array([np.nan if x is None else getattr(x, k) for x in s], float)
    flags = np.array([','.join(x.quality_flags) if x is not None else '' for x in s], dtype='<U200')
    return dict(p=np.asarray(m.quasar_probability(s), float), logr=get('log_r_per_unit_z'), eligible=np.asarray(d['eligible'], bool),
                reason=np.asarray(d['reason']).astype('<U40'), flags=flags)


def score(path, flux, var, bands, ra, dec, z, workers, chunk=100):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    jobs = [(str(path), flux[i:i+chunk], var[i:i+chunk], bands, ra[i:i+chunk], dec[i:i+chunk], z[i:i+chunk])
            for i in range(0, len(ra), chunk)]
    if workers <= 1:
        parts = [score_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as ex:
            parts = list(ex.map(score_chunk, jobs))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def legacy_arrays(get, n, release, maskbits):
    """(n, 10) Legacy South/North g r z w1 w2 fluxes and variances; masked or unobserved -> NaN / inf."""
    rel = np.asarray(release).astype(str); south = np.char.startswith(rel, '901') & (rel != '9011')
    lf = np.full((n, 10), np.nan); lv = np.full((n, 10), np.inf)
    for j, b in enumerate(('g', 'r', 'z', 'w1', 'w2')):
        f, iv = get(b)
        ok = np.isfinite(f) & (iv > 0) & (np.asarray(maskbits, float) == 0)
        for h, col in (('south', j), ('north', 5 + j)):
            hm = ok & (south if h == 'south' else ~south); lf[hm, col] = f[hm]; lv[hm, col] = 1/iv[hm]
    return lf, lv


def euclid_arrays(rows):
    f = np.column_stack([np.asarray(rows[f'flux_{c}'], float) for c in ECOL])*UJY
    e = np.column_stack([np.asarray(rows[f'fluxerr_{c}'], float) for c in ECOL])*UJY
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    return np.where(ok, f, np.nan), np.where(ok, e**2, np.inf)


def wide_variance_model(ef, ev):
    """Per band (a, c) of var = a + c f fitted to median variance in flux bins of wide rows (f > 0)."""
    out = []
    for j in range(ef.shape[1]):
        f, v = ef[:, j], ev[:, j]; ok = np.isfinite(f) & np.isfinite(v) & (f > 0)
        edges = np.quantile(f[ok], np.linspace(0, .99, 40)); cen, med = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = ok & (f >= lo) & (f < hi)
            if s.sum() > 50:
                cen.append(np.median(f[s])); med.append(np.median(v[s]))
        X = np.c_[np.ones(len(cen)), cen]; coef = np.linalg.lstsq(X, np.asarray(med), rcond=None)[0]
        out.append([float(max(coef[0], 0)), float(max(coef[1], 0))])
    return np.array(out)


def load_background(settings, per_cone, rng, smoke):
    from cones import deployment
    rows = []
    paths = sorted(glob.glob(str(BASE/'cones/eval_*.npz')))
    for p in paths[:2] if smoke else paths:
        r, _ = deployment(p, settings)
        keep = np.flatnonzero(~r['known_qso']); keep = np.sort(rng.choice(keep, min(per_cone, len(keep)), replace=False))
        r = {k: v[keep] for k, v in r.items()}; r['cone'] = np.full(len(keep), int(Path(p).stem.split('_')[1])); rows.append(r)
    bk = {k: np.concatenate([o[k] for o in rows]) for k in rows[0]}
    ef, ev = euclid_arrays(bk)
    lf, lv = legacy_arrays(lambda b: (bk[f'legacy_flux_{b}'].astype(float), np.where(bk[f'legacy_nobs_{b}'].astype(float) > 0,
                                       bk[f'legacy_flux_ivar_{b}'].astype(float), 0.)), len(ef), bk['legacy_release'], bk['legacy_maskbits'])
    return bk, ef, ev, lf, lv, len(paths[:2] if smoke else paths)


def load_qsos(settings, model_wide, rng):
    from qso_pcolor.euclid_bundle import clean, point_like
    d = dict(np.load(BASE/'edfn_qso/edfn_qso.npz', allow_pickle=False))
    mer = {k[4:]: v for k, v in d.items() if k.startswith('mer_')}
    m = np.asarray(mer['matched'], bool)
    mm = {k: np.asarray(v)[m] for k, v in mer.items()}; dd = {k[5:]: np.asarray(v)[m] for k, v in d.items() if k.startswith('desi_')}
    ok = clean(dict(mm, matched=np.ones(m.sum(), bool)), neighbours=np.zeros(m.sum(), int), bad_bits=settings['bad_bits'],
               vis_bright=settings['vis_bright']) & point_like(dict(mm, matched=np.ones(m.sum(), bool)), settings['p_min'])
    zq = dd['z'].astype(float); ok &= (zq >= 0.15) & (zq <= 4.35)          # quasar-model redshift range, as validation
    mm = {k: v[ok] for k, v in mm.items()}; dd = {k: v[ok] for k, v in dd.items()}
    ef, ev = euclid_arrays(mm)
    vw = model_wide[:, 0] + model_wide[:, 1]*np.clip(np.nan_to_num(ef), 0, None)
    add = np.clip(vw - np.where(np.isfinite(ev), ev, 0), 0, None)
    efw = ef + rng.normal(0, 1, ef.shape)*np.sqrt(add); evw = np.where(np.isfinite(ev), np.maximum(vw, ev), np.inf)
    lf, lv = legacy_arrays(lambda b: (dd[f'flux_{b}'].astype(float), dd[f'flux_ivar_{b}'].astype(float)), len(ef),
                           dd['release'], dd['maskbits'])
    return dict(ra=mm['e_ra'], dec=mm['e_dec'], z=dd['z'].astype(float), ef=efw, ev=evw, lf=lf, lv=lv,
                n_matched=int(m.sum()), n_clean=int(ok.sum()))


def auc(pos, neg):
    a = np.where(np.isfinite(pos), pos, -np.inf); b = np.where(np.isfinite(neg), neg, -np.inf)
    allv = np.concatenate([a, b]); order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks)/cnt)[inv]
    return float((ranks[:len(a)].sum() - len(a)*(len(a) + 1)/2)/(len(a)*len(b)))


def ranked_p(s, old):
    elig = s['eligible'].copy()
    if old:
        elig &= ~np.array([any(f in x.split(',') for f in OLD_REFUSED) for x in s['flags']])
    return np.where(elig, s['p'], np.nan)


def metrics(pq, pb, unfl):
    return dict(auc=auc(pq, pb[unfl]), recall=float(np.mean(np.nan_to_num(pq) > .5)),
                bkg_unflagged=float(np.mean(np.nan_to_num(pb[unfl]) > .5)), finite_bkg=float(np.mean(np.isfinite(pb))))


def compare(q, b, unfl, qg, bg, n_boot=1000, seed=7):
    pn, po = ranked_p(q, False), ranked_p(q, True); bn, bo = ranked_p(b, False), ranked_p(b, True)
    val = dict(new=metrics(pn, bn, unfl), old=metrics(po, bo, unfl))
    rng = np.random.default_rng(seed); uq, ub = np.unique(qg), np.unique(bg)
    iq = {g: np.flatnonzero(qg == g) for g in uq}; ib = {g: np.flatnonzero(bg == g) for g in ub}; diffs = []
    for _ in range(n_boot):
        i = np.concatenate([iq[g] for g in rng.choice(uq, len(uq))]); j = np.concatenate([ib[g] for g in rng.choice(ub, len(ub))])
        a, c = metrics(pn[i], bn[j], unfl[j]), metrics(po[i], bo[j], unfl[j]); diffs.append({k: a[k] - c[k] for k in a})
    ci = {k: [float(np.percentile([d[k] for d in diffs], 2.5)), float(np.percentile([d[k] for d in diffs], 97.5))] for k in diffs[0]}
    val['difference'] = {k: val['new'][k] - val['old'][k] for k in val['new']}; val['ci95'] = ci
    val['T5'] = bool(ci['auc'][0] > -0.002 and ci['recall'][0] > -0.02)
    val['T6'] = bool(ci['bkg_unflagged'][1] < 0.001)
    return val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--per-cone', type=int, default=600)
    ap.add_argument('--seed', type=int, default=20261119); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); OUT.mkdir(parents=True, exist_ok=True)
    import healpy as hp
    settings = json.loads((ROOT/'docs/euclid/settings.json').read_text())
    bk, ef, ev, lf, lv, n_cones = load_background(settings, 50 if a.smoke else a.per_cone, rng, a.smoke)
    wide = wide_variance_model(ef, ev)
    Q = load_qsos(settings, wide, rng)
    if a.smoke:
        Q = {k: (v[:60] if np.ndim(v) else v) for k, v in Q.items()}
    vis_q = 22.5 - 2.5*np.log10(np.clip(Q['ef'][:, 0], 1e-9, None)); vis_b = 22.5 - 2.5*np.log10(np.clip(ef[:, 0], 1e-9, None))
    zb = rng.choice(Q['z'], len(ef)); unfl_all = ~np.asarray(bk['likely_qso'], bool)
    qg = hp.ang2pix(64, Q['ra'], Q['dec'], lonlat=True, nest=True); bg = np.asarray(bk['cone'])
    report = dict(n_eval_cones=n_cones, n_bkg=int(len(ef)), n_qso_matched=Q['n_matched'], n_qso_clean=Q['n_clean'],
                  n_qso_scored=int(len(Q['z'])), qso_isolation_applied=False, wide_variance_model=wide.tolist(),
                  vis_lt_23=dict(qso=int((vis_q < 23).sum()), bkg=int((vis_b < 23).sum())), results={})
    import time
    for name, bundle in BUNDLES.items():
        ext = MODELS/f'euclid_stage1_{bundle}'
        for mode in ('euclid', 'combined'):
            t0 = time.time()
            if mode == 'euclid':
                sq = score(ext, Q['ef'], Q['ev'], EUC, Q['ra'], Q['dec'], Q['z'], a.workers)
                sb = score(ext, ef, ev, EUC, bk['e_ra'], bk['e_dec'], zb, a.workers)
            else:
                sq = score(ext, np.c_[Q['lf'], Q['ef']], np.c_[Q['lv'], Q['ev']], LEG + list(EUC), Q['ra'], Q['dec'], Q['z'], a.workers)
                sb = score(ext, np.c_[lf, ef], np.c_[lv, ev], LEG + list(EUC), bk['e_ra'], bk['e_dec'], zb, a.workers)
            np.save(OUT/f'scores_{name}_{mode}{"_smoke" if a.smoke else ""}.npy', dict(q=sq, b=sb, vis_q=vis_q, vis_b=vis_b), allow_pickle=True)
            mq, mb = vis_q < 23, vis_b < 23
            sub = lambda s, m: {k: v[m] for k, v in s.items()}
            r = dict(seconds=round(time.time() - t0, 1),
                     vis_lt_23=compare(sub(sq, mq), sub(sb, mb), unfl_all[mb], qg[mq], bg[mb], 100 if a.smoke else 1000),
                     all_vis=compare(sq, sb, unfl_all, qg, bg, 100 if a.smoke else 1000))
            r['T1b_finite_p_bkg_vis_lt_23'] = r['vis_lt_23']['new']['finite_bkg']; r['T1b'] = bool(r['T1b_finite_p_bkg_vis_lt_23'] >= .995)
            report['results'][f'{name}_{mode}'] = r
            print(name, mode, f"{r['seconds']} s", 'T5', r['vis_lt_23']['T5'], 'T6', r['vis_lt_23']['T6'], 'T1b', r['T1b'],
                  json.dumps(r['vis_lt_23']['difference']), flush=True)
    (OUT/f'report{"_smoke" if a.smoke else ""}.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
