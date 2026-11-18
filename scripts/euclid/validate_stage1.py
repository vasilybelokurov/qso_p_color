#!/usr/bin/env python
"""Stage-1 validation (docs/euclid/PLAN.md T-V5, T-V6, T-V8, T-V9, T-D1 galaxies) on the Euclid test/calib roles.

QSOs: eligible spectroscopic QSO targets in Euclid TEST cells (T-V5, T-V8, T-V9) or CALIB cells (T-V6) passing the
deployment selection, scored at their own redshift. Background: deployment rows of the new TEST cones without a
known-QSO match (<= --per-cone per cone, seeded), scored at a redshift drawn from the test QSOs; "unflagged" rows
also lack a likely-QSO flag (Quaia or WISE). Each object is scored three ways:
  euclid   - Euclid VIS, Y, J, H only (Legacy r integrated over), stage-1 bundle;
  legacy   - Legacy g, r, z, W1, W2 only, the PARENT bundle (the baseline; objects without Legacy r S/N >= 10 are
             refused there as in production);
  combined - Legacy + Euclid, stage-1 bundle.
Metrics with bootstrap over sky cells (QSOs) and cones (background), 1000 resamples.

    python scripts/euclid/validate_stage1.py --bundle 20261005_magdep_qlf --workers 6
"""
import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import warnings
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cones import deployment                                  # noqa: E402

BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
MODELS = Path('models/multisurvey_psf')
RUN = MODELS/'work/unified_full/20261001/5f4002492dbb849c'
LEG = [f'decals_dr9_{h}:{b}' for h in ('south', 'north') for b in ('g', 'r', 'z', 'w1', 'w2')]
EUC = ('euclid:vis', 'euclid:y', 'euclid:j', 'euclid:h')
UJY = 10**((22.5 - 23.9)/2.5)
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
                       match=RedshiftMatch(half_width_kms=2000.),
                       blend_policy=BlendPolicy(min_separation_arcsec=3., max_fracflux=.2),
                       separation_arcsec=np.full(n, 6.), fracflux=np.zeros(n), ood_flag_sigma=4.)
    get = lambda k: np.array([np.nan if x is None else getattr(x, k) for x in s], float)
    return dict(p=m.quasar_probability(s), logr=get('log_r_per_unit_z'), eligible=d['eligible'], reason=d['reason'].astype('<U40'),
                support=np.asarray(d['qso_support']['percentile'], float), ref=get('ref_mag'),
                p_sameq=get('p_sameq'), dz=get('dz_match_eff'), qso_ood_p=get('qso_ood_p'), bkg_ood_p=get('bkg_ood_p'))


def score(path, flux, var, bands, ra, dec, z, workers, chunk=200):
    jobs = [(str(path), flux[i:i+chunk], var[i:i+chunk], bands, ra[i:i+chunk], dec[i:i+chunk], z[i:i+chunk])
            for i in range(0, len(ra), chunk)]
    with Pool(workers) as pool:
        parts = pool.map(score_chunk, jobs)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def auc(pos, neg):
    """AUC with NaN (unranked) at the bottom; ties count half."""
    a = np.where(np.isfinite(pos), pos, -np.inf); b = np.where(np.isfinite(neg), neg, -np.inf)
    allv = np.concatenate([a, b]); order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums/cnt)[inv]
    return float((ranks[:len(a)].sum() - len(a)*(len(a) + 1)/2)/(len(a)*len(b)))


def metrics(q, bk, unfl):
    """q, bk: dicts of arrays for QSOs and background; unfl: unflagged mask over bk."""
    pq = np.where(q['eligible'], q['p'], np.nan); pb = np.where(bk['eligible'], bk['p'], np.nan)
    return dict(auc_p=auc(pq, pb[unfl]), auc_logr=auc(np.where(q['eligible'], q['logr'], np.nan), np.where(bk['eligible'], bk['logr'], np.nan)[unfl]),
                recall_p05=float(np.mean(np.nan_to_num(pq) > .5)), bkg_p05_unflagged=float(np.mean(np.nan_to_num(pb[unfl]) > .5)),
                bkg_p05_all=float(np.mean(np.nan_to_num(pb) > .5)), qso_unranked=float(np.mean(~q['eligible'])),
                bkg_unranked=float(np.mean(~bk['eligible'])))


def boot(q, bk, unfl, qg, bg, n=1000, seed=1):
    rng = np.random.default_rng(seed); uq, ub = np.unique(qg), np.unique(bg); out = []
    iq = {g: np.flatnonzero(qg == g) for g in uq}; ib = {g: np.flatnonzero(bg == g) for g in ub}
    for _ in range(n):
        a = np.concatenate([iq[g] for g in rng.choice(uq, len(uq))]); b = np.concatenate([ib[g] for g in rng.choice(ub, len(ub))])
        out.append(metrics({k: v[a] for k, v in q.items()}, {k: v[b] for k, v in bk.items()}, unfl[b]))
    return {k: [float(np.percentile([o[k] for o in out], 2.5)), float(np.percentile([o[k] for o in out], 97.5))] for k in out[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--per-cone', type=int, default=600); ap.add_argument('--seed', type=int, default=20261118)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    s = json.loads(Path('docs/euclid/settings.json').read_text())
    roles = {c['cell']: c['role'] for c in json.loads(Path('docs/euclid/sky_roles.json').read_text())['cells']}
    ext = MODELS/f'euclid_stage1_{a.bundle}'; par = MODELS/a.bundle
    out_dir = MODELS/'work/euclid'/f'validation_{a.bundle}'; out_dir.mkdir(parents=True, exist_ok=True)
    # QSOs (test and calib cells)
    sys.path.insert(0, str(Path(__file__).parent)); from fit_stage1 import load_rows
    qt = load_rows('qso', s); qrole = np.array([roles.get(int(c), '') for c in qt['euclid_cell']])
    lay = json.loads((RUN/'layout.json').read_text()); lcol = [lay['native_labels'].index(b) for b in LEG]
    pflux = np.load(RUN/'qso/flux.npy', mmap_mode='r'); pvar = np.load(RUN/'qso/variance.npy', mmap_mode='r')
    res = {}
    for role in ('test', 'calib'):
        idx = np.flatnonzero(qt['clean'] & (qrole == role))
        prow = qt['parent_row'][idx]
        ef, ev = qt['euclid_flux'][idx], qt['euclid_variance'][idx]
        lf, lv = np.asarray(pflux[prow][:, lcol]), np.asarray(pvar[prow][:, lcol])
        z = qt['zspec'][idx]; ra, dec = qt['ra'][idx], qt['dec'][idx]
        res[f'qso_{role}'] = dict(idx=idx, z=z, cell=qt['euclid_cell'][idx], vis=22.5 - 2.5*np.log10(np.clip(ef[:, 0], 1e-9, None)),
            euclid=score(ext, ef, ev, EUC, ra, dec, z, a.workers),
            **({} if role == 'calib' else dict(
                legacy=score(par, lf, lv, LEG, ra, dec, z, a.workers),
                combined=score(ext, np.c_[lf, ef], np.c_[lv, ev], LEG + list(EUC), ra, dec, z, a.workers))))
        print('scored QSOs', role, len(idx), flush=True)
    # background (test cones)
    rows = []; gal = []
    for p in sorted(glob.glob(str(BASE/'cones'/'test_*.npz'))):
        r, g = deployment(p, s); gal.append(g)
        keep = np.flatnonzero(~r['known_qso'])
        keep = np.sort(rng.choice(keep, min(a.per_cone, len(keep)), replace=False))
        r = {k: v[keep] for k, v in r.items()}; r['cone'] = np.full(len(keep), int(Path(p).stem.split('_')[1])); rows.append(r)
    bk = {k: np.concatenate([o[k] for o in rows]) for k in rows[0]}
    ef = np.column_stack([bk[f'flux_{c}'] for c in ('vis_to_y_templfit', 'y_templfit', 'j_templfit', 'h_templfit')])*UJY
    ee = np.column_stack([bk[f'fluxerr_{c}'] for c in ('vis_to_y_templfit', 'y_templfit', 'j_templfit', 'h_templfit')])*UJY
    okE = np.isfinite(ef) & np.isfinite(ee) & (ee > 0); ef = np.where(okE, ef, np.nan); ev = np.where(okE, ee**2, np.inf)
    south = np.char.startswith(np.asarray(bk['legacy_release']).astype(str), '901') & (np.asarray(bk['legacy_release']).astype(str) != '9011')
    lf = np.full((len(ef), 10), np.nan); lv = np.full((len(ef), 10), np.inf)
    for j, bnd in enumerate(('g', 'r', 'z', 'w1', 'w2')):
        f = bk[f'legacy_flux_{bnd}'].astype(float); iv = bk[f'legacy_flux_ivar_{bnd}'].astype(float)
        nob = bk[f'legacy_nobs_{bnd}'].astype(float)
        ok = np.isfinite(f) & (iv > 0) & (nob > 0) & (bk['legacy_maskbits'].astype(float) == 0)
        for h, col in (('south', j), ('north', 5 + j)):
            hm = ok & (south if h == 'south' else ~south)
            lf[hm, col] = f[hm]; lv[hm, col] = 1/iv[hm]
    zq = res['qso_test']['z']; zb = rng.choice(zq, len(ef))
    res['bkg'] = dict(cone=bk['cone'], likely=bk['likely_qso'], legacy_type=bk['legacy_type'], z=zb,
                      vis=22.5 - 2.5*np.log10(np.clip(ef[:, 0], 1e-9, None)),
                      euclid=score(ext, ef, ev, EUC, bk['e_ra'], bk['e_dec'], zb, a.workers),
                      legacy=score(par, lf, lv, LEG, bk['e_ra'], bk['e_dec'], zb, a.workers),
                      combined=score(ext, np.c_[lf, ef], np.c_[lv, ev], LEG + list(EUC), bk['e_ra'], bk['e_dec'], zb, a.workers))
    print('scored background', len(ef), flush=True)
    np.save(out_dir/'scores.npy', res, allow_pickle=True)
    # metrics
    Q, B = res['qso_test'], res['bkg']; unfl = ~B['likely']
    report = dict(n_qso_test=len(Q['z']), n_bkg=len(B['z']), n_bkg_unflagged=int(unfl.sum()), modes={})
    for mode in ('euclid', 'legacy', 'combined'):
        report['modes'][mode] = dict(value=metrics(Q[mode], B[mode], unfl), ci95=boot(Q[mode], B[mode], unfl, Q['cell'], B['cone']),
                                     qso_reasons=dict(zip(*map(lambda x: x.tolist(), np.unique(Q[mode]['reason'], return_counts=True)))),
                                     bkg_reasons=dict(zip(*map(lambda x: x.tolist(), np.unique(B[mode]['reason'], return_counts=True)))))
    # same objects scored by both: euclid vs legacy where the legacy baseline ranks
    both_q = Q['legacy']['eligible'] & Q['euclid']['eligible']; both_b = B['legacy']['eligible'] & B['euclid']['eligible']
    report['common_ranked'] = {m: metrics({k: v[both_q] for k, v in Q[m].items()}, {k: v[both_b] for k, v in B[m].items()}, unfl[both_b])
                               for m in ('euclid', 'legacy', 'combined')}
    report['common_ranked']['n'] = [int(both_q.sum()), int(both_b.sum())]
    # recall at the Legacy baseline's background rate
    thr_rate = report['modes']['legacy']['value']['bkg_p05_unflagged']
    pb = np.where(B['euclid']['eligible'], B['euclid']['p'], -1)[unfl]
    t = np.quantile(pb, 1 - thr_rate) if thr_rate > 0 else 1.
    report['euclid_recall_at_legacy_bkg_rate'] = dict(rate=thr_rate, threshold=float(t),
                                                      recall=float(np.mean(np.where(Q['euclid']['eligible'], Q['euclid']['p'], -1) > t)))
    # by redshift and VIS (Euclid mode)
    by = {}
    for zl, zh in ((0, .8), (.8, 1.5), (1.5, 2.2), (2.2, 3.), (3., 4.4)):
        m_ = (Q['z'] >= zl) & (Q['z'] < zh)
        by[f'z{zl}-{zh}'] = dict(n=int(m_.sum()), recall_p05=float(np.mean(np.nan_to_num(np.where(Q['euclid']['eligible'], Q['euclid']['p'], np.nan))[m_] > .5)),
                                 unranked=float(np.mean(~Q['euclid']['eligible'][m_])))
    for vl, vh in ((18, 20), (20, 21), (21, 22), (22, 23), (23, 26)):
        mq = (Q['vis'] >= vl) & (Q['vis'] < vh); mb = (B['vis'] >= vl) & (B['vis'] < vh) & unfl
        by[f'VIS{vl}-{vh}'] = dict(n_qso=int(mq.sum()), n_bkg=int(mb.sum()),
                                   auc=auc(np.where(Q['euclid']['eligible'], Q['euclid']['p'], np.nan)[mq], np.where(B['euclid']['eligible'], B['euclid']['p'], np.nan)[mb]) if mq.sum() and mb.sum() else None,
                                   recall_p05=float(np.mean(np.nan_to_num(np.where(Q['euclid']['eligible'], Q['euclid']['p'], np.nan))[mq] > .5)) if mq.sum() else None,
                                   bkg_p05=float(np.mean(np.nan_to_num(np.where(B['euclid']['eligible'], B['euclid']['p'], np.nan))[mb] > .5)) if mb.sum() else None)
    report['euclid_by'] = by
    # T-V6: support retention, calib-cell QSOs, Euclid only
    C = res['qso_calib']['euclid']
    report['T-V6'] = dict(n=int(len(C['support'])), retention_support=float(np.mean(C['reason'] != 'qso_support_rejected')),
                          retention_outside_both=float(np.mean(C['reason'] != 'outside_both_models')), eligible=float(np.mean(C['eligible'])),
                          reasons=dict(zip(*map(lambda x: x.tolist(), np.unique(C['reason'], return_counts=True)))))
    # T-V9 (model B): Euclid-only p_quasar / log R must not trend with VIS beyond the prior; reported as slope of logit p
    # T-D1 galaxy contamination
    pl = np.concatenate([g.get('point_like_prob', []) for g in gal]) if gal else np.array([])
    report['T-D1_galaxies'] = dict(n=int(len(pl)), frac_point=float(np.mean(np.nan_to_num(pl) >= s['p_min'])) if len(pl) else None)
    (out_dir/'report.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != 'modes'}, indent=1, default=float)[:4000])
    for mode, v in report['modes'].items():
        print(mode, json.dumps(v['value']), json.dumps(v['ci95']))


if __name__ == '__main__':
    main()
