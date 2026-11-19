#!/usr/bin/env python
"""Phase E step 3b: VIS-referenced background colour model on EDF-North deep (D6), unidentified QSOs held fixed.

Design and tests fixed before fitting: docs/score_all/PHASE_AB.md, "Phase E step 3: design and tests".
Per 0.25-mag VIS bin (21.0-26.5), rows = EDF-North deep candidates (tile_rows.deep_candidates) in training tiles
(id mod 5 != 0), weighted by the mimic w (phase_e3a_mimic.py; rows with w < W_MIN dropped), known QSOs removed.
Features: dereddened (VIS-Y, VIS-J, VIS-H) luptitudes; noise covariance n_VIS + diag(n_Y, n_J, n_H); a band with no
valid luptitude is unobserved (XD handles it exactly).

Fixed QSO term per bin: f_Q p_Q(c), p_Q from phase_e3_qso_colours.py (model B primary, A sensitivity), convolved with
each row's noise; f_Q = max(0, Sigma_Q A C_rel - W_known) / W, A = owned deep area of the training tiles, C_rel = the
mimic completeness of point sources (mean w at point_like_prob > 0.9) relative to its VIS 21-22 value (c_Q in Sigma_Q
already carries the bright-end selection losses), W_known = sum of w over removed known QSOs, W = sum of w over the
fitted rows.
Fit: K from {2,3,4,6,8} by 4-fold held-out log density over tiles (background only, weights w, <= N_KSEL rows); then
OUTER iterations, each: q_Q per row from the current background, refit with weights w (1 - q_Q) (warm start).

Tests (per bin): H1 held-out tiles, w-weighted coverage of the 68/95% highest-density regions of each row's
predictive (QSO + background, convolved), Monte Carlo with M_PRED draws per row, within +-3 pp (bins with >= 500 rows);
H2 adjacent-bin change of the background median colours <= 0.05 mag; H3 VIS 21.0-21.5 background median colours within
0.05 mag of the stage-1 background (noise-free draws at the EDF-North centre, phase_e2 method).

Outputs: ~/data/qso_p_color/score_all/phase_e/bkg_vis_<qso model>.npz, docs/score_all/phase_e3b_fit_<qso model>.json.
Workers over bins (default 4, single-threaded BLAS each); measure one bin first (--bins 0 --workers 1).

    OMP_NUM_THREADS=1 nice -n 19 python scripts/score_all/phase_e3b_fit.py [--qso B|A] [--workers 4] [--bins i,j]
        [--provisional]   # w = 1[point_like_prob >= 0.1] instead of the mimic (smoke test only, never reported)
"""
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/euclid')); sys.path.insert(0, str(ROOT/'scripts/score_all'))
import tile_rows as T  # noqa: E402

BINS = np.round(np.arange(21.0, 26.5001, 0.25), 2)
K_GRID = (2, 3, 4, 6, 8)
K_EXTEND = (10, 12, 16)       # tried only if the best K is the largest of K_GRID (amendment before fitting)
W_MIN = 0.005
N_KSEL = 40_000
N_FIT = 200_000
OUTER = 4
REG = 1e-4
M_PRED = 200
N_H1 = 5_000
SEED = 20261119
CENTRE_N = (269.73, 66.02)


def conv_logpdf(w, mu, C, x, S, obs):
    """log sum_k w_k N(x | mu_k, C_k + S_i) on the observed dims of each row; returns [n]."""
    from scipy.special import logsumexp
    n, d = x.shape; out = np.full(n, -np.inf)
    pats, inv = np.unique(obs, axis=0, return_inverse=True)
    for ip, p in enumerate(pats):
        r = np.flatnonzero(inv.ravel() == ip); dims = np.flatnonzero(p)
        if not len(dims):
            out[r] = 0.0; continue
        xx = x[np.ix_(r, dims)]; SS = S[np.ix_(r, dims, dims)]
        lk = np.empty((len(r), len(w)))
        for k in range(len(w)):
            T_ = C[k][np.ix_(dims, dims)][None] + SS; L = np.linalg.cholesky(T_)
            dv = xx - mu[k, dims][None]
            z = np.linalg.solve(L, dv[..., None])[..., 0]
            lk[:, k] = np.log(w[k]) - 0.5*np.sum(z**2, 1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1) \
                - 0.5*len(dims)*np.log(2*np.pi)
        out[r] = logsumexp(lk, 1)
    return out


def features(d):
    lup, var = d['lup'].astype(float), d['lupvar'].astype(float)
    x = lup[:, :1] - lup[:, 1:]
    obs = np.isfinite(x) & (var[:, 1:] > 0) & np.isfinite(lup[:, :1])
    S = var[:, 0][:, None, None]*np.ones((1, 3, 3)) + np.einsum('ij,jk->ijk', var[:, 1:], np.eye(3))
    return np.where(obs, x, 0.0), S, obs


def prepare(qso, provisional, limit):
    settings = json.loads((ROOT/'docs/euclid/settings.json').read_text())
    d = T.field_table('deep', 'N', settings, limit)
    if provisional:
        w = (d['point_like_prob'] >= settings['p_min']).astype(float); crel = np.ones(len(BINS) - 1)
    else:
        m = pickle.loads((T.OUT/'mimic.pkl').read_bytes())
        w = m['model'].predict_proba(np.column_stack([np.asarray(d[f], float) for f in m['features']]))[:, 1]
        pt = d['point_like_prob'] > 0.9
        c = np.array([w[pt & (d['vis'] >= lo) & (d['vis'] < lo + 0.25)].mean() for lo in BINS[:-1]])
        ref = w[pt & (d['vis'] >= 21) & (d['vis'] < 22)].mean(); crel = np.minimum(c/ref, 1.0)
    d['w'] = w
    # owned training area: covered pixels whose nearest tile centre is a training tile
    import healpy as hp
    from cones import unit
    from scipy.spatial import cKDTree
    pix, _ = T.coverage('deep', 'N', limit)
    cen = np.asarray(json.loads((T.DATA/'deep_tiles/N/tiles.json').read_text())['centres'])
    have = np.array(sorted(int(p.stem.split('_')[1]) for p in (T.DATA/'deep_tiles/N').glob('tile_*.npz')))[:limit]
    lon, lat = hp.pix2ang(T.NSIDE, pix, lonlat=True, nest=True)
    _, own = cKDTree(unit(cen[:, 0], cen[:, 1])).query(unit(lon, lat))
    area_train = float(np.isin(own, have[have % 5 != 0]).sum()*hp.nside2pixarea(T.NSIDE, degrees=True))
    qc = dict(np.load(T.OUT/f'qso_colour_{qso}.npz'))                 # plain arrays: picklable for the workers
    return d, crel, area_train, qc


def fit_bin(args):
    """One VIS bin: K selection, outer QSO-fixed iterations, H1 on held-out tiles."""
    ib, d, crel, area, qc = args
    from qso_pcolor.xd import fit_xd, select_n_components
    t0 = time.monotonic(); rng = np.random.default_rng(SEED + ib)
    lo, hi = BINS[ib], BINS[ib + 1]
    s = (d['vis'] >= lo) & (d['vis'] < hi) & (d['w'] >= W_MIN)
    train = s & (d['tile'] % 5 != 0); hold = s & (d['tile'] % 5 == 0)
    known = train & d['known_qso']; fitr = np.flatnonzero(train & ~d['known_qso'])
    W = float(d['w'][fitr].sum()); W_known = float(d['w'][known].sum())
    sq = float(qc['sigma_q_bin'][ib])
    fq = max(0.0, sq*area*crel[ib] - W_known)/W if W > 0 else 0.0
    if len(fitr) > N_FIT:                                       # subsample, keep weights (unbiased)
        fitr = np.sort(rng.choice(fitr, N_FIT, replace=False))
    x, S, obs = features({k: d[k][fitr] for k in ('lup', 'lupvar')}); w = d['w'][fitr].astype(float)
    ks = fitr if len(fitr) <= N_KSEL else np.sort(rng.choice(len(fitr), N_KSEL, replace=False))
    ks = np.arange(len(fitr)) if len(fitr) <= N_KSEL else ks
    kbest, kscores = select_n_components(x[ks], S[ks], list(K_GRID), observed=obs[ks], weights=w[ks],
                                         groups=d['tile'][fitr][ks], n_folds=4, seed=SEED, regularization=REG, max_iter=300)
    if kbest == max(K_GRID):
        k2, s2 = select_n_components(x[ks], S[ks], list(K_EXTEND), observed=obs[ks], weights=w[ks],
                                     groups=d['tile'][fitr][ks], n_folds=4, seed=SEED, regularization=REG, max_iter=300)
        kscores.update(s2); kbest = max(kscores, key=kscores.get)
    qw, qm, qC = qc['weights'][ib], qc['means'][ib], qc['covs'][ib]
    mix = None; hist = []
    lq = np.log(max(fq, 1e-300)) + conv_logpdf(qw, qm, qC, x, S, obs)
    qq = np.zeros(len(x))
    for it in range(OUTER):
        r = fit_xd(x, S, n_components=kbest, observed=obs, weights=w*(1 - qq), regularization=REG, init=mix,
                   seed=SEED, max_iter=500)
        mix = r.mixture
        lb = np.log(max(1 - fq, 1e-300)) + conv_logpdf(mix.weights, mix.means, mix.covs, x, S, obs)
        qq = np.exp(lq - np.logaddexp(lq, lb)) if fq > 0 else np.zeros(len(x))
        hist.append(dict(it=it, mean_loglike=round(float(r.mean_loglike), 5), mean_qQ=round(float(np.average(qq, weights=w)), 5),
                         converged=bool(r.converged), n_iter=int(r.n_iter)))
    # H1: HPD coverage on held-out tiles (w-weighted), Monte Carlo per row
    hr = np.flatnonzero(hold)
    if len(hr) > N_H1:
        hr = np.sort(rng.choice(hr, N_H1, replace=False))
    h1 = None
    if len(hr) >= 50:
        xh, Sh, oh = features({k: d[k][hr] for k in ('lup', 'lupvar')}); wh = d['w'][hr].astype(float)
        ww = np.concatenate([fq*qw, (1 - fq)*mix.weights]); mm = np.concatenate([qm, mix.means]); CC = np.concatenate([qC, mix.covs])
        ww = ww/ww.sum()
        u = np.empty(len(hr))
        for i in range(len(hr)):
            k = rng.choice(len(ww), M_PRED, p=ww)
            y = np.array([rng.multivariate_normal(mm[kk], CC[kk] + Sh[i]) for kk in k])
            oi = np.repeat(oh[i][None], M_PRED, 0)
            ly = conv_logpdf(ww, mm, CC, np.where(oi, y, 0), np.repeat(Sh[i][None], M_PRED, 0), oi)
            l0 = conv_logpdf(ww, mm, CC, xh[i][None], Sh[i][None], oh[i][None])[0]
            u[i] = np.mean(ly > l0)
        h1 = dict(n=int(len(hr)), cov68=round(float(np.average(u < 0.68, weights=wh)), 4),
                  cov95=round(float(np.average(u < 0.95, weights=wh)), 4))
        h1['pass_'] = bool(len(hr) < 500 or (abs(h1['cov68'] - 0.68) <= 0.03 and abs(h1['cov95'] - 0.95) <= 0.03))
    samp = mix.sample(20000, rng) if hasattr(mix, 'sample') else None
    if samp is None:
        k = rng.choice(len(mix.weights), 20000, p=mix.weights)
        samp = np.array([rng.multivariate_normal(mix.means[kk], mix.covs[kk]) for kk in k])
    return dict(bin=ib, vis=f'{lo:.2f}-{hi:.2f}', n_rows=int(train.sum()), n_fit=int(len(fitr)), n_known=int(known.sum()),
                sum_w=round(W, 1), f_Q=round(fq, 5), k=int(kbest), k_scores={int(a): round(float(b), 5) for a, b in kscores.items()},
                outer=hist, H1=h1, bkg_median=np.round(np.median(samp, 0), 4).tolist(), seconds=round(time.monotonic() - t0, 1),
                mixture=dict(weights=mix.weights.tolist(), means=mix.means.tolist(), covs=mix.covs.tolist()))


def stage1_median(lo, hi):
    """Noise-free stage-1 background median colours at VIS in [lo, hi) at the EDF-North centre (phase_e2 draws)."""
    import phase_e2_background_check as E2
    from qso_pcolor.unified import galactic_from_equatorial
    mo, b, ir, ie, medges, gdens = E2.model_parts()
    l, bb = galactic_from_equatorial(np.array([CENTRE_N[0]]), np.array([CENTRE_N[1]]))
    x = E2.draw(mo, b, ir, ie, medges, gdens, float(l[0]), float(bb[0]), 400_000, np.random.default_rng(SEED))
    x = x[(x[:, 0] >= lo) & (x[:, 0] < hi)]
    return np.median(x[:, :1] - x[:, 1:], 0), len(x)


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--qso', default='B', choices=('A', 'B'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--bins', default=None, help='comma-separated bin indices (default all)')
    ap.add_argument('--limit', type=int, default=None, help='first N EDF-North tiles (smoke test)')
    ap.add_argument('--provisional', action='store_true')
    a = ap.parse_args()
    d, crel, area, qc = prepare(a.qso, a.provisional, a.limit)
    keep = ('vis', 'w', 'tile', 'known_qso', 'lup', 'lupvar')
    d = {k: d[k] for k in keep}
    bins = [int(i) for i in a.bins.split(',')] if a.bins else list(range(len(BINS) - 1))
    print(f'rows {len(d["vis"])}, training area {area:.2f} deg2, bins {bins}', flush=True)
    jobs = [(ib, {k: v[(d['vis'] >= BINS[ib]) & (d['vis'] < BINS[ib + 1])] for k, v in d.items()}, crel, area, qc) for ib in bins]
    res = []
    if a.workers == 1:
        res = [fit_bin(j) for j in jobs]
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(fit_bin, jobs))
    for r in res:
        print(r['vis'], 'n', r['n_rows'], 'fQ', r['f_Q'], 'K', r['k'], 'H1', r['H1'], 'median', r['bkg_median'], f"{r['seconds']} s", flush=True)
    rep = dict(qso_model=a.qso, provisional=a.provisional, area_train_deg2=round(area, 3), c_rel=np.round(crel, 4).tolist(), bins=res)
    med = {r['bin']: np.array(r['bkg_median']) for r in res}
    rep['H2'] = [dict(bins=[i, i + 1], max_abs_dmedian=round(float(np.abs(med[i + 1] - med[i]).max()), 4),
                      pass_=bool(np.abs(med[i + 1] - med[i]).max() <= 0.05)) for i in sorted(med) if i + 1 in med]
    if 0 in med and 1 in med:
        s1, ns = stage1_median(21.0, 21.5); m = 0.5*(med[0] + med[1])
        rep['H3'] = dict(stage1_median=np.round(s1, 4).tolist(), model_median=np.round(m, 4).tolist(), n_stage1=int(ns),
                         pass_=bool(np.abs(m - s1).max() <= 0.05))
    tag = ('provisional_' if a.provisional else '') + a.qso
    if not a.provisional and a.bins is None and a.limit is None:
        (ROOT/f'docs/score_all/phase_e3b_fit_{tag}.json').write_text(json.dumps(rep, indent=1))
        np.savez(T.OUT/f'bkg_vis_{a.qso}.npz', bins=BINS, report=json.dumps(rep))
    else:
        (T.OUT/f'phase_e3b_fit_{tag}_trial.json').write_text(json.dumps(rep, indent=1))
    print('H2', [x['pass_'] for x in rep['H2']]); print('H3', rep.get('H3'))


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
