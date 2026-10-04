#!/usr/bin/env python
"""Illustrations of the blend mode: what blends look like and where QQ and QS can be told apart.

All blends are built from real held-out (role 3) objects with Legacy r S/N >= 50, so each component's
colours are well measured: a test quasar (the known primary) plus either a second test quasar with
|dz| < 0.02 (QQ) or a cleaned test-cone background object (QS), same Legacy hemisphere. Background
objects are grouped by dereddened Legacy g-r ("star class": blue < 0.45, G-K 0.45-0.9, K-M 0.9-1.3,
M > 1.3).

Controlled blends: both components are rescaled so that the blend has Legacy r = ``r_blend`` and the
companion carries a share ``alpha`` of the r flux (the same factor in every band, so colours are kept).
Noise: the blend variance per band is max(V_p, k1^2 V_p + k2^2 V_c) (V_p the primary's variance, i.e. the
depth at its position, sky-limited); the rescaled components keep their own noise and fresh Gaussian
noise makes up the difference. Components are drawn preferentially from objects at least half as bright
as their target flux, so they are rarely scaled up.

Figures (plots/method_unified):
  F21_blend_colours_vs_z.png       blend colours (Legacy, noiseless) vs redshift: single quasar, QQ and
                                   QS for each star class at companion shares 0.1, 0.3, 0.5
  F22_blend_colour_tracks.png      colour-colour planes at four redshifts: quasars, background objects,
                                   and mixing tracks quasar -> star (alpha 0 -> 1) for each star class
  F23_blend_separability_z.png     AUC(QQ vs QS) on a (z, alpha) grid per star class, r_blend = 19.5,
                                   plus median log BF of QQ blends
  F24_blend_separability_mag.png   the same on a (r_blend, alpha) grid, all redshifts
Numbers go to docs/method_unified/blend_illustrations.json.
Cost: ~6,400 blends x ~0.25 CPU-s (n_alpha 24) ~ 25 CPU-min; run niced with few workers.
Usage: nice python scripts/blend_illustrations.py [--n 20] [--workers 3] [--n-alpha 24] [--model current]
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qso_pcolor.blend import BlendModel, luptitude
from qso_pcolor.multisurvey_data import Photometry
from run_unified_pilot import arrays
from validate_blend_mode import MODELS, RUN, auc

OUT = Path('models/multisurvey_psf/work/blend_illustrations')
PLOTS = Path('plots/method_unified')
CLASSES = (('blue  g-r<0.45', -9, .45), ('G-K  0.45-0.9', .45, .9), ('K-M  0.9-1.3', .9, 1.3), ('M  g-r>1.3', 1.3, 9))
CCOL = ('#2a78d6', '#1baa7d', '#e8a33d', '#c2410c')
QCOL, QQCOL = '0.35', '#8b5cf6'
Z_EDGES = np.array([.2, .8, 1.3, 1.8, 2.3, 2.8, 3.4, 4.1])
A_QS = (.05, .1, .2, .3, .5, .7)
A_QQ = (.05, .1, .2, .3, .5)
R_GRID = (18.5, 19.5, 20.5, 21.5)
Z_FINE = np.round(np.arange(.2, 4.11, .2), 2)        # F21 only (cheap)
R_Z = 19.5
_s = {}


class Data:
    """Held-out components with their dereddened fluxes, variances, hemisphere and Legacy g-r."""

    def __init__(self, snr_min=50.):
        self.lab = json.loads((RUN/'layout.json').read_text())['native_labels']
        q, st = arrays(RUN, 'qso'), arrays(RUN, 'stars'); flag = np.load(RUN/'stars'/'qso_flag.npy')
        self.qF, self.qV = np.asarray(q['flux_dered'], float), np.asarray(q['variance_dered'], float)
        self.sF, self.sV = np.asarray(st['flux_dered'], float), np.asarray(st['variance_dered'], float)
        self.qz = np.asarray(q['zspec'], float); self.sl, self.sb = np.asarray(st['l'], float), np.asarray(st['b'], float)
        self.soft = np.asarray(BlendModel.load(MODELS['current']).soft, float) if 'soft' not in _s else _s['soft']
        self.qh, qsnr = self._hemi(self.qF, self.qV); self.sh, ssnr = self._hemi(self.sF, self.sV)
        self.q_ok = np.flatnonzero((np.asarray(q['role']) == 3) & (qsnr >= snr_min) & (self.qz > .1) & (self.qz < 4.4))
        self.s_ok = np.flatnonzero((np.asarray(st['role']) == 3) & (ssnr >= snr_min) & ~flag)
        self.s_gr = self.colour(self.sF, 'g', 'r', self.sh)
        self.s_class = np.full(len(self.sF), -1)
        for c, (_, lo, hi) in enumerate(CLASSES):
            self.s_class[np.isfinite(self.s_gr) & (self.s_gr >= lo) & (self.s_gr < hi)] = c
        self.pos = np.flatnonzero((np.asarray(st['role']) == 3) & ~flag)       # sky positions for both classes

    def idx(self, band, h):
        return self.lab.index(f'decals_dr9_{h}:{band}')

    def _hemi(self, F, V):
        s, n = self.idx('r', 'south'), self.idx('r', 'north')
        os_, on = np.isfinite(V[:, s]) & (V[:, s] > 0), np.isfinite(V[:, n]) & (V[:, n] > 0)
        h = np.where(os_, 'south', np.where(on, 'north', ''))
        with np.errstate(invalid='ignore', divide='ignore'):
            snr = np.where(os_, F[:, s]/np.sqrt(V[:, s]), np.where(on, F[:, n]/np.sqrt(V[:, n]), 0.))
        return h, np.nan_to_num(snr)

    def ref(self, h):
        return self.idx('r', h)

    def colour(self, F, b1, b2, h):
        out = np.full(len(F), np.nan)
        for hh in ('south', 'north'):
            m = h == hh; i, j = self.idx(b1, hh), self.idx(b2, hh)
            out[m] = luptitude(F[m, i], self.soft[i]) - luptitude(F[m, j], self.soft[j])
        return out


def pick(rng, cand, flux_r, target):
    """Prefer candidates at least half as bright as the target flux (rarely scaled up)."""
    bright = cand[flux_r[cand] >= .5*target]
    return rng.choice(bright if len(bright) >= 5 else cand)


def make_blend(d, rng, a, src, b, alpha, r_blend):
    """Rescale primary quasar ``a`` and companion ``b`` (from 'q' or 's') to blend r and share alpha."""
    h = d.qh[a]; r = d.ref(h); T = 10**(-.4*(r_blend - 22.5))
    fa, va = d.qF[a], d.qV[a]
    fb, vb = (d.qF[b], d.qV[b]) if src == 'q' else (d.sF[b], d.sV[b])
    k1, k2 = (1 - alpha)*T/fa[r], alpha*T/fb[r]
    ok = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(va) & np.isfinite(vb) & (va > 0) & (vb > 0)
    vs = k1**2*va + k2**2*vb; V = np.maximum(va, vs)
    F = k1*fa + k2*fb + rng.standard_normal(len(fa))*np.sqrt(np.clip(V - vs, 0, None))
    return np.where(ok, F, 0.), np.where(ok, V, np.inf)


def build_jobs(d, n, seed=20261004):
    rng = np.random.default_rng(seed); jobs, meta = [], []
    qz, qok = d.qz, d.q_ok; ref_flux = lambda F, H: np.where(H == 'south', F[:, d.idx('r', 'south')], F[:, d.idx('r', 'north')])
    qfr, sfr = ref_flux(d.qF, d.qh), ref_flux(d.sF, d.sh)
    s_by = {(c, h): d.s_ok[(d.s_class[d.s_ok] == c) & (d.sh[d.s_ok] == h)] for c in range(len(CLASSES)) for h in ('south', 'north')}

    def one(grid, zlo, zhi, kind, c, alpha, rb):
        T = 10**(-.4*(rb - 22.5)); prim = qok[(qz[qok] >= zlo) & (qz[qok] < zhi)]
        for _ in range(200):
            a = pick(rng, prim, qfr, (1 - alpha)*T); h = d.qh[a]
            cand = (qok[(np.abs(qz[qok] - qz[a]) < .02) & (d.qh[qok] == h) & (qok != a)] if kind == 'qq' else s_by[(c, h)])
            if len(cand):
                break
        b = pick(rng, cand, qfr if kind == 'qq' else sfr, alpha*T); p = rng.choice(d.pos)
        F, V = make_blend(d, rng, a, 'q' if kind == 'qq' else 's', b, alpha, rb)
        jobs.append((F, V, float(qz[a]), float(d.sl[p]), float(d.sb[p])))
        meta.append(dict(grid=grid, kind=kind, cls=c, z=float(qz[a]), zbin=[float(zlo), float(zhi)], alpha=alpha, r=rb))

    for zlo, zhi in zip(Z_EDGES[:-1], Z_EDGES[1:]):
        for _ in range(n):
            for al in A_QQ:
                one('z', zlo, zhi, 'qq', -1, al, R_Z)
            for c in range(len(CLASSES)):
                for al in A_QS:
                    one('z', zlo, zhi, 'qs', c, al, R_Z)
    for rb in R_GRID:
        for _ in range(n):
            for al in A_QQ:
                one('r', .1, 4.4, 'qq', -1, al, rb)
            for c in range(len(CLASSES)):
                for al in A_QS:
                    one('r', .1, 4.4, 'qs', c, al, rb)
    return jobs, meta


def _init(path, n_alpha):
    _s['m'] = BlendModel.load(path, n_alpha=n_alpha)


def _score(chunk):
    m = _s['m']; out = []
    for F, V, z, l, b in chunk:
        o = m.score(Photometry(F[None], V[None], tuple(m.bands)), ra_deg=0., dec_deg=0., z_qso=z, dereddened=True,
                    l_deg=l, b_deg=b)[0]
        out.append((o.status, o.log_bf_qq_qs, o.p_qq))
    return out


def score(jobs, path, workers, key, n_alpha):
    cache = OUT/f'scores_{key}.npz'
    if cache.exists():
        return dict(np.load(cache, allow_pickle=False))
    chunks = [jobs[i:i+8] for i in range(0, len(jobs), 8)]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init, initargs=(path, n_alpha)) as pool:
        res = [r for c in pool.map(_score, chunks) for r in c]
    out = dict(status=np.array([r[0] for r in res]), log_bf=np.array([r[1] for r in res], float), p_qq=np.array([r[2] for r in res], float))
    np.savez(cache, **out); return out


# ---- figures -------------------------------------------------------------------------------------
COLOURS = (('g', 'r'), ('r', 'z'), ('z', 'w1'), ('w1', 'w2'))


def legacy_sed(d, F, h):
    """Legacy g,r,z,W1,W2 fluxes (rows), each object normalised to r = 1, in its own hemisphere."""
    out = np.full((len(F), 5), np.nan)
    for hh in ('south', 'north'):
        m = h == hh
        out[m] = F[m][:, [d.idx(b, hh) for b in ('g', 'r', 'z', 'w1', 'w2')]]
    return out/out[:, [1]]


def sed_colours(d, sed):
    s = d.soft[[d.idx(b, 'south') for b in ('g', 'r', 'z', 'w1', 'w2')]]
    T = 10**(-.4*(R_Z - 22.5)); lup = luptitude(sed*T, s)         # at r = 19.5, softening negligible
    k = {b: i for i, b in enumerate(('g', 'r', 'z', 'w1', 'w2'))}
    return np.stack([lup[..., k[a]] - lup[..., k[b]] for a, b in COLOURS], axis=-1)


def fig_colours_vs_z(d, rng, report):
    import matplotlib.pyplot as plt
    qs = d.q_ok[rng.permutation(len(d.q_ok))[:60000]]; qsed = legacy_sed(d, d.qF[qs], d.qh[qs]); qz = d.qz[qs]
    good = np.all(np.isfinite(qsed) & (qsed > 0), axis=1); qsed, qz = qsed[good], qz[good]
    zc = .5*(Z_FINE[:-1] + Z_FINE[1:]); shares = (.1, .3, .5)
    ssed = {c: legacy_sed(d, d.sF[d.s_ok[d.s_class[d.s_ok] == c]], d.sh[d.s_ok[d.s_class[d.s_ok] == c]]) for c in range(len(CLASSES))}
    ssed = {c: v[np.all(np.isfinite(v) & (v > 0), axis=1)] for c, v in ssed.items()}
    fig, ax = plt.subplots(len(COLOURS), len(shares), figsize=(11, 10), sharex=True, sharey='row')
    rows = {}
    for j, al in enumerate(shares):
        for zi, (lo, hi) in enumerate(zip(Z_FINE[:-1], Z_FINE[1:])):
            sel = np.flatnonzero((qz >= lo) & (qz < hi))
            if len(sel) < 50:
                continue
            sel = rng.choice(sel, 2000); prim = qsed[sel]
            single = sed_colours(d, prim)
            comp_q = qsed[rng.choice(np.flatnonzero((qz >= lo) & (qz < hi)), len(sel))]
            qq = sed_colours(d, (1 - al)*prim + al*comp_q)
            rows.setdefault(('single', -1, al), []).append(np.percentile(single, [16, 50, 84], axis=0))
            rows.setdefault(('qq', -1, al), []).append(np.percentile(qq, [16, 50, 84], axis=0))
            for c in range(len(CLASSES)):
                st = ssed[c][rng.integers(0, len(ssed[c]), len(sel))]
                rows.setdefault(('qs', c, al), []).append(np.percentile(sed_colours(d, (1 - al)*prim + al*st), [16, 50, 84], axis=0))
        for i, (a, b) in enumerate(COLOURS):
            x = ax[i, j]; p = np.array(rows[('single', -1, al)])
            x.fill_between(zc, p[:, 0, i], p[:, 2, i], color=QCOL, alpha=.18, lw=0)
            x.plot(zc, p[:, 1, i], color=QCOL, lw=2, label='single quasar (16-84%)')
            p = np.array(rows[('qq', -1, al)]); x.plot(zc, p[:, 1, i], color=QQCOL, lw=1.6, ls='--', label='QQ (same z)')
            for c in range(len(CLASSES)):
                p = np.array(rows[('qs', c, al)])
                x.plot(zc, p[:, 1, i], color=CCOL[c], lw=1.5, label=f'QS, star {CLASSES[c][0]}')
                x.fill_between(zc, p[:, 0, i], p[:, 2, i], color=CCOL[c], alpha=.08, lw=0)
            if i == 0:
                x.set_title(f'companion share of r flux = {al}')
            if j == 0:
                x.set_ylabel(f'{a}-{b} (Legacy, dereddened)')
            if i == len(COLOURS) - 1:
                x.set_xlabel('quasar redshift $z_0$')
    ax[0, 0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.suptitle('Blend colours vs redshift (noiseless; real held-out quasars and background objects, medians and 16-84%)', fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS/'F21_blend_colours_vs_z.png', dpi=140); plt.close(fig)
    report['colours_vs_z'] = {f'{k[0]}_{k[1]}_{k[2]}': np.array(v)[:, 1].tolist() for k, v in rows.items()}


def fig_tracks(d, rng):
    import matplotlib.pyplot as plt
    qsed = legacy_sed(d, d.qF[d.q_ok], d.qh[d.q_ok]); good = np.all(np.isfinite(qsed) & (qsed > 0), axis=1)
    qsed, qz = qsed[good], d.qz[d.q_ok][good]
    ss = d.s_ok[d.s_class[d.s_ok] >= 0]; ssed = legacy_sed(d, d.sF[ss], d.sh[ss]); sg = np.all(np.isfinite(ssed) & (ssed > 0), axis=1)
    ssed, scls = ssed[sg], d.s_class[ss][sg]
    zs = (.8, 1.6, 2.4, 3.2); planes = ((0, 1), (1, 2))       # (g-r, r-z), (r-z, z-W1)
    alphas = np.linspace(0, 1, 41); marks = (.1, .3, .5, .7)
    fig, ax = plt.subplots(len(planes), len(zs), figsize=(13, 6.6))
    pick_s = rng.permutation(len(ssed))[:4000]; sc = sed_colours(d, ssed[pick_s]); scl = scls[pick_s]
    for j, z0 in enumerate(zs):
        sel = np.flatnonzero(np.abs(qz - z0) < .05); qc = sed_colours(d, qsed[sel]); med = np.median(qsed[sel], axis=0)
        for i, (u, v) in enumerate(planes):
            x = ax[i, j]
            x.scatter(sc[:, u], sc[:, v], s=2, c=[CCOL[k] for k in scl], alpha=.35, lw=0, rasterized=True)
            x.scatter(qc[:, u], qc[:, v], s=2, color=QCOL, alpha=.35, lw=0, rasterized=True)
            for c in range(len(CLASSES)):
                smed = np.median(ssed[scls == c], axis=0)
                tr = sed_colours(d, (1 - alphas)[:, None]*med + alphas[:, None]*smed)
                x.plot(tr[:, u], tr[:, v], color=CCOL[c], lw=1.8)
                mk = sed_colours(d, np.array([(1 - a)*med + a*smed for a in marks]))
                x.scatter(mk[:, u], mk[:, v], s=14, color=CCOL[c], edgecolor='white', lw=.6, zorder=3)
            m0 = sed_colours(d, med[None])[0]; x.scatter(m0[u], m0[v], marker='*', s=90, color='k', zorder=4)
            nm = ['g-r', 'r-z', 'z-W1', 'W1-W2']
            x.set(xlabel=nm[u], ylabel=nm[v]); x.set_xlim(-.6, 2.0) if u == 0 else x.set_xlim(-.6, 1.8)
            x.set_ylim(-.6, 1.8) if v == 1 else x.set_ylim(-2.2, 3.0)
            if i == 0:
                x.set_title(f'$z_0$ = {z0}', fontsize=10)
    for c in range(len(CLASSES)):
        ax[0, 0].plot([], [], color=CCOL[c], label=f'track to {CLASSES[c][0]} star')
    ax[0, 0].legend(frameon=False, fontsize=7, loc='lower right')
    fig.suptitle('Mixing tracks: median quasar at $z_0$ (black star; grey = quasars at $z_0\\pm0.05$) blended with the median star of each class '
                 '(coloured points = background objects);\ndots at star share 0.1, 0.3, 0.5, 0.7. QQ blends at the same redshift stay inside the grey quasar cloud.', fontsize=9.5)
    fig.tight_layout(); fig.savefig(PLOTS/'F22_blend_colour_tracks.png', dpi=140); plt.close(fig)


def auc2(x, y):
    """P(log BF of a QQ blend > that of a QS blend), ties counted half."""
    d = x[:, None] - y[None, :]
    return float(np.mean(d > 0) + .5*np.mean(d == 0))


def cell_stats(meta, s, grid, key_x, xs):
    """AUC and decisive fractions per (class, x, alpha) cell of one grid."""
    kind = np.array([m['kind'] for m in meta]); cls = np.array([m['cls'] for m in meta]); g = np.array([m['grid'] for m in meta])
    al = np.array([m['alpha'] for m in meta]); ok = s['status'] == 'ok'; bf = np.where(ok, s['log_bf'], np.nan)
    xv = np.array([m['zbin'][0] if key_x == 'z' else m['r'] for m in meta])
    res = {'qq': {}, 'qs': {}}
    for x in xs:
        for a in A_QQ:
            sel = (g == grid) & (kind == 'qq') & (xv == x) & (al == a)
            res['qq'][(x, a)] = dict(n=int(sel.sum()), scored=float(ok[sel].mean()), median_log_bf=float(np.nanmedian(bf[sel])) if ok[sel].any() else np.nan,
                                     p_gt_0p9=float(np.mean(s['p_qq'][sel & ok] > .9)) if ok[sel].any() else np.nan)
        for c in range(len(CLASSES)):
            for a in A_QS:
                sel = (g == grid) & (kind == 'qs') & (cls == c) & (xv == x) & (al == a)
                qq = (g == grid) & (kind == 'qq') & (xv == x) & (al == min(a, round(1 - a, 2)))
                res['qs'][(c, x, a)] = dict(n=int(sel.sum()), scored=float(ok[sel].mean()),
                                            median_log_bf=float(np.nanmedian(bf[sel])) if ok[sel].any() else np.nan,
                                            p_lt_0p1=float(np.mean(s['p_qq'][sel & ok] < .1)) if ok[sel].any() else np.nan,
                                            auc=auc2(bf[qq & ok], bf[sel & ok]) if (ok[sel].sum() >= 15 and ok[qq].sum() >= 15) else np.nan)
    return res


def fig_maps(res, xs, xlabel, fname, title):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    fig, ax = plt.subplots(1, len(CLASSES) + 1, figsize=(18, 4.2), sharey=True, layout='constrained')
    cm = ListedColormap(['#7f1d1d', '#dc2626', '#f59e0b', '#fde68a', '#a7f3d0', '#10b981', '#065f46'])
    norm = BoundaryNorm([.5, .6, .7, .8, .9, .95, .99, 1.], cm.N)
    xe = np.arange(len(xs) + 1) - .5; ye = np.arange(len(A_QS) + 1) - .5
    for c in range(len(CLASSES)):
        M = np.array([[res['qs'][(c, x, a)]['auc'] for x in xs] for a in A_QS])
        im = ax[c].pcolormesh(xe, ye, M, cmap=cm, norm=norm)
        for (i, j), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax[c].text(j, i, f'{v:.2f}'.replace('0.', '.') if f'{v:.2f}' != '1.00' else '1', ha='center', va='center', fontsize=6, color='white' if v < .65 or v > .95 else 'k')
        ax[c].set_title(f'QS star {CLASSES[c][0]}: AUC vs QQ', fontsize=9)
    Q = np.array([[res['qq'][(x, a)]['median_log_bf'] for x in xs] for a in A_QQ] + [[np.nan]*len(xs)]*(len(A_QS) - len(A_QQ)))
    im2 = ax[-1].pcolormesh(xe, ye, Q, cmap='PuOr_r', vmin=-8, vmax=8)
    for (i, j), v in np.ndenumerate(Q):
        if np.isfinite(v):
            ax[-1].text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6)
    ax[-1].set_title('QQ: median log BF (QQ:QS)', fontsize=9)
    for x in ax:
        x.set_xticks(range(len(xs))); x.set_xticklabels([f'{v:g}' for v in xs], rotation=90 if len(xs) > 6 else 0, fontsize=7)
        x.set_yticks(range(len(A_QS))); x.set_yticklabels([str(a) for a in A_QS]); x.set_xlabel(xlabel)
    ax[0].set_ylabel('companion share of r flux')
    fig.colorbar(im, ax=list(ax[:-1]), location='left', fraction=.012, pad=.04, label='AUC: QS cell vs QQ at the same minor share')
    fig.colorbar(im2, ax=ax[-1], fraction=.06, pad=.02, label='median log BF')
    fig.suptitle(title, fontsize=10); fig.savefig(PLOTS/fname, dpi=140, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=20); ap.add_argument('--n-alpha', type=int, default=24); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--model', default='current')
    ap.add_argument('--scores', help='reuse a saved scores_*.npz (same --n and seed) instead of scoring; no workers')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    d = Data(); rng = np.random.default_rng(7); report = dict(definition=__doc__, model=a.model, star_classes=[c[0] for c in CLASSES],
                                                                n_components=dict(qso=int(len(d.q_ok)), background=int(len(d.s_ok))))
    report['star_class_counts'] = {CLASSES[c][0]: int((d.s_class[d.s_ok] == c).sum()) for c in range(len(CLASSES))}
    fig_colours_vs_z(d, rng, report); fig_tracks(d, rng); print('colour figures done', flush=True)
    jobs, meta = build_jobs(d, a.n)
    from qso_pcolor import blend as B
    root = Path(MODELS[a.model]); root = root.parent/json.loads(root.read_text())['bundle'] if root.is_file() else root
    key = hashlib.sha256(b''.join(j[0].tobytes() + j[1].tobytes() for j in jobs) + json.dumps(meta).encode()
                         + Path(B.__file__).read_bytes() + (root/'manifest.json').read_bytes()).hexdigest()[:16]
    print(len(jobs), 'blends to score', flush=True)
    if a.scores:
        s = dict(np.load(a.scores, allow_pickle=False))
        if len(s['status']) != len(jobs):
            raise SystemExit(f'{a.scores} has {len(s["status"])} rows, the grid has {len(jobs)}')
    else:
        s = score(jobs, MODELS[a.model], a.workers, f'{a.model}_a{a.n_alpha}_{key}', a.n_alpha)
    zs = [float(z) for z in Z_EDGES[:-1]]
    rz = cell_stats(meta, s, 'z', 'z', zs); rr = cell_stats(meta, s, 'r', 'r', list(R_GRID))
    zl = [f'{lo:.1f}-{hi:.1f}' for lo, hi in zip(Z_EDGES[:-1], Z_EDGES[1:])]
    fig_maps(rz, zs, 'quasar redshift bin (lower edge)', 'F23_blend_separability_z.png',
             f'Where QQ and QS separate: synthetic blends of held-out objects at Legacy r = {R_Z}, {a.n} blends per cell ({a.model})')
    fig_maps(rr, list(R_GRID), 'blend Legacy r', 'F24_blend_separability_mag.png',
             f'Separation vs blend magnitude (all redshifts), {a.n} blends per cell ({a.model}); blank = too faint for both components')
    ser = lambda r: {k: {'|'.join(map(str, kk)): v for kk, v in r[k].items()} for k in r}
    report['z_bins'] = zl; report['grid_z'] = ser(rz); report['grid_r'] = ser(rr)
    report['status'] = {str(k): int(v) for k, v in zip(*np.unique(s['status'], return_counts=True))}
    Path('docs/method_unified/blend_illustrations.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report['status']))


if __name__ == '__main__':
    main()
