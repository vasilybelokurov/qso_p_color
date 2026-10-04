#!/usr/bin/env python
"""Validate the blend mode (qso_pcolor.blend) on synthetic blends of real held-out objects.

Sample (role 3 = test, never fitted):
  QQ  a test quasar + a second test quasar with |dz| < 0.02 (same slice), same Legacy hemisphere;
  QS  a test quasar + a test-cone background object (likely QSOs removed), same hemisphere.
  Every component has Legacy r S/N >= 10 on its own; the blend flux is the sum of the components'
  extinction-corrected fluxes in each band both observed (variances added); other bands unobserved.
  Blends are placed at the sky position of a random test-cone object for BOTH classes, so position
  carries no class information. The blend redshift is the first quasar's.
Checks:
  mc     the full likelihood (same alpha grid, priors, Jacobian) with the colour term from brute-force
         flux-space Monte Carlo instead of the linearisation, on a subset restricted to Legacy g,r,z,W1,W2
         (both hypotheses): differences in log L and log BF.
  perf   AUC of log BF (QQ vs QS), reliability of P(QQ) at prior odds 1 (balanced sample), and AUC by true
         companion flux share, blend magnitude, redshift and number of bands, for both promoted bundles.
Writes docs/method_unified/blend_validation.json and plots/method_unified/F20_blend_mode.png.
Usage: python scripts/validate_blend_mode.py [--n 1200] [--workers 12]
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
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qso_pcolor.blend import A, BlendModel, condition_mixture, flux_of, luptitude
from qso_pcolor.multisurvey_data import Photometry
from run_unified_pilot import arrays

RUN = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')
OUT = Path('models/multisurvey_psf/work/blend_validation')
MODELS = {'current': 'models/multisurvey_psf/current', 'current_magindep': 'models/multisurvey_psf/current_magindep'}
LEGACY = ('g', 'r', 'z', 'w1', 'w2')
_s = {}


def snr_r(d, rows, lab):
    s, n = lab.index('decals_dr9_south:r'), lab.index('decals_dr9_north:r'); obs = np.asarray(d['observed'][rows])
    j = np.where(obs[:, s], s, n); f = np.asarray(d['flux_dered'])[rows, j]; v = np.asarray(d['variance_dered'])[rows, j]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(obs[:, s] | obs[:, n], f/np.sqrt(v), 0.), np.where(obs[:, s], 'south', np.where(obs[:, n], 'north', ''))


def build_sample(n, seed=20261004):
    lab = json.loads((RUN/'layout.json').read_text())['native_labels']; rng = np.random.default_rng(seed)
    q, st = arrays(RUN, 'qso'), arrays(RUN, 'stars'); flag = np.load(RUN/'stars'/'qso_flag.npy')
    qs, qh = snr_r(q, np.arange(len(q['role'])), lab); ss, sh = snr_r(st, np.arange(len(st['role'])), lab)
    qok = np.flatnonzero((np.asarray(q['role']) == 3) & (qs >= 10) & (np.asarray(q['zspec']) > .1) & (np.asarray(q['zspec']) < 4.4))
    sok = np.flatnonzero((np.asarray(st['role']) == 3) & (ss >= 10) & ~flag)
    z = np.asarray(q['zspec'])
    pairs = []
    for kind in ('qq', 'qs'):
        got = 0
        while got < n:
            a = rng.choice(qok); h = qh[a]
            if kind == 'qq':
                cand = qok[(np.abs(z[qok] - z[a]) < .02) & (qh[qok] == h) & (qok != a)]
            else:
                cand = sok[sh[sok] == h]
            if not len(cand):
                continue
            b = rng.choice(cand); pos = rng.choice(sok)
            pairs.append((kind, int(a), int(b), int(pos))); got += 1
    return pairs, lab


def blend_photometry(pairs, lab):
    q, st = arrays(RUN, 'qso'), arrays(RUN, 'stars')
    F, V, info = [], [], []
    for kind, a, b, pos in pairs:
        fa, va = np.asarray(q['flux_dered'][a]), np.asarray(q['variance_dered'][a])
        src = q if kind == 'qq' else st
        fb, vb = np.asarray(src['flux_dered'][b]), np.asarray(src['variance_dered'][b])
        ok = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(va) & np.isfinite(vb) & (va > 0) & (vb > 0)
        F.append(np.where(ok, fa + fb, 0.)); V.append(np.where(ok, va + vb, np.inf))
        r = [lab.index(f'decals_dr9_{h}:r') for h in ('south', 'north')]
        rr = next(j for j in r if ok[j])
        info.append(dict(kind=kind, z=float(q['zspec'][a]), alpha_true=float(fb[rr]/(fa[rr] + fb[rr])),
                         l=float(st['l'][pos]), b=float(st['b'][pos]), nbands=int(ok.sum())))
    return np.array(F), np.array(V), info


def _init(path):
    _s['m'] = BlendModel.load(path)


def _score(args):
    F, V, bands, z, l, b = args; m = _s['m']
    out = m.score(Photometry(F, V, bands), ra_deg=0., dec_deg=0., z_qso=z, dereddened=True, l_deg=l, b_deg=b)
    return [(o.status, o.log_like_qq, o.log_like_qs, o.log_bf_qq_qs, o.p_qq, o.alpha_mean_qq, o.alpha_mean_qs) for o in out]


def score_all(name, path, F, V, info, bands, workers):
    import hashlib
    from qso_pcolor import blend as B
    root = Path(path); root = root.parent/json.loads(root.read_text())['bundle'] if root.is_file() else root
    key = hashlib.sha256(F.tobytes() + V.tobytes() + json.dumps(info).encode() + Path(B.__file__).read_bytes()
                         + (root/'manifest.json').read_bytes()).hexdigest()[:16]
    cache = OUT/f'scores_{name}_{key}.npz'      # keyed on sample, blend code and bundle
    if cache.exists():
        return dict(np.load(cache, allow_pickle=False))
    jobs = [(F[i:i+1], V[i:i+1], bands, info[i]['z'], info[i]['l'], info[i]['b']) for i in range(len(info))]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init, initargs=(path,)) as pool:
        res = [r[0] for r in pool.map(_score, jobs, chunksize=4)]
    out = dict(status=np.array([r[0] for r in res]), **{k: np.array([r[i] for r in res], float) for i, k in
               enumerate(('log_like_qq', 'log_like_qs', 'log_bf', 'p_qq', 'alpha_qq', 'alpha_qs'), start=1)})
    np.savez(cache, **out); return out


# ---- Monte Carlo version of the full likelihood (validation only) ------------------------------------
def mc_loglike(m, F, V, z0, l, b, n_draw, rng):
    bands = m.bands; obs = np.isfinite(V) & (V > 0) & np.isfinite(V)
    ref = next(bands.index(r) for r in ('decals_dr9_south:r', 'decals_dr9_north:r') if obs[bands.index(r)])
    label = bands[ref]; T, sig = F[ref], np.sqrt(V[ref])
    qprior, bdens = m.base.priors[label]
    m_lim = float(luptitude(m.faint_snr*sig, m.soft[ref]))
    lpq = m._log_mag_density(lambda x: qprior(np.array([z0]), x)[0], qprior.mag_edges)
    lpq2 = m._log_mag_density(lambda x: qprior(np.array([z0]), x)[0], qprior.mag_edges, upper=m_lim)
    lpb = m._log_mag_density(lambda x: bdens(np.array([x]), l, b)[0], bdens.mag_edges, upper=m_lim)
    others = np.flatnonzero(obs & (np.arange(len(obs)) != ref))
    f = m.model.transform(Photometry(np.where(obs, F, 0.)[None], np.where(obs, V, np.inf)[None], bands))
    y = f.x[0, others]; noise = f.cov[0][np.ix_(others, others)] + m.extra[np.ix_(others, others)]
    s_o, s_r = m.soft[others], m.soft[ref]; amin = m.faint_snr*sig/T
    t = np.linspace(np.log(amin/(1 - amin)), np.log((1 - amin)/amin), m.n_alpha); alpha = 1/(1 + np.exp(-t))
    inv = np.linalg.inv(noise); ldet = np.linalg.slogdet(2*np.pi*noise)[1]
    def draw(comp, k):
        lw, mu, c = comp; idx = rng.choice(len(lw), k, p=np.exp(lw - logsumexp(lw)))   # intrinsic conditionals
        c = .5*(c + c.swapaxes(1, 2)) + 1e-12*np.eye(len(others))
        return mu[idx] + np.einsum('nij,nj->ni', np.linalg.cholesky(c)[idx], rng.standard_normal((k, len(others))))
    out = {}
    for hyp, mix, lp2 in (('qq', m._qso_mixture(z0), lpq2), ('qs', m._background_mixture(l, b), lpb)):
        lg = np.full(len(alpha), -np.inf)
        for i, a in enumerate(alpha):
            f1, f2 = (1 - a)*T, a*T; m1, m2 = luptitude(f1, s_r), luptitude(f2, s_r)
            prior = lpq(m1) + lp2(m2)
            if not np.isfinite(prior):
                continue
            logj = np.log(A*T*np.hypot(T, 2*s_r)) - np.log(np.hypot(f1, 2*s_r)) - np.log(np.hypot(f2, 2*s_r))
            c1 = condition_mixture(*m._qso_mixture(z0), ref, m1, others); c2 = condition_mixture(*mix, ref, m2, others)
            l1, l2 = draw(c1, n_draw), draw(c2, n_draw)
            lb = luptitude(flux_of(l1, s_o) + flux_of(l2, s_o), s_o); r = y - lb
            ln = -.5*np.einsum('ni,ij,nj->n', r, inv, r) - .5*ldet
            lg[i] = prior + logj + logsumexp(ln) - np.log(n_draw) + np.log(a*(1 - a))
        dt = t[1] - t[0]; w = np.full(len(t), dt); w[[0, -1]] = dt/2
        out[hyp] = float(logsumexp(lg + np.log(w)))
    return out


def auc(x, y):
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    return float(mannwhitneyu(x, y).statistic/(len(x)*len(y))) if len(x) > 20 and len(y) > 20 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=1200); ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--n-mc', type=int, default=80); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    pairs, lab = build_sample(a.n); F, V, info = blend_photometry(pairs, lab); bands = tuple(lab)
    kind = np.array([i['kind'] for i in info]); alpha_t = np.array([i['alpha_true'] for i in info]); z = np.array([i['z'] for i in info])
    rref = np.array([luptitude(F[i, lab.index('decals_dr9_south:r')] if np.isfinite(V[i, lab.index('decals_dr9_south:r')])
                                else F[i, lab.index('decals_dr9_north:r')], .3) for i in range(len(info))])
    nb = np.array([i['nbands'] for i in info]); report = dict(definition=__doc__, n_per_class=a.n, models={})
    # --- Monte Carlo check (Legacy bands only), current bundle
    m = BlendModel.load(MODELS['current']); rng = np.random.default_rng(11); mc = []
    keep = np.array([b.startswith('decals_') for b in bands])
    for i in list(np.flatnonzero(kind == 'qq')[:a.n_mc//2]) + list(np.flatnonzero(kind == 'qs')[:a.n_mc//2]):
        Fi, Vi = np.where(keep, F[i], 0.), np.where(keep, V[i], np.inf)
        lin = m.score(Photometry(Fi[None], Vi[None], bands), ra_deg=0., dec_deg=0., z_qso=z[i], dereddened=True,
                      l_deg=info[i]['l'], b_deg=info[i]['b'])[0]
        if lin.status != 'ok':
            continue
        mcv = mc_loglike(m, Fi, Vi, z[i], info[i]['l'], info[i]['b'], 20000, rng)
        mc.append(dict(kind=kind[i], alpha=alpha_t[i], d_qq=lin.log_like_qq - mcv['qq'], d_qs=lin.log_like_qs - mcv['qs'],
                       d_bf=lin.log_bf_qq_qs - (mcv['qq'] - mcv['qs']), bf_lin=lin.log_bf_qq_qs, bf_mc=mcv['qq'] - mcv['qs']))
    dbf = np.array([x['d_bf'] for x in mc]); dqq = np.array([x['d_qq'] for x in mc]); dqs = np.array([x['d_qs'] for x in mc])
    bfl = np.array([x['bf_lin'] for x in mc]); bfm = np.array([x['bf_mc'] for x in mc]); mid = np.abs(bfl) < 10
    report['mc_check'] = dict(n=len(mc), n_abs_bf_lt_10=int(mid.sum()),
                              median_abs_dlogBF_abs_bf_lt_10=float(np.median(np.abs(dbf[mid]))) if mid.any() else None,
                              max_abs_dlogBF_abs_bf_lt_10=float(np.abs(dbf[mid]).max()) if mid.any() else None,
                              sign_agreement=float(np.mean(np.sign(bfl) == np.sign(bfm))),
                              median_abs_dlogBF=float(np.median(np.abs(dbf))), p90_abs_dlogBF=float(np.percentile(np.abs(dbf), 90)),
                              max_abs_dlogBF=float(np.abs(dbf).max()), median_dlogL_qq=float(np.median(dqq)), median_dlogL_qs=float(np.median(dqs)),
                              rows=mc)
    print('MC check:', {k: v for k, v in report['mc_check'].items() if k != 'rows'}, flush=True)
    # --- performance, both bundles
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6)); col = {'current': '#e8663d', 'current_magindep': '#2a78d6'}
    for name, path in MODELS.items():
        s = score_all(name, path, F, V, info, bands, a.workers); ok = s['status'] == 'ok'
        bf = np.where(ok, s['log_bf'], np.nan); pq = np.where(ok, s['p_qq'], np.nan); isqq = kind == 'qq'
        rep = dict(scored=float(ok.mean()), status={str(k): int(v) for k, v in zip(*np.unique(s['status'], return_counts=True))},
                   auc=auc(bf[isqq], bf[~isqq]))
        edges = np.array([0, .05, .2, .35, .5, .65, .8, .95, 1.]); rel = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = ok & (pq >= lo) & (pq < hi if hi < 1 else pq <= hi)
            rel.append(dict(bin=[float(lo), float(hi)], n=int(sel.sum()), mean_p=float(np.nanmean(pq[sel])) if sel.any() else None,
                            frac_qq=float(isqq[sel].mean()) if sel.any() else None))
        rep['reliability'] = rel
        rep['decisive'] = dict(qq_p_gt_0p9=float(np.nanmean(pq[isqq & ok] > .9)), qs_p_lt_0p1=float(np.nanmean(pq[~isqq & ok] < .1)),
                               qq_p_lt_0p1=float(np.nanmean(pq[isqq & ok] < .1)), qs_p_gt_0p9=float(np.nanmean(pq[~isqq & ok] > .9)))
        def by(x, ed):
            return [dict(bin=[float(lo), float(hi)], n_qq=int((isqq & ok & (x >= lo) & (x < hi)).sum()),
                         n_qs=int((~isqq & ok & (x >= lo) & (x < hi)).sum()),
                         auc=auc(bf[isqq & (x >= lo) & (x < hi)], bf[~isqq & (x >= lo) & (x < hi)])) for lo, hi in zip(ed[:-1], ed[1:])]
        rep['by_alpha_true'] = by(alpha_t, [0, .1, .25, .5, .75, .9, 1.])
        rep['by_blend_r'] = by(rref, [15, 19, 20, 21, 22, 24])
        rep['by_z'] = by(z, [0, .8, 1.5, 2.2, 3., 4.5])
        rep['by_nbands'] = by(nb, [0, 8, 14, 20, 42])
        rep['minor_share_qq_error_median_abs'] = float(np.nanmedian(np.abs(s['alpha_qq'] - np.minimum(alpha_t, 1 - alpha_t))[isqq & ok]))
        rep['star_share_qs_error_median_abs'] = float(np.nanmedian(np.abs(s['alpha_qs'] - alpha_t)[~isqq & ok]))
        report['models'][name] = rep
        print(name, json.dumps({k: rep[k] for k in ('scored', 'auc', 'decisive', 'minor_share_qq_error_median_abs',
                                                    'star_share_qs_error_median_abs')}), flush=True)
        for row in rep['by_alpha_true']:
            print('   alpha', row['bin'], row['n_qq'], row['n_qs'], None if row['auc'] != row['auc'] else round(row['auc'], 3))
        x = [r['mean_p'] for r in rel if r['n'] >= 10]; yv = [r['frac_qq'] for r in rel if r['n'] >= 10]
        ax[1].plot(x, yv, 'o-', color=col[name], label=name)
        ba = rep['by_alpha_true']; ax[2].plot([.5*(r['bin'][0] + r['bin'][1]) for r in ba], [r['auc'] for r in ba], 'o-', color=col[name], label=name)
        h0 = np.histogram(bf[isqq & ok], bins=np.linspace(-15, 15, 61))[0]; h1 = np.histogram(bf[~isqq & ok], bins=np.linspace(-15, 15, 61))[0]
        if name == 'current':
            ax[0].step(np.linspace(-15, 15, 61)[:-1], h0, where='post', color='#e8663d', label='true QQ')
            ax[0].step(np.linspace(-15, 15, 61)[:-1], h1, where='post', color='#1baa7d', label='true QS')
    ax[0].set(xlabel='log Bayes factor QQ : QS (current)', ylabel='blends', title='synthetic blends of held-out objects'); ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot([0, 1], [0, 1], ':', color='0.5'); ax[1].set(xlabel='predicted P(QQ), prior odds 1', ylabel='fraction truly QQ', title='reliability (balanced sample)')
    ax[1].legend(frameon=False, fontsize=7)
    ax[2].set(xlabel='true companion share of r flux', ylabel='AUC (QQ vs QS)', title='separation vs flux ratio'); ax[2].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig('plots/method_unified/F20_blend_mode.png', dpi=150)
    Path('docs/method_unified/blend_validation.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
