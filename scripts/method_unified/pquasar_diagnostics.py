#!/usr/bin/env python
"""Diagnostics of p_quasar: which real quasars are left unranked, and who the overconfident objects are.

Part 1 (unranked quasars). Held-out test quasars of the performance panels (score_performance.py, main
panel, 2 x 20,000 rows per model above the faint limit), scored by both promoted bundles. The cached
scores used support threshold 6/256; the promoted threshold 2/256 is re-applied from the stored support
percentiles. For every quasar: redshift, reference magnitude, hemisphere, |b|, E(B-V), number of bands,
survey coverage, Legacy colours, inter-survey r differences (a variability / calibration proxy) and the
per-band standardised residual against the QSO model at its redshift (conditional on Legacy r).
--rescore additionally rescores the unranked quasars (i) with Legacy g,r,z,W1,W2 only and (ii) with
each survey removed in turn (3 workers, niced), to find which survey's photometry makes them unranked.

Part 2 (top-end overconfidence). The natural test-cone population of calibrate_probability.py (1,640 known
QSOs + 40,000 weighted background objects; cached scores 'independent'/'dependent'). Reliability at
p > 0.9 with binomial intervals, and the composition of the high-p background objects (flag reasons,
magnitude, WISE detection and W1-W2, colours).

Writes plots/pquasar_diagnostics/P*.png and docs/pquasar_diagnostics/pquasar_diagnostics.json.
Usage: nice python scripts/method_unified/pquasar_diagnostics.py [--rescore] [--workers 3]
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor import Photometry
from run_unified_pilot import arrays

ROOT = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')
PERF = Path('models/multisurvey_psf/work/method_unified/performance_stellar_binned')
CAL = Path('models/multisurvey_psf/work/probability_calibration')
WORK = Path('models/multisurvey_psf/work/pquasar_diagnostics')
PLOTS = Path('plots/pquasar_diagnostics'); DOCS = Path('docs/pquasar_diagnostics')
POINTERS = {'dependent': 'models/multisurvey_psf/current', 'independent': 'models/multisurvey_psf/current_magindep'}
NAME = {'dependent': 'A (magnitude-dependent)', 'independent': 'B (magnitude-independent)'}
COL = {'dependent': '#e8663d', 'independent': '#2a78d6'}
THRESHOLD = 2/256
SURVEYS = ('sdss', 'ps1', 'nsc', 'skymapper', 'vhs', 'allwise', 'legacy_wise')
_s = {}


def labels():
    return list(json.loads((ROOT/'layout.json').read_text())['native_labels'])


def survey_mask(lab, name):
    if name == 'legacy_wise':
        return np.array([b.startswith('decals_') and b.endswith((':w1', ':w2')) for b in lab])
    if name == 'legacy_only':
        return np.array([not b.startswith('decals_') for b in lab])          # bands to REMOVE
    return np.array([b.startswith(name + ':') for b in lab])


# ---------------------------------------------------------------------------------------------------
def load_main(model):
    """Concatenate the main-panel chunks of one model; re-apply the promoted support threshold."""
    out = {}
    for h in ('south', 'north'):
        for f in sorted(glob.glob(str(PERF/f'{model}_qso_main_{h}_*.npz'))):
            if f.endswith('.rows.npz'):
                continue
            s = np.load(f, allow_pickle=True); r = np.load(f[:-4] + '.rows.npz')
            part = dict(rows=r['rows'], z=r['z'], hemi=np.full(len(r['rows']), h),
                        **{k: s[k] for k in ('status', 'support', 'p_quasar', 'qso_ood_sigma_any_z', 'bkg_ood_sigma', 'ref_mag', 'reference')})
            for k, v in part.items():
                out.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in out.items()}
    st = out['status'].astype(str)
    base_ok = np.isin(st, ['ok', 'qso_support_rejected'])
    supp_fail = np.isfinite(out['support']) & (out['support'] < THRESHOLD)
    reason = np.where(base_ok & supp_fail, 'qso_support_rejected', np.where(base_ok, 'ranked', st))
    out['reason'] = reason; out['ranked'] = reason == 'ranked'
    return out


def properties(rows):
    """Photometric properties of test quasars (dereddened native luptitudes from the prepared arrays)."""
    d = arrays(ROOT, 'qso'); lab = labels()
    y = np.asarray(d['y'][rows]); obs = np.asarray(d['observed'][rows]).astype(bool)
    F = np.asarray(d['flux_dered'][rows]); V = np.asarray(d['variance_dered'][rows])
    p = dict(z=np.asarray(d['zspec'][rows]), b=np.asarray(d['b'][rows]), ebv=np.asarray(d['ebv'][rows]),
             nb=obs.sum(axis=1))
    south = obs[:, lab.index('decals_dr9_south:r')]
    def leg(band):
        i, j = lab.index('decals_dr9_south:' + band), lab.index('decals_dr9_north:' + band)
        v = np.where(south, y[:, i], y[:, j]); o = np.where(south, obs[:, i], obs[:, j])
        return np.where(o, v, np.nan)
    for a in ('g', 'r', 'z', 'w1', 'w2'):
        p['L' + a] = leg(a)
    for a, b in (('g', 'r'), ('r', 'z'), ('z', 'w1'), ('w1', 'w2')):
        p[f'{a}-{b}'] = p['L' + a] - p['L' + b]
    for s in ('sdss', 'ps1', 'nsc', 'skymapper', 'vhs', 'allwise'):
        p['has_' + s] = obs[:, survey_mask(lab, s)].any(axis=1)
    for s in ('sdss', 'ps1'):
        i = lab.index(s + ':r'); p[f'dr_{s}'] = np.where(obs[:, i], y[:, i] - p['Lr'], np.nan)
    rs = np.where(south, lab.index('decals_dr9_south:r'), lab.index('decals_dr9_north:r'))
    with np.errstate(invalid='ignore', divide='ignore'):
        p['snr_r'] = F[np.arange(len(rows)), rs]/np.sqrt(V[np.arange(len(rows)), rs])
    return p


def band_residuals(model_name, rows, z):
    """Standardised residual per observed non-reference band under the QSO model at z (conditional on Legacy r)."""
    from qso_pcolor.unified import UnifiedPSFModel, conditional_predictive_mixture
    m = UnifiedPSFModel.load(POINTERS[model_name]).base.model; d = arrays(ROOT, 'qso'); lab = labels()
    phot = Photometry(np.asarray(d['flux_dered'][rows]), np.asarray(d['variance_dered'][rows]), tuple(lab))
    f = m.transform(phot); anchors = m.reference_indices(phot)
    R = np.full((len(rows), len(lab)), np.nan)
    for i in range(len(rows)):
        if not m.qso.in_support(z[i]):
            continue
        mix, dims = conditional_predictive_mixture(m.qso, float(z[i]), f.x[i], f.cov[i], f.observed[i], int(anchors[i]))
        w = mix.weights[:, None]; mu = (w*mix.means).sum(0)
        var = (w*(np.diagonal(mix.covs, axis1=1, axis2=2) + mix.means**2)).sum(0) - mu**2
        R[i, dims] = (f.x[i, dims] - mu)/np.sqrt(var)
    return R


# ---------------------------------------------------------------------------------------------------
def _init(path):
    from qso_pcolor.unified import UnifiedPSFModel
    _s['m'] = UnifiedPSFModel.load(path); _s['cfg'] = json.loads(Path('configs/full_sample_release.json').read_text())


def _rescore(job):
    from qso_pcolor import BlendPolicy, RedshiftMatch
    F, V, ra, dec, z = job; m, cfg = _s['m'], _s['cfg']; lab = tuple(m.base.model.transform.bands)
    sc, dec_ = m.score(Photometry(F, V, lab), ra_deg=ra, dec_deg=dec, z_primary=z, morphology=['PSF']*len(z),
                       match=RedshiftMatch(half_width_kms=cfg['window_kms']), blend_policy=BlendPolicy(**cfg['blend_policy']),
                       ood_flag_sigma=cfg['ood_flag_sigma'], separation_arcsec=cfg['fixture_separation_arcsec'],
                       fracflux=cfg['fixture_fracflux'])
    return list(dec_['eligible']), [s.status if s is not None else r for s, r in zip(sc, dec_['reason'])]


def rescore(model_name, rows, z, workers):
    """Rescore rows with masked surveys. Returns {config: eligible array}."""
    cache = WORK/f'rescore_{model_name}_{len(rows)}_{int(np.sum(rows)) % 100000}.npz'
    if cache.exists():
        return dict(np.load(cache))
    d = arrays(ROOT, 'qso'); lab = labels()
    F0, V0 = np.asarray(d['flux'][rows], float), np.asarray(d['variance'][rows], float)
    ra, dec = np.asarray(d['ra'][rows]), np.asarray(d['dec'][rows])
    configs = {'all': np.zeros(len(lab), bool), 'legacy_only': survey_mask(lab, 'legacy_only'),
               **{f'minus_{s}': survey_mask(lab, s) for s in SURVEYS}}
    jobs, keys = [], []
    for k, drop in configs.items():
        V = V0.copy(); V[:, drop] = np.inf; F = np.where(np.isfinite(V), F0, np.nan)
        for i in range(0, len(rows), 16):
            jobs.append((F[i:i+16], V[i:i+16], ra[i:i+16], dec[i:i+16], z[i:i+16])); keys.append(k)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init,
                             initargs=(POINTERS[model_name],)) as pool:
        res = list(pool.map(_rescore, jobs))
    out = {}
    for k, (e, _) in zip(keys, res):
        out.setdefault(k, []).extend(e)
    out = {k: np.array(v, bool) for k, v in out.items()}
    np.savez(cache, **out); return out


# ---------------------------------------------------------------------------------------------------
def calibration_population(name):
    """Cached natural-population scores (calibrate_probability.py) with labels and properties."""
    from calibrate_probability import sample
    s = sample(40000)
    def cat(kind):
        fs = sorted(glob.glob(str(CAL/f'{name}_{kind}_[0-9]*.npz')))
        return {k: np.concatenate([np.load(f, allow_pickle=True)[k] for f in fs]) for k in ('p_quasar', 'eligible', 'status', 'ref_mag')}
    return s, cat('qso'), cat('stars')


def wilson(k, n, zc=1.):
    if n <= 0:
        return np.nan, np.nan
    p = k/n; den = 1 + zc**2/n; c = (p + zc**2/(2*n))/den; h = zc*np.sqrt(p*(1 - p)/n + zc**2/(4*n**2))/den
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rescore', action='store_true'); ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for p in (WORK, PLOTS, DOCS):
        p.mkdir(parents=True, exist_ok=True)
    rep = dict(definition=__doc__, threshold=THRESHOLD, part1={}, part2={})
    main_ = {m: load_main(m) for m in ('dependent', 'independent')}
    assert np.array_equal(main_['dependent']['rows'], main_['independent']['rows'])
    rows, z = main_['dependent']['rows'], main_['dependent']['z']; P = properties(rows)
    P['ref_mag'] = main_['dependent']['ref_mag']; P['north'] = main_['dependent']['hemi'] == 'north'
    for m, s in main_.items():
        u = ~s['ranked']; rr, cc = np.unique(s['reason'][u], return_counts=True)
        rep['part1'][m] = dict(n=int(len(u)), unranked=int(u.sum()), unranked_frac=float(u.mean()),
                               reasons={str(k): int(v) for k, v in zip(rr, cc)})
        print(m, rep['part1'][m], flush=True)
    uA, uB = ~main_['dependent']['ranked'], ~main_['independent']['ranked']
    rep['part1']['overlap'] = dict(both=int((uA & uB).sum()), only_A=int((uA & ~uB).sum()), only_B=int((~uA & uB).sum()))

    # -- P1: unranked fraction vs properties ------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2)); ax = ax.ravel()
    specs = [('z', 'quasar redshift', np.arange(0, 4.6, .25)), ('ref_mag', 'Legacy r (dereddened)', np.arange(16, 24.1, .5)),
             ('nb', 'number of observed bands', np.arange(1.5, 42, 2)), ('ebv', 'E(B-V) (SFD)', np.array([0, .02, .04, .06, .08, .1, .15, .2, .3])),
             ('absb', '|b| (deg)', np.arange(25, 91, 5)), ('snr_r', 'Legacy r S/N', np.array([10, 15, 20, 30, 50, 100, 200, 1e4]))]
    P['absb'] = np.abs(P['b']); rep['part1']['by'] = {}
    for x, (key, lbl, ed) in zip(ax, specs):
        v = P[key]; cen = .5*(ed[:-1] + ed[1:]) if key != 'snr_r' else np.sqrt(ed[:-1]*ed[1:])
        for m, s in main_.items():
            fr, lo, hi, n = [], [], [], []
            for l, h in zip(ed[:-1], ed[1:]):
                sel = (v >= l) & (v < h); k = int((~s['ranked'][sel]).sum()); nn = int(sel.sum())
                fr.append(k/nn if nn else np.nan); w = wilson(k, nn); lo.append(w[0]); hi.append(w[1]); n.append(nn)
            fr = np.array(fr); ok = np.array(n) >= 30
            x.errorbar(cen[ok], 100*fr[ok], yerr=100*np.abs(np.array([np.array(lo)[ok], np.array(hi)[ok]]) - fr[ok]), fmt='o-', ms=3,
                       color=COL[m], label=NAME[m], lw=1.2, capsize=0)
            rep['part1']['by'].setdefault(key, {})[m] = dict(edges=ed.tolist(), frac=fr.tolist(), n=n)
        x.set(xlabel=lbl, ylabel='unranked test quasars (%)'); x.set_ylim(bottom=0)
        if key == 'snr_r':
            x.set_xscale('log')
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'P1. Fraction of held-out test quasars left unranked ({len(rows):,} quasars, Legacy r S/N $\\geq$ 10; 68% intervals)', fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS/'P1_unranked_fraction.png', dpi=140); plt.close(fig)

    # -- P2: distances to the two models -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for x, (m, s) in zip(ax, main_.items()):
        dq, db = s['qso_ood_sigma_any_z'], s['bkg_ood_sigma']
        x.scatter(dq[s['ranked']][::5], db[s['ranked']][::5], s=1, color='0.7', alpha=.4, rasterized=True, label='ranked (1 in 5)')
        for r_, c in (('outside_both_models', '#c2410c'), ('qso_support_rejected', '#7c3aed')):
            sel = s['reason'] == r_
            x.scatter(dq[sel], db[sel], s=4, color=c, alpha=.7, label=f'{r_} ({sel.sum()})', rasterized=True)
        x.axvline(4, ls=':', color='k', lw=.8); x.axhline(4, ls=':', color='k', lw=.8)
        x.set(xscale='log', yscale='log', xlabel='distance from QSO model, any z ($\\sigma$)', ylabel='distance from background model ($\\sigma$)',
              title=NAME[m]); x.legend(frameon=False, fontsize=7, loc='lower right', markerscale=3)
    fig.tight_layout(); fig.savefig(PLOTS/'P2_distances.png', dpi=140); plt.close(fig)

    # -- P3: colours of unranked vs ranked quasars, by redshift ------------------------------------
    zb = ((0.1, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.4)); planes = (('g-r', 'r-z'), ('r-z', 'z-w1'), ('z-w1', 'w1-w2'))
    fig, ax = plt.subplots(len(planes), len(zb), figsize=(14, 10)); s = main_['dependent']
    for j, (zl, zh) in enumerate(zb):
        inz = (z >= zl) & (z < zh)
        for i, (u, v) in enumerate(planes):
            x = ax[i, j]; good = inz & s['ranked'] & np.isfinite(P[u]) & np.isfinite(P[v])
            x.hist2d(P[u][good], P[v][good], bins=60, range=[[-1, 2.5], [-1.5, 3]], cmap='Greys', cmin=1, rasterized=True)
            for r_, c, mk in (('outside_both_models', '#c2410c', 'o'), ('qso_support_rejected', '#7c3aed', 's')):
                sel = inz & (s['reason'] == r_)
                x.scatter(P[u][sel], P[v][sel], s=7, color=c, marker=mk, alpha=.75, lw=0, label=r_ if (i == 0 and j == 0) else None)
            x.set(xlim=(-1, 2.5), ylim=(-1.5, 3), xlabel=u.replace('w', 'W'), ylabel=v.replace('w', 'W'))
            if i == 0:
                x.set_title(f'{zl} $\\leq z <$ {zh}  (model A)', fontsize=10)
    ax[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle('P3. Legacy colours (dereddened) of unranked test quasars over the ranked ones (grey density)', fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS/'P3_unranked_colours.png', dpi=140); plt.close(fig)

    # -- P4: per-band residuals under the QSO model at z -------------------------------------------
    rng = np.random.default_rng(3); lab = labels()
    cache = WORK/'band_residuals.npz'
    if cache.exists():
        br = dict(np.load(cache))
    else:
        br = {}
        for m, s in main_.items():
            un = np.flatnonzero(~s['ranked']); rk = rng.choice(np.flatnonzero(s['ranked']), 3000, replace=False)
            br[f'{m}_un_idx'], br[f'{m}_rk_idx'] = un, rk
            br[f'{m}_un'] = band_residuals(m, rows[un], z[un]); br[f'{m}_rk'] = band_residuals(m, rows[rk], z[rk])
            print('residuals', m, flush=True)
        np.savez(cache, **br)
    show = [b for b in lab if not b.startswith('decals_dr9_north')
            and np.isfinite(br['dependent_rk'][:, lab.index(b)]).sum() >= 20]
    fig, axs = plt.subplots(2, 2, figsize=(15, 7.5), gridspec_kw=dict(width_ratios=[3, 1])); rep['part1']['band_residuals'] = {}
    for x, xc, m in zip(axs[:, 0], axs[:, 1], main_):
        for key, c, l in (('rk', '0.5', 'ranked'), ('un', COL[m], 'unranked')):
            mx = np.nanmax(np.abs(np.where(np.isfinite(br[f'{m}_{key}']), br[f'{m}_{key}'], np.nan)), axis=1); mx = np.sort(mx[np.isfinite(mx)])
            xc.plot(mx, 1 - np.arange(len(mx))/len(mx), color=c, label=l)
        xc.set(xscale='log', xlabel='largest |residual| of the object ($\\sigma$)', ylabel='fraction above', xlim=(.3, 100))
        xc.axvline(4, ls=':', color='k', lw=.8); xc.legend(frameon=False, fontsize=8)
        fu, fr_ = [], []
        for b in show:
            cols = [lab.index(b)] + ([lab.index(b.replace('south', 'north'))] if 'south' in b else [])
            def frac(Rm):
                v = np.concatenate([Rm[:, c] for c in cols]); v = v[np.isfinite(v)]
                return (np.mean(np.abs(v) > 4) if len(v) >= 20 else np.nan), len(v)
            a1, n1 = frac(br[f'{m}_un']); a2, n2 = frac(br[f'{m}_rk']); fu.append(a1); fr_.append(a2)
            rep['part1']['band_residuals'].setdefault(m, {})[b.replace('decals_dr9_south', 'legacy')] = dict(unranked=a1, n_unranked=n1, ranked=a2, n_ranked=n2)
        X = np.arange(len(show))
        x.bar(X - .2, 100*np.array(fu), .4, color=COL[m], label='unranked'); x.bar(X + .2, 100*np.array(fr_), .4, color='0.6', label='ranked (3,000 random)')
        x.set(ylabel='bands with |residual| > 4$\\sigma$ (%)', title=f'P4. {NAME[m]}: per-band residuals under the QSO model at the quasar redshift (given Legacy r)')
        x.legend(frameon=False, fontsize=8)
        x.set_xticks(np.arange(len(show))); x.set_xticklabels([b.replace('decals_dr9_south', 'legacy') for b in show], rotation=90, fontsize=7)
    fig.tight_layout(); fig.savefig(PLOTS/'P4_band_residuals.png', dpi=140); plt.close(fig)
    # number of discrepant bands per object
    for m in main_:
        nu = (np.abs(np.nan_to_num(br[f'{m}_un'])) > 4).sum(axis=1); nr = (np.abs(np.nan_to_num(br[f'{m}_rk'])) > 4).sum(axis=1)
        rep['part1'].setdefault('n_discrepant', {})[m] = dict(unranked={str(k): float(np.mean(nu == k)) for k in range(4)} | {'4+': float(np.mean(nu >= 4))},
                                                              ranked={str(k): float(np.mean(nr == k)) for k in range(4)} | {'4+': float(np.mean(nr >= 4))})

    # -- P5: inter-survey r differences (variability / calibration) and survey coverage ------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2)); s = main_['dependent']
    for x, key in zip(ax[:2], ('dr_sdss', 'dr_ps1')):
        for sel, c, l in ((s['ranked'], '0.5', 'ranked'), (~s['ranked'], COL['dependent'], 'unranked')):
            v = P[key][sel]; v = v[np.isfinite(v)]
            x.hist(v, bins=np.linspace(-1.5, 1.5, 61), density=True, histtype='step', color=c, lw=1.5, label=f'{l} (n={len(v)})')
        x.set(xlabel=key.replace('dr_', '').upper() + ' r $-$ Legacy r (mag)', ylabel='density', yscale='log', title='different epochs and systems')
        x.legend(frameon=False, fontsize=8)
        rep['part1'].setdefault('dr', {})[key] = dict(ranked_mad=float(np.nanmedian(np.abs(P[key][s['ranked']]))),
                                                     unranked_mad=float(np.nanmedian(np.abs(P[key][~s['ranked']]))),
                                                     ranked_gt_0p5=float(np.nanmean(np.abs(P[key][s['ranked']]) > .5)),
                                                     unranked_gt_0p5=float(np.nanmean(np.abs(P[key][~s['ranked']]) > .5)))
    sv = ('sdss', 'ps1', 'nsc', 'skymapper', 'vhs', 'allwise'); X = np.arange(len(sv))
    ax[2].bar(X - .2, [100*P['has_' + k][s['ranked']].mean() for k in sv], .4, color='0.6', label='ranked')
    ax[2].bar(X + .2, [100*P['has_' + k][~s['ranked']].mean() for k in sv], .4, color=COL['dependent'], label='unranked')
    ax[2].set_xticks(X); ax[2].set_xticklabels(sv); ax[2].set(ylabel='objects with the survey (%)', title='survey coverage (model A)'); ax[2].legend(frameon=False, fontsize=8)
    rep['part1']['coverage'] = {k: dict(ranked=float(P['has_' + k][s['ranked']].mean()), unranked=float(P['has_' + k][~s['ranked']].mean())) for k in sv}
    fig.suptitle('P5. Survey-to-survey consistency of the unranked quasars (model A)', fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS/'P5_survey_consistency.png', dpi=140); plt.close(fig)

    # -- P6: rescoring with surveys removed --------------------------------------------------------
    if a.rescore:
        rep['part1']['rescore'] = {}
        fig, ax = plt.subplots(figsize=(9, 4.2)); keys = ['all', 'legacy_only'] + [f'minus_{k}' for k in SURVEYS]
        for off, m in ((-.2, 'dependent'), (.2, 'independent')):
            un = np.flatnonzero(~main_[m]['ranked'])
            r = rescore(m, rows[un], z[un], a.workers)
            fr = {k: float(r[k].mean()) for k in keys}; rep['part1']['rescore'][m] = dict(n=int(len(un)), ranked_fraction=fr)
            ax.bar(np.arange(len(keys)) + off, [100*fr[k] for k in keys], .4, color=COL[m], label=NAME[m])
            print('rescore', m, fr, flush=True)
        ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels([k.replace('minus_', '$-$') for k in keys], rotation=30, fontsize=8)
        ax.set(ylabel='now ranked (%)', title='P6. Unranked test quasars rescored with surveys removed'); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(PLOTS/'P6_rescore_surveys.png', dpi=140); plt.close(fig)

    # -- P8: the outside-both-models cut against the number of observed bands ---------------------
    from scipy.stats import chi2
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4)); K = np.arange(2, 30); rep['part1']['distance_vs_bands'] = {}
    st = arrays(ROOT, 'stars')
    for kind in ('qso', 'stars'):
        fs = sorted(f for f in glob.glob(str(PERF/f'dependent_{kind}_main_*_[0-9]*.npz')) if not f.endswith('.rows.npz'))
        dq = np.concatenate([np.load(f, allow_pickle=True)['qso_ood_sigma_any_z'] for f in fs])
        db = np.concatenate([np.load(f, allow_pickle=True)['bkg_ood_sigma'] for f in fs])
        rr = np.concatenate([np.load(f[:-4] + '.rows.npz')['rows'] for f in fs])
        src = arrays(ROOT, 'qso') if kind == 'qso' else st
        k = np.asarray(src['observed'][rr]).astype(bool).sum(1)
        lab_ = 'test quasars' if kind == 'qso' else 'background objects'; c = COL['dependent'] if kind == 'qso' else '0.3'
        for x, dist, nm in ((ax[0], dq, 'QSO model'), (ax[1], db, 'background model')):
            med = [np.nanmedian(dist[k == kk]) if (k == kk).sum() >= 30 else np.nan for kk in K]
            p99 = [np.nanpercentile(dist[k == kk], 99) if (k == kk).sum() >= 30 else np.nan for kk in K]
            x.plot(K, med, 'o-', color=c, ms=3, label=f'{lab_}: median'); x.plot(K, p99, 's--', color=c, ms=3, label=f'{lab_}: 99th percentile')
            x.set(xlabel='number of observed bands k', ylabel=f'distance from {nm} ($\\sigma$)', title=f'distance from the {nm}')
        out = (dq > 4) & (db > 4)
        fr = [out[k == kk].mean() if (k == kk).sum() >= 30 else np.nan for kk in K]
        ax[2].plot(K, 100*np.array(fr), 'o-', color=c, ms=3, label=lab_)
        rep['part1']['distance_vs_bands'][kind] = dict(k=K.tolist(), outside_frac=fr, n=[int((k == kk).sum()) for kk in K])
    for x in ax[:2]:
        x.axhline(4, ls=':', color='k', lw=.8); x.plot(K, np.sqrt(chi2.ppf(.5, K)), color='0.6', lw=.8, label='$\\sqrt{\\chi^2_k}$ median (one Gaussian)')
        x.legend(frameon=False, fontsize=7)
    ax[2].set(xlabel='number of observed bands k', ylabel='"outside both models" (%)', title='fixed 4$\\sigma$ cut on both distances (model A)')
    ax[2].legend(frameon=False, fontsize=8)
    fig.suptitle('P8. The raw Mahalanobis distance grows with the number of observed bands; the 4$\\sigma$ cut does not', fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS/'P8_distance_vs_bands.png', dpi=140); plt.close(fig)

    # -- Part 2: top-end reliability on the natural population --------------------------------------
    flag_reason = np.asarray(arrays(ROOT, 'stars')['qso_flag_reason']).astype(str)
    st = arrays(ROOT, 'stars'); lab = labels()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    edges = np.array([.5, .7, .8, .9, .95, .97, .99, .999, 1.0000001])
    for m in ('dependent', 'independent'):
        smp, q, b = calibration_population(m); w = smp['weight_star']
        pq, pb = np.where(q['eligible'], q['p_quasar'], np.nan), np.where(b['eligible'], b['p_quasar'], np.nan)
        flag = smp['sflag'].astype(bool); cen, lo_, hi_, elo, ehi, rows_ = [], [], [], [], [], []
        for l, h in zip(edges[:-1], edges[1:]):
            nq = int(((pq >= l) & (pq < h)).sum()); sb = (pb >= l) & (pb < h); nf = int((sb & flag).sum()); nu = int((sb & ~flag).sum())
            tot = nq + w*(nf + nu); f_lo = nq/tot if tot else np.nan; f_hi = (nq + w*nf)/tot if tot else np.nan
            neff = nq + nf + nu
            mp = float(np.nanmean(np.concatenate([pq[(pq >= l) & (pq < h)], pb[sb]]))) if neff else np.nan
            e1 = wilson(f_lo*neff, neff)[0] if neff else np.nan; e2 = wilson(f_hi*neff, neff)[1] if neff else np.nan
            cen.append(mp); lo_.append(f_lo); hi_.append(f_hi); elo.append(e1); ehi.append(e2)
            rows_.append(dict(bin=[float(l), float(min(h, 1))], mean_p=mp, n_known_qso=nq, n_flagged=nf, n_unflagged=nu, star_weight=w,
                              frac_lower=f_lo, frac_upper=f_hi, lower_68=e1, upper_68=e2))
        rep['part2'][m] = dict(top_bins=rows_)
        cen = np.array(cen); k = np.isfinite(cen)
        ax[0].fill_between(cen[k], np.array(elo)[k], np.array(ehi)[k], color=COL[m], alpha=.15, lw=0)
        ax[0].plot(cen[k], np.array(lo_)[k], 'o-', color=COL[m], ms=3, lw=1, label=f'{NAME[m]}: known QSOs only')
        ax[0].plot(cen[k], np.array(hi_)[k], 's--', color=COL[m], ms=3, lw=1, label='   + flagged likely QSOs')
        # composition of high-p background objects
        if m == 'dependent':
            hp = np.flatnonzero(pb > .97); srow = smp['stars'][hp]
            reasons = flag_reason[srow]; rr, cc = np.unique(np.where(reasons == '', 'none', reasons), return_counts=True)
            rep['part2']['high_p_background_reasons'] = {str(k_): int(v) for k_, v in zip(rr, cc)}
            obs = np.asarray(st['observed'][srow]).astype(bool); y = np.asarray(st['y'][srow])
            south = obs[:, lab.index('decals_dr9_south:r')]
            def leg(bn):
                i, j = lab.index('decals_dr9_south:' + bn), lab.index('decals_dr9_north:' + bn)
                return np.where(south, np.where(obs[:, i], y[:, i], np.nan), np.where(obs[:, j], y[:, j], np.nan))
            r_, w1, w2 = leg('r'), leg('w1'), leg('w2')
            Fw = np.asarray(st['flux_dered'][srow]); Vw = np.asarray(st['variance_dered'][srow])
            iw = np.where(south, lab.index('decals_dr9_south:w1'), lab.index('decals_dr9_north:w1'))
            with np.errstate(invalid='ignore', divide='ignore'):
                snr_w1 = Fw[np.arange(len(srow)), iw]/np.sqrt(Vw[np.arange(len(srow)), iw])
            fl = np.char.find(reasons, 'quaia') >= 0; fl |= np.char.find(reasons, 'wise') >= 0     # the flag is quaia OR wise; pm is diagnostic only
            ax[1].scatter(r_[~fl], (w1 - w2)[~fl], s=8, color='0.3', label=f'unflagged ({(~fl).sum()})')
            ax[1].scatter(r_[fl], (w1 - w2)[fl], s=8, color='#c2410c', label=f'flagged likely QSO ({fl.sum()})')
            qq = q['eligible'] & (q['p_quasar'] > .97); qrow = smp['qso'][qq]; dq = arrays(ROOT, 'qso')
            oq = np.asarray(dq['observed'][qrow]).astype(bool); yq = np.asarray(dq['y'][qrow]); sq = oq[:, lab.index('decals_dr9_south:r')]
            def legq(bn):
                i, j = lab.index('decals_dr9_south:' + bn), lab.index('decals_dr9_north:' + bn)
                return np.where(sq, np.where(oq[:, i], yq[:, i], np.nan), np.where(oq[:, j], yq[:, j], np.nan))
            ax[1].scatter(legq('r'), legq('w1') - legq('w2'), s=3, color=COL['dependent'], alpha=.3, label=f'known QSOs, p > 0.97 ({qq.sum()})')
            ax[1].axhline(.8 - (3.339 - 2.699) + 0, ls=':', color='k', lw=.8)
            ax[1].set(xlabel='Legacy r (dereddened)', ylabel='Legacy W1 $-$ W2 (AB)', title='objects with p > 0.97 (model A)', ylim=(-1.5, 2.5))
            ax[1].legend(frameon=False, fontsize=7)
            rmag = np.concatenate([r_[~fl], r_[fl]])
            ed = np.arange(16, 24.1, .5)
            ax[2].hist(r_[~fl], bins=ed, histtype='step', color='0.3', lw=1.5, label='background, unflagged')
            ax[2].hist(r_[fl], bins=ed, histtype='step', color='#c2410c', lw=1.5, label='background, flagged')
            ax[2].hist(legq('r'), bins=ed, histtype='step', color=COL['dependent'], lw=1.5, label='known QSOs')
            ax[2].set(xlabel='Legacy r (dereddened)', ylabel='objects with p > 0.97', title='magnitudes (model A, unweighted counts)'); ax[2].legend(frameon=False, fontsize=7)
            rep['part2']['high_p_background'] = dict(n=int(len(hp)), flagged=int(fl.sum()), unflagged=int((~fl).sum()),
                                                     unflagged_w1_snr_lt_5=float(np.nanmean(snr_w1[~fl] < 5)) if (~fl).any() else None,
                                                     unflagged_w1w2_gt_0p16=float(np.nanmean((w1 - w2)[~fl] > .16)) if (~fl).any() else None,
                                                     unflagged_median_r=float(np.nanmedian(r_[~fl])) if (~fl).any() else None,
                                                     flagged_median_r=float(np.nanmedian(r_[fl])) if fl.any() else None,
                                                     known_qso_median_r=float(np.nanmedian(legq('r'))), n_known_qso=int(qq.sum()))
    ax[0].plot([.5, 1], [.5, 1], ':', color='k', lw=.8)
    ax[0].set(xlabel='mean predicted p_quasar (ranked objects)', ylabel='observed quasar fraction', xlim=(.5, 1.005), ylim=(0, 1.02),
              title='top-end reliability (natural population, weighted)'); ax[0].legend(frameon=False, fontsize=6.5, loc='upper left')
    fig.suptitle('P7. Overconfidence above p = 0.97: reliability with 68% intervals, and who the high-p objects are', fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS/'P7_top_end.png', dpi=140); plt.close(fig)
    (DOCS/'pquasar_diagnostics.json').write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep['part2'].items() if k != 'dependent' and k != 'independent'}, indent=1, default=float))


if __name__ == '__main__':
    main()
