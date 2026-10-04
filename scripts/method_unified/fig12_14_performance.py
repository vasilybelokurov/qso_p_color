#!/usr/bin/env python
"""F12-F14: performance of both promoted models across sky, magnitude, redshift and band coverage.

Input: scores written by score_performance.py (clean role-3 rows outside every development
exclusion and outside the release-check rows; support cut at 99.5% calibration retention).
Metrics (defined in the note):
  AUC        P(random test QSO ranks above random test background object on log R);
             unranked objects at the bottom.
  density    mean ln p(colours | Q, z_true, m_ref) of test QSOs [nat/object].
  retention  fraction of test QSOs not rejected by the support cut.
  incidence  fraction of background test objects ranked with p_quasar > 0.5.
Writes figures and docs/method_unified/performance.json (also the independent check of the cut).
"""
import glob
import json
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from common import COLOURS, LABELS, data, idx, save_figure, use_paper_style

from common import PERF as OUT
MODELS = ('independent', 'dependent')
SURVEYS = ('sdss', 'decals_dr9_south', 'decals_dr9_north', 'allwise', 'ps1', 'nsc', 'skymapper', 'vhs')
SHORT = dict(sdss='SDSS', decals_dr9_south='LS-S', decals_dr9_north='LS-N', allwise='AllWISE', ps1='PS1',
             nsc='NSC', skymapper='SkyMapper', vhs='VHS')


def load(model, kind, panel, suffix=''):
    """Concatenate scored chunks; returns dict of arrays including prepared rows and z."""
    parts = []
    for p in sorted(glob.glob(str(OUT/f'{model}_{kind}_{panel}_*{suffix}_[0-9]*.npz'))):
        if p.endswith('.rows.npz'):
            continue
        tag = Path(p).name[len(f'{model}_{kind}_{panel}_'):-len('_000000.npz')]
        if suffix == '' and ('_z' in tag or any(tag.endswith(m) for m in ('grz', 'all', 'sdss', 'ps1'))):
            continue
        d = dict(np.load(p)); r = dict(np.load(p[:-4] + '.rows.npz')); d.update(rows=r['rows'], z=r['z'])
        d['hemi'] = np.full(len(r['rows']), 'north' if '_north' in p else 'south'); parts.append(d)
    if not parts:
        return None
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0] if k not in ('flags', 'reference')}


def rank(d):
    return np.where(d['eligible'], d['log_r_per_unit_z'], -np.inf)


def auc(q, b):
    nq, nb = len(q['rows']), len(b['rows'])
    if nq < 30 or nb < 30:
        return np.nan
    return float(mannwhitneyu(rank(q), rank(b)).statistic/(nq*nb))


def sub(d, m):
    return {k: v[m] for k, v in d.items()}


def summary(q, b):
    return dict(n_qso=int(len(q['rows'])), n_bkg=int(len(b['rows'])), auc=auc(q, b),
                density=float(np.nanmean(np.where(np.isfinite(q['loglike_qso_zprimary']), q['loglike_qso_zprimary'], np.nan))),
                retention=float(1 - q['support_rejected'].mean()),
                incidence=float((b['eligible'] & (b['p_quasar'] > .5)).mean()))


def coverage_label(kind, rows):
    obs = np.asarray(data(kind)['observed'][rows]); labs = json.loads(Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059/layout.json').read_text())['native_labels']
    has = np.stack([obs[:, [i for i, l in enumerate(labs) if l.startswith(s+':')]].any(axis=1) for s in SURVEYS], 1)
    return np.array(['+'.join(SHORT[s] for s, h in zip(SURVEYS, row) if h) for row in has])


def f12(main, report):
    fig = plt.figure(figsize=(11, 8.2)); nside = 4
    for i, model in enumerate(MODELS):
        q, b = main[model]
        pq = hp.ang2pix(nside, data('qso')['l'][q['rows']], data('qso')['b'][q['rows']], lonlat=True)
        pb = hp.ang2pix(nside, data('stars')['l'][b['rows']], data('stars')['b'][b['rows']], lonlat=True)
        amap = np.full(hp.nside2npix(nside), hp.UNSEEN); imap = amap.copy()
        for c in np.union1d(pq, pb):
            mq, mb = pq == c, pb == c
            if mq.sum() >= 30 and mb.sum() >= 30:
                amap[c] = auc(sub(q, mq), sub(b, mb)); imap[c] = (b['eligible'][mb] & (b['p_quasar'][mb] > .5)).mean()
        hp.mollview(amap, fig=fig.number, sub=(3, 2, 2*i+1), title=f'AUC per nside-4 cell: {LABELS[model]}', min=.9, max=1., cmap='viridis', unit='', badcolor='white')
        hp.mollview(100*imap, fig=fig.number, sub=(3, 2, 2*i+2), title=f'high-QSO incidence [%]: {LABELS[model]}', min=0, max=1., cmap='Greys', unit='', badcolor='white')
    for j, (key, edges, xlabel) in enumerate((('b', [25, 30, 35, 45, 55, 70, 90], '$|b|$ [deg]'), ('mag', [16, 18, 19, 20, 20.5, 21, 21.5, 22, 22.5, 23.5], 'reference magnitude [mag]'))):
        ax = fig.add_axes((.08 + .48*j, .06, .38, .24)); report['by_'+key] = {}
        for model in MODELS:
            q, b = main[model]
            xq = np.abs(data('qso')['b'][q['rows']]) if key == 'b' else q['ref_mag']
            xb = np.abs(data('stars')['b'][b['rows']]) if key == 'b' else b['ref_mag']
            vals = [summary(sub(q, (xq >= lo) & (xq < hi)), sub(b, (xb >= lo) & (xb < hi))) for lo, hi in zip(edges[:-1], edges[1:])]
            report['by_'+key][model] = dict(edges=edges, bins=vals)
            cen = .5*(np.array(edges[1:]) + np.array(edges[:-1]))
            ax.plot(cen, [v['auc'] for v in vals], 'o-', color=COLOURS[model], label=LABELS[model], ms=4)
        ax.set(xlabel=xlabel, ylabel='AUC'); ax.legend(frameon=False, fontsize=7)
    return fig


def f13(main, zgrid, report):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3)); report['by_z'] = {}
    for model in MODELS:
        q, _ = main[model]; zq = q['z']; rows = []
        for zt in sorted(zgrid[model]):
            b = zgrid[model][zt]; m = np.abs(zq - zt) < .1
            rows.append(dict(z=zt, **summary(sub(q, m), b)))
        report['by_z'][model] = rows
        z = [r['z'] for r in rows]
        ax[0].plot(z, [r['auc'] for r in rows], 'o-', color=COLOURS[model], label=LABELS[model])
        edges = np.arange(.1, 4.45, .2); cen = .5*(edges[1:] + edges[:-1])
        dens = [np.nanmean(q['loglike_qso_zprimary'][(zq >= lo) & (zq < hi)]) if ((zq >= lo) & (zq < hi)).sum() > 30 else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
        ret = [1 - q['support_rejected'][(zq >= lo) & (zq < hi)].mean() if ((zq >= lo) & (zq < hi)).sum() > 30 else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
        ax[1].plot(cen, dens, 'o-', color=COLOURS[model], ms=3); ax[2].plot(cen, ret, 'o-', color=COLOURS[model], ms=3)
    ax[0].set(xlabel='redshift $z_0$', ylabel='AUC', title='QSOs at $z_0\\pm0.1$ vs background scored at $z_0$')
    ax[1].set(xlabel='QSO redshift', ylabel='held-out log density [nat]', title='mean $\\ln p$(colours | Q, $z$)')
    ax[2].set(xlabel='QSO redshift', ylabel='fraction kept', title='support retention')
    ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); return fig


def f14(main, masks, report):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=(1.6, 1)))
    q0, _ = main['independent']; labq = coverage_label('qso', q0['rows'])
    combos = [c for c, n in zip(*np.unique(labq, return_counts=True)) if n >= 300]
    combos = sorted(combos, key=lambda c: -(labq == c).sum())[:10]; report['by_coverage'] = {}
    for k, model in enumerate(MODELS):
        q, b = main[model]; lq, lb = coverage_label('qso', q['rows']), coverage_label('stars', b['rows'])
        vals = [summary(sub(q, lq == c), sub(b, lb == c)) for c in combos]; report['by_coverage'][model] = dict(zip(combos, vals))
        y = np.arange(len(combos)) + (k - .5)*.35
        ax[0].barh(y, [v['auc'] - .9 for v in vals], left=.9, height=.33, color=COLOURS[model], label=LABELS[model])
    ax[0].set_yticks(np.arange(len(combos))); ax[0].set_yticklabels([f'{c} ({(labq == c).sum():,} QSOs)' for c in combos], fontsize=7)
    ax[0].set(xlabel='AUC', xlim=(.9, 1.), title='actual survey coverage (≥300 test QSOs)'); ax[0].invert_yaxis(); ax[0].legend(frameon=False, fontsize=7, loc='lower center', bbox_to_anchor=(.5, -.28), ncol=2)
    report['by_mask'] = {}; names = list(next(iter(masks.values())).keys())
    for k, model in enumerate(MODELS):
        vals = [summary(*masks[model][m]) for m in names]; report['by_mask'][model] = dict(zip(names, vals))
        y = np.arange(len(names)) + (k - .5)*.35
        ax[1].barh(y, [v['auc'] - .85 for v in vals], left=.85, height=.33, color=COLOURS[model])
    for i, m in enumerate(names):
        if not np.isfinite(report['by_mask']['independent'][m]['auc']):
            ax[1].text(.86, i, 'not scored: no Legacy $r$\n(faint-limit band)', va='center', fontsize=7, color='0.35')
    ax[1].set_yticks(np.arange(len(names))); ax[1].set_yticklabels(names, fontsize=8); ax[1].invert_yaxis()
    ax[1].set(xlabel='AUC', xlim=(.85, 1.), title='same objects, artificial band masks')
    fig.tight_layout(); return fig


def main():
    use_paper_style(); report = dict(definition=__doc__)
    main_ = {m: (load(m, 'qso', 'main'), load(m, 'stars', 'main')) for m in MODELS}
    report['overall'] = {m: summary(*main_[m]) for m in MODELS}
    for m in MODELS:
        for h in ('south', 'north'):
            q, b = main_[m]; report['overall'][m+'_'+h] = summary(sub(q, q['hemi'] == h), sub(b, b['hemi'] == h))
    import re
    zgrid = {m: {float(re.search(r'_z([0-9.]+)_[0-9]{6}\.npz$', p).group(1)): None
                 for p in glob.glob(str(OUT/f'{m}_stars_zgrid_*_z*_000000.npz'))} for m in MODELS}
    for m in MODELS:
        for zt in list(zgrid[m]):
            zgrid[m][zt] = load(m, 'stars', 'zgrid', f'_z{zt}')
    masks = {m: {mk: (load(m, 'qso', 'masks', '_'+mk), load(m, 'stars', 'masks', '_'+mk))
                 for mk in ('legacy_grz', 'legacy_all', 'sdss', 'ps1', 'optical_all')} for m in MODELS}
    print(save_figure(f12(main_, report), 'method_unified/F12_performance_sky_magnitude'))
    print(save_figure(f13(main_, zgrid, report), 'method_unified/F13_performance_redshift'))
    print(save_figure(f14(main_, masks, report), 'method_unified/F14_performance_coverage'))
    Path('docs/method_unified/performance.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report['overall'], indent=1))


if __name__ == '__main__':
    main()
