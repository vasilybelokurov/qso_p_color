#!/usr/bin/env python
"""F16: worked examples, real test objects scored by both promoted models.

Each panel: observed extinction-corrected luptitudes minus the reference-band luptitude,
against approximate central wavelength (display only), with errors; model prediction
for the same quantity at the object's redshift (QSOs: spectroscopic z; background objects:
the z they were scored at), conditioned on the reference band: mean and 1-sigma of the
moment-matched conditional mixture, both quasar models. Titles list both models' log R,
p_quasar and support percentile from score_performance.py.
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import COLOURS, LABELS, bands, condition_on, data, idx, model, save_figure, use_paper_style

from common import PERF as OUT
WAVE = {'sdss': dict(u=355, g=469, r=617, i=748, z=893), 'decals_dr9_south': dict(g=480, r=640, z=920, w1=3400, w2=4600),
        'decals_dr9_north': dict(g=477, r=623, z=918, w1=3400, w2=4600), 'allwise': dict(w1=3400, w2=4600, w3=12000, w4=22000),
        'ps1': dict(g=487, r=622, i=755, z=868, y=963), 'nsc': dict(u=380, g=480, r=640, i=780, z=920, y=990, vr=630),
        'skymapper': dict(u=349, v=384, g=510, r=617, i=779, z=916), 'vhs': dict(y=1020, j=1250, h=1650, ks=2150)}


def scored(kind, model_name):
    out = {}
    for p in sorted(glob.glob(str(OUT/f'{model_name}_{kind}_main_*_[0-9]*.npz'))):
        if p.endswith('.rows.npz'):
            continue
        d = np.load(p); r = np.load(p[:-4] + '.rows.npz')
        for i, row in enumerate(r['rows']):
            out[int(row)] = dict(z=float(r['z'][i]), logR=float(d['log_r_per_unit_z'][i]), pq=float(d['p_quasar'][i]),
                                 support=float(d['support'][i]), eligible=bool(d['eligible'][i]), reference=str(d['reference'][i]),
                                 rejected=bool(d['support_rejected'][i]))
    return out


def pick(kind, s, rule):
    rows = np.array(sorted(s)); keep = [r for r in rows if rule(r, s[r])]
    return keep[len(keep)//2] if keep else None


def predict(name, z, ref, yref, vref):
    mod = model(name); k = int(np.abs(np.asarray(mod.qso.z_centres) - z).argmin())
    c = condition_on(mod.qso.mixtures[k], ref, yref, vref)
    mean = c.weights @ c.means; second = np.einsum('k,kij->ij', c.weights, c.covs + c.means[:, :, None]*c.means[:, None, :])
    return mean, np.sqrt(np.clip(np.diag(second - np.outer(mean, mean)), 0, None))


def panel(ax, kind, row, sc, title):
    d = data(kind); lab = bands(); obs = np.asarray(d['observed'][row]); y = np.asarray(d['y'][row]); v = np.asarray(d['noise'][row])
    ref = sc['independent']['reference']; a = idx(ref); z = sc['independent']['z']
    for name in ('independent', 'dependent'):
        mean, sd = predict(name, z, ref, y[a], v[a])
        xs, ms, ss = [], [], []
        for j, l in enumerate(lab):
            s, bnd = l.split(':')
            if obs[j] and j != a:
                xs.append(WAVE[s][bnd]); ms.append(mean[j] - y[a]); ss.append(sd[j])
        shift = 1.035 if name == 'independent' else 1/1.035          # small offset in wavelength for legibility
        ax.errorbar(np.array(xs)*shift, ms, ss, fmt='s', ms=3, color=COLOURS[name], lw=1, alpha=.85,
                    label=LABELS[name] + ' (mean $\\pm1\\sigma$)')
    xs = [WAVE[l.split(':')[0]][l.split(':')[1]] for j, l in enumerate(lab) if obs[j] and j != a]
    ys = [y[j] - y[a] for j in range(len(lab)) if obs[j] and j != a]; es = [np.sqrt(v[j]) for j in range(len(lab)) if obs[j] and j != a]
    ax.errorbar(xs, ys, es, fmt='o', ms=3.5, color='k', lw=.8, label='observed')
    ax.set_xscale('log'); ax.invert_yaxis(); ax.set_xlabel('approximate central wavelength [nm]')
    ax.set_ylabel(f'luptitude minus reference ({ref.replace("decals_dr9_", "LS-").replace(":", " ")}) [mag]', fontsize=7.5)
    lines = [f"{'indep' if n == 'independent' else 'dep'}: log R={sc[n]['logR']:.1f}, p_Q={sc[n]['pq']:.2f}, support={sc[n]['support']:.2f}"
             + ('' if sc[n]['eligible'] else ' (not ranked)') for n in ('independent', 'dependent')]
    ax.set_title(f'{title}, $z_0$ = {z:.2f}\n' + '\n'.join(lines), fontsize=7.5)


def main():
    use_paper_style()
    sq = {n: scored('qso', n) for n in ('independent', 'dependent')}; sb = {n: scored('stars', n) for n in ('independent', 'dependent')}
    common_q = {r: {n: sq[n][r] for n in sq} for r in set(sq['independent']) & set(sq['dependent'])}
    common_b = {r: {n: sb[n][r] for n in sb} for r in set(sb['independent']) & set(sb['dependent'])}
    nobs = lambda kind, r: int(np.asarray(data(kind)['observed'][r]).sum())
    rmag = lambda kind, r: float(data(kind)['y'][r][idx('decals_dr9_south:r')])
    cases = [('qso', pick('qso', common_q, lambda r, s: abs(s['independent']['z'] - 1.5) < .05 and s['independent']['eligible'] and nobs('qso', r) >= 14 and rmag('qso', r) < 19.5), 'bright QSO'),
             ('qso', pick('qso', common_q, lambda r, s: abs(s['independent']['z'] - 3.0) < .05 and s['independent']['eligible'] and nobs('qso', r) >= 10), 'QSO at z~3'),
             ('qso', pick('qso', common_q, lambda r, s: abs(s['independent']['z'] - .6) < .05 and s['independent']['eligible'] and nobs('qso', r) >= 10), 'low-redshift QSO'),
             ('qso', pick('qso', common_q, lambda r, s: s['independent']['rejected'] and nobs('qso', r) >= 8), 'QSO rejected by support cut'),
             ('stars', pick('stars', common_b, lambda r, s: s['independent']['eligible'] and s['independent']['pq'] < .01 and nobs('stars', r) >= 10 and rmag('stars', r) < 20), 'background star'),
             ('stars', pick('stars', common_b, lambda r, s: s['independent']['eligible'] and s['independent']['pq'] > .5), 'background object with $p_Q>0.5$')]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4)); chosen = []
    for ax, (kind, row, title) in zip(axes.ravel(), cases):
        if row is None:
            ax.set_visible(False); continue
        sc = (common_q if kind == 'qso' else common_b)[row]; panel(ax, kind, row, sc, title)
        d = data(kind); chosen.append(dict(kind=kind, prepared_row=int(row), source_row=int(d['source_row'][row]), title=title,
                                           ra=float(d['ra'][row]), dec=float(d['dec'][row]), scores=sc))
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .95))
    Path('docs/method_unified/examples.json').write_text(json.dumps(chosen, indent=1))
    print(save_figure(fig, 'method_unified/F16_examples'))


if __name__ == '__main__':
    main()
