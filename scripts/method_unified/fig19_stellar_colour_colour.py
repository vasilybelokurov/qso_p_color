#!/usr/bin/env python
"""F19: stellar colour-colour density in magnitude bins, data vs the stellar (background) model.

Data (grey, log density): held-out stellar-sample objects (role 3: test cones, never fitted), Legacy
South r S/N >= 10 (the faint limit), likely QSOs removed (Quaia / WISE AGN colours), all plane bands
observed with S/N >= 5. Extinction-corrected luptitudes (AB).
Model (contours enclosing 50% and 90%): posterior-predictive draws for the same objects. For each
object with the plane bands observed, DRAWS samples are taken from the background mixture conditioned
on its measured Legacy r (component weights from the bundle's sky gate at its l, b); each drawn
luptitude is converted to flux, the object's own flux error (extinction-corrected variance) is added
in flux, the result is converted back to luptitude, and the same S/N >= 5 cut is applied to the noisy
drawn flux. This reproduces how the data were measured and selected (a fixed magnitude error would
overstate the scatter of bright draws in low-S/N bands such as SDSS u, W1, W2). Panels with fewer
than 100 data objects are left empty. Titles: data N, fraction of data inside the model's 50% / 90% regions (ideal 0.50 /
0.90), and the fraction of model draws in colour cells (45 x 45) with no data.

Usage: python scripts/method_unified/fig19_stellar_colour_colour.py [bundle_dir] [output_tag]
(default: models/multisurvey_psf/work/stellar_binned/20261003/bundles/independent, F19_stellar_colour_colour)
"""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
from common import ROOT, data, idx, save_figure, use_paper_style

BUNDLE = Path(sys.argv[1] if len(sys.argv) > 1 else 'models/multisurvey_psf/work/stellar_binned/20261003/bundles/independent')
TAG = sys.argv[2] if len(sys.argv) > 2 else 'F19_stellar_colour_colour'
S = 'decals_dr9_south:'
REF = S + 'r'
PLANES = [  # (x colour, y colour, x range, y range)
    ((S+'g', S+'r'), (S+'r', S+'z'), (-.5, 2.), (-.6, 2.2)),
    (('sdss:u', 'sdss:g'), ('sdss:g', 'sdss:r'), (-.5, 3.5), (-.5, 2.)),
    ((S+'r', S+'z'), (S+'z', S+'w1'), (-.6, 2.2), (-2., 3.)),
    ((S+'z', S+'w1'), (S+'w1', S+'w2'), (-2., 3.), (-1.2, 1.2)),
]
LABEL = {S+'g': 'g', S+'r': 'r', S+'z': 'z', S+'w1': 'W1', S+'w2': 'W2', 'sdss:u': 'u_{SDSS}', 'sdss:g': 'g_{SDSS}', 'sdss:r': 'r_{SDSS}'}
MAG_BINS = [(16, 18), (18, 19.5), (19.5, 20.5), (20.5, 21.5), (21.5, 22.5), (22.5, 23.5)]
DRAWS = 3
MIN_SNR = 5.


def rows_for(plane_bands):
    d = data('stars'); flag = np.load(ROOT/'stars'/'qso_flag.npy'); obs = np.asarray(d['observed'])
    cols = [idx(b) for b in [REF, *plane_bands]]
    v = np.asarray(d['noise']); snr_r = np.where(obs[:, idx(REF)], 1.0857/np.sqrt(v[:, idx(REF)]), 0)
    return np.flatnonzero((np.asarray(d['role']) == 3) & ~flag & (snr_r >= 10) & obs[:, cols].all(axis=1))


def predictive(model, rows, plane_bands, rng):
    """Draws (rows*DRAWS, bands) of the plane bands for each row, conditioned on its Legacy r."""
    d = data('stars'); b = [idx(x) for x in [REF, *plane_bands]]
    mix = model.background; mu, V = mix.means[:, b], mix.covs[:, b][:, :, b]
    y = np.asarray(d['y'][rows])[:, b]; nv = np.asarray(d['noise'][rows])[:, b]
    w = model.spatial_background.evaluate(np.asarray(d['l'])[rows], np.asarray(d['b'])[rows])[0]
    s00 = V[:, 0, 0][None] + nv[:, :1]                                           # (n, K)
    lw = np.log(np.maximum(w, 1e-300)) - .5*np.log(2*np.pi*s00) - .5*(y[:, :1] - mu[None, :, 0])**2/s00
    p = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
    out = np.empty((len(rows), DRAWS, len(plane_bands)))
    for i in range(len(rows)):
        ks = rng.choice(len(p[i]), DRAWS, p=p[i])
        for j, k in enumerate(ks):
            g = V[k, 1:, 0]/s00[i, k]
            m = mu[k, 1:] + g*(y[i, 0] - mu[k, 0]); c = V[k, 1:, 1:] - np.outer(g, V[k, 0, 1:])
            out[i, j] = rng.multivariate_normal(m, c)
    # Noise in flux space with each object's own flux error, then back to luptitude.
    soft = model.transform.softening[b[1:]]; a = 2.5/np.log(10)
    flux = 2*soft*np.sinh((22.5 - out)/a - np.log(soft))
    ferr = np.sqrt(np.asarray(d['variance_dered'][rows])[:, b[1:]])[:, None, :]
    noisy = flux + ferr*rng.standard_normal(flux.shape)
    lup = 22.5 - a*(np.arcsinh(noisy/(2*soft)) + np.log(soft))
    fobs = np.asarray(d['flux_dered'][rows])[:, b[1:]]
    return lup, noisy/ferr, y[:, 1:], fobs/ferr[:, 0, :]


def levels(h, fractions=(.5, .9)):
    f = np.sort(h.ravel())[::-1]; c = np.cumsum(f)/f.sum()
    return sorted(f[np.searchsorted(c, q)] for q in fractions)


def main():
    from qso_pcolor.multisurvey import MultiSurveyModel
    use_paper_style(); rng = np.random.default_rng(19); model = MultiSurveyModel.load(BUNDLE/'model.json'); d = data('stars')
    fig, axes = plt.subplots(len(PLANES), len(MAG_BINS), figsize=(2.35*len(MAG_BINS), 2.35*len(PLANES)))
    report = []
    for pi, (cx, cy, xl, yl) in enumerate(PLANES):
        pb = list(dict.fromkeys([*cx, *cy])); rows = rows_for(pb); r = np.asarray(d['y'][rows])[:, idx(REF)]
        for j, (lo, hi) in enumerate(MAG_BINS):
            ax = axes[pi, j]; sel = rows[(r >= lo) & (r < hi)]
            if len(sel) > 6000:
                sel = np.sort(rng.choice(sel, 6000, replace=False))
            if len(sel) < 30:
                ax.set_visible(False); continue
            draws, snr_draw, yobs, snr_obs = predictive(model, sel, pb, rng)
            keep_d = (snr_obs >= MIN_SNR).all(axis=1)
            keep_m = (snr_draw >= MIN_SNR).all(axis=2)
            if keep_d.sum() < 100:
                ax.set_visible(False); continue
            col = lambda a, pair: a[..., pb.index(pair[0])] - a[..., pb.index(pair[1])]
            xd, yd = col(yobs, cx)[keep_d], col(yobs, cy)[keep_d]
            xm, ym = col(draws, cx)[keep_m], col(draws, cy)[keep_m]
            ex, ey = np.linspace(*xl, 46), np.linspace(*yl, 46)
            hd, _, _ = np.histogram2d(xd, yd, bins=[ex, ey]); hm, _, _ = np.histogram2d(xm, ym, bins=[ex, ey])
            fx, fy = np.linspace(*xl, 91), np.linspace(*yl, 91)
            hf, _, _ = np.histogram2d(xd, yd, bins=[fx, fy])
            ax.pcolormesh(fx, fy, np.log10(hf.T + 1), cmap='Greys', rasterized=True)
            sm = gaussian_filter(np.histogram2d(xm, ym, bins=[fx, fy])[0], 1.)
            lev = levels(sm); xc, yc = .5*(fx[1:] + fx[:-1]), .5*(fy[1:] + fy[:-1])
            ax.contour(xc, yc, sm.T, levels=lev, colors='#c0392b', linewidths=(1.3, .8))
            ix = np.clip(np.searchsorted(fx, xd) - 1, 0, len(xc) - 1); iy = np.clip(np.searchsorted(fy, yd) - 1, 0, len(yc) - 1)
            inside = (xd >= xl[0]) & (xd < xl[1]) & (yd >= yl[0]) & (yd < yl[1]); val = np.where(inside, sm[ix, iy], 0)
            in50, in90 = float(np.mean(val >= lev[1])), float(np.mean(val >= lev[0]))
            empty = float(hm[hd == 0].sum()/max(hm.sum(), 1))
            report.append(dict(plane=[cx, cy], bin=[lo, hi], n_data=int(keep_d.sum()), n_draws=int(keep_m.sum()),
                               inside50=in50, inside90=in90, draws_in_empty=empty))
            ax.set_title(f'{lo}$\\leq r<${hi}  N={keep_d.sum()}\nin 50/90%: {in50:.2f}/{in90:.2f}; empty {empty:.2f}', fontsize=7)
            ax.set_xlim(xl); ax.set_ylim(yl); ax.tick_params(labelsize=7)
            ax.set_xlabel(f'${LABEL[cx[0]]}-{LABEL[cx[1]]}$', fontsize=8)
            if j == 0:
                ax.set_ylabel(f'${LABEL[cy[0]]}-{LABEL[cy[1]]}$', fontsize=8)
    fig.suptitle('Held-out stellar objects (grey, log density) and stellar model (red: 50%, 90% of posterior-predictive draws, same S/N cut)',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .97))
    Path(f'docs/method_unified/{TAG}.json').write_text(json.dumps(dict(definition=__doc__, bundle=str(BUNDLE), panels=report), indent=1))
    print(save_figure(fig, f'method_unified/{TAG}'))
    for e in report:
        print(e['plane'][0][0].split(':')[-1], e['bin'], e['n_data'], round(e['inside50'], 2), round(e['inside90'], 2), round(e['draws_in_empty'], 3))


if __name__ == '__main__':
    main()
