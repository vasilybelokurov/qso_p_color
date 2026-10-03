#!/usr/bin/env python
"""Stellar panels of Fig. 9 (F7) for the three background designs of test_background_designs.py.

Data: role-3 background objects with S/N > 5 in the plane's bands (as F7), grey histogram.
Models (all-sky weights), conditioned on the bin's central Legacy South r: current joint
background, binned XD (bin containing that r), tied colour shapes. Contours enclose 50% and
90% of the noise-convolved model (median noise of the bin's objects); titles give the data
fraction inside them (ideal 0.50/0.90) and the fraction of model draws landing in colour cells
(45x45 grid) that contain no data.

Usage: python scripts/method_unified/fig07d_background_designs.py [designs_dir [output_tag]]
(default designs test1, tag F07d_background_designs)
"""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from common import COLOURS, colour_matrix, condition_on, contour_levels, density_2d, project, save_figure, use_paper_style
import fig07_09_colour_colour as F
from test_background_designs import load_models

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'models/multisurvey_psf/work/background_designs/test1')
TAG = sys.argv[2] if len(sys.argv) > 2 else 'F07d_background_designs'
EDGE = dict(current='#5c5c57', binned=COLOURS['background'], tied='#b5562b')


def at(model, r0):
    for sel, nat in model(np.array([r0])):
        if sel[0]:
            return nat
    raise ValueError('no model for r0')


def main():
    use_paper_style(); rng = np.random.default_rng(5); mods, _ = load_models(OUT); report = []
    names = list(mods); nrow = 2*len(names)
    fig, axes = plt.subplots(nrow, len(F.MAG_BINS), figsize=(11, 2.6*nrow))
    for pi, plane in enumerate(('P1', 'P2')):
        b = F.selection('stars', plane); r = F.r_of('stars', b); a = colour_matrix(F.PLANES[plane][0]); (xl, yl) = F.PLANES[plane][2]
        for j, (lo, hi) in enumerate(F.MAG_BINS):
            br = b[(r >= lo) & (r < hi)]; c = F.plane_values('stars', br, plane); noise = F.plane_noise('stars', br, plane)
            hd, xe, ye = np.histogram2d(c[:, 0], c[:, 1], bins=[np.linspace(*xl, 46), np.linspace(*yl, 46)])
            h, xe2, ye2 = np.histogram2d(c[:, 0], c[:, 1], bins=[np.linspace(*xl, 90), np.linspace(*yl, 90)])
            for mi, name in enumerate(names):
                ax = axes[len(names)*pi + mi, j]; r0 = .5*(lo + hi)
                m2 = project(condition_on(at(mods[name], r0), F.S+'r', r0), a)
                ax.pcolormesh(xe2, ye2, np.log10(h.T + 1), cmap='Greys', rasterized=True)
                xx, yy = np.meshgrid(np.linspace(*xl, 160), np.linspace(*yl, 160)); z = density_2d(m2, xx, yy, noise); lev = contour_levels(z)
                ax.contour(xx, yy, z, levels=lev, colors=EDGE[name], linewidths=(1.4, .9))
                dz = density_2d(m2, c[:, 0][:, None], c[:, 1][:, None], noise).ravel()
                k = rng.choice(len(m2.weights), 20000, p=m2.weights/m2.weights.sum())
                s = np.array([rng.multivariate_normal(m2.means[i], m2.covs[i] + noise) for i in k])
                hs, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[xe, ye]); empty = hs[hd == 0].sum()/max(hs.sum(), 1)
                cov50, cov90 = (dz >= lev[1]).mean(), (dz >= lev[0]).mean()
                report.append(dict(plane=plane, bin=[lo, hi], model=name, n=int(len(br)), inside50=float(cov50),
                                   inside90=float(cov90), draws_in_empty=float(empty)))
                ax.set_title(f'{name}, {lo} $\\leq r <$ {hi}\nin 50%/90%: {cov50:.2f}/{cov90:.2f}; draws in empty: {empty:.2f}', fontsize=7.5)
                ax.set_xlim(xl); ax.set_ylim(yl)
                if j == 0:
                    ax.set_ylabel(F.AXIS[plane][1] + f'\n({name})')
                ax.set_xlabel(F.AXIS[plane][0])
    fig.suptitle('Background objects (grey) and background models: ' + ', '.join(names) + '; all-sky weights', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, .985))
    Path(f'docs/method_unified/{TAG}.json' if TAG != 'F07d_background_designs' else 'docs/method_unified/background_designs_test.json').write_text(json.dumps(dict(test=str(OUT), panels=report,
        score=json.loads((OUT/'score.json').read_text()) if (OUT/'score.json').exists() else None), indent=1))
    print(save_figure(fig, f'method_unified/{TAG}'))
    for e in report:
        print(e['plane'], e['bin'], e['model'], round(e['inside50'], 2), round(e['inside90'], 2), round(e['draws_in_empty'], 3))


if __name__ == '__main__':
    main()
