#!/usr/bin/env python
"""Stellar panels of Fig. 9 (F7) redrawn: current background against the decoupled test model.

Data: role-3 background objects (as F7). Models, both with all-sky component weights and
conditioned on the bin's central Legacy South r: the current joint background, and the light-test
background with fixed colour shapes and magnitude-dependent amplitudes
(test_background_decoupled.py). Contours enclose 50% and 90%; titles give the data fraction inside
them (ideal 0.50/0.90) and the fraction of model draws landing in colour cells with no data.
"""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from common import COLOURS, colour_matrix, condition_on, contour_levels, data, density_2d, project, save_figure, use_paper_style
import fig07_09_colour_colour as F
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.projected_xd import native_mixture

TEST = Path(sys.argv[1] if len(sys.argv) > 1 else 'models/multisurvey_psf/work/background_decoupled/test1')/'result.json'


def models():
    from common import model
    cur = model('independent').background
    cur = GaussianMixture(model('independent').spatial_background.global_weights, cur.means, cur.covs)
    r = json.loads(TEST.read_text()); op = r['operator']
    new = native_mixture(GaussianMixture.from_dict(r['mixture_u']), np.array(op['matrix']), np.array(op['offset']), np.array(op['covariance']))
    return {'current joint background': cur, 'decoupled test background': new}, r


def main():
    use_paper_style(); rng = np.random.default_rng(5); mods, res = models(); report = []
    fig, axes = plt.subplots(4, len(F.MAG_BINS), figsize=(11, 10.5))
    for pi, plane in enumerate(('P1', 'P2')):
        b = F.selection('stars', plane); r = F.r_of('stars', b); a = colour_matrix(F.PLANES[plane][0]); (xl, yl) = F.PLANES[plane][2]
        for j, (lo, hi) in enumerate(F.MAG_BINS):
            br = b[(r >= lo) & (r < hi)]; c = F.plane_values('stars', br, plane); noise = F.plane_noise('stars', br, plane)
            hd, xe, ye = np.histogram2d(c[:, 0], c[:, 1], bins=[np.linspace(*xl, 46), np.linspace(*yl, 46)])
            for mi, (name, mix) in enumerate(mods.items()):
                ax = axes[2*pi + mi, j]; m2 = project(condition_on(mix, F.S+'r', .5*(lo + hi)), a)
                h, xe2, ye2 = np.histogram2d(c[:, 0], c[:, 1], bins=[np.linspace(*xl, 90), np.linspace(*yl, 90)])
                ax.pcolormesh(xe2, ye2, np.log10(h.T + 1), cmap='Greys', rasterized=True)
                xx, yy = np.meshgrid(np.linspace(*xl, 160), np.linspace(*yl, 160)); z = density_2d(m2, xx, yy, noise); lev = contour_levels(z)
                ax.contour(xx, yy, z, levels=lev, colors=COLOURS['background'] if mi else '#5c5c57', linewidths=(1.4, .9))
                dz = density_2d(m2, c[:, 0][:, None], c[:, 1][:, None], noise).ravel()
                k = rng.choice(len(m2.weights), 20000, p=m2.weights)
                s = np.array([rng.multivariate_normal(m2.means[i], m2.covs[i] + noise) for i in k])
                hs, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[xe, ye]); empty = hs[hd == 0].sum()/max(hs.sum(), 1)
                cov50, cov90 = (dz >= lev[1]).mean(), (dz >= lev[0]).mean()
                report.append(dict(plane=plane, bin=[lo, hi], model=name, inside50=float(cov50), inside90=float(cov90), draws_in_empty=float(empty)))
                ax.set_title(f'{name.split()[0]}, {lo} $\\leq r <$ {hi}\nin 50%/90%: {cov50:.2f}/{cov90:.2f}; draws in empty: {empty:.2f}', fontsize=7.5)
                ax.set_xlim(xl); ax.set_ylim(yl)
                if j == 0: ax.set_ylabel(F.AXIS[plane][1] + f'\n({name.split()[0]} model)')
                ax.set_xlabel(F.AXIS[plane][0])
    fig.suptitle('Background objects: current joint model (grey) vs fixed-colour-shape model (aqua); all-sky weights', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, .97))
    Path('docs/method_unified/background_decoupled_test.json').write_text(json.dumps(dict(test=str(TEST), heldout_current=res['current_heldout'],
        trace=res['trace'], rows=res['rows'], panels=report), indent=1))
    print(save_figure(fig, 'method_unified/F07c_background_decoupled'))
    for e in report: print(e['plane'], e['bin'], e['model'][:8], round(e['inside50'], 2), round(e['inside90'], 2), round(e['draws_in_empty'], 3))


if __name__ == '__main__':
    main()
