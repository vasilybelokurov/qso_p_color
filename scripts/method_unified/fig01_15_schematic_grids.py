#!/usr/bin/env python
"""F1: schematic of the four hypotheses. F15: artificial colour-grid stress maps.

F1 is illustrative only: synthetic two-dimensional Gaussians, not fitted models.
F15: 15 x 15 grids in Legacy (g-r, r-z) from -6 to 8 mag at fixed r (18.5, 21), placed at
(l, b) = (180, 45) deg with 0.03 mag luptitude errors. Colour: total QSO probability
p_quasar for scorable points; grey: not scorable (outside both models, or rejected by the
support cut, threshold 2/256, recalibrated under the faint limit). Black outline: the original low-density mask
in which the 30 September candidate placed 72/79 high-QSO points.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from common import SERIES, save_figure, use_paper_style

OLD = Path(json.loads(Path('docs/FULL_SAMPLE_RELEASE_2026-09-30.json').read_text())['cache'])
RUNS = {'magnitude-independent': Path('models/multisurvey_psf/work/method_unified/grids_note/independent'),
        'magnitude-dependent': Path('models/multisurvey_psf/work/method_unified/grids_note/dependent')}   # grid_scores.py
THRESHOLD = 0.0078125


def f1():
    fig, ax = plt.subplots(figsize=(6.4, 4.6)); rng = np.random.default_rng(1)
    t = np.linspace(-.3, 1.8, 400); ax.plot(t, .55*t + .25*t**2 - .1, color=SERIES['background'], lw=10, alpha=.25, solid_capstyle='round')
    ax.text(1.55, 1.25, 'background $B$\n(stars, other PSF sources;\nsky-dependent)', color=SERIES['background'], fontsize=8.5)
    for z, (x, y) in zip((0.5, 1.0, 1.5, 2.2, 2.8, 3.4), ((.35, .55), (.15, .35), (.05, .1), (.2, .45), (.5, .25), (.9, .05))):
        ax.add_patch(Ellipse((x, y), .32, .2, angle=25, fc=SERIES['field_q'], alpha=.18, ec=SERIES['field_q']))
        ax.text(x, y - .17, f'$z={z}$', fontsize=7, color=SERIES['field_q'], ha='center')
    ax.add_patch(Ellipse((.05, .1), .34, .22, angle=25, fc=SERIES['same_z'], alpha=.45, ec=SERIES['same_z'], lw=1.5))
    ax.text(-.55, -.25, 'same-redshift QSO $Q(z_0)$\n(here $z_0=1.5$)', color=SERIES['same_z'], fontsize=8.5)
    ax.text(.62, -.42, 'QSOs at other redshifts $Q(z\\neq z_0)$', color=SERIES['field_q'], fontsize=8.5)
    for s in (1.4, 2.2, 3.0):
        ax.add_patch(Ellipse((.6, .5), 2.2*s, 1.6*s, angle=30, fc='none', ec=SERIES_NEUTRAL, ls=':', lw=1))
    ax.text(-.9, 1.75, 'catch-all $U$: broad Student-$t$\n(anything none of the above explains)', color=SERIES_NEUTRAL, fontsize=8.5)
    for (x, y), lab in (((.08, .12), 'A'), ((1.25, .98), 'B'), ((-.6, 1.4), 'C')):
        ax.plot(x, y, '*', ms=12, color='k'); ax.text(x - .12, y + .06, lab, fontsize=10, weight='bold')
    ax.set(xlim=(-1, 2.4), ylim=(-.6, 2.1), xlabel='colour 1 (e.g. $g-r$)', ylabel='colour 2 (e.g. $r-z$)')
    ax.set_title('Four hypotheses for a companion of a QSO at $z_0$ (schematic)', fontsize=9.5)
    fig.tight_layout(); return fig


SERIES_NEUTRAL = '#5c5c57'


def low_mask(h, mag):
    m = None
    for lab in ('baseline', 'candidate'):
        s = np.load(OLD/f'grid_{h}_{mag}_{lab}.npz'); q = np.logaddexp(s['log_lambda_sameq'], s['log_lambda_fieldq']); b = s['log_lambda_bkg']
        k = (q < np.nanmax(q) + np.log(.01)) & (b < np.nanmax(b) + np.log(.01)); m = k if m is None else m & k
    return m


def f15():
    axes_vals = np.linspace(-6, 8, 15); models = ['previous (28 Sep)'] + list(RUNS)
    fig, axes = plt.subplots(len(models), 4, figsize=(11, 8.4), sharex=True, sharey=True)
    for j, (h, mag) in enumerate(((h, m) for h in ('south', 'north') for m in (18.5, 21.0))):
        mask = low_mask(h, mag).reshape(15, 15)
        for i, name in enumerate(models):
            if i == 0:
                r = np.load(OLD/f'grid_{h}_{mag}_baseline.npz'); ok = r['eligible']
            else:
                raw = np.load(RUNS[name]/f'grid_{h}_{mag}_raw.npz'); uni = np.load(RUNS[name]/f'grid_{h}_{mag}_unified.npz')
                r = raw; ok = raw['eligible'] & ~(np.isfinite(uni['support']) & (uni['support'] < THRESHOLD))
            p = np.where(ok, r['p_quasar'], np.nan).reshape(15, 15); ax = axes[i, j]
            ax.pcolormesh(axes_vals, axes_vals, np.where(np.isnan(p), 1, np.nan), cmap='Greys', vmin=0, vmax=4, shading='nearest')
            im = ax.pcolormesh(axes_vals, axes_vals, p, cmap='viridis', vmin=0, vmax=1, shading='nearest')
            ax.contour(axes_vals, axes_vals, mask.astype(float), levels=[.5], colors='k', linewidths=.8)
            n = int(np.nansum((p > .5) & mask))
            ax.set_title(f'{name}\n{h}, $r$={mag}: {n} in mask', fontsize=7.5)
            if j == 0: ax.set_ylabel('$r-z$ [mag]')
            if i == len(models) - 1: ax.set_xlabel('$g-r$ [mag]')
    fig.colorbar(im, ax=axes, shrink=.6, label='$p_\\mathrm{quasar}$ (grey: not scorable)')
    return fig


def main():
    use_paper_style()
    print(save_figure(f1(), 'method_unified/F01_schematic'))
    print(save_figure(f15(), 'method_unified/F15_colour_grids'))


if __name__ == '__main__':
    main()
