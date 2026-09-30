#!/usr/bin/env python
"""Plot saved prototype results; no fitting or rescoring."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qso_pcolor.plotting import save_figure


def main():
    report = json.loads(Path('docs/POOLED_OPTICAL_PROTOTYPE_2026-09-30.json').read_text())
    root = Path(report['root']); figures = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for hemi, color in [('north', '#356ab6'), ('south', '#cd632a')]:
        rows = report['predictive']['qso_'+hemi]['redshift']
        use = [r for r in rows if r['n'] >= report['validation_config']['minimum_summary_rows']]
        axes[0].plot([r['z'] for r in use], [r['mean'] for r in use], marker='.', label=hemi, color=color)
        rows = report['predictive']['stars_'+hemi]['magnitude']
        use = [r for r in rows if r['n'] >= report['validation_config']['minimum_summary_rows']]
        axes[1].plot([(r['lo']+r['hi'])/2 for r in use], [r['mean'] for r in use], marker='o', label=hemi, color=color)
    for ax in axes:
        ax.axhline(0, color='black', lw=.8); ax.axhline(-report['config']['maximum_mean_log_density_loss'], color='grey', ls='--', lw=.8)
        ax.set_ylabel('Mean log-density change: pooled − separate [nats]', labelpad=10); ax.legend(); ax.grid(alpha=.15)
    axes[0].set(xlabel='Spectroscopic redshift', title='Reserved QSOs: native conditional density')
    axes[1].set(xlabel='Measured reference asinh magnitude', title='Reserved stellar background: spatial density model')
    fig.suptitle('Does pooling preserve predictions for real objects? Positive values favour pooling')
    figures.append(str(save_figure(fig, 'pooled_optical_prototype/predictive_comparison')))
    plt.close(fig)

    old = json.loads(Path('docs/FULL_SAMPLE_RELEASE_2026-09-30.json').read_text()); grid = old['config']['grid']
    mag = grid['reference_magnitudes'][0]
    paths = [Path(old['cache'])/f'grid_north_{mag}_candidate.npz', root/f'grid_separate_north_{mag}.npz', root/f'grid_pooled_north_{mag}.npz']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
    nodes = np.linspace(*grid['grid_colour_range'], grid['grid_size']); step = nodes[1]-nodes[0]
    cmap = plt.get_cmap('magma').copy(); cmap.set_bad('#d9dde0')
    for ax, p, title in zip(axes, paths, ['Original full-data candidate', 'New separate optical fits', 'New pooled optical fit']):
        d = np.load(p); values = np.ma.array(d['p_quasar'], mask=~d['eligible']).reshape(grid['grid_size'], grid['grid_size'])
        im = ax.imshow(values, origin='lower', extent=[nodes[0]-step/2, nodes[-1]+step/2]*2, vmin=0, vmax=1, cmap=cmap)
        ax.set(title=title, xlabel='g − r [asinh mag]', ylabel='r − z [asinh mag]')
    fig.colorbar(im, ax=axes, label='Total-QSO diagnostic score; uncalibrated', shrink=.8)
    fig.suptitle('Northern bright colour grid; grey = excluded by existing guards\nIdentical native priors and catch-all; only photometric distributions and their spatial weights change')
    figures.append(str(save_figure(fig, 'pooled_optical_prototype/northern_tail_comparison')))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout='constrained')
    for ax, mode in zip(axes, ['separate', 'pooled']):
        n, s = (np.load(root/f'paired_{mode}_{h}.npz') for h in ('north', 'south'))
        use = n['eligible'] & s['eligible']
        ax.scatter(s['p_quasar'][use], n['p_quasar'][use], s=9, alpha=.35, rasterized=True)
        ax.plot([0, 1], [0, 1], color='black', lw=.8)
        value = report['scores']['paired'][mode]['common_eligible_absolute_score_difference']['mean']
        ax.set(xlabel='Score from southern measurement', ylabel='Score from northern measurement',
            title=f'{mode.capitalize()}: mean |difference| = {value:.3f}', xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    fig.suptitle('Same reserved objects measured in both systems\nScore differences include native errors, population priors and possible epoch differences')
    figures.append(str(save_figure(fig, 'pooled_optical_prototype/paired_scores')))
    plt.close(fig)
    print('\n'.join(figures))


if __name__ == '__main__': main()
