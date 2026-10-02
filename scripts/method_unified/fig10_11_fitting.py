#!/usr/bin/env python
"""F10: fitting behaviour. F11: support-cut threshold sweep.

F10a  Training objective per object minus its starting value, z=1.75 control slice:
      additive update (plain log likelihood) and MAP update (log posterior).
F10b  Held-out conditional log density per object for six test slices, both updates
      (MAP-update test run, identical rows and warm starts).
F10c  Updates at which each production fit stopped, against slice redshift, both refits.
F10d  Held-out gain of the kept checkpoint over the warm start, both refits.
F11   AUC minus the previous model's AUC (worst of eight panels), test-QSO retention and
      artificial-grid stray points against the support-cut calibration retention.
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import COLOURS, LABELS, save_figure, use_paper_style

TEST = Path('models/multisurvey_psf/work/map_update_test/20260930/c108415a6a7a695f')
RUNS = {'independent': Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059'),
        'dependent': Path('models/multisurvey_psf/work/unified_full/20261001/5f4002492dbb849c')}
UPD = {'additive': '#5c5c57', 'map': '#2a78d6'}


def f10():
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.8))
    for mode, lab in (('additive', 'additive update (log likelihood)'), ('map', 'MAP update (log posterior)')):
        h = np.array(json.loads((TEST/f'qso_16_{mode}'/'state.json').read_text())['history'])
        ax[0, 0].plot(np.arange(len(h)), h - h[0], color=UPD[mode], lw=1.5, label=lab)
        dec = np.flatnonzero(np.diff(h) < 0)
        if len(dec):
            ax[0, 0].plot(dec + 1, (h - h[0])[dec + 1], 'v', ms=3, color=UPD[mode], label=f'{len(dec)} decreasing steps')
    ax[0, 0].set(xlabel='update', ylabel='objective per object minus start [nat]', title='z = 1.75 control slice: training objective')
    ax[0, 0].legend(frameon=False, fontsize=7.5)
    zs = {'qso_13': 1.45, 'qso_15': 1.65, 'qso_16': 1.75, 'qso_22': 2.35, 'qso_31': 3.25, 'qso_39': 4.05}
    for name, z in zs.items():
        for mode, ls in (('additive', '--'), ('map', '-')):
            pts = sorted((int(p[-7:-4]), float(np.load(p)['values'].mean())) for p in glob.glob(str(TEST/f'{name}_{mode}'/'prediction_*.npz')))
            it, v = np.array(pts).T; line = ax[0, 1].plot(it, v, ls, lw=1.3, color=plt.cm.viridis((z - 1.3)/3), label=f'z = {z}' if mode == 'map' else None)
    ax[0, 1].plot([], [], 'k-', lw=1.3, label='MAP'); ax[0, 1].plot([], [], 'k--', lw=1.3, label='additive')
    ax[0, 1].set(xlabel='update', ylabel='held-out log density per QSO [nat]', title='held-out density, six test slices')
    ax[0, 1].legend(frameon=False, fontsize=7, ncol=2)
    for name, run in RUNS.items():
        rows = []
        tasks = {t['name']: t for t in json.loads((run/'tasks.json').read_text())}
        for p in glob.glob(str(run/'fits'/'qso_*.json')):
            f = json.loads(Path(p).read_text()); v = [e['mean'] for e in f['stopping_evaluations']]
            rows.append((tasks[f['task']]['z'], f['n_iter'], f['selected_iteration'], max(v) - v[0]))
        z, n, sel, gain = np.array(sorted(rows)).T
        ax[1, 0].plot(z, sel, 'o-', ms=3.5, lw=1, color=COLOURS[name], label=LABELS[name])
        ax[1, 1].plot(z, gain, 'o-', ms=3.5, lw=1, color=COLOURS[name], label=LABELS[name])
    ax[1, 0].set(xlabel='slice redshift', ylabel='kept checkpoint [updates]', title='where each production fit stopped')
    ax[1, 1].set(xlabel='slice redshift', ylabel='held-out gain over warm start [nat]', title='gain of the kept checkpoint')
    ax[1, 0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); return fig


def f11():
    r = json.loads(Path('docs/SUPPORT_CUT_SWEEP_2026-10-01.json').read_text())
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for key, name in (('baseline_corrected_mag_independent', 'independent'), ('corrected_mag_dependent', 'dependent')):
        sw = r[key]['sweep']; x = np.arange(len(sw)); ticks = [f"{s['target_retention']:.3f}" if s['target_retention'] < 1 else 'no cut' for s in sw]
        ax[0].plot(x, [s['min_auc_minus_active'] for s in sw], 'o-', color=COLOURS[name], label=LABELS[name])
        ax[1].plot(x, [s['min_qso_retention'] for s in sw], 'o-', color=COLOURS[name])
        ax[2].plot(x, [sum(s['grid'].values()) for s in sw], 'o-', color=COLOURS[name])
    ax[0].axhline(-.005, color='0.4', lw=.8, ls=':'); ax[0].text(0, -.0055, 'release tolerance', fontsize=7, va='top')
    titles = ('worst AUC minus previous model', 'lowest test-QSO retention', 'stray points on artificial grids')
    for a, t in zip(ax, titles):
        a.set_xticks(x); a.set_xticklabels(ticks); a.set_xlabel('support-cut calibration retention'); a.set_title(t, fontsize=9)
    ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); return fig


def main():
    use_paper_style()
    print(save_figure(f10(), 'method_unified/F10_fitting'))
    print(save_figure(f11(), 'method_unified/F11_support_cut'))


if __name__ == '__main__':
    main()
