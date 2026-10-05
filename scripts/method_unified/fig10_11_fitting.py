#!/usr/bin/env python
"""F10: the final fits. F11: support-cut threshold sweep.

F10a  Update at which each quasar-slice fit kept its checkpoint, against slice redshift, both models.
F10b  Held-out gain of the kept checkpoint over the warm start.
F10c  Held-out density gain over the warm start against update, for the eight background bins.
F11   Support-cut sweep for the note's bundles: held-out AUC, test-QSO retention, background
      p_Q>0.5 incidence and artificial-grid stray points against the support threshold (scores of
      fig12_14 / grid_scores.py, no rescoring); dotted: the recalibrated threshold.
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


STELLAR = Path('models/multisurvey_psf/work/stellar_binned/20261003')


def f10():
    """Final fits only: where each quasar slice stopped and what it gained; held-out traces of the stellar bins."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for name, run in RUNS.items():
        rows = []
        tasks = {t['name']: t for t in json.loads((run/'tasks.json').read_text())}
        for p in glob.glob(str(run/'fits'/'qso_*.json')):
            f = json.loads(Path(p).read_text()); v = [e['mean'] for e in f['stopping_evaluations']]
            rows.append((tasks[f['task']]['z'], f['n_iter'], f['selected_iteration'], max(v) - v[0]))
        z, n, sel, gain = np.array(sorted(rows)).T
        ax[0].plot(z, sel, 'o-', ms=3.5, lw=1, color=COLOURS[name], label=LABELS[name])
        ax[1].plot(z, gain, 'o-', ms=3.5, lw=1, color=COLOURS[name], label=LABELS[name])
    ax[0].set(xlabel='slice redshift', ylabel='kept checkpoint [updates]', title='quasar slices: where each fit stopped')
    ax[1].set(xlabel='slice redshift', ylabel='held-out gain over warm start [nat]', title='quasar slices: gain of the kept checkpoint')
    ax[0].legend(frameon=False, fontsize=7.5)
    for j in range(8):
        r = json.loads((STELLAR/f'binned_{j}.json').read_text()); it = [e['iteration'] for e in r['trace']]; v = np.array([e['heldout'] for e in r['trace']])
        ax[2].plot(it, v - v[0], 'o-', ms=3, lw=1, color=plt.cm.viridis(j/7), label=f"{r['bin'][0]:g}--{r['bin'][1]:g}")
        ax[2].plot(r['best_iteration'], v[it.index(r['best_iteration'])] - v[0], 'k*', ms=6)
    ax[2].set(xlabel='update', ylabel='held-out gain over warm start [nat]', title='background bins (Legacy $r$); $\\star$ kept')
    ax[2].legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout(); return fig


THRESHOLDS = [0., 1/256, 2/256, 4/256, 8/256, 16/256]
CHOSEN = 2/256


def f11():
    """Support-cut sweep on the note's held-out main panels and artificial grids (no rescoring)."""
    import fig12_14_performance as P
    from fig01_15_schematic_grids import RUNS as GRIDS, low_mask
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2)); x = np.arange(len(THRESHOLDS))
    ticks = ['no cut' if t == 0 else f'{round(t*256)}/256' for t in THRESHOLDS]; report = {}
    for name, gname in (('independent', 'magnitude-independent'), ('dependent', 'magnitude-dependent')):
        q, b = P.load(name, 'qso', 'main'), P.load(name, 'stars', 'main')
        pre = lambda d: d['eligible'] | (d['status'] == 'qso_support_rejected')
        out = []
        for t in THRESHOLDS:
            cut = lambda d: pre(d) & ~(np.isfinite(d['support']) & (d['support'] < t))
            qq, bb = dict(q, eligible=cut(q)), dict(b, eligible=cut(b))
            stray = 0
            for h in ('south', 'north'):
                for mag in (18.5, 21.0):
                    raw = np.load(GRIDS[gname]/f'grid_{h}_{mag}_raw.npz'); uni = np.load(GRIDS[gname]/f'grid_{h}_{mag}_unified.npz')
                    ok = (uni['eligible'] | (uni['status'] == 'qso_support_rejected')) & ~(np.isfinite(uni['support']) & (uni['support'] < t))
                    stray += int((ok & (raw['p_quasar'] > .5) & low_mask(h, mag)).sum())
            # Rank by p_quasar: the scorer blanks log R for support-rejected rows, so it cannot rank them
            # at looser thresholds; unranked rows go to the bottom as in fig12_14.
            rq = np.where(qq['eligible'], np.nan_to_num(q['p_quasar']), -1.); rb = np.where(bb['eligible'], np.nan_to_num(b['p_quasar']), -1.)
            from scipy.stats import mannwhitneyu
            out.append(dict(threshold=t, auc=float(mannwhitneyu(rq, rb).statistic/(len(rq)*len(rb))), retention=float(np.mean(~(np.isfinite(q['support']) & (q['support'] < t)))),
                            incidence=float((bb['eligible'] & (bb['p_quasar'] > .5)).mean()), grid_stray=stray))
        report[name] = out
        for k, key in enumerate(('auc', 'retention', 'incidence', 'grid_stray')):
            ax[k].plot(x, [o[key] for o in out], 'o-', color=COLOURS[name], label=LABELS[name])
    titles = ('AUC of $p_Q$ (held-out main panel)', 'test-QSO retention', 'background with $p_Q>0.5$', 'stray points on artificial grids')
    for a, t in zip(ax, titles):
        a.set_xticks(x); a.set_xticklabels(ticks, fontsize=7); a.set_xlabel('support-cut threshold (percentile)'); a.set_title(t, fontsize=9)
        a.axvline(THRESHOLDS.index(CHOSEN), color='0.6', lw=.8, ls=':')
    ax[0].legend(frameon=False, fontsize=7); fig.tight_layout()
    Path('docs/method_unified/support_sweep_note.json').write_text(json.dumps(report, indent=1, default=float))
    return fig


def main():
    use_paper_style()
    print(save_figure(f10(), 'method_unified/F10_fitting'))
    print(save_figure(f11(), 'method_unified/F11_support_cut'))


if __name__ == '__main__':
    main()
