#!/usr/bin/env python
"""Classification performance against photometric S/N, to choose a training/scoring S/N limit.

Input: the held-out performance scores of both promoted models (score_performance.py, main
panel: clean role-3 QSOs and background objects). S/N = 1.0857/sigma with sigma the luptitude error
(prepared noise, extinction corrected) of either the band the scorer used as reference ('reference';
SDSS r has priority, so this tracks survey depth as much as brightness) or Legacy r ('legacy_r', South
if observed else North; the magnitude the binned background model uses).
Per S/N bin and model:
  AUC         P(test QSO ranks above test background object on log R); unranked at the bottom.
  incidence   fraction of background objects ranked with p_quasar > 0.5.
  recall      fraction of QSOs ranked with p_quasar > 0.5.
  n_qso/n_bkg objects in the bin (note: the background panel is not a random sample in S/N).
Also cumulative versions for objects with S/N >= threshold.

Usage: python scripts/method_unified/snr_cut_check.py   (writes docs/method_unified/snr_cut_check.json,
plots/method_unified/F17_snr_cut_check.png)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import COLOURS, LABELS, data, save_figure, use_paper_style
import fig12_14_performance as _P
from fig12_14_performance import MODELS, auc, load, sub
import glob

OUT = Path('models/multisurvey_psf/work/method_unified/performance')   # scores WITHOUT the faint limit (promoted bundles)
_P.OUT = OUT
EDGES = [5, 7, 10, 15, 20, 30, 50, 100, np.inf]
THRESHOLDS = [5, 7, 10, 15, 20]


def references(model, kind):
    """Reference-band names in the same concatenation order as fig12_14_performance.load."""
    out = []
    for p in sorted(glob.glob(str(OUT/f'{model}_{kind}_main_*_[0-9]*.npz'))):
        if p.endswith('.rows.npz'):
            continue
        tag = Path(p).name[len(f'{model}_{kind}_main_'):-len('_000000.npz')]
        if '_z' in tag or any(tag.endswith(m) for m in ('grz', 'all', 'sdss', 'ps1')):
            continue
        out.append(np.load(p)['reference'])
    return np.concatenate(out)


def legacy_r(kind, rows):
    labs = json.loads(Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059/layout.json').read_text())['native_labels']
    obs = np.asarray(data(kind)['observed'][rows]); s, n = labs.index('decals_dr9_south:r'), labs.index('decals_dr9_north:r')
    return np.where(obs[:, s], 'decals_dr9_south:r', np.where(obs[:, n], 'decals_dr9_north:r', ''))


def snr(kind, rows, ref):
    ok = ref != ''; out = np.zeros(len(rows))
    if not ok.all():
        out[~ok] = 0.; out[ok] = snr(kind, rows[ok], ref[ok]); return out
    labs = json.loads(Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059/layout.json').read_text())['native_labels']
    j = np.array([labs.index(r) for r in ref]); v = np.asarray(data(kind)['noise'][rows])[np.arange(len(rows)), j]
    return 1.0857/np.sqrt(v)


def stats(q, b):
    return dict(n_qso=int(len(q['rows'])), n_bkg=int(len(b['rows'])), auc=auc(q, b),
                incidence=float((b['eligible'] & (b['p_quasar'] > .5)).mean()) if len(b['rows']) else np.nan,
                recall=float((q['eligible'] & (q['p_quasar'] > .5)).mean()) if len(q['rows']) else np.nan)


def panel(ax, which, report):
    """One row of the figure: S/N measured in the reference band or in Legacy r."""
    for model in MODELS:
        q, b = load(model, 'qso', 'main'), load(model, 'stars', 'main')
        rq = references(model, 'qso') if which == 'reference' else legacy_r('qso', q['rows'])
        rb = references(model, 'stars') if which == 'reference' else legacy_r('stars', b['rows'])
        sq, sb = snr('qso', q['rows'], rq), snr('stars', b['rows'], rb); key = f'{model}_{which}'
        per = [stats(sub(q, (sq >= lo) & (sq < hi)), sub(b, (sb >= lo) & (sb < hi))) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        cum = {t: stats(sub(q, sq >= t), sub(b, sb >= t)) for t in THRESHOLDS}
        report['models'][key] = dict(per_bin=per, cumulative=cum)
        x = np.sqrt(np.array(EDGES[:-2])*np.array(EDGES[1:-1])).tolist() + [150]
        for k, name in enumerate(('auc', 'incidence', 'recall')):
            ax[k].plot(x, [p[name] for p in per], 'o-', color=COLOURS[model], label=LABELS[model], ms=4)
        print(key)
        for (lo, hi), p in zip(zip(EDGES[:-1], EDGES[1:]), per):
            print(f'  S/N {lo:>4}-{hi:<4}: nQ {p["n_qso"]:6d} nB {p["n_bkg"]:6d}  AUC {p["auc"]:.4f}  incidence {p["incidence"]:.4f}  recall {p["recall"]:.3f}')
        for t, p in cum.items():
            print(f'  S/N >= {t:>2}: nQ {p["n_qso"]:6d} nB {p["n_bkg"]:6d}  AUC {p["auc"]:.4f}  incidence {p["incidence"]:.4f}  recall {p["recall"]:.3f}')
    for k, (yl, t) in enumerate((('AUC', 'ranking quality'), ('fraction with $p_Q>0.5$', 'background objects'), ('fraction with $p_Q>0.5$', 'QSOs'))):
        ax[k].set(xscale='log', xlabel=('reference-band' if which == 'reference' else 'Legacy $r$') + ' S/N', ylabel=yl, title=t)
        for e in (5, 10):
            ax[k].axvline(e, color='0.7', lw=.8, ls=':')
    ax[0].legend(frameon=False, fontsize=7)


def main():
    use_paper_style(); report = dict(definition=__doc__, edges=[e if np.isfinite(e) else None for e in EDGES], models={})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    panel(axes, 'legacy_r', report)
    fig.tight_layout()
    Path('docs/method_unified/snr_cut_check.json').write_text(json.dumps(report, indent=1, default=float))
    print(save_figure(fig, 'method_unified/F17_snr_cut_check'))


if __name__ == '__main__':
    main()
