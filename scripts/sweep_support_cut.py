"""Support-cut threshold sweep from cached release scores (no rescoring)."""
import json, glob, sys
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

def auc(q, s):
    return float(mannwhitneyu(q, s).statistic/(len(q)*len(s)))

def ranks(raw, unified, thr):
    r = np.where(raw['eligible'], raw['log_r_per_unit_z'], -np.inf)
    rej = np.isfinite(unified['support']) & (unified['support'] < thr)
    return np.where(rej, -np.inf, r), rej

OLD = Path(json.load(open('docs/FULL_SAMPLE_RELEASE_2026-09-30.json'))['cache'])
runs = {'baseline_corrected_mag_independent': 'models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059',
        'sep30_uncorrected_mag_dependent': 'models/multisurvey_psf/work/unified_full/20260930/a637f155f7f48d00'}
out = {}
for name, root in runs.items():
    root = Path(root); rel = root/'release'
    vals = np.concatenate([np.load(p)['percentile'] for p in glob.glob(str(root/'support_calibration_*.npz'))]); vals = vals[np.isfinite(vals)]
    current = json.load(open(root/'bundle'/'support.json'))['threshold']
    rows = []
    for target in (0.98, 0.99, 0.995, 1.0):
        thr = 0. if target >= 1 else float(np.sort(vals)[int(np.floor((1-target)*len(vals)))])
        panels = {}
        for m in ('all', 'legacy_optical', 'sdss', 'ps1'):
            for h in ('south', 'north'):
                L = lambda k: dict(np.load(rel/f'{m}_{h}_{k}.npz'))
                qk, qrej = ranks(L('qso_raw'), L('qso_unified'), thr); sk, _ = ranks(L('stars_raw'), L('stars_unified'), thr)
                qa, sa = L('qso_active'), L('stars_active')
                active = auc(np.where(qa['eligible'], qa['log_r_per_unit_z'], -np.inf), np.where(sa['eligible'], sa['log_r_per_unit_z'], -np.inf))
                panels[f'{m}_{h}'] = dict(auc=auc(qk, sk), active=active, retention=float(1-qrej.mean()))
        grid = {}
        for h in ('south', 'north'):
            for mag in (18.5, 21.0):
                low = None
                for lab in ('baseline', 'candidate'):
                    o = np.load(OLD/f'grid_{h}_{mag}_{lab}.npz'); qq = np.logaddexp(o['log_lambda_sameq'], o['log_lambda_fieldq']); bb = o['log_lambda_bkg']
                    m_ = (qq < np.nanmax(qq)+np.log(.01)) & (bb < np.nanmax(bb)+np.log(.01)); low = m_ if low is None else low & m_
                gu = dict(np.load(rel/f'grid_{h}_{mag}_unified.npz')); gr = dict(np.load(rel/f'grid_{h}_{mag}_raw.npz'))
                keep = gr['eligible'] & ~(np.isfinite(gu['support']) & (gu['support'] < thr))
                grid[f'{h}_{mag}'] = int((low & keep & (gr['p_quasar'] > .5)).sum())
        d = np.array([p['auc']-p['active'] for p in panels.values()])
        rows.append(dict(target_retention=target, threshold=thr, panels=panels, grid=grid,
                         min_auc_minus_active=float(d.min()), failed_panels=int((d < -0.005).sum()),
                         min_qso_retention=min(p['retention'] for p in panels.values())))
    out[name] = dict(current_threshold=current, sweep=rows)
    print(f'\n{name} (current threshold {current:.4f})')
    print('target  thr     worst dAUC vs active  failed  min QSO retention  grid stray S18.5/S21/N18.5/N21')
    for r in rows:
        print(f"{r['target_retention']:<7} {r['threshold']:.4f}  {r['min_auc_minus_active']:+.4f}               {r['failed_panels']}       {r['min_qso_retention']:.3f}              {list(r['grid'].values())}")
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
