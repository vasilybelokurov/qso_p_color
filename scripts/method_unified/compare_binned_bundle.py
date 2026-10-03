#!/usr/bin/env python
"""Step 3: promoted bundles vs the binned-background test bundles, both with the faint limit.

Inputs: main-panel scores of score_performance.py with --faint-limit 10 for the promoted bundles
(performance_faint10/) and for the test bundles (performance_binned/, build_binned_test_bundle.py).
Same rows in both; background rows used by the background light tests (test1, test2 rows.npz) are
removed from both, as they took part in model selection. Metrics (fig12_14_performance.py):
AUC, background p_Q>0.5 incidence, QSO support retention, QSO recall (p_Q>0.5); overall, per
hemisphere and per Legacy r bin (reference magnitude, extinction corrected).
Writes docs/method_unified/binned_bundle_comparison.json.
"""
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig12_14_performance as P

RUNS = dict(current=Path('models/multisurvey_psf/work/method_unified/performance_faint10'),
            binned=Path('models/multisurvey_psf/work/method_unified/performance_binned'))
LIGHT = [Path('models/multisurvey_psf/work/background_designs')/t/'rows.npz' for t in ('test1', 'test2', 'test3')]
MAG = [15.5, 18, 19, 20, 21, 22, 23, 24.5]


def metrics(q, b):
    s = P.summary(q, b); s['recall'] = float((q['eligible'] & (q['p_quasar'] > .5)).mean()) if len(q['rows']) else np.nan
    return s


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--binned', type=Path, default=RUNS['binned'])
    ap.add_argument('--json', type=Path, default=Path('docs/method_unified/binned_bundle_comparison.json')); args = ap.parse_args()
    RUNS['binned'] = args.binned
    used = np.unique(np.concatenate([v for f in LIGHT if f.exists() for v in np.load(f).values()]))
    report = dict(definition=__doc__, removed_light_test_rows=int(len(used)), models={})
    for model in P.MODELS:
        sc = {}
        for run, path in RUNS.items():
            P.OUT = path; sc[run] = (P.load(model, 'qso', 'main'), P.load(model, 'stars', 'main'))
        rq = np.intersect1d(sc['current'][0]['rows'], sc['binned'][0]['rows'])
        rb = np.setdiff1d(np.intersect1d(sc['current'][1]['rows'], sc['binned'][1]['rows']), used)
        rep = {}
        for run, (q, b) in sc.items():
            q = P.sub(q, np.isin(q['rows'], rq)); b = P.sub(b, np.isin(b['rows'], rb))
            out = dict(all=metrics(q, b))
            for h in ('south', 'north'):
                out[h] = metrics(P.sub(q, q['hemi'] == h), P.sub(b, b['hemi'] == h))
            out['by_mag'] = [dict(bin=[lo, hi], **metrics(P.sub(q, (q['ref_mag'] >= lo) & (q['ref_mag'] < hi)),
                                                           P.sub(b, (b['ref_mag'] >= lo) & (b['ref_mag'] < hi))))
                             for lo, hi in zip(MAG[:-1], MAG[1:])]
            rep[run] = out
        report['models'][model] = rep
        print(f'{model}: QSOs {len(rq)}, background {len(rb)}')
        for key in ('all', 'south', 'north'):
            for run in RUNS:
                m = rep[run][key]
                print(f"  {key:5s} {run:7s} AUC {m['auc']:.4f} incidence {m['incidence']:.4f} retention {m['retention']:.4f} recall {m['recall']:.3f}")
        print('  by reference magnitude: AUC current/binned, incidence current/binned, recall current/binned (nQ, nB)')
        for c, bn in zip(rep['current']['by_mag'], rep['binned']['by_mag']):
            print(f"   {c['bin'][0]:5.1f}-{c['bin'][1]:4.1f}: {c['auc']:.4f}/{bn['auc']:.4f}  {c['incidence']:.4f}/{bn['incidence']:.4f}"
                  f"  {c['recall']:.3f}/{bn['recall']:.3f}  ({c['n_qso']}, {c['n_bkg']})")
    args.json.write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
