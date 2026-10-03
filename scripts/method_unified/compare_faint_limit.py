#!/usr/bin/env python
"""Effect of the faint limit (Legacy r S/N >= 10, Legacy r reference) on the promoted bundles.

Compares the main-panel scores of score_performance.py without the limit (performance/) and with it
(performance_faint10/) on the same held-out rows. 'before, same rows' restricts the original scores to
the rows the limit keeps, isolating the reference-band change from the removal of faint objects.
Metrics as in fig12_14_performance.py (AUC, background p_Q>0.5 incidence, QSO support retention) plus
QSO recall (p_Q>0.5). Writes docs/method_unified/faint_limit_effect.json.
"""
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig12_14_performance as P

BEFORE, AFTER = P.OUT, Path('models/multisurvey_psf/work/method_unified/performance_faint10')


def metrics(q, b):
    s = P.summary(q, b); s['recall'] = float((q['eligible'] & (q['p_quasar'] > .5)).mean()); return s


def main():
    report = {}
    for model in P.MODELS:
        P.OUT = BEFORE; q0, b0 = P.load(model, 'qso', 'main'), P.load(model, 'stars', 'main')
        P.OUT = AFTER; q1, b1 = P.load(model, 'qso', 'main'), P.load(model, 'stars', 'main')
        keep_q = np.isin(q0['rows'], q1['rows']); keep_b = np.isin(b0['rows'], b1['rows'])
        rep = dict(kept_qso=float(keep_q.mean()), kept_bkg=float(keep_b.mean()),
                   before_all=metrics(q0, b0), before_same_rows=metrics(P.sub(q0, keep_q), P.sub(b0, keep_b)), after=metrics(q1, b1))
        for h in ('south', 'north'):
            rep['after_'+h] = metrics(P.sub(q1, q1['hemi'] == h), P.sub(b1, b1['hemi'] == h))
            rep['before_same_rows_'+h] = metrics(P.sub(q0, keep_q & (q0['hemi'] == h)), P.sub(b0, keep_b & (b0['hemi'] == h)))
        report[model] = rep
        print(f"{model}: kept QSO {rep['kept_qso']:.3f}, kept background {rep['kept_bkg']:.3f}")
        for k in ('before_all', 'before_same_rows', 'after', 'before_same_rows_south', 'after_south', 'before_same_rows_north', 'after_north'):
            m = rep[k]
            print(f"  {k:24s} nQ {m['n_qso']:6d} nB {m['n_bkg']:6d} AUC {m['auc']:.4f} incidence {m['incidence']:.4f} retention {m['retention']:.4f} recall {m['recall']:.3f}")
    Path('docs/method_unified/faint_limit_effect.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
