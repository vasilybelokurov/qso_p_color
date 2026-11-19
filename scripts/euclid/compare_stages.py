#!/usr/bin/env python
"""Paired comparison of stage-2 vs stage-1 Euclid-only scores on the same Euclid test rows (docs/euclid/PLAN.md 2.5).

Both validations (scripts/euclid/validate_stage1.py --stage 1|2, same seed) score identical QSO test rows and background
cone rows; this script checks the rows match and reports stage2 - stage1 for each metric of validate_stage1.metrics
with a paired bootstrap (QSOs resampled by Euclid sky cell, background by cone; 1000 resamples).

    python scripts/euclid/compare_stages.py --bundle 20261005_magdep_qlf
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_stage1 import metrics  # noqa: E402

WORK = Path('models/multisurvey_psf/work/euclid')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True); ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--mode', default='euclid', choices=('euclid', 'combined'))
    ap.add_argument('--tag', default='', help='validation-directory prefix, e.g. d5_')
    a = ap.parse_args()
    s1 = np.load(WORK/f'{a.tag}validation_{a.bundle}'/'scores.npy', allow_pickle=True).item()
    s2 = np.load(WORK/f'{a.tag}validation_stage2_{a.bundle}'/'scores.npy', allow_pickle=True).item()
    q1, q2, b1, b2 = s1['qso_test'], s2['qso_test'], s1['bkg'], s2['bkg']
    if not (np.array_equal(q1['idx'], q2['idx']) and np.array_equal(b1['cone'], b2['cone'])
            and np.allclose(b1['vis'], b2['vis']) and np.allclose(b1['z'], b2['z'])):
        raise SystemExit('stage-1 and stage-2 validations scored different rows')
    unfl = ~b1['likely']; m = a.mode
    v1, v2 = metrics(q1[m], b1[m], unfl), metrics(q2[m], b2[m], unfl)
    rng = np.random.default_rng(11); uq, ub = np.unique(q1['cell']), np.unique(b1['cone'])
    iq = {g: np.flatnonzero(q1['cell'] == g) for g in uq}; ib = {g: np.flatnonzero(b1['cone'] == g) for g in ub}
    diffs = []
    for _ in range(a.n_boot):
        i = np.concatenate([iq[g] for g in rng.choice(uq, len(uq))]); j = np.concatenate([ib[g] for g in rng.choice(ub, len(ub))])
        sub = lambda d, s: {k: v[s] for k, v in d.items()}
        x1 = metrics(sub(q1[m], i), sub(b1[m], j), unfl[j]); x2 = metrics(sub(q2[m], i), sub(b2[m], j), unfl[j])
        diffs.append({k: x2[k] - x1[k] for k in x1})
    out = {k: dict(stage1=v1[k], stage2=v2[k], difference=v2[k] - v1[k],
                   ci95=[float(np.percentile([d[k] for d in diffs], q)) for q in (2.5, 97.5)]) for k in v1}
    (WORK/f'{a.tag}compare_stages_{m}_{a.bundle}.json').write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:18s} {v['stage1']:.4f} -> {v['stage2']:.4f}  diff {v['difference']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == '__main__':
    main()
