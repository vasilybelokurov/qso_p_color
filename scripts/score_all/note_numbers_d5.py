#!/usr/bin/env python
"""Headline numbers of the method note's Section "Performance" under the D5 rules (19 Nov 2026).

Cohort: the held-out main panels (score_performance.py --faint-limit 10 --d5 --keep-weak), rows the scorer handles
(fig12_14_performance.load drops rows refused for faintness). Same definitions as the note:
  recall          fraction of test QSOs with p_quasar > 0.5 (unranked count as below)
  unranked        not eligible (no score): prior empty at the magnitude
  flagged         eligible with qso_support_rejected / outside_both_models / weak_legacy_r_in_likelihood
  incidence       fraction of background objects with p_quasar > 0.5; *_clean after removing objects flagged as
                  likely QSOs (stars/qso_flag.npy, as bootstrap_model_difference.py)
  auc_clean       AUC of log R against the clean background
Also recall and incidence restricted to unflagged objects (what a user cutting on the flags gets) and recall at
23 < r < 24.

    PYTHONPATH=scripts/method_unified python scripts/score_all/note_numbers_d5.py
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/method_unified'))
import fig12_14_performance as P   # noqa: E402
from common import ROOT as RUN      # noqa: E402

FLAGS = ('qso_support_rejected', 'outside_both_models', 'weak_legacy_r_in_likelihood')


def load(m, kind):
    """P.load plus the flags column (P.load drops it), same row order and cohort filter."""
    import glob
    d = P.load(m, kind, 'main'); fl = []
    for p in sorted(glob.glob(str(P.OUT/f'{m}_{kind}_main_*_[0-9]*.npz'))):
        if p.endswith('.rows.npz'):
            continue
        tag = Path(p).name[len(f'{m}_{kind}_main_'):-len('_000000.npz')]
        if '_z' in tag or any(tag.endswith(x) for x in ('grz', 'all', 'sdss', 'ps1')):
            continue
        z = np.load(p, allow_pickle=True); keep = np.asarray(z['status']).astype(str) != ''
        fl.append(np.asarray(z['flags']).astype(str)[keep])
    d['flags'] = np.concatenate(fl); assert len(d['flags']) == len(d['rows'])
    return d


def flagged(d, name):
    return np.array([name in f.split(',') for f in d['flags']])


def main():
    out = {}
    for m in P.MODELS:
        q, b = load(m, 'qso'), load(m, 'stars')
        likely = np.load(RUN/'stars'/'qso_flag.npy')[b['rows']].astype(bool)
        fq = {k: flagged(q, k) for k in FLAGS}; fb = {k: flagged(b, k) for k in FLAGS}
        anyq = np.any(list(fq.values()), axis=0); anyb = np.any(list(fb.values()), axis=0)
        hi_q = q['eligible'] & (q['p_quasar'] > .5); hi_b = b['eligible'] & (b['p_quasar'] > .5)
        clean = {k: v[~likely] for k, v in b.items() if np.ndim(v) and len(v) == len(likely)}
        faint = (q['ref_mag'] > 23) & (q['ref_mag'] < 24)
        out[m] = dict(
            n_qso=int(len(q['rows'])), n_bkg=int(len(b['rows'])), n_bkg_clean=int((~likely).sum()),
            auc=P.auc(q, b), auc_clean=P.auc(q, clean),
            recall=float(hi_q.mean()), recall_23_24=float(hi_q[faint].mean()), n_23_24=int(faint.sum()),
            recall_unflagged_only=float((hi_q & ~anyq).mean()),
            unranked_qso=float((~q['eligible']).mean()), unranked_bkg=float((~b['eligible']).mean()),
            flagged_qso={k: float(v.mean()) for k, v in fq.items()}, flagged_qso_any=float(anyq.mean()),
            flagged_bkg={k: float(v.mean()) for k, v in fb.items()}, flagged_bkg_any=float(anyb.mean()),
            incidence=float(hi_b.mean()), incidence_clean=float(hi_b[~likely].mean()),
            incidence_clean_unflagged=float((hi_b & ~anyb)[~likely].mean()),
            incidence_flagged_share=float((hi_b & anyb).sum()/max(hi_b.sum(), 1)))
    path = ROOT/'docs/score_all/note_numbers_d5.json'; path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
