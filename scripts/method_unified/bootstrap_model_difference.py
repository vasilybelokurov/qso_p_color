#!/usr/bin/env python
"""Is the magnitude-dependent QSO model significantly better? Paired sky bootstrap of the differences.

Held-out main panels of the note (score_performance.py with the faint limit, PERF in common.py): both
models score the same objects, so differences are paired. Resampling is by sky region to respect spatial
correlation: quasars by HEALPix nside-4 cell, background objects by stellar-sample cone (field).
Metrics: AUC (log R; unranked at the bottom), QSO recall at p_quasar > 0.5, background incidence at
p_quasar > 0.5 (flagged likely QSOs removed), overall and by reference magnitude.
Writes docs/method_unified/model_difference_bootstrap.json.
"""
import json
from pathlib import Path
import sys

import healpy as hp
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig12_14_performance as P
from common import data, ROOT

NB = 300
MAG = [(15.5, 18), (18, 19), (19, 21), (21, 24.5)]


def auc_fast(sq, sb):
    r = rankdata(np.concatenate([sq, sb])); n1 = len(sq)
    return (r[:n1].sum() - n1*(n1 + 1)/2)/(n1*len(sb))


def main():
    q = {m: P.load(m, 'qso', 'main') for m in P.MODELS}; b = {m: P.load(m, 'stars', 'main') for m in P.MODELS}
    assert (q['independent']['rows'] == q['dependent']['rows']).all() and (b['independent']['rows'] == b['dependent']['rows']).all()
    flag = np.load(ROOT/'stars'/'qso_flag.npy')[b['independent']['rows']]
    qc = hp.ang2pix(4, data('qso')['l'][q['independent']['rows']], data('qso')['b'][q['independent']['rows']], lonlat=True)
    bc = np.asarray(data('stars')['field'])[b['independent']['rows']]
    rank = {m: (np.where(q[m]['eligible'], q[m]['log_r_per_unit_z'], -np.inf), np.where(b[m]['eligible'], b[m]['log_r_per_unit_z'], -np.inf)) for m in P.MODELS}
    rec = {m: q[m]['eligible'] & (q[m]['p_quasar'] > .5) for m in P.MODELS}
    inc = {m: b[m]['eligible'] & (b[m]['p_quasar'] > .5) for m in P.MODELS}
    mq, mb = q['independent']['ref_mag'], b['independent']['ref_mag']
    def stats(iq, ib):
        out = {}
        for lab, (lo, hi) in [('all', (0, 99))] + [(f'{lo}-{hi}', (lo, hi)) for lo, hi in MAG]:
            sq = iq[(mq[iq] >= lo) & (mq[iq] < hi)]; sb = ib[(mb[ib] >= lo) & (mb[ib] < hi)]
            if lab != 'all' and (len(sq) < 30 or len(sb) < 30):
                out[lab] = dict(auc=np.nan); continue
            a = {m: auc_fast(rank[m][0][sq], rank[m][1][sb]) for m in P.MODELS}
            out[lab] = dict(auc=a['dependent'] - a['independent'])
        out['recall'] = rec['dependent'][iq].mean() - rec['independent'][iq].mean()
        clean = ib[~flag[ib]]
        out['incidence_clean'] = inc['dependent'][clean].mean() - inc['independent'][clean].mean()
        return out
    point = stats(np.arange(len(mq)), np.arange(len(mb)))
    rng = np.random.default_rng(7); uq, ub = np.unique(qc), np.unique(bc)
    gq = {c: np.flatnonzero(qc == c) for c in uq}; gb = {c: np.flatnonzero(bc == c) for c in ub}
    boots = []
    for _ in range(NB):
        iq = np.concatenate([gq[c] for c in rng.choice(uq, len(uq))]); ib = np.concatenate([gb[c] for c in rng.choice(ub, len(ub))])
        boots.append(stats(iq, ib))
    rep = {}
    for key in point:
        if isinstance(point[key], dict):
            v = np.array([bb[key]['auc'] for bb in boots]); p0 = point[key]['auc']
        else:
            v = np.array([bb[key] for bb in boots]); p0 = point[key]
        v = v[np.isfinite(v)]
        rep[key] = dict(point=float(p0), lo95=float(np.percentile(v, 2.5)), hi95=float(np.percentile(v, 97.5)),
                        frac_positive=float(np.mean(v > 0)))
        print(f"{key:15s} dependent - independent = {p0:+.4f}  95% [{rep[key]['lo95']:+.4f}, {rep[key]['hi95']:+.4f}]  P(>0) {rep[key]['frac_positive']:.3f}")
    Path('docs/method_unified/model_difference_bootstrap.json').write_text(json.dumps(dict(definition=__doc__, n_boot=NB,
        n_qso_cells=int(len(uq)), n_bkg_cones=int(len(ub)), results=rep), indent=1))


if __name__ == '__main__':
    main()
