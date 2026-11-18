#!/usr/bin/env python
"""Stage-2 acceptance on the parent's 41-band held-out main panels (docs/euclid/PLAN.md 2.5, T-V10).

Comparator: the promoted bundles' scores in models/multisurvey_psf/work/method_unified/performance_qlf (current
bundles, eBOSS QLF prior, faint limit S/N >= 10). Candidate: the stage-2 'legacy_psf' bundles scored on the same rows
by scripts/method_unified/score_performance.py --panels main --faint-limit 10 --bundles ... --out <PERF2>.
Paired bootstrap by sky region (QSOs: HEALPix nside-4 cell; background: stellar cone), 1000 resamples.
Pass if the lower 95% bound of (candidate - parent) exceeds -margin for AUC (log R, unranked at the bottom; 0.002)
and recall at p > 0.5 (1 pp), and the upper bound of the change of the unflagged background p > 0.5 rate is below
+0.1 pp. The bootstrap SE of each difference is reported (a margin below 2 SE is unresolvable).

    python scripts/euclid/stage2_acceptance.py --candidate models/multisurvey_psf/work/euclid/performance_stage2
"""
import argparse
import glob
import json
from pathlib import Path
import healpy as hp
import numpy as np
from scipy.stats import rankdata

PARENT = Path('models/multisurvey_psf/work/method_unified/performance_qlf')
RUN = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')
MARGIN = dict(auc=0.002, recall=0.01, bkg=0.001)


def load(root, model, kind):
    parts = []
    for p in sorted(glob.glob(str(Path(root)/f'{model}_{kind}_main_*_[0-9]*.npz'))):
        if p.endswith('.rows.npz'):
            continue
        d = dict(np.load(p)); r = dict(np.load(p[:-4] + '.rows.npz')); d.update(rows=r['rows'])
        d['hemi'] = np.full(len(r['rows']), 1 if '_north' in p else 0); parts.append(d)
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0] if k not in ('flags', 'reference')}


def align(a, b):
    ka = a['hemi']*10**8 + a['rows']; kb = b['hemi']*10**8 + b['rows']
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    return {k: v[ia] for k, v in a.items()}, {k: v[ib] for k, v in b.items()}


def stats(q, b, flag):
    sq = np.where(q['eligible'], q['log_r_per_unit_z'], -np.inf); sb = np.where(b['eligible'], b['log_r_per_unit_z'], -np.inf)
    r = rankdata(np.concatenate([sq, sb])); n1 = len(sq)
    return dict(auc=(r[:n1].sum() - n1*(n1 + 1)/2)/(n1*len(sb)),
                recall=float(np.mean(q['eligible'] & (q['p_quasar'] > .5))),
                bkg=float(np.mean((b['eligible'] & (b['p_quasar'] > .5))[~flag])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidate', type=Path, required=True); ap.add_argument('--n-boot', type=int, default=1000)
    a = ap.parse_args(); rng = np.random.default_rng(3); report = {}
    l = np.load(RUN/'qso/l.npy', mmap_mode='r'); bq = np.load(RUN/'qso/b.npy', mmap_mode='r')
    field = np.load(RUN/'stars/field.npy', mmap_mode='r'); qflag = np.load(RUN/'stars/qso_flag.npy', mmap_mode='r')
    for model in ('independent', 'dependent'):
        pq, cq = align(load(PARENT, model, 'qso'), load(a.candidate, model, 'qso'))
        pb, cb = align(load(PARENT, model, 'stars'), load(a.candidate, model, 'stars'))
        flag = np.asarray(qflag[pb['rows']]).astype(bool)
        qcell = hp.ang2pix(4, np.asarray(l[pq['rows']]), np.asarray(bq[pq['rows']]), lonlat=True)
        bcell = np.asarray(field[pb['rows']])
        base = {k: stats(cq, cb, flag)[k] - stats(pq, pb, flag)[k] for k in MARGIN}
        uq, ub = np.unique(qcell), np.unique(bcell); iq = {c: np.flatnonzero(qcell == c) for c in uq}
        ib = {c: np.flatnonzero(bcell == c) for c in ub}; boot = []
        for _ in range(a.n_boot):
            s1 = np.concatenate([iq[c] for c in rng.choice(uq, len(uq))]); s2 = np.concatenate([ib[c] for c in rng.choice(ub, len(ub))])
            sub = lambda d, s: {k: v[s] for k, v in d.items()}
            c_, p_ = stats(sub(cq, s1), sub(cb, s2), flag[s2]), stats(sub(pq, s1), sub(pb, s2), flag[s2])
            boot.append({k: c_[k] - p_[k] for k in MARGIN})
        out = {}
        for k, mg in MARGIN.items():
            v = np.array([x[k] for x in boot]); lo, hi = np.percentile(v, [2.5, 97.5])
            ok = (lo > -mg) if k != 'bkg' else (hi < mg)
            out[k] = dict(difference=float(base[k]), ci95=[float(lo), float(hi)], se=float(v.std()), margin=mg,
                          resolvable=bool(mg >= 2*v.std()), pass_=bool(ok))
        report[model] = dict(n_qso=len(pq['rows']), n_bkg=len(pb['rows']), parent=stats(pq, pb, flag),
                             candidate=stats(cq, cb, flag), differences=out, accepted=all(x['pass_'] for x in out.values()))
    (a.candidate/'acceptance.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
