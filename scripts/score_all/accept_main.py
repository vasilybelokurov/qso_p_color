#!/usr/bin/env python
"""Main-scorer acceptance of the D5 changes (docs/euclid/PLAN_score_all.md, Phase D-main, tests M1 and M2).

Comparator: the promoted bundles' held-out main panels scored with the pre-D5 rules
(models/multisurvey_psf/work/method_unified/performance_qlf; faint limit S/N >= 10; rows below it left out).
Candidate: the same panels scored with the D5 defaults by

    python scripts/method_unified/score_performance.py --panels main --faint-limit 10 --d5 --keep-weak \
        --out models/multisurvey_psf/work/score_all/performance_d5

which also scores rows whose Legacy r is below the limit (weak r). The cohort is the union of rows; a row absent
from the comparator (weak r) counts as refused there. Refused rows rank at the bottom (log R = -inf) and are never
above p = 0.5.

M1: every row eligible in the comparator keeps log R and p_quasar exactly (max abs difference reported; < 1e-9).
M2: paired bootstrap by sky region (QSOs: HEALPix nside-4 cell; background: stellar cone), 1000 resamples.
Pass if the lower 95% bound of (candidate - comparator) exceeds -margin for AUC (0.002) and recall at p > 0.5
(1 pp), and the upper bound of the unflagged-background p > 0.5 rate change is below +0.1 pp. A margin below 2 SE
is reported as unresolvable. Newly scored rows are also summarised per stratum (support-flagged, outside-both,
weak r).

    python scripts/score_all/accept_main.py --candidate models/multisurvey_psf/work/score_all/performance_d5
"""
import argparse
import json
from pathlib import Path
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'euclid'))
from stage2_acceptance import MARGIN, PARENT, RUN, load, stats  # noqa: E402

STRATA = dict(support='qso_support_rejected', outside_both='outside_both_models', weak_r='weak_legacy_r_in_likelihood')


def union(a, b):
    """Rows of both runs on one key; rows missing from a run are refused there."""
    ka = a['hemi']*10**8 + a['rows']; kb = b['hemi']*10**8 + b['rows']
    keys = np.union1d(ka, kb)
    def expand(d, k):
        idx = np.searchsorted(keys, k)
        out = {}
        for name, v in d.items():
            if name == 'flags' or v.dtype.kind in 'US':
                fill = np.full(len(keys), '', dtype=v.dtype if v.dtype.kind == 'U' else '<U200')
            elif v.dtype == bool:
                fill = np.zeros(len(keys), bool)
            else:
                fill = np.full(len(keys), np.nan)
            fill[idx] = v; out[name] = fill
        out['present'] = np.isin(keys, k)
        return out
    a2, b2 = expand(a, ka), expand(b, kb)
    for d in (a2, b2):
        d['rows'] = (keys % 10**8).astype(int); d['hemi'] = (keys // 10**8).astype(int)
    return a2, b2


def load_flags(root, model, kind):
    import glob
    parts = []
    for p in sorted(glob.glob(str(Path(root)/f'{model}_{kind}_main_*_[0-9]*.npz'))):
        if p.endswith('.rows.npz'):
            continue
        with np.load(p) as d:
            parts.append(d['flags'] if 'flags' in d.files else np.full(len(d['eligible']), ''))
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidate', type=Path, required=True); ap.add_argument('--n-boot', type=int, default=1000)
    a = ap.parse_args(); rng = np.random.default_rng(3); report = {}
    l = np.load(RUN/'qso/l.npy', mmap_mode='r'); bq = np.load(RUN/'qso/b.npy', mmap_mode='r')
    field = np.load(RUN/'stars/field.npy', mmap_mode='r'); qflag = np.load(RUN/'stars/qso_flag.npy', mmap_mode='r')
    for model in ('independent', 'dependent'):
        rep = {}
        loaded = {}
        for kind in ('qso', 'stars'):
            par, cand = load(PARENT, model, kind), load(a.candidate, model, kind)
            cand['flags'] = load_flags(a.candidate, model, kind)
            loaded[kind] = union(par, cand)
        (pq, cq), (pb, cb) = loaded['qso'], loaded['stars']
        for d in (pq, cq, pb, cb):                       # absent rows: refused
            d['eligible'] = d['eligible'] & d['present']
        # M1: rows eligible in the comparator
        m1 = {}
        for kind, (p, c) in loaded.items():
            e = p['eligible']
            m1[kind] = dict(n=int(e.sum()), lost=int((e & ~c['eligible']).sum()),
                            max_abs_dlogr=float(np.nanmax(np.abs(c['log_r_per_unit_z'][e] - p['log_r_per_unit_z'][e]))),
                            max_abs_dp=float(np.nanmax(np.abs(c['p_quasar'][e] - p['p_quasar'][e]))))
        rep['M1'] = dict(rows=m1, pass_=all(v['lost'] == 0 and v['max_abs_dlogr'] < 1e-9 and v['max_abs_dp'] < 1e-9
                                             for v in m1.values()))
        # newly scored strata
        strata = {}
        for kind, (p, c) in loaded.items():
            new = c['eligible'] & ~p['eligible']
            flags = c['flags'].astype(str)
            s = {}
            for name, tag in STRATA.items():
                m = new & (np.char.find(flags, tag) >= 0)
                s[name] = dict(n=int(m.sum()), frac_p_gt_half=float(np.mean(c['p_quasar'][m] > .5)) if m.any() else None)
            s['all_new'] = dict(n=int(new.sum()), frac_p_gt_half=float(np.mean(c['p_quasar'][new] > .5)) if new.any() else None)
            strata[kind] = s
        rep['new_strata'] = strata
        # M2
        flag = np.asarray(qflag[pb['rows']]).astype(bool)
        qcell = hp.ang2pix(4, np.asarray(l[pq['rows']]), np.asarray(bq[pq['rows']]), lonlat=True)
        bcell = np.asarray(field[pb['rows']])
        for d in (pq, cq, pb, cb):
            d['p_quasar'] = np.nan_to_num(d['p_quasar'], nan=0.)
            d['log_r_per_unit_z'] = np.nan_to_num(d['log_r_per_unit_z'], nan=-np.inf)
        base = {k: stats(cq, cb, flag)[k] - stats(pq, pb, flag)[k] for k in MARGIN}
        uq, ub = np.unique(qcell), np.unique(bcell); iq = {c: np.flatnonzero(qcell == c) for c in uq}
        ib = {c: np.flatnonzero(bcell == c) for c in ub}; boot = []
        sub = lambda d, s: {k: v[s] for k, v in d.items()}
        for _ in range(a.n_boot):
            s1 = np.concatenate([iq[c] for c in rng.choice(uq, len(uq))]); s2 = np.concatenate([ib[c] for c in rng.choice(ub, len(ub))])
            c_, p_ = stats(sub(cq, s1), sub(cb, s2), flag[s2]), stats(sub(pq, s1), sub(pb, s2), flag[s2])
            boot.append({k: c_[k] - p_[k] for k in MARGIN})
        out = {}
        for k, mg in MARGIN.items():
            v = np.array([x[k] for x in boot]); lo, hi = np.percentile(v, [2.5, 97.5])
            ok = (lo > -mg) if k != 'bkg' else (hi < mg)
            out[k] = dict(difference=float(base[k]), ci95=[float(lo), float(hi)], se=float(v.std()), margin=mg,
                          resolvable=bool(mg >= 2*v.std()), pass_=bool(ok))
        rep['M2'] = dict(n_qso=len(pq['rows']), n_bkg=len(pb['rows']), comparator=stats(pq, pb, flag),
                         candidate=stats(cq, cb, flag), differences=out, pass_=all(x['pass_'] for x in out.values()))
        report[model] = rep
    a.candidate.joinpath('acceptance_d5.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
