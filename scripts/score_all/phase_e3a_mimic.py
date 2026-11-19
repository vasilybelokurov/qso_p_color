#!/usr/bin/env python
"""Phase E step 3a: wide-selection mimic w = P(selected by the wide deployment selection | deep measurements).

Design and tests fixed before running: docs/score_all/PHASE_AB.md, "Phase E step 3: design and tests".
- Candidate pool: deep rows (scripts/score_all/tile_rows.py deep_candidates), restricted to HEALPix pixels covered by
  BOTH the deep and the wide tiles (otherwise a deep row outside wide coverage would be labelled 0 wrongly).
- Label: nearest wide-selected row (cones.deployment on the wide tile, owned by the tile) within 0.5".
- Classifier: HistGradientBoostingClassifier on tile_rows.FEATURES (no colours), trained on EDF-South tiles with
  id mod 5 != 0; at most N_TRAIN rows drawn at random (seed fixed).
- M1 reliability on EDF-South held-out tiles (id mod 5 == 0): |observed - predicted| <= 0.03 per decile of w.
- M2 transfer to EDF-North (wide-covered part): per 0.5-mag VIS bin 21.5-26, |sum w - n_selected| <= max(10%, 3 sqrt(n)).
- M3 colours, EDF-North wide-covered part, VIS 21.5-25.5: w-weighted deep colours vs deep colours of the wide-selected
  matched rows: |d median| <= 0.05 mag, |d half-width| <= 20%, VIS-Y, VIS-J, VIS-H.
- Reported: fraction of wide-selected rows (in the deep-covered area) with no deep candidate within 0.5"; completeness
  C(V) = mean w of deep rows with point_like_prob > 0.9 (step 4 input), both fields.
Outputs: model ~/data/qso_p_color/score_all/phase_e/mimic.pkl, docs/score_all/phase_e3a_mimic.json.

    nice -n 19 python scripts/score_all/phase_e3a_mimic.py [--limit N]
"""
import json
import pickle
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/euclid')); sys.path.insert(0, str(ROOT/'scripts/score_all'))
import tile_rows as T  # noqa: E402

OUT = ROOT/'docs/score_all/phase_e3a_mimic.json'
MODEL = T.OUT/'mimic.pkl'
N_TRAIN = 3_000_000
MATCH_ARCSEC = 0.5
SEED = 20261119
V_EDGES = np.arange(21.0, 26.51, 0.5)


def labelled(field, settings, limit):
    """Deep candidates in the deep & wide covered area, with the wide-selected label; and the unmatched-wide fraction."""
    from cones import nearest
    d = T.field_table('deep', field, settings, limit); w = T.field_table('wide', field, settings, limit)
    pd, _ = T.coverage('deep', field, limit); pw, _ = T.coverage('wide', field, limit)
    both = np.intersect1d(pd, pw)
    keep = T.in_pixels(d['ra'], d['dec'], both)
    d = {k: v[keep] for k, v in d.items()}
    ww = T.in_pixels(w['ra'], w['dec'], both); wra, wdec = w['ra'][ww], w['dec'][ww]
    iw = nearest(d['ra'], d['dec'], wra, wdec, MATCH_ARCSEC)
    # one deep row per wide row: if several deep rows claim the same wide row keep the nearest (rare)
    lab = iw >= 0
    if lab.any():
        from cones import unit
        sep = np.full(len(iw), np.inf)
        sep[lab] = np.sum((unit(d['ra'][lab], d['dec'][lab]) - unit(wra[iw[lab]], wdec[iw[lab]]))**2, 1)
        order = np.lexsort((sep, iw)); first = np.ones(len(order), bool)
        first[1:] = iw[order][1:] != iw[order][:-1]
        dup = order[~first]; lab[dup] = False
    d['label'] = lab
    unmatched = float(1 - lab.sum()/max(len(wra), 1))
    return d, dict(n_deep=int(len(lab)), n_wide_selected=int(len(wra)), n_matched=int(lab.sum()),
                   wide_unmatched_fraction=round(unmatched, 4),
                   area_both_deg2=round(len(both)*0.000819, 3))


def X(d):
    return np.column_stack([np.asarray(d[f], float) for f in T.FEATURES])


def colour_check(d, w):
    rep, ok_all = [], True
    for lo in np.arange(21.5, 25.51, 0.5)[:-1]:
        s = (d['vis'] >= lo) & (d['vis'] < lo + 0.5) & np.isfinite(d['lup']).all(1)
        sel = s & d['label']
        row = dict(vis=f'{lo:.1f}-{lo+0.5:.1f}', n_selected=int(sel.sum()), sum_w=round(float(w[s].sum()), 1), colours={})
        for j, name in ((1, 'VIS-Y'), (2, 'VIS-J'), (3, 'VIS-H')):
            c = (d['lup'][:, 0] - d['lup'][:, j]).astype(float)
            qo = np.percentile(c[sel], [16, 50, 84])
            o = np.argsort(c[s]); cw = np.cumsum(w[s][o]); cw /= cw[-1]
            qp = np.interp([0.16, 0.5, 0.84], cw, c[s][o])
            dmed = float(qp[1] - qo[1]); dw = float((qp[2] - qp[0])/(qo[2] - qo[0]) - 1)
            ok = abs(dmed) <= 0.05 and abs(dw) <= 0.2; ok_all &= ok
            row['colours'][name] = dict(selected_q=np.round(qo, 3).tolist(), weighted_q=np.round(qp, 3).tolist(),
                                        d_median=round(dmed, 3), d_halfwidth_frac=round(dw, 3), pass_=bool(ok))
        rep.append(row)
    return rep, bool(ok_all)


def main():
    import argparse
    from sklearn.ensemble import HistGradientBoostingClassifier
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=None, help='first N tiles per field (smoke test)')
    a = ap.parse_args()
    settings = json.loads((ROOT/'docs/euclid/settings.json').read_text())
    rng = np.random.default_rng(SEED)
    S, infoS = labelled('S', settings, a.limit)
    N, infoN = labelled('N', settings, a.limit)
    rep = dict(features=list(T.FEATURES), match_arcsec=MATCH_ARCSEC, fields=dict(S=infoS, N=infoN))
    hold = (S['tile'] % 5) == 0
    tr = np.flatnonzero(~hold)
    if len(tr) > N_TRAIN:
        tr = rng.choice(tr, N_TRAIN, replace=False)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
    clf.fit(X({k: v[tr] for k, v in S.items()}), S['label'][tr])
    MODEL.write_bytes(pickle.dumps(dict(model=clf, features=T.FEATURES)))
    rep['n_train'] = int(len(tr)); rep['n_iter'] = int(clf.n_iter_)
    # M1
    ph = clf.predict_proba(X({k: v[hold] for k, v in S.items()}))[:, 1]; yh = S['label'][hold]
    q = np.quantile(ph, np.linspace(0, 1, 11)); m1 = []
    for lo, hi in zip(q[:-1], q[1:]):
        s = (ph >= lo) & (ph <= hi)
        m1.append(dict(pred=round(float(ph[s].mean()), 4), obs=round(float(yh[s].mean()), 4), n=int(s.sum())))
    # deciles of all rows are dominated by w ~ 0; also report deciles among rows with w > 0.05
    s5 = ph > 0.05; q5 = np.quantile(ph[s5], np.linspace(0, 1, 11)); m1b = []
    for lo, hi in zip(q5[:-1], q5[1:]):
        s = s5 & (ph >= lo) & (ph <= hi)
        m1b.append(dict(pred=round(float(ph[s].mean()), 4), obs=round(float(yh[s].mean()), 4), n=int(s.sum())))
    rep['M1'] = dict(all_deciles=m1, deciles_w_gt_0.05=m1b,
                     pass_=bool(all(abs(r['pred'] - r['obs']) <= 0.03 for r in m1 + m1b)))
    # M2
    wN = clf.predict_proba(X(N))[:, 1]; m2 = []
    for lo, hi in zip(V_EDGES[:-1], V_EDGES[1:]):
        if lo < 21.5:
            continue
        s = (N['vis'] >= lo) & (N['vis'] < hi); n = int(N['label'][s].sum()); sw = float(wN[s].sum())
        tol = max(0.1*n, 3*np.sqrt(max(n, 1)))
        m2.append(dict(vis=f'{lo:.1f}-{hi:.1f}', n_selected=n, sum_w=round(sw, 1), pass_=bool(abs(sw - n) <= tol)))
    rep['M2'] = dict(bins=m2, pass_=bool(all(r['pass_'] for r in m2)))
    # M3
    m3, ok3 = colour_check(N, wN)
    rep['M3'] = dict(bins=m3, pass_=ok3)
    # completeness of point sources (step 4 input)
    rep['completeness_point'] = {}
    for f, d in (('S', S), ('N', N)):
        w = clf.predict_proba(X(d))[:, 1]; pt = d['point_like_prob'] > 0.9; cc = []
        for lo in np.arange(21.0, 26.51, 0.25)[:-1]:
            s = pt & (d['vis'] >= lo) & (d['vis'] < lo + 0.25)
            cc.append(dict(vis=round(lo + 0.125, 3), n=int(s.sum()), C_model=round(float(w[s].mean()), 4) if s.any() else None,
                           C_observed=round(float(d['label'][s].mean()), 4) if s.any() else None))
        rep['completeness_point'][f] = cc
    OUT.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ('fields', 'n_train', 'n_iter')}, indent=1))
    print('M1', rep['M1']['pass_']); [print('  ', r) for r in m1 + m1b]
    print('M2', rep['M2']['pass_']); [print('  ', r) for r in m2]
    print('M3', ok3)
    for r in m3:
        print('  ', r['vis'], r['n_selected'], r['sum_w'], {k: (v['d_median'], v['d_halfwidth_frac']) for k, v in r['colours'].items()})
    print('completeness (N, point_like_prob > 0.9):', [(r['vis'], r['C_model'], r['C_observed']) for r in rep['completeness_point']['N']])


if __name__ == '__main__':
    main()
