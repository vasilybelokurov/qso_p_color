#!/usr/bin/env python
"""Validate the calibrated outside-both-models test (qso_pcolor.ood_calibration) against the fixed 4-sigma cut.

alpha = 2/256 is set a priori (as for the QSO-support cut) and checked on calibration-role objects (role 2;
random quasars and background objects above the faint limit) before the test panel is examined.

Test panel: the main performance panel (score_performance.py; held-out role 3; both promoted bundles). Only
the outside-both-models decision changes, so p_Q and p_B are computed for every object; objects whose flag
clears are rescored through the scorer with ``ood_calibration`` (which also checks the integration), and
objects newly flagged become unranked. Metrics as in the method note: AUC on log R with unranked objects at
the bottom; incidence = background objects ranked with p_quasar > 0.5; QSO recall = ranked with p > 0.5.

Writes docs/method_unified/ood_calibration_validation.json and plots/pquasar_diagnostics/P9_calibrated_ood.png.
Usage: nice python scripts/method_unified/validate_ood_calibration.py [--workers 3]
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor import Photometry
from run_unified_pilot import arrays

ROOT = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')
PERF = Path('models/multisurvey_psf/work/method_unified/performance_stellar_binned')
WORK = Path('models/multisurvey_psf/work/ood_calibration')
POINTERS = {'dependent': 'models/multisurvey_psf/current', 'independent': 'models/multisurvey_psf/current_magindep'}
NAME = {'dependent': 'A (magnitude-dependent)', 'independent': 'B (magnitude-independent)'}
COL = {'dependent': '#e8663d', 'independent': '#2a78d6'}
CFG = dict(alpha=2/256, draws=256, seed=20261005)
_s = {}


def _init(model):
    from qso_pcolor.unified import UnifiedPSFModel
    _s['m'] = UnifiedPSFModel.load(POINTERS[model]); _s['lab'] = tuple(json.loads((ROOT/'layout.json').read_text())['native_labels'])


def _pvalues(job):
    """p_Q, p_B for rows of one kind, from the prepared extinction-corrected arrays."""
    from qso_pcolor.gaussmix import GaussianMixture
    from qso_pcolor.ood_calibration import _components, row_seed, tail_pvalue
    kind, rows = job; um = _s['m']; m = um.base.model; d = arrays(ROOT, kind)
    phot = Photometry(np.asarray(d['flux_dered'][rows]), np.asarray(d['variance_dered'][rows]), _s['lab'])
    f = m.transform(phot); anchors = m.reference_indices(phot)
    cq = _s.setdefault('cq', _components(m.qso.mixtures))
    w_bg = m.spatial_background.evaluate(np.asarray(d['l'][rows], float), np.asarray(d['b'][rows], float))[0]
    out = np.full((len(rows), 2), np.nan)
    for i in range(len(rows)):
        if anchors[i] < 0 or not f.observed[i, anchors[i]]:
            continue
        rs = row_seed(CFG['seed'], f.x[i], f.cov[i], f.observed[i])
        out[i, 0] = tail_pvalue(cq, f.x[i], f.cov[i], f.observed[i], int(anchors[i]), draws=CFG['draws'], seed=rs)[0]
        cb = _components([GaussianMixture(w_bg[i], m.background.means, m.background.covs)], [1.])
        out[i, 1] = tail_pvalue(cb, f.x[i], f.cov[i], f.observed[i], int(anchors[i]), draws=CFG['draws'], seed=rs + 1)[0]
    return out


def _rescore(job):
    from qso_pcolor import BlendPolicy, RedshiftMatch
    kind, rows, z = job; um = _s['m']; d = arrays(ROOT, kind); cfg = json.loads(Path('configs/full_sample_release.json').read_text())
    phot = Photometry(np.asarray(d['flux'][rows]), np.asarray(d['variance'][rows]), _s['lab'])
    sc, dec = um.score(phot, ra_deg=np.asarray(d['ra'][rows]), dec_deg=np.asarray(d['dec'][rows]), z_primary=z,
                       morphology=['PSF']*len(rows), match=RedshiftMatch(half_width_kms=cfg['window_kms']),
                       blend_policy=BlendPolicy(**cfg['blend_policy']), ood_flag_sigma=cfg['ood_flag_sigma'],
                       separation_arcsec=cfg['fixture_separation_arcsec'], fracflux=cfg['fixture_fracflux'], ood_calibration=CFG)
    pq = um.quasar_probability(sc)
    return [(bool(e), np.nan if s is None else s.log_r_per_unit_z, p, np.nan if s is None else s.qso_ood_p,
             np.nan if s is None else s.bkg_ood_p) for e, s, p in zip(dec['eligible'], sc, pq)]


def pool_map(model, fn, jobs, workers):
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init, initargs=(model,)) as pool:
        return list(pool.map(fn, jobs))


def load_panel(model, kind):
    out = {}
    for f in sorted(glob.glob(str(PERF/f'{model}_{kind}_main_*_[0-9]*.npz'))):
        if f.endswith('.rows.npz'):
            continue
        s = np.load(f, allow_pickle=True); r = np.load(f[:-4] + '.rows.npz')
        for k, v in dict(rows=r['rows'], z=r['z'], status=s['status'].astype(str), support=s['support'], p_quasar=s['p_quasar'],
                         log_r=s['log_r_per_unit_z'], eligible=s['eligible']).items():
            out.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in out.items()}
    # promoted support threshold (2/256; the cached panel used 6/256)
    th = 2/256; base_ok = np.isin(out['status'], ['ok', 'qso_support_rejected'])
    out['old_ranked'] = base_ok & ~(np.isfinite(out['support']) & (out['support'] < th))
    out['support_ok'] = ~(np.isfinite(out['support']) & (out['support'] < th))
    return out


def chunks(kind, rows, n=200):
    return [(kind, rows[i:i+n]) for i in range(0, len(rows), n)]


def auc(rq, rb):
    return float(mannwhitneyu(rq, rb).statistic/(len(rq)*len(rb)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=3); a = ap.parse_args(); WORK.mkdir(parents=True, exist_ok=True)
    lab = json.loads((ROOT/'layout.json').read_text())['native_labels']
    rep = dict(definition=__doc__, config=CFG, calibration_role={}, test={})
    # -- calibration role: a-priori check of alpha --------------------------------------------------
    rng = np.random.default_rng(5)
    for kind in ('qso', 'stars'):
        d = arrays(ROOT, kind); role = np.asarray(d['role'])
        flag = np.load(ROOT/'stars'/'qso_flag.npy') if kind == 'stars' else np.zeros(len(role), bool)
        F, V = np.asarray(d['flux_dered']), np.asarray(d['variance_dered'])
        rs_, rn_ = lab.index('decals_dr9_south:r'), lab.index('decals_dr9_north:r')
        with np.errstate(invalid='ignore', divide='ignore'):
            snr = np.where(np.isfinite(V[:, rs_]) & (V[:, rs_] > 0), F[:, rs_]/np.sqrt(V[:, rs_]), F[:, rn_]/np.sqrt(V[:, rn_]))
        cand = np.flatnonzero((role == 2) & (np.nan_to_num(snr) >= 10) & ~flag)
        rows = np.sort(rng.choice(cand, min(4000, len(cand)), replace=False)); obs = np.asarray(d['observed'][rows]).astype(bool)
        for model in POINTERS:
            cache = WORK/f'calib_{model}_{kind}.npy'
            if cache.exists():
                p = np.load(cache)
            else:
                p = np.concatenate(pool_map(model, _pvalues, chunks(kind, rows), a.workers)); np.save(cache, p)
            out = (p[:, 0] < CFG['alpha']) & (p[:, 1] < CFG['alpha']); k = obs.sum(1)
            rep['calibration_role'].setdefault(model, {})[kind] = dict(
                n=int(len(rows)), outside_new=float(out.mean()),
                p_own_below_alpha=float(np.nanmean(p[:, 0 if kind == 'qso' else 1] < CFG['alpha'])),
                outside_new_k_lt16=float(out[k < 16].mean()), outside_new_k_ge20=float(out[k >= 20].mean()) if (k >= 20).any() else None)
            print('calib', model, kind, rep['calibration_role'][model][kind], flush=True)
    # -- test panel ----------------------------------------------------------------------------------
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4)); K = np.arange(4, 30)
    for model in POINTERS:
        res = {}
        for kind in ('qso', 'stars'):
            s = load_panel(model, kind); cache = WORK/f'test_{model}_{kind}.npy'
            if cache.exists():
                p = np.load(cache)
            else:
                p = np.concatenate(pool_map(model, _pvalues, chunks(kind, s['rows']), a.workers)); np.save(cache, p)
            new_out = (p[:, 0] < CFG['alpha']) & (p[:, 1] < CFG['alpha'])
            old_out = s['status'] == 'outside_both_models'
            # objects whose flag clears: rescore through the scorer (needs log R)
            clear = np.flatnonzero(old_out & ~new_out); cache2 = WORK/f'rescore_{model}_{kind}.npz'
            if cache2.exists():
                rr = dict(np.load(cache2))
            else:
                jobs = [(kind, s['rows'][clear[i:i+50]], s['z'][clear[i:i+50]]) for i in range(0, len(clear), 50)]
                out = [x for c in pool_map(model, _rescore, jobs, a.workers) for x in c]
                rr = dict(eligible=np.array([o[0] for o in out], bool), log_r=np.array([o[1] for o in out]), p=np.array([o[2] for o in out]),
                          pq=np.array([o[3] for o in out]), pb=np.array([o[4] for o in out]))
                np.savez(cache2, **rr)
            new_ranked = s['old_ranked'] & ~new_out
            log_r_new = np.where(s['old_ranked'], s['log_r'], np.nan); p_new = s['p_quasar'].copy()
            new_ranked[clear] = rr['eligible']; log_r_new[clear] = rr['log_r']; p_new[clear] = rr['p']
            log_r_new = np.where(new_ranked, log_r_new, -np.inf)
            log_r_old = np.where(s['old_ranked'], s['log_r'], -np.inf)
            k = np.asarray(arrays(ROOT, kind)['observed'][s['rows']]).astype(bool).sum(1)
            res[kind] = dict(s=s, p=p, new_out=new_out, old_out=old_out, new_ranked=new_ranked, k=k, log_r_new=log_r_new,
                             log_r_old=log_r_old, p_new=p_new, rr=rr, clear=clear)
            # integration check: scorer p-values vs offline ones for the rescored rows
            ok = np.isfinite(rr['pq'])
            rep['test'].setdefault(model, {})[kind] = dict(
                n=int(len(k)), old_unranked=float((~s['old_ranked']).mean()), new_unranked=float((~new_ranked).mean()),
                old_outside=float(old_out.mean()), new_outside=float(new_out.mean()), cleared=int(len(clear)),
                newly_flagged=int((~old_out & new_out & s['old_ranked']).sum()),
                rescored_still_unranked=int((~rr['eligible']).sum()),
                scorer_vs_offline_p_median_abs=float(np.median(np.abs(rr['pq'][ok] - p[clear][ok, 0]))) if ok.any() else None)
        q, b = res['qso'], res['stars']
        for tag, lq, lb, rq, rb, pq_, pb_ in (('old', q['log_r_old'], b['log_r_old'], q['s']['old_ranked'], b['s']['old_ranked'], q['s']['p_quasar'], b['s']['p_quasar']),
                                              ('new', q['log_r_new'], b['log_r_new'], q['new_ranked'], b['new_ranked'], q['p_new'], b['p_new'])):
            rep['test'][model][tag] = dict(auc=auc(lq, lb), recall=float((rq & (pq_ > .5)).mean()), incidence=float((rb & (pb_ > .5)).mean()),
                                           qso_unranked=float((~rq).mean()), bkg_unranked=float((~rb).mean()))
        print(model, json.dumps({k: rep['test'][model][k] for k in ('old', 'new')}), flush=True)
        print('   ', {kk: {x: y for x, y in v.items()} for kk, v in rep['test'][model].items() if kk in ('qso', 'stars')}, flush=True)
        for kind, ls in (('qso', '-'), ('stars', '--')):
            r = res[kind]; nm = 'test quasars' if kind == 'qso' else 'background objects'
            fo = [r['old_out'][r['k'] == kk].mean() if (r['k'] == kk).sum() >= 30 else np.nan for kk in K]
            fn = [r['new_out'][r['k'] == kk].mean() if (r['k'] == kk).sum() >= 30 else np.nan for kk in K]
            x = ax[0] if kind == 'qso' else ax[1]
            x.plot(K, 100*np.array(fo), 'o:', color=COL[model], ms=3, label=f'{NAME[model]}: fixed 4$\\sigma$')
            x.plot(K, 100*np.array(fn), 's-', color=COL[model], ms=3, label=f'{NAME[model]}: calibrated, $\\alpha$=2/256')
            x.set(xlabel='number of observed bands k', ylabel='outside both models (%)', title=nm); x.legend(frameon=False, fontsize=7)
            rep['test'][model][kind]['outside_by_k'] = dict(k=K.tolist(), old=fo, new=fn)
        pq = res['qso']['p'][:, 0]; pq = np.sort(pq[np.isfinite(pq)])
        ax[2].plot(pq, np.arange(1, len(pq) + 1)/len(pq), color=COL[model], label=f'{NAME[model]}: p$_Q$ of test quasars')
    ax[2].plot([0, 1], [0, 1], ':', color='k', lw=.8, label='uniform'); ax[2].set(xscale='log', yscale='log', xlim=(3e-3, 1), ylim=(1e-3, 1),
                                                                                   xlabel='p$_Q$', ylabel='fraction below', title='calibration of p$_Q$ on real quasars')
    ax[2].axvline(CFG['alpha'], ls='--', color='0.5', lw=.8); ax[2].legend(frameon=False, fontsize=7)
    fig.suptitle('P9. Outside both models: fixed 4$\\sigma$ cut against the calibrated test (held-out test panel)', fontsize=10)
    fig.tight_layout(); Path('plots/pquasar_diagnostics').mkdir(parents=True, exist_ok=True)
    fig.savefig('plots/pquasar_diagnostics/P9_calibrated_ood.png', dpi=140); plt.close(fig)
    Path('docs/method_unified/ood_calibration_validation.json').write_text(json.dumps(rep, indent=1, default=float))


if __name__ == '__main__':
    main()
