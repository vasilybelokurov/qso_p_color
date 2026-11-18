#!/usr/bin/env python
"""Build the stage-1 Euclid bundle (selection 'euclid_point') of one promoted bundle (docs/euclid/PLAN.md 2.2-2.4).

1. QSO extension: candidate with the highest held-out (select-cell QSO) Euclid marginal density (fit_report.json).
2. Background extension: candidate with the highest Euclid marginal density of select-cone background rows
   (deployment selection, known QSOs removed; sky-gate weights).
3. Fallback window: Legacy r - VIS = c0 + c1 (VIS - Y) + c2 (VIS - Y)^2 fitted on fit-role rows (QSO fit cells +
   background training rows, Legacy r S/N >= 10, dereddened luptitudes); half-widths = residual quantiles
   (0.05%, 99.95%) pooled over both populations, recorded per population. Eligibility: window centre <= 23.0 and
   posterior Legacy-r mass above 23.5 (training range) <= 1%.
4. Bundle: thinning f_k(r), Sigma_Q x c_Q (scripts/euclid/selection.py), parent catch-all fraction and support.

    python scripts/euclid/build_stage1.py --bundle 20261005_magdep_qlf
"""
import argparse
import glob
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cones import deployment                                    # noqa: E402
from fit_stage1 import BASE, EUCLID_BANDS, euclid_marginal_logp, gate_log_weights, load_rows, operator  # noqa: E402
from qso_pcolor.euclid_bundle import EXTINCTION_R, build_bundle, extend_layout, luptitudes  # noqa: E402
from qso_pcolor.euclid_extension import Extension, extend_mixture, regressor_matrix  # noqa: E402
from qso_pcolor.extinction import sfd_ebv  # noqa: E402
from qso_pcolor.gaussmix import GaussianMixture  # noqa: E402
from qso_pcolor.multisurvey import MultiSurveyModel  # noqa: E402

MODELS = Path('models/multisurvey_psf'); WORK = MODELS/'work/euclid'
R_MAX = 23.5


def cone_rows(role, settings):
    out = []
    for p in sorted(glob.glob(str(BASE/'cones'/f'{role}_*.npz'))):
        r, _ = deployment(p, settings); r['cone'] = np.full(len(r['e_ra']), int(Path(p).stem.split('_')[1])); out.append(r)
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def euclid_y(rows, settings, l, b):
    from qso_pcolor.euclid_bundle import photometry
    p = photometry(dict(rows, matched=np.ones(len(l), bool)))
    fac = 10**(0.4*np.outer(sfd_ebv(l, b), [EXTINCTION_R[x] for x in EUCLID_BANDS]))
    return luptitudes(p.flux*fac, p.variance*fac**2, np.asarray(settings['softening']))


def galactic(ra, dec):
    from astropy.coordinates import SkyCoord; import astropy.units as u
    g = SkyCoord(ra=ra*u.deg, dec=dec*u.deg).galactic
    return g.l.deg, g.b.deg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True)
    a = ap.parse_args()
    s = json.loads(Path('docs/euclid/settings.json').read_text())
    root = MODELS/a.bundle; latent = json.loads((root/'latent.json').read_text())
    model = MultiSurveyModel.load(root/'model.json'); st = WORK/f'stage1_{a.bundle}'
    rep = json.loads((st/'fit_report.json').read_text())['results']
    labels = latent['layout']['latent_labels']; r_index = labels.index('legacy:r')
    magdep = 'qso_coordinates' not in latent['layout']
    t_q = regressor_matrix(labels, ('legacy:g', 'legacy:z', 'legacy:w1', 'legacy:w2'), 'legacy:r', include_magnitude=magdep)
    t_b = regressor_matrix(labels, ('legacy:g', 'legacy:z', 'legacy:w1', 'legacy:w2'), 'legacy:r', include_magnitude=False)
    qbest = max((k for k in rep if k.startswith('qso:')), key=lambda k: rep[k]['select_mean_logp'])[4:]
    qext = [Extension.from_dict(e) for e in json.loads((st/f'qso_{qbest}.json').read_text())['per_slice']]
    lay = extend_layout(latent['layout'], s['softening'])
    # background candidate on select cones
    sel = cone_rows('select', s); bgs = ~sel['known_qso'] & ~sel['likely_qso']
    l, b = galactic(sel['e_ra'][bgs], sel['e_dec'][bgs])
    ey, en = euclid_y({k: v[bgs] for k, v in sel.items()}, s, l, b)
    y = np.concatenate([np.zeros((len(l), 10)), np.nan_to_num(ey)], 1); noise = np.concatenate([np.ones((len(l), 10)), np.where(en > 0, en, 1.)], 1)
    obs = np.concatenate([np.zeros((len(l), 10), bool), np.isfinite(ey) & (en > 0)], 1)
    h, off, t = operator(lay, 'stars'); lw = gate_log_weights(model, l, b)
    bscore = {}
    for k in [k for k in rep if k.startswith('bkg:')]:
        e = Extension.from_dict(json.loads((st/f'bkg_{k[4:]}.json').read_text()))
        mx = extend_mixture(GaussianMixture.from_dict(latent['background']), e, t_b, r_index, EUCLID_BANDS)
        v = euclid_marginal_logp(mx, h, off, t, y, noise, obs, log_w=lw)
        bscore[k[4:]] = float(np.mean(v[np.isfinite(v)]))
    bbest = max(bscore, key=bscore.get)
    bext = Extension.from_dict(json.loads((st/f'bkg_{bbest}.json').read_text()))
    # window: Legacy r - VIS as a quadratic in VIS - Y on fit-role rows
    resid_by = {}; X = []; Y = []; pop = []
    roles = {c['cell']: c['role'] for c in json.loads(Path('docs/euclid/sky_roles.json').read_text())['cells']}
    for name in ('qso', 'background'):
        d = load_rows(name, s)
        use = d['clean'].copy()
        if name == 'qso':
            use &= np.array([roles.get(int(c)) == 'fit' for c in d['euclid_cell']])
        r_ = np.where(d['observed'][:, 1], d['y'][:, 1], np.where(d['observed'][:, 6], d['y'][:, 6], np.nan))
        rn = np.where(d['observed'][:, 1], d['noise'][:, 1], d['noise'][:, 6])
        use &= np.isfinite(r_) & (1.0857/np.sqrt(np.where(rn > 0, rn, np.inf)) >= 10) & d['observed'][:, 10] & d['observed'][:, 11]
        X.append(d['y'][use, 10] - d['y'][use, 11]); Y.append(r_[use] - d['y'][use, 10]); pop += [name]*int(use.sum())
    X = np.concatenate(X); Y = np.concatenate(Y); pop = np.array(pop)
    coef = np.polyfit(X, Y, 2)[::-1]; res = Y - np.polyval(coef[::-1], X)
    lo, hi = float(-np.quantile(res, 0.0005)), float(np.quantile(res, 0.9995))
    for name in ('qso', 'background'):
        rr = res[pop == name]
        resid_by[name] = dict(n=int(len(rr)), q0005=float(np.quantile(rr, .0005)), q9995=float(np.quantile(rr, .9995)),
                              nmad=float(1.4826*np.median(np.abs(rr - np.median(rr)))), median=float(np.median(rr)))
    window = dict(colour=['euclid:vis', 'euclid:y'], coef=coef.tolist(), lo=lo, hi=hi, r_max=R_MAX, centre_max=23.0,
                  post_mass_max=0.01,
                  nisp=dict(bands=['euclid:y', 'euclid:j', 'euclid:h'], min_bands=2, min_snr=5.))
    fallback = dict(bands=['ps1:r', 'nsc:r', 'skymapper:r', 'euclid:vis'], n_grid=81, windows={'euclid:vis': window},
                    rule='Legacy r integrated over given Euclid VIS (docs/euclid/PLAN.md 2.4)')
    selj = json.loads((WORK/f'selection_{a.bundle}'/'selection.json').read_text())
    qtab = load_rows('qso', s); infit = np.array([roles.get(int(c)) == 'fit' for c in qtab['euclid_cell']]) & qtab['clean']
    zc = np.asarray(model.qso.z_centres); near = np.argmin(np.abs(qtab['zspec'][infit][:, None] - zc), 1)
    counts = np.zeros((len(zc), 4), int)
    for j in range(4):
        counts[:, j] = np.bincount(near[qtab['observed'][infit][:, 10 + j]], minlength=len(zc))
    bounds = [[float(np.nanpercentile(qtab['y'][qtab['clean']][:, 10 + j], .1)) - 1, 27.] for j in range(4)]
    out = build_bundle(root, MODELS/f'euclid_stage1_{a.bundle}', qso_ext=qext, bkg_ext=bext, t_d_qso=t_q, t_d_bkg=t_b,
                       softening_euclid=s['softening'], bounds_euclid=bounds,
                       selection=dict(name='euclid_point', thinning=selj['thinning'],
                                      rule='Euclid DR1 MER clean + isolated + point_like_prob >= p_min (docs/euclid/settings.json)'),
                       c_q=selj['c_q'], fallback=fallback, euclid_band_counts=counts,
                       meta=dict(stage=1, qso_candidate=qbest, bkg_candidate=bbest, bkg_select_logp=bscore,
                                 qso_select_logp={k: v['select_mean_logp'] for k, v in rep.items() if k.startswith('qso:')},
                                 window_residuals=resid_by, settings=s, plan='docs/euclid/PLAN.md'))
    print(json.dumps(dict(bundle=str(out), qso=qbest, bkg=bbest, bkg_scores=bscore, window=window, residuals=resid_by), indent=1))


if __name__ == '__main__':
    main()
