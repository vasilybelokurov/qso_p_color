#!/usr/bin/env python
"""Stage 1 of docs/euclid/PLAN.md: fit the frozen Euclid extension of one promoted bundle and write candidates.

    python scripts/euclid/fit_stage1.py --bundle 20261005_magdep_qlf --out models/multisurvey_psf/work/euclid/<tag>

Inputs: ~/data/qso_p_color/catalogues/euclid_dr1/tables/{qso,background}.npz (scripts/euclid/assemble.py),
docs/euclid/sky_roles.json, docs/euclid/settings.json (cleaning decisions from the data tests).
QSO training rows: matched, clean, point-like targets in Euclid FIT cells (all parent roles; the parent old block is
frozen, so parent roles do not matter for the extension). Background rows: the 33 parent cones overlapping DR1
(parent training material; every Euclid evaluation uses new cones only). Writes the fitted extensions per candidate,
held-out (select-cell QSOs) Euclid marginal densities, and the chosen candidate.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp

from qso_pcolor.euclid_bundle import EUCLID_BANDS, EXTINCTION_R, clean, extend_layout, luptitudes, point_like
from qso_pcolor.euclid_extension import Extension, extend_mixture, fit_extension, regressor_matrix
from qso_pcolor.gaussmix import GaussianMixture, log_gauss_batch
from qso_pcolor.multisurvey import MultiSurveyModel

BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
MODELS = Path('models/multisurvey_psf')
LEGACY = [f'decals_dr9_{h}:{b}' for h in ('south', 'north') for b in ('g', 'r', 'z', 'w1', 'w2')]
EDGES = np.array([15.5, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.5])


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_rows(name, settings, roles=None):
    """Arrays for the extension fit: y (N, 14) = Legacy native 10 + Euclid 4 (dereddened luptitudes), noise, observed."""
    d = dict(np.load(BASE/'tables'/f'{name}.npz', allow_pickle=False))
    m = {k: d[k] for k in ('vis_det', 'spurious_flag', 'det_quality_flag', 'point_like_prob')}
    m['matched'] = np.ones(len(d['ra']), bool); m['flux_vis_to_y_templfit'] = d['euclid_flux'][:, 0]/0.27542287033381663
    ok = clean(m, neighbours=d['neighbours'], bad_bits=settings['bad_bits'], vis_bright=settings['vis_bright'])
    ok &= point_like(m, settings['p_min'])
    r = np.array([EXTINCTION_R[b] for b in EUCLID_BANDS])
    fac = 10**(0.4*np.outer(d['ebv'], r))
    ef, ev = d['euclid_flux']*fac, d['euclid_variance']*fac**2*np.asarray(settings.get('error_scale', [1.]*4))**2
    ey, en = luptitudes(ef, ev, np.asarray(settings['softening']))
    y = np.concatenate([d['legacy_y'], ey], axis=1); noise = np.concatenate([d['legacy_noise'], en], axis=1)
    obs = np.isfinite(y) & (noise > 0)
    y = np.where(obs, y, 0.); noise = np.where(obs, noise, 1.)
    d.update(y=y, noise=noise, observed=obs, clean=ok)
    return d


def operator(layout_ext, kind):
    labels = layout_ext['native_labels']; cols = [labels.index(b) for b in LEGACY + list(EUCLID_BANDS)]
    op = layout_ext['operators'][kind]
    h = np.asarray(op['matrix'])[cols]; b = np.asarray(op['offset'])[cols]
    t = np.asarray(op['covariance'])[np.ix_(cols, cols)]
    return h, b, t


def euclid_marginal_logp(mix_ext: GaussianMixture, h, b, t, y, noise, obs, log_w=None):
    """log p(observed Euclid luptitudes) under an extended latent mixture (Legacy and other bands marginalised)."""
    e = np.arange(10, 14)
    out = np.full(len(y), np.nan)
    keys, inv = np.unique(obs[:, e], axis=0, return_inverse=True)
    for g, mask in enumerate(keys):
        rows = np.flatnonzero(inv.ravel() == g); cols = e[mask]
        if not len(cols):
            continue
        ho = h[cols]; mean = mix_ext.means @ ho.T + b[cols]; cov = ho @ mix_ext.covs @ ho.T + t[np.ix_(cols, cols)]
        tot = cov[None] + noise[np.ix_(rows, cols)][:, None, :, None]*np.eye(len(cols))
        lp = log_gauss_batch(y[np.ix_(rows, cols)][:, None, :], mean[None], tot)
        lw = np.log(mix_ext.weights)[None] if log_w is None else log_w[rows]
        out[rows] = logsumexp(lp + lw, axis=1)
    return out


def run_qso(latent, z_centres, layout_ext, d, train, *, t_d, r_index, h_z, kappa_a, b_zero, n_iter, init=None, kappa_w=np.inf):
    mixes = [GaussianMixture.from_dict(m) for m in latent['qso']]
    sizes = [m.n_components for m in mixes]; starts = np.cumsum([0] + sizes)
    big = GaussianMixture(np.concatenate([m.weights for m in mixes])/len(mixes), np.concatenate([m.means for m in mixes]),
                          np.concatenate([m.covs for m in mixes]), mixes[0].labels)
    groups = [np.arange(starts[i], starts[i+1]) for i in range(len(mixes))]
    z = d['zspec'][train]; idx = np.clip(np.searchsorted(z_centres, z) - 1, 0, len(z_centres) - 2)
    frac = np.clip((z - z_centres[idx])/(z_centres[idx+1] - z_centres[idx]), 0, 1)
    rows = np.concatenate([train, train]); slc = np.concatenate([idx, idx + 1]); w = np.concatenate([1 - frac, frac])
    keep = w > 0; rows, slc, w = rows[keep], slc[keep], w[keep]
    y, noise, obs = d['y'][rows], d['noise'][rows], d['observed'][rows]
    blocks = [(np.flatnonzero(slc == s), groups[s]) for s in range(len(mixes)) if (slc == s).any()]
    hw = np.where(z_centres > 3, np.minimum(h_z, 0.2), h_z)
    pool = np.exp(-0.5*((z_centres[None, :] - z_centres[:, None])/hw[:, None])**2)
    n_slice = np.bincount(slc, weights=w, minlength=len(mixes))
    kap = np.where(pool @ n_slice < 30, np.inf, kappa_a)
    h, b, t = operator(layout_ext, 'qso')
    ext, hist = fit_extension(big, y, noise, obs, h=h, offset=b, extra=t, t_d=t_d, r_index=r_index,
                              new_labels=EUCLID_BANDS, groups=groups, pooling=pool, kappa_a=kap, kappa_b=np.inf,
                              kappa_w=kappa_w, row_weights=w, blocks=blocks, n_iter=n_iter, tol=1e-4, b_zero=b_zero,
                              init=init, log=log)
    per = [Extension(ext.a[g], ext.b[g], ext.w[g], dict(ext.meta, slice=i)) for i, g in enumerate(groups)]
    return per, ext, hist, dict(rows_per_slice=n_slice.tolist())


def gate_log_weights(model: MultiSurveyModel, l, b):
    w, _ = model.spatial_background.evaluate(l, b)
    with np.errstate(divide='ignore'):
        return np.log(w)


def run_bkg(latent, model, layout_ext, d, train, *, t_d, r_index, kappa, b_zero, n_iter, init=None):
    mix = GaussianMixture.from_dict(latent['background'])
    rr = mix.means[:, r_index]; comp_bin = np.clip(np.searchsorted(EDGES, rr) - 1, 0, len(EDGES) - 2)
    groups = [np.flatnonzero(comp_bin == j) for j in range(len(EDGES) - 1)]
    y, noise, obs = d['y'][train], d['noise'][train], d['observed'][train]
    # each row's Legacy r: south if observed, else north
    rs = np.where(obs[:, 1], y[:, 1], y[:, 6]); rbin = np.clip(np.searchsorted(EDGES, rs) - 1, 0, len(EDGES) - 2)
    blocks = [(np.flatnonzero(rbin == j), np.flatnonzero(np.abs(comp_bin - j) <= 1)) for j in range(len(EDGES) - 1)
              if (rbin == j).any()]
    lw = gate_log_weights(model, d['l'][train], d['b'][train])
    h, b, t = operator(layout_ext, 'stars')
    # bins without training rows (r < 18 after the VIS bright limit) borrow the nearest populated bin's regression
    nbin = np.bincount(rbin, minlength=len(EDGES) - 1); pop = np.flatnonzero(nbin >= 200)
    pool = np.eye(len(EDGES) - 1)
    for j in np.flatnonzero(nbin < 200):
        pool[j] = 0.; pool[j, pop[np.argmin(np.abs(pop - j))]] = 1.
    ext, hist = fit_extension(mix, y, noise, obs, h=h, offset=b, extra=t, t_d=t_d, r_index=r_index,
                              new_labels=EUCLID_BANDS, groups=groups, pooling=pool, kappa_a=1., kappa_b=kappa, kappa_w=kappa,
                              blocks=blocks, row_log_weights=lw, n_iter=n_iter, tol=1e-4, b_zero=b_zero, init=init, log=log)
    return ext, hist, dict(rows_per_bin=nbin.tolist(), pooling=pool.tolist(),
                           components_per_bin=[len(g) for g in groups])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-iter', type=int, default=60)
    ap.add_argument('--only-qso', action='store_true', help='fit only QSO candidates missing from the report; keep the rest')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path('docs/euclid/settings.json').read_text())
    roles = json.loads(Path('docs/euclid/sky_roles.json').read_text())
    role_of = {c['cell']: c['role'] for c in roles['cells']}
    root = MODELS/a.bundle; latent = json.loads((root/'latent.json').read_text())
    model = MultiSurveyModel.load(root/'model.json')
    lay = extend_layout(latent['layout'], settings['softening'])
    labels = latent['layout']['latent_labels']; r_index = labels.index('legacy:r')
    magdep = 'qso_coordinates' not in latent['layout']
    t_q = regressor_matrix(labels, ('legacy:g', 'legacy:z', 'legacy:w1', 'legacy:w2'), 'legacy:r', include_magnitude=magdep,
                         colour_coordinates=not magdep)   # model A latent: (magnitude, colour) coordinates
    t_b = regressor_matrix(labels, ('legacy:g', 'legacy:z', 'legacy:w1', 'legacy:w2'), 'legacy:r', include_magnitude=False)
    q = load_rows('qso', settings); bg = load_rows('background', settings)
    qrole = np.array([role_of.get(int(c), 'none') for c in q['euclid_cell']])
    q_train = np.flatnonzero(q['clean'] & (qrole == 'fit')); q_sel = np.flatnonzero(q['clean'] & (qrole == 'select'))
    b_train = np.flatnonzero(bg['clean'])
    log('rows: QSO fit', len(q_train), 'select', len(q_sel), '; background', len(b_train), '; magdep', magdep)
    zc = np.asarray(model.qso.z_centres, float)
    h_q, b_q, t_qop = operator(lay, 'qso')
    old = json.loads((a.out/'fit_report.json').read_text())['results'] if (a.out/'fit_report.json').exists() else {}
    results = dict(old) if a.only_qso else {}
    for name, kw in settings['qso_candidates'].items():
        if a.only_qso and f'qso:{name}' in old:
            continue
        per, ext, hist, info = run_qso(latent, zc, lay, q, q_train, t_d=t_q, r_index=r_index, n_iter=a.n_iter, **kw)
        # held-out Euclid marginal density of select-cell QSOs at their slice (interpolation as in the scorer)
        zs = q['zspec'][q_sel]; idx = np.clip(np.searchsorted(zc, zs) - 1, 0, len(zc) - 2)
        f = np.clip((zs - zc[idx])/(zc[idx+1] - zc[idx]), 0, 1); lp = np.full(len(q_sel), -np.inf)
        for s in np.unique(np.r_[idx, idx + 1]):
            mx = extend_mixture(GaussianMixture.from_dict(latent['qso'][s]), per[s], t_q, r_index, EUCLID_BANDS)
            for which, wgt in ((idx == s, 1 - f), (idx + 1 == s, f)):
                rows = np.flatnonzero(which & (wgt > 0))
                if len(rows):
                    v = euclid_marginal_logp(mx, h_q, b_q, t_qop, q['y'][q_sel[rows]], q['noise'][q_sel[rows]], q['observed'][q_sel[rows]])
                    lp[rows] = np.logaddexp(lp[rows], np.log(wgt[rows]) + v)
        results[f'qso:{name}'] = dict(select_mean_logp=float(np.mean(lp[np.isfinite(lp)])), n_select=int(np.isfinite(lp).sum()),
                                      iterations=len(hist), final_train_loglike=hist[-1], **info)
        json.dump(dict(per_slice=[e.to_dict() for e in per]), open(a.out/f'qso_{name}.json', 'w'))
        log('QSO candidate', name, results[f'qso:{name}'])
    for name, kw in ({} if a.only_qso else settings['bkg_candidates']).items():
        ext, hist, info = run_bkg(latent, model, lay, bg, b_train, t_d=t_b, r_index=r_index, n_iter=a.n_iter, **kw)
        results[f'bkg:{name}'] = dict(iterations=len(hist), final_train_loglike=hist[-1], **info)
        json.dump(ext.to_dict(), open(a.out/f'bkg_{name}.json', 'w'))
        log('background candidate', name, results[f'bkg:{name}'])
    (a.out/'fit_report.json').write_text(json.dumps(dict(bundle=a.bundle, settings=settings, results=results), indent=1))


if __name__ == '__main__':
    main()
