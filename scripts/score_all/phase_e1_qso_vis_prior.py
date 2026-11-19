#!/usr/bin/env python
"""Phase E step 1: quasar abundance per unit VIS, Sigma_Q(z, VIS), for the VIS-referenced faint model (design 2B).

Definition (docs/euclid/PLAN_score_all.md section 2):

    Sigma_Q(z, V) = \\int Sigma_Q(z, r) p_Q(V | r, z) dr

- Sigma_Q(z, r): the PARENT (Legacy-PSF, unthinned) QSO prior grid (priors.json of the bundle named in the Euclid
  model's meta.euclid.parent_bundle, anchor decals_dr9_south:r; eBOSS QLF, per deg^2 per unit z per mag; axis =
  Legacy South r luptitude, as passed by the scorer). Beyond the last grid centre (r ~ 25.66) log Sigma is
  extrapolated linearly in r with the slope of the last LAST_N centres (ASSUMED; the fraction of each VIS bin from
  extrapolated r is reported).
- c_Q(z, r) = P(Euclid point and clean | Legacy PSF QSO, z, r): the Euclid bundle's factor (its prior / the parent
  prior; scripts/euclid/selection.py, a logistic quadratic in r fitted on spectroscopic QSOs, almost all r < 23).
  Beyond R_CQ_MAX it is unmeasured (PLAN_score_all.md section 2), so three scenarios are carried:
    flat    c_Q(z, r > R_CQ_MAX) = c_Q(z, R_CQ_MAX)       (primary: no extrapolation of the fit)
    bundle  the bundle's own quadratic extrapolation     (what the promoted Euclid bundle uses; falls to ~1% at r 25.7)
    none    c_Q = 1                                      (upper bound: every faint QSO passes the point cut)
  First run (19 Nov) used the bundle prior directly; its faint-end fall is the quadratic extrapolation.
- p_Q(V | r, z): the conditional of the stage-1 QSO slice mixture (noise-free native luptitudes, labels
  decals_dr9_south:r and euclid:vis) on r, per z slice centre. Model B (magdep) carries an r-dependent colour by
  construction; model A (magindep) does not. Beyond the r range of the QSO training set the conditional is an
  extrapolation of the mixture (reported as the fraction of each VIS bin from r > R_TRAIN_MAX).
- V is the VIS luptitude (softening 0.0078 nmgy); it differs from the VIS magnitude by < 0.01 mag at V < 26.

Checks (pre-stated):
  C1 conservation: sum_V Sigma_Q(z, V) dV = sum_r Sigma_Q(z, r) dr within 1e-3 for V spanning 10-40.
  C2 bright-end reality check: model Sigma_Q(V) (flat c_Q), integrated over z, against the density of DESI DR1 QSOs
     in EDF-North (scripts/score_all/fetch_edfn_qsos.py) matched to deep MER and passing the cleaning and point-like
     cuts, VIS 19-21.5, restricted to probes with full DESI DR1 coverage (area = their number x 0.25 deg^2, coarse).
     DESI completeness makes known/model <= ~1; a gross mismatch (outside 0.3-1.5) stops the step. (The wide fit cones were the first
     choice and are unusable: most lie outside the DESI DR1 footprint, known/model = 0.09.)
Scenarios Sigma_Q x 0.5 and x 2 are uniform factors (logit p shift of -/+0.69) and are not tabulated separately.

Output: ~/data/qso_p_color/score_all/phase_e/sigma_q_vis_<A|B>.npz (z_centres, vis_edges, sigma [n_z, n_vis] per
deg^2 per unit z per mag, frac_r_extrap, frac_r_beyond_train) and docs/score_all/phase_e1_qso_vis_prior.json.
Light: one process, < 1 GB.

    nice -n 19 python scripts/score_all/phase_e1_qso_vis_prior.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path.home()/'data/qso_p_color/score_all/phase_e'
REPORT = ROOT/'docs/score_all/phase_e1_qso_vis_prior.json'
MODELS = {'B': 'current_euclid', 'A': 'current_euclid_magindep'}     # B magdep, A magindep (project convention)
R_BAND, V_BAND = 'decals_dr9_south:r', 'euclid:vis'
R_GRID = np.arange(14.0, 30.0001, 0.02)
V_EDGES = np.arange(10.0, 40.0001, 0.25)
LAST_N = 4
DESI_FULL = 200         # DESI targets per 0.1-deg probe marking full DR1 coverage (bimodal histogram)
R_CQ_MAX = 23.0         # faint end of the measured c_Q (selection.py fit data, almost all r < 23)
R_TRAIN_MAX = 23.5      # ASSUMED faint end of the QSO colour training set in Legacy r (PLAN_score_all.md table)


def sigma_r(prior: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sigma_Q(z, r) on R_GRID: log-linear interpolation inside, slope extrapolation beyond, zero below the grid."""
    zc = np.asarray(prior['z_centres']); mc = np.asarray(prior['mag_centres']); s = np.asarray(prior['sigma'])
    out = np.zeros((len(zc), len(R_GRID))); extrap = R_GRID > mc[-1]
    for j in range(len(zc)):
        ok = s[j] > 0
        ls = np.log(np.where(ok, s[j], np.nan))
        inside = (R_GRID >= mc[ok][0]) & (R_GRID <= mc[-1])
        out[j, inside] = np.exp(np.interp(R_GRID[inside], mc[ok], ls[ok]))
        slope = np.polyfit(mc[-LAST_N:], ls[-LAST_N:], 1)[0]
        out[j, extrap] = np.exp(ls[-1] + slope*(R_GRID[extrap] - mc[-1]))
    return zc, out, extrap


def p_v_given_r(mix, ir: int, iv: int) -> np.ndarray:
    """P(V in each V_EDGES bin | r) on R_GRID for one slice mixture (shape [n_r, n_v])."""
    from scipy.stats import norm
    w, mu, C = mix.weights, mix.means[:, [ir, iv]], mix.covs[:, [ir, iv]][:, :, [ir, iv]]
    crr, cvv, crv = C[:, 0, 0], C[:, 1, 1], C[:, 0, 1]
    # component responsibilities given r, conditional mean and sd of V
    with np.errstate(divide='ignore'):
        lw = np.log(w)
    lr = lw[None] + norm.logpdf(R_GRID[:, None], mu[None, :, 0], np.sqrt(crr)[None])
    resp = np.exp(lr - lr.max(1, keepdims=True)); resp /= resp.sum(1, keepdims=True)
    m = mu[None, :, 1] + (crv/crr)[None]*(R_GRID[:, None] - mu[None, :, 0])
    sd = np.sqrt(cvv - crv**2/crr)[None]
    cdf = norm.cdf((V_EDGES[None, None, :] - m[..., None])/sd[..., None])        # [n_r, K, n_edges]
    return np.einsum('rk,rkv->rv', resp, np.diff(cdf, axis=-1))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return np.log(p/(1 - p))


def c_q_grid(cq_tab: np.ndarray, mc: np.ndarray, scenario: str) -> np.ndarray:
    """c_Q(z, r) on R_GRID from the table on the prior centres, per scenario (see docstring)."""
    out = np.ones((cq_tab.shape[0], len(R_GRID)))
    if scenario == 'none':
        return out
    for j in range(cq_tab.shape[0]):
        lg = logit(cq_tab[j])
        if scenario == 'flat':
            r = np.minimum(R_GRID, R_CQ_MAX)
            out[j] = 1/(1 + np.exp(-np.interp(r, mc, lg)))
        else:
            slope = (lg[-1] - lg[-2])/(mc[-1] - mc[-2])
            v = np.interp(R_GRID, mc, lg); b = R_GRID > mc[-1]; v[b] = lg[-1] + slope*(R_GRID[b] - mc[-1])
            out[j] = 1/(1 + np.exp(-v))
    return out


def build(name: str) -> dict:
    from qso_pcolor.unified import UnifiedPSFModel
    root = ROOT/'models/multisurvey_psf'
    model = UnifiedPSFModel.load(root/name).base.model
    bundle = root/json.loads((root/name).read_text())['bundle']
    parent = ROOT/model.meta['euclid']['parent_bundle']
    pe = json.loads((bundle/'priors.json').read_text())['anchors'][R_BAND]['qso_prior']
    pp = json.loads((parent/'priors.json').read_text())['anchors'][R_BAND]['qso_prior']
    assert np.allclose(pe['mag_centres'], pp['mag_centres']) and np.allclose(pe['z_centres'], pp['z_centres'])
    se, sp = np.asarray(pe['sigma']), np.asarray(pp['sigma'])
    with np.errstate(invalid='ignore', divide='ignore'):
        cq_tab = np.where(sp > 0, se/sp, np.nan)
    cq_tab = np.where(np.isfinite(cq_tab), cq_tab, np.nanmedian(cq_tab, axis=1, keepdims=True))
    mc = np.asarray(pp['mag_centres'])
    zc, sig, extrap = sigma_r(pp)
    q = model.qso; ir, iv = q.labels.index(R_BAND), q.labels.index(V_BAND)
    assert np.allclose(q.z_centres, zc), 'prior and QSO slices must share z centres'
    dr = np.gradient(R_GRID); dv = np.diff(V_EDGES)
    out = dict(z_centres=zc, z_edges=np.asarray(pp['z_edges']), parent_bundle=str(parent.name))
    Ps = [p_v_given_r(q.mixtures[j], ir, iv) for j in range(len(zc))]      # [n_r, n_v] per slice
    for sc in ('flat', 'bundle', 'none'):
        cq = c_q_grid(cq_tab, mc, sc)
        S = np.zeros((len(zc), len(dv))); fx = np.zeros_like(S); ft = np.zeros_like(S); cons = []
        for j in range(len(zc)):
            w = sig[j]*cq[j]*dr
            lam = w[:, None]*Ps[j]
            tot = lam.sum(0); S[j] = tot/dv
            with np.errstate(invalid='ignore', divide='ignore'):
                fx[j] = np.where(tot > 0, lam[extrap].sum(0)/tot, np.nan)
                ft[j] = np.where(tot > 0, lam[R_GRID > R_TRAIN_MAX].sum(0)/tot, np.nan)
            cons.append(float(abs(tot.sum()/w.sum() - 1)) if w.sum() > 0 else 0.)
        out.update({f'sigma_{sc}': S, f'frac_r_extrap_{sc}': fx, f'frac_r_beyond_train_{sc}': ft,
                    f'conservation_{sc}': np.array(cons)})
    return out


def edfn_known(settings) -> tuple[np.ndarray, float]:
    """VIS (mag) of EDF-N DESI QSOs matched to deep MER passing cleaning + point-like cuts, and the deep area."""
    from qso_pcolor.euclid_bundle import clean, point_like
    d = dict(np.load(Path.home()/'data/qso_p_color/catalogues/euclid_dr1/edfn_qso/edfn_qso.npz', allow_pickle=False))
    m = {k[4:]: v for k, v in d.items() if k.startswith('mer_')}
    ok = np.asarray(m['matched'], bool)
    ok &= clean(m, bad_bits=settings['bad_bits'], vis_bright=settings['vis_bright']) & point_like(m, settings['p_min'])
    with np.errstate(invalid='ignore', divide='ignore'):
        vis = 23.9 - 2.5*np.log10(np.asarray(m['flux_vis_to_y_templfit'], float))
    # restrict to the part of EDF-N with full DESI DR1 coverage: probes (0.5-deg grid) with >= DESI_FULL DESI targets
    # within 0.1 deg (scripts/score_all/edfn_desi_coverage.py; the counts are bimodal, ~35% of probes full).
    # Coverage is decided from all DESI targets, not from the QSOs, so the check is not selected on its own outcome.
    cov = json.loads((Path.home()/'data/qso_p_color/catalogues/euclid_dr1/deep_coverage/edfn_desi_probes.json').read_text())
    pts = np.asarray(cov['pts']); full = np.asarray(cov['n_desi']) >= DESI_FULL
    u = lambda ra, de: np.c_[np.cos(np.deg2rad(de))*np.cos(np.deg2rad(ra)), np.cos(np.deg2rad(de))*np.sin(np.deg2rad(ra)),
                             np.sin(np.deg2rad(de))]
    from scipy.spatial import cKDTree
    ra_, de_ = np.nan_to_num(np.asarray(m['e_ra'], float)), np.nan_to_num(np.asarray(m['e_dec'], float))   # unmatched rows: ok is already False
    _, near = cKDTree(u(pts[:, 0], pts[:, 1])).query(u(ra_, de_))
    ok &= full[near]
    return vis[ok], float(full.sum()*0.25)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT/'docs/euclid/settings.json').read_text())
    census = json.loads((ROOT/'docs/score_all/census_fit.json').read_text())
    cb = census['bins']; cv = 0.5*(np.asarray(cb['vis_lo']) + np.asarray(cb['vis_hi']))
    bkg = np.asarray(cb['density_sel'], float)
    kvis, karea = edfn_known(settings)
    vc = 0.5*(V_EDGES[1:] + V_EDGES[:-1]); dv = np.diff(V_EDGES)
    rep = dict(definition='Sigma_Q(z,V) = int Sigma_Q(z,r) c_Q(z,r) p_Q(V|r,z) dr; see script docstring',
               r_cq_max=R_CQ_MAX, r_train_max_assumed=R_TRAIN_MAX, edfn_area_deg2=karea,
               extrapolation=f'log-linear slope of the last {LAST_N} prior centres beyond r = last centre', models={})
    for key, name in MODELS.items():
        d = build(name)
        np.savez_compressed(OUT/f'sigma_q_vis_{key}.npz', vis_edges=V_EDGES, **d)
        dz = np.diff(d['z_edges'])[:, None]
        sv = {sc: (d[f'sigma_{sc}']*dz).sum(0) for sc in ('flat', 'bundle', 'none')}     # per deg^2 per mag
        lam = d['sigma_flat']*dz*dv[None]
        w = lambda f: np.nansum(f*lam, 0)/np.maximum(lam.sum(0), 1e-300)
        sel = (vc >= 19) & (vc < 21.5)
        nk = float(((kvis >= 19) & (kvis < 21.5)).sum())
        ratio = (nk/karea)/float((sv['flat'][sel]*dv[sel]).sum())
        rows = []
        for lo in np.arange(19, 27, 0.5):
            b = (vc >= lo) & (vc < lo + 0.5); bc = (cv >= lo) & (cv < lo + 0.5)
            nq = {sc: float((sv[sc][b]*dv[b]).sum()) for sc in sv}; nb = float(bkg[bc].sum())
            rows.append(dict(vis=f'{lo:.1f}-{lo+0.5:.1f}', **{f'sigma_q_{sc}': round(v, 2) for sc, v in nq.items()},
                             edfn_desi_qso_per_deg2=round(float(((kvis >= lo) & (kvis < lo + .5)).sum()/karea), 2),
                             selected_per_deg2=round(nb, 1),
                             prior_frac_q_flat=round(nq['flat']/nb, 5) if nb > 0 else None,
                             frac_from_extrapolated_r=round(float(w(d['frac_r_extrap_flat'])[b].mean()), 3),
                             frac_from_r_beyond_train=round(float(w(d['frac_r_beyond_train_flat'])[b].mean()), 3)))
        cons = max(float(d[f'conservation_{sc}'].max()) for sc in sv)
        rep['models'][key] = dict(bundle_pointer=name, parent_bundle=d['parent_bundle'], C1_max_conservation_error=cons,
                                  C1_pass=bool(cons < 1e-3), C2_edfn_known_over_model_vis19_21p5=round(ratio, 3),
                                  C2_n_known=int(nk), C2_within_0p3_1p5=bool(0.3 <= ratio <= 1.5), per_half_mag=rows)
    REPORT.write_text(json.dumps(rep, indent=1))
    for k, v in rep['models'].items():
        print(f"model {k}: C1 max err {v['C1_max_conservation_error']:.1e}  C2 EDF-N known/model {v['C2_edfn_known_over_model_vis19_21p5']}"
              f" (n={v['C2_n_known']}, area {karea} deg2)")
        print('  VIS       SigQ_flat SigQ_bund SigQ_none  EDFN_DESI  selected  fQ_flat  f_extrap f_r>train')
        for r in v['per_half_mag']:
            print(f"  {r['vis']}  {r['sigma_q_flat']:8.2f} {r['sigma_q_bundle']:8.2f} {r['sigma_q_none']:8.2f}  {r['edfn_desi_qso_per_deg2']:8.2f}"
                  f"  {r['selected_per_deg2']:8.1f}  {r['prior_frac_q_flat'] if r['prior_frac_q_flat'] is not None else float('nan'):7.4f}"
                  f"  {r['frac_from_extrapolated_r']:6.3f}  {r['frac_from_r_beyond_train']:6.3f}")


if __name__ == '__main__':
    main()
