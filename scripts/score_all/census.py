#!/usr/bin/env python
"""Phase B census of the Euclid point-source deployment selection (docs/euclid/PLAN_score_all.md, Phase B).

Input: the cones of one role (default select; fit and eval cones from docs/score_all/new_cones.json), each r = 0.3 deg, cached by
scripts/euclid/fetch_cones.py; deployment selection as in scripts/euclid/cones.py (cleaning cuts, isolation,
point-like). Per VIS bin of 0.25 mag (VIS = 22.5 - 2.5 log10 f, f in nanomaggies, not dereddened):
  density_sel   selected sources per deg^2            density_all  all MER rows (vis_det=1, spurious_flag=0) per deg^2
  snr_<band>    median S/N in VIS, Y, J, H            f_legacy     fraction with a Legacy DR9 source within 1"
  f_r_snr5/10   fraction with Legacy r S/N >= 5 / 10  plp_median   median point_like_prob
  known_qso     known quasars (DESI DR1 + DR16Q within 1")
Standard errors: bootstrap over cones (1000 resamples), which carries the sky-to-sky scatter. Area: the full cone
area pi r^2 per cone (ASSUMED fully covered: sky_roles.py placed cone centres where the whole cone is covered; no
star-mask correction). The turnover VIS is the bin centre of maximum selected density.

Output: docs/score_all/census_<role>.json and plots/score_all/census_<role>.png.

    python scripts/score_all/census.py [--role fit]
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/euclid'))
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/cones'
OUT_TMPL = 'docs/score_all/census_{role}.json'
EDGES = np.arange(17.0, 27.51, 0.25)
BANDS = ('vis', 'y', 'j', 'h')


def vis_mag(flux_ujy):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 23.9 - 2.5*np.log10(np.asarray(flux_ujy, float))


def per_cone(path, settings):
    from cones import deployment
    from qso_pcolor.euclid_bundle import photometry
    c = dict(np.load(path, allow_pickle=False)); R = float(c['radius_deg']); ra0, dec0 = c['centre']
    area = np.pi*R**2
    # all MER rows inside the cone (fetch kept vis_det = 1, spurious_flag = 0)
    m = {k[4:]: v for k, v in c.items() if k.startswith('mer_')}
    d = np.deg2rad
    cosd = np.sin(d(dec0))*np.sin(d(m['e_dec'])) + np.cos(d(dec0))*np.cos(d(m['e_dec']))*np.cos(d(m['e_ra'] - ra0))
    inside = cosd >= np.cos(d(R))
    p_all = photometry(dict({k: np.asarray(v)[inside] for k, v in m.items()}, matched=np.ones(int(inside.sum()), bool)))
    rows, _ = deployment(path, settings)
    p = photometry(dict(rows, matched=np.ones(len(rows['e_ra']), bool)))
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = p.flux/np.sqrt(p.variance)
        lsn = rows['legacy_flux_r']*np.sqrt(rows['legacy_flux_ivar_r'])
    vis = 22.5 - 2.5*np.log10(np.where(p.flux[:, 0] > 0, p.flux[:, 0], np.nan))
    vis_all = 22.5 - 2.5*np.log10(np.where(p_all.flux[:, 0] > 0, p_all.flux[:, 0], np.nan))
    return dict(area=area, vis=vis, vis_all=vis_all, snr=snr, f_legacy=rows['legacy_index'] >= 0,
                lsn=np.nan_to_num(lsn, nan=-np.inf), plp=np.asarray(rows['point_like_prob'], float),
                known_qso=np.asarray(rows['known_qso'], bool))


def bin_stats(cones, idx):
    """Statistics per VIS bin for the cones in ``idx`` (pooled rows, total area)."""
    area = sum(cones[i]['area'] for i in idx)
    cat = lambda k: np.concatenate([cones[i][k] for i in idx])
    vis, vis_all = cat('vis'), cat('vis_all'); snr = np.concatenate([cones[i]['snr'] for i in idx])
    fl, lsn, plp, kq = cat('f_legacy'), cat('lsn'), cat('plp'), cat('known_qso')
    b = np.digitize(vis, EDGES) - 1; ba = np.digitize(vis_all, EDGES) - 1; nb = len(EDGES) - 1
    out = dict(density_sel=np.bincount(b[(b >= 0) & (b < nb)], minlength=nb)/area,
               density_all=np.bincount(ba[(ba >= 0) & (ba < nb)], minlength=nb)/area)
    for name in ('f_legacy', 'f_r_snr5', 'f_r_snr10', 'plp_median', 'known_qso') + tuple(f'snr_{x}' for x in BANDS):
        out[name] = np.full(nb, np.nan)
    for k in range(nb):
        s = b == k
        if s.sum() < 5:
            continue
        out['f_legacy'][k] = fl[s].mean(); out['f_r_snr5'][k] = (lsn[s] >= 5).mean()
        out['f_r_snr10'][k] = (lsn[s] >= 10).mean(); out['plp_median'][k] = np.nanmedian(plp[s])
        out['known_qso'][k] = kq[s].sum()
        for j, x in enumerate(BANDS):
            out[f'snr_{x}'][k] = np.nanmedian(snr[s, j])
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--role', default='select', help='cone role prefix: select (default), fit, eval, ...')
    role = ap.parse_args().role
    out_path = ROOT/OUT_TMPL.format(role=role)
    settings = json.loads((ROOT/'docs/euclid/settings.json').read_text())
    paths = sorted(glob.glob(str(BASE/f'{role}_*.npz')))
    cones = [per_cone(p, settings) for p in paths]
    n = len(cones); full = bin_stats(cones, range(n))
    rng = np.random.default_rng(20261119); boots = [bin_stats(cones, rng.integers(0, n, n)) for _ in range(1000)]
    se = {k: np.nanstd(np.array([bb[k] for bb in boots]), axis=0) for k in full}
    cen = 0.5*(EDGES[1:] + EDGES[:-1])
    turn = float(cen[np.argmax(full['density_sel'])])
    area = float(sum(c['area'] for c in cones)); nsel = int(sum(np.isfinite(c['vis']).sum() for c in cones))
    tot = lambda lo, hi: float(full['density_sel'][(cen >= lo) & (cen < hi)].sum())
    report = dict(
        cones=[Path(p).name for p in paths], area_deg2=area, n_selected=nsel,
        assumption='area = pi r^2 per cone (whole cone covered by construction; no star mask)',
        vis_turnover_selected=turn,
        density_selected_by_range={f'{lo}-{hi}': tot(lo, hi) for lo, hi in ((18, 22), (22, 23), (23, 24), (24, 25), (25, 27.5))},
        bins=dict(vis_lo=EDGES[:-1].tolist(), vis_hi=EDGES[1:].tolist(),
                  **{k: np.round(v, 5).tolist() for k, v in full.items()},
                  **{f'se_{k}': np.round(v, 5).tolist() for k, v in se.items()}))
    out_path.write_text(json.dumps(report, indent=1, allow_nan=True))
    plot(cen, full, se, turn, role)
    print(json.dumps({k: report[k] for k in ('area_deg2', 'n_selected', 'vis_turnover_selected', 'density_selected_by_range')}, indent=1))


def plot(cen, full, se, turn, role='select'):
    sys.path.insert(0, str(ROOT/'scripts/method_unified'))
    import matplotlib.pyplot as plt
    from common import save_figure, use_paper_style
    use_paper_style()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].errorbar(cen, full['density_all'], se['density_all'], fmt='.-', color='#888', label='all MER rows', ms=3)
    ax[0].errorbar(cen, full['density_sel'], se['density_sel'], fmt='.-', color='k', label='deployment selection', ms=3)
    ax[0].axvline(turn, ls=':', color='k'); ax[0].set(yscale='log', xlabel='VIS [mag]', ylabel='sources / deg$^2$ / 0.25 mag')
    ax[0].legend(frameon=False, fontsize=7)
    for k, lab, c in (('f_legacy', 'Legacy match', '#1b6ca8'), ('f_r_snr5', 'Legacy r S/N$\\geq$5', '#e08a2c'),
                      ('f_r_snr10', 'Legacy r S/N$\\geq$10', '#b5342c')):
        ax[1].errorbar(cen, full[k], se[k], fmt='.-', color=c, label=lab, ms=3)
    ax[1].set(xlabel='VIS [mag]', ylabel='fraction of selected', ylim=(0, 1.02)); ax[1].legend(frameon=False, fontsize=7)
    for x, c in zip(BANDS, ('k', '#1b6ca8', '#e08a2c', '#b5342c')):
        ax[2].plot(cen, full[f'snr_{x}'], '.-', color=c, label=x.upper(), ms=3)
    ax[2].axhline(10, ls=':', color='#888'); ax[2].axhline(5, ls=':', color='#888')
    ax[2].set(yscale='log', xlabel='VIS [mag]', ylabel='median S/N'); ax[2].legend(frameon=False, fontsize=7)
    fig.tight_layout(); print(save_figure(fig, f'score_all/census_{role}'))


if __name__ == '__main__':
    main()
