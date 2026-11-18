#!/usr/bin/env python
"""Data tests T-D1 (QSO side), T-D3, T-D4, T-D5, T-D6, T-D7 of docs/euclid/PLAN.md on the assembled tables.

Stellar locus: for background rows, each Euclid colour (VIS-Y, Y-J, J-H; dereddened luptitudes) is predicted from
Legacy colours (g-r, r-z, z-W1) by a cubic polynomial fitted (least squares, 3 sigma-clipping passes) to clean,
isolated, point-like rows with every band at S/N >= 30 and no det_quality_flag bit set. Residual = data - locus.
Writes docs/euclid/data_tests.json.
"""
from itertools import combinations_with_replacement
import json
from pathlib import Path
import numpy as np

BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/tables'
A = 2.5/np.log(10)
R = np.array([1.917, 0.921, 0.629, 0.420])


def mags(t, deredden=True):
    f = t['euclid_flux']*(10**(0.4*np.outer(t['ebv'], R)) if deredden else 1.)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = 22.5 - 2.5*np.log10(np.where(f > 0, f, np.nan))
        snr = t['euclid_flux']/np.sqrt(t['euclid_variance'])
    return m, snr


def legacy(t):
    y, n = t['legacy_y'], t['legacy_noise']
    south = n[:, 1] > 0
    pick = lambda j: np.where(south, y[:, j], y[:, 5 + j])
    snr = lambda j: np.where(south, A/np.sqrt(np.where(n[:, j] > 0, n[:, j], np.nan)),
                             A/np.sqrt(np.where(n[:, 5 + j] > 0, n[:, 5 + j], np.nan)))
    return {b: pick(j) for j, b in enumerate('grzWV')}, {b: snr(j) for j, b in enumerate('grzWV')}


def design(x):
    cols = [np.ones(len(x))]
    for deg in (1, 2, 3):
        for c in combinations_with_replacement(range(x.shape[1]), deg):
            cols.append(np.prod(x[:, c], axis=1))
    return np.column_stack(cols)


def nmad(x):
    x = x[np.isfinite(x)]
    return float(1.4826*np.median(np.abs(x - np.median(x)))) if len(x) > 10 else np.nan


if __name__ == '__main__':
    bg = dict(np.load(BASE/'background.npz')); q = dict(np.load(BASE/'qso.npz'))
    out = {}
    m, snr = mags(bg); L, Ls = legacy(bg)
    x = np.column_stack([L['g'] - L['r'], L['r'] - L['z'], L['z'] - L['W']])
    euc = np.column_stack([m[:, 0] - m[:, 1], m[:, 1] - m[:, 2], m[:, 2] - m[:, 3]])
    base = (bg['vis_det'] == 1) & (bg['spurious_flag'] == 0) & (bg['neighbours'] == 0)
    pt = np.nan_to_num(bg['point_like_prob']) >= 0.3
    hi = base & pt & (bg['det_quality_flag'] == 0) & (snr >= 30).all(1) & np.isfinite(x).all(1) & np.isfinite(euc).all(1)
    hi &= np.all([Ls[b] >= 30 for b in 'grzW'], axis=0)
    X = design(x); resid = np.full_like(euc, np.nan)
    for j in range(3):
        use = hi.copy()
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(X[use], euc[use, j], rcond=None)
            r = euc[:, j] - X @ coef; s = nmad(r[use]); use = hi & (np.abs(r) < 3*s)
        resid[:, j] = r
    out['locus'] = dict(n_fit=int(hi.sum()), nmad=[nmad(resid[hi, j]) for j in range(3)], colours=['VIS-Y', 'Y-J', 'J-H'])
    vis = m[:, 0]
    # T-D3: per bit, residual scatter at matched S/N vs bit-free rows; QSO retention per bit and VIS
    bits = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    ok3 = base & pt & np.isfinite(resid).all(1) & (snr[:, 1:] >= 10).all(1)
    snr_bins = [10, 20, 50, 100, 1e9]; d3 = {}
    clean = ok3 & (bg['det_quality_flag'] == 0)
    for bit in bits:
        has = ok3 & ((bg['det_quality_flag'] & bit) > 0)
        if has.sum() < 50:
            continue
        ratios = []
        for lo, hi_ in zip(snr_bins[:-1], snr_bins[1:]):
            a_ = has & (snr[:, 1] >= lo) & (snr[:, 1] < hi_); c_ = clean & (snr[:, 1] >= lo) & (snr[:, 1] < hi_)
            if a_.sum() >= 30 and c_.sum() >= 30:
                ratios.append(max(nmad(resid[a_, j])/nmad(resid[c_, j]) for j in range(3)))
        qv = 22.5 - 2.5*np.log10(np.where(q['euclid_flux'][:, 0] > 0, q['euclid_flux'][:, 0], np.nan))
        qret = {f'{lo}-{lo+2}': float(np.mean((q['det_quality_flag'][(qv >= lo) & (qv < lo + 2)] & bit) == 0))
                for lo in (15, 17, 19, 21, 23)}
        d3[str(bit)] = dict(n_background=int(has.sum()), max_scatter_ratio=float(max(ratios)) if ratios else None,
                            qso_kept_if_cut=qret)
    out['T-D3_bits'] = d3
    # flag_band = 2 and blended_prob
    out['T-D3_blended_prob'] = {f'>{t}': float(np.nanmean(bg['blended_prob'][ok3] > t)) for t in (0.1, 0.3, 0.5)}
    # T-D5: bright limit, median residual by VIS
    d5 = {}
    for lo in np.arange(14., 22., 0.5):
        s = base & pt & np.isfinite(resid).all(1) & (vis >= lo) & (vis < lo + 0.5)
        if s.sum() >= 30:
            d5[f'{lo:.1f}'] = dict(n=int(s.sum()), median=[float(np.median(resid[s, j])) for j in range(3)],
                                   nmad=[nmad(resid[s, j]) for j in range(3)])
    out['T-D5_bright'] = d5
    # T-D6: per-tile median residual (tiles with >= 200 locus rows)
    use = base & pt & np.isfinite(resid).all(1) & (snr[:, 1:] >= 20).all(1)
    tiles = {}
    for tile in np.unique(bg['tile_index'][use]):
        s = use & (bg['tile_index'] == tile)
        if s.sum() >= 200:
            tiles[int(tile)] = [float(np.median(resid[s, j])) for j in range(3)] + [int(s.sum())]
    arr = np.array([v[:3] for v in tiles.values()])
    out['T-D6_tiles'] = dict(n_tiles=len(tiles), rms=arr.std(0).tolist() if len(arr) else None,
                             max_abs=np.abs(arr - np.median(arr, 0)).max(0).tolist() if len(arr) else None, per_tile=tiles)
    # T-D7: residual vs E(B-V) slope (our dereddening applied); slope -(R_a - R_b) if MER were already dereddened
    slopes = []
    for j in range(3):
        cf = np.polyfit(bg['ebv'][use], resid[use, j], 1)
        slopes.append(float(cf[0]))
    out['T-D7_dereddening'] = dict(slope_per_unit_ebv=slopes, expected_if_double=(-(R[:-1] - R[1:])).tolist(),
                                   ebv_range=[float(np.percentile(bg['ebv'][use], 1)), float(np.percentile(bg['ebv'][use], 99))])
    # T-D4: match statistics
    out['T-D4_match'] = dict(qso_sep_pct=np.percentile(q['sep_arcsec'], [50, 90, 99, 99.9]).tolist(),
                             qso_within_0p3=float(np.mean(q['sep_arcsec'] < 0.3)),
                             bkg_paths={p: dict(n=int((bg['match_path'] == p).sum()),
                                                sep_pct=np.percentile(bg['sep_arcsec'][bg['match_path'] == p], [50, 90, 99]).tolist())
                                        for p in np.unique(bg['match_path'])},
                             qso_isolated=float(np.mean(q['neighbours'] == 0)), bkg_isolated=float(np.mean(bg['neighbours'] == 0)))
    # T-D1 (QSO side): completeness of the point rule by z and VIS for several thresholds (isolated, clean rows)
    qm, _ = mags(q); qbase = (q['vis_det'] == 1) & (q['spurious_flag'] == 0) & (q['neighbours'] == 0)
    d1 = {}
    for pmin in (0.1, 0.2, 0.3, 0.5):
        tab = {}
        for zl, zh in ((0, .5), (.5, .8), (.8, 1.5), (1.5, 2.5), (2.5, 5.)):
            for vl, vh in ((15, 21), (21, 23), (23, 26)):
                s = qbase & (q['zspec'] >= zl) & (q['zspec'] < zh) & (qm[:, 0] >= vl) & (qm[:, 0] < vh)
                if s.sum() >= 20:
                    tab[f'z{zl}-{zh} VIS{vl}-{vh}'] = [float(np.mean(np.nan_to_num(q['point_like_prob'][s]) >= pmin)), int(s.sum())]
        d1[str(pmin)] = tab
    out['T-D1_qso_completeness'] = d1
    out['isolation_qso_loss_by_z'] = {f'{zl}-{zh}': float(np.mean(q['neighbours'][(q['zspec'] >= zl) & (q['zspec'] < zh)] > 0))
                                       for zl, zh in ((0, .8), (.8, 1.5), (1.5, 2.5), (2.5, 5.))}
    Path('docs/euclid/data_tests.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ('T-D6_tiles',)}, indent=1)[:6000])
    print('T-D6', {k: v for k, v in out['T-D6_tiles'].items() if k != 'per_tile'})
