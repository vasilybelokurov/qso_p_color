#!/usr/bin/env python
"""Flag likely quasars in the stellar (non-QSO) training sample: known QSO, Gaia astrometry, WISE colour.

Cuts, applied to the prepared stellar rows of a unified run (extinction-corrected luptitudes, AB):
  quaia   Gaia DR3 source_id in Quaia G<20 (Storey-Fisher et al. 2024, ApJ 964, 69,
          https://doi.org/10.3847/1538-4357/ad1328), on top of the DESI DR1 + DR16Q removal already
          done in the stellar preparation.
  pm      DIAGNOSTIC ONLY, not used in the flag: Gaia DR3 astrometry consistent with zero motion
          (proper-motion chi^2, 2 dof with the pmra/pmdec correlation, < PM_CHI2 and |parallax|/sigma < 3).
          Measured on 3 October: 33-56% of such objects have stellar W1-W2 even with sigma_pm < 0.2-0.5
          mas/yr (distant, slow halo stars), and the cut only reaches r < 21 where contamination is small,
          so it removes stars about as often as quasars.
  wise    Legacy W1-W2 > WISE_AB (AB; = 0.8 Vega, Stern et al. 2012, ApJ 753, 30,
          https://doi.org/10.1088/0004-637X/753/1/30) with W1, W2 S/N >= 5, EXCEPT very red optical
          objects (r-z > RED_RZ, AB), kept as brown-dwarf candidates.
Diagnostic for stars removed by mistake: among flagged objects with W1, W2 S/N >= 5, the fraction
with stellar infrared colour (W1-W2 < 0.4 Vega); for the pm cut this measures how many removed
objects are stars with small motion rather than quasars.

Writes <run>/stars/qso_flag.npy (bool, aligned with the prepared rows), <run>/stars/qso_flag_reason.npy
and docs/stellar_qso_cleaning.json.
Usage: python scripts/clean_stellar_qso_contamination.py [--run models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

PM_CHI2 = 11.83          # 2-dof chi^2 at 3 sigma (p = 0.0027)
WISE_AB = 0.8 - (3.339 - 2.699)   # W1-W2 Vega 0.8 in AB (W1, W2 AB-Vega offsets 2.699, 3.339)
STAR_IR_AB = 0.4 - (3.339 - 2.699)
RED_RZ = 2.0
EDGES = [15.5, 18, 19, 20, 21, 22, 23, 24.5]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, default=Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059'))
    a = p.parse_args()
    cfg = json.loads((a.run/'config.json').read_text()); lab = json.loads((a.run/'layout.json').read_text())['native_labels']
    root = Path(cfg['stellar_root']); st = a.run/'stars'
    src = np.load(st/'source_row.npy'); obs = np.load(st/'observed.npy', mmap_mode='r'); y = np.load(st/'y.npy', mmap_mode='r')
    noise = np.load(st/'noise.npy', mmap_mode='r'); role = np.load(st/'role.npy')
    gaia = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(str(root/'gaia_dr3'/'cone_*.npz')))])
            for k in ('dr3_source_id', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr', 'parallax', 'parallax_error')}
    gaia = {k: v[src] for k, v in gaia.items()}
    quaia = fits.getdata(Path('~/data/catalogues/quaia_G20.0.fits').expanduser(), 1)['source_id']
    f_quaia = (gaia['dr3_source_id'] >= 0) & np.isin(gaia['dr3_source_id'], quaia)
    ex, ey, rho = gaia['pmra_error'], gaia['pmdec_error'], np.nan_to_num(gaia['pmra_pmdec_corr'])
    det = (ex*ey)**2*(1 - rho**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        chi2 = (gaia['pmra']**2*ey**2 - 2*rho*ex*ey*gaia['pmra']*gaia['pmdec'] + gaia['pmdec']**2*ex**2)/det
        f_pm = np.isfinite(chi2) & (chi2 < PM_CHI2) & (np.abs(gaia['parallax']/gaia['parallax_error']) < 3)
    o = np.asarray(obs); Y = np.asarray(y); V = np.asarray(noise)
    hemi = np.where(o[:, lab.index('decals_dr9_south:r')], 'south', 'north')
    col = lambda b: np.array([lab.index(f'decals_dr9_{h}:{b}') for h in hemi])
    ir, iz, i1, i2 = col('r'), col('z'), col('w1'), col('w2'); rows = np.arange(len(src))
    sn = lambda j: np.where(o[rows, j], 1.0857/np.sqrt(V[rows, j]), 0.)
    w_ok = (sn(i1) >= 5) & (sn(i2) >= 5); w12 = Y[rows, i1] - Y[rows, i2]; rz = Y[rows, ir] - Y[rows, iz]
    f_wise = w_ok & (w12 > WISE_AB) & ~(o[rows, iz] & (rz > RED_RZ))
    flag = f_quaia | f_wise
    reason = np.array([','.join(n for n, f in (('quaia', a_), ('pm', b_), ('wise', c_)) if f)
                       for a_, b_, c_ in zip(f_quaia, f_pm, f_wise)])
    np.save(st/'qso_flag.npy', flag); np.save(st/'qso_flag_reason.npy', reason)
    r = Y[rows, ir]; snr_r = sn(ir); train = np.isin(role, cfg['fit_roles']) & (snr_r >= 10)
    report = dict(definition=__doc__, pm_chi2=PM_CHI2, wise_ab=WISE_AB, red_rz=RED_RZ, bins=[])
    print('training rows (fit/select, Legacy r S/N>=10) per r bin: N, Gaia, flagged by quaia / pm / wise / any;'
          ' stellar-IR fraction among pm-flagged and among all flagged (W1,W2 S/N>=5)')
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = train & (r >= lo) & (r < hi); n = int(m.sum())
        star_ir = lambda f: float(np.mean(w12[m & f & w_ok] < STAR_IR_AB)) if (m & f & w_ok).any() else np.nan
        b = dict(bin=[lo, hi], n=n, gaia=int((m & (gaia['dr3_source_id'] >= 0)).sum()),
                 quaia=float(f_quaia[m].mean()), pm=float(f_pm[m].mean()), wise=float(f_wise[m].mean()), any=float(flag[m].mean()),
                 pm_star_ir=star_ir(f_pm), any_star_ir=star_ir(flag))
        report['bins'].append(b)
        print(f"  r {lo:4.1f}-{hi:4.1f}: N {n:6d} Gaia {b['gaia']:6d}  quaia {b['quaia']:.4f} pm {b['pm']:.4f} wise {b['wise']:.4f}"
              f" any {b['any']:.4f}  | stellar-IR among pm {b['pm_star_ir']:.2f}, among all {b['any_star_ir']:.2f}")
    Path('docs/stellar_qso_cleaning.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
