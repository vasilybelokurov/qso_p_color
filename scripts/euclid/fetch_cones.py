#!/usr/bin/env python
"""Per-cone data for the Euclid select/calib/test cones (docs/euclid/PLAN.md 3.3).

For each cone of docs/euclid/sky_roles.json (r = 0.3 deg):
- every MER source (both DR1 wide tables) with vis_det = 1 and spurious_flag = 0 within r + 2" (the margin gives
  complete isolation-neighbour counts at the edge);
- Legacy DR9 sources within r + 2" (type, fluxes, ivar, nobs, maskbits, release, Gaia pm/ref_epoch);
- DESI DR1 spectroscopic galaxies (spectype GALAXY, zwarn = 0) within r (point-rule contamination, T-D1).
Known QSOs (DESI DR1 + DR16Q master) and Quaia are joined locally later. Cached per cone under
~/data/qso_p_color/catalogues/euclid_dr1/cones/<role>_<id>.npz; resumable.
"""
import json
from pathlib import Path
import time
import numpy as np
import sqlutilpy as sqlutil
from qso_pcolor.euclid_data import COLUMNS, TABLES

OUT = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/cones'
ROLES = Path('docs/euclid/sky_roles.json')
MARGIN = 2/3600.
LEGACY_COLS = ('ra, dec, release, brickid, objid, type, maskbits, fracflux_r, ref_cat, ref_id, ref_epoch, pmra, pmdec, '
               + ', '.join(f'flux_{b}, flux_ivar_{b}, nobs_{b}' for b in ('g', 'r', 'z', 'w1', 'w2')))
PRE = "SET statement_timeout = '1800s'"


def fetch(cone):
    ra, dec, r = cone['ra'], cone['dec'], cone['radius_deg']
    mer = []
    for code, t in enumerate(TABLES):
        d = sqlutil.get(f"""SELECT right_ascension AS e_ra, declination AS e_dec, {', '.join(COLUMNS)} FROM {t}
            WHERE q3c_radial_query(right_ascension, declination, {ra}, {dec}, {r + MARGIN})
              AND vis_det = 1 AND spurious_flag = 0""", asDict=True, preamb=PRE)
        d = {k: np.asarray(v) for k, v in d.items()}; d['table'] = np.full(len(d['e_ra']), code); mer.append(d)
    mer = {k: np.concatenate([m[k] for m in mer]) for k in mer[0]}
    leg = sqlutil.get(f"""SELECT {LEGACY_COLS} FROM decals_dr9.main
        WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {r + MARGIN})""", asDict=True, preamb=PRE)
    gal = sqlutil.get(f"""SELECT p.ra, p.dec, z.z, z.targetid FROM desi_dr1.photometry p JOIN desi_dr1.zpix z
        ON z.targetid = p.targetid WHERE q3c_radial_query(p.ra, p.dec, {ra}, {dec}, {r})
        AND z.spectype = 'GALAXY' AND z.zwarn = 0 AND z.zcat_primary""", asDict=True, preamb=PRE)
    out = {f'mer_{k}': v for k, v in mer.items()}
    out.update({f'legacy_{k}': np.asarray(v) for k, v in leg.items()})
    out.update({f'desigal_{k}': np.asarray(v) for k, v in gal.items()})
    return out


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    roles = json.loads(ROLES.read_text())
    for cone in roles['cones']:
        path = OUT/f"{cone['role']}_{cone['id']:03d}.npz"
        if path.exists():
            continue
        t = time.monotonic(); d = fetch(cone)
        np.savez_compressed(path, **d, centre=np.array([cone['ra'], cone['dec']]), radius_deg=cone['radius_deg'])
        print(path.name, 'MER', len(d['mer_e_ra']), 'Legacy', len(d['legacy_ra']), 'DESI gal', len(d['desigal_ra']),
              f'{time.monotonic()-t:.0f} s', flush=True)
    print('DONE', flush=True)
