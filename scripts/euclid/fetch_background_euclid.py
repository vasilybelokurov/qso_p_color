#!/usr/bin/env python
"""Euclid DR1 MER matches for every parent background row in the cones that overlap DR1 wide.

All rows are matched positionally (nearest MER source within 1.5"). Rows with a Gaia DR3 proper motion are matched
again from their positions propagated from 2016.0 to EUCLID_EPOCH (stars move up to ~1" between Gaia DR3 and DR1;
review D6); that match replaces the plain one for them. A MER gaia_id join was tried first and abandoned: it is a
full scan of the 888M-row table (> 25 min, cancelled).
Output: ~/data/qso_p_color/catalogues/euclid_dr1/background/{rows.npz, positional/<hash>, propagated/<hash>}.
"""
from pathlib import Path
import json
import numpy as np
import sqlutilpy as sqlutil
from qso_pcolor.euclid_data import match, TABLES

# DR1 wide observations, 2024 Feb - 2025 (approximate mid-epoch); 0.5 yr error = 0.05" at 100 mas/yr
EUCLID_EPOCH = 2024.6

RUN = Path('models/multisurvey_psf/work/unified_full/20261001/5f4002492dbb849c/stars')
PREP = Path('models/multisurvey_psf/work/stellar_preparation/2d43d81d0b5f2af0')
OUT = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/background'

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    regions = json.loads((PREP/'spatial_roles.json').read_text())['regions']
    cone = np.array([r['cone'] for r in regions]); ra = np.array([r['ra'] for r in regions]); dec = np.array([r['dec'] for r in regions])
    n = np.zeros(len(cone), int)
    for t in TABLES:
        r = sqlutil.local_join(f"select m.c, (select count(*) from {t} e where q3c_radial_query(e.right_ascension,"
                               f"e.declination,m.ra,m.dec,0.3)) n from mytab m", 'mytab', (cone, ra, dec), ('c', 'ra', 'dec'), asDict=True)
        n += np.asarray(r['n'])[np.argsort(r['c'])][np.argsort(np.argsort(cone))]
    fields = cone[n > 0]
    print('overlapping cones', len(fields), flush=True)
    field = np.load(RUN/'field.npy'); src = np.load(RUN/'source_row.npy')
    rows = np.flatnonzero(np.isin(field, fields))
    # source_row indexes the concatenation of clean/cone_*.npz in file order (verified on 4 cones by RA)
    files = sorted((PREP/'clean').glob('cone_*.npz'))
    offset = dict(zip([int(f.stem[5:]) for f in files], np.cumsum([0]+[len(np.load(f)['ra']) for f in files])[:-1]))
    gid = np.full(len(rows), -1, np.int64); pmra = np.full(len(rows), np.nan); pmdec = np.full(len(rows), np.nan)
    for f in fields:
        g = np.load(PREP/'gaia_dr3'/f'cone_{f:03d}.npz')
        sel = field[rows] == f; loc = src[rows[sel]] - offset[int(f)]
        gid[sel] = g['dr3_source_id'][loc]; pmra[sel] = g['pmra'][loc]; pmdec[sel] = g['pmdec'][loc]
    np.savez_compressed(OUT/'rows.npz', rows=rows, field=field[rows], gaia_id=gid, pmra=pmra, pmdec=pmdec,
                        cone_euclid_counts=n, cones=cone, euclid_epoch=EUCLID_EPOCH)
    print('rows', len(rows), 'with Gaia id', int((gid > 0).sum()), flush=True)
    ra, dec = np.load(RUN/'ra.npy')[rows], np.load(RUN/'dec.npy')[rows]
    pos = match(ra, dec, OUT/'positional', radius_arcsec=1.5, log=lambda s: print(s, flush=True))
    # NOTE: Legacy DR9 positions of Gaia stars are at the Gaia epoch (Tractor fixes Gaia astrometry at 2015.5/2016);
    # propagate with Gaia DR3 proper motions.
    has = np.isfinite(pmra) & np.isfinite(pmdec)
    dt = EUCLID_EPOCH - 2016.0
    dra = pmra[has]*dt/3.6e6/np.cos(np.deg2rad(dec[has])); ddec = pmdec[has]*dt/3.6e6
    prop = match(ra[has] + dra, dec[has] + ddec, OUT/'propagated', radius_arcsec=1.5, log=lambda s: print(s, flush=True))
    print('DONE positional', int(pos['matched'].sum()), 'of', len(rows), '; propagated', int(prop['matched'].sum()),
          'of', int(has.sum()), flush=True)
