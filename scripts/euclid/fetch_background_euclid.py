#!/usr/bin/env python
"""Euclid DR1 MER matches for every parent background row in the cones that overlap DR1 wide.

Rows with a Gaia DR3 id are matched through MER's own ``gaia_id`` (proper-motion safe); all rows are also matched
positionally (nearest within 1.5", with the count of MER sources inside 1.5" for the isolation cut).
Output: ~/data/qso_p_color/catalogues/euclid_dr1/background/{rows.npz, positional/<hash>, gaia/<hash>}.
"""
from pathlib import Path
import json
import numpy as np
import sqlutilpy as sqlutil
from qso_pcolor.euclid_data import match, match_gaia, TABLES

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
    gid = np.full(len(rows), -1, np.int64)
    for f in fields:
        g = np.load(PREP/'gaia_dr3'/f'cone_{f:03d}.npz')['dr3_source_id']
        sel = field[rows] == f; gid[sel] = g[src[rows[sel]] - offset[int(f)]]
    np.savez_compressed(OUT/'rows.npz', rows=rows, field=field[rows], gaia_id=gid, cone_euclid_counts=n, cones=cone)
    print('rows', len(rows), 'with Gaia id', int((gid > 0).sum()), flush=True)
    pos = match(np.load(RUN/'ra.npy')[rows], np.load(RUN/'dec.npy')[rows], OUT/'positional', radius_arcsec=1.5,
                log=lambda s: print(s, flush=True))
    gai = match_gaia(gid, OUT/'gaia', log=lambda s: print(s, flush=True))
    print('DONE positional', int(pos['matched'].sum()), 'gaia', int(gai['matched'].sum()), flush=True)
