#!/usr/bin/env python
"""Gaia DR3 astrometry for the stellar (non-QSO) training cones, aligned to the cleaned rows.

Legacy Surveys DR9 (``decals_dr9.main``) carries Gaia DR2 identifiers (``ref_cat='G2'``,
``ref_id``). Per cone (same centre as the stellar preparation, radius + 0.01 deg) this maps
DR2 -> DR3 with ``gaia_edr3.dr2_neighbourhood`` (closest DR3 neighbour by angular distance)
and reads ``gaia_dr3.gaia_source``. Rows are matched to the cleaned cone by
(release, brickid, objid); rows without a Gaia DR2 reference get NaN.

Output, one file per cone, aligned with ``clean/cone_XXX.npz``::

    <stellar_root>/gaia_dr3/cone_XXX.npz: dr3_source_id (int64, -1 if none), pmra, pmdec,
        pmra_error, pmdec_error, pmra_pmdec_corr [mas/yr], parallax, parallax_error [mas],
        phot_g_mean_mag, ruwe, astrometric_params_solved, dr2_dr3_sep_mas, dr2_dr3_dmag

Usage: python scripts/fetch_stellar_gaia_dr3.py [--root models/multisurvey_psf/work/stellar_preparation/2d43d81d0b5f2af0]
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

SQL = """
WITH c AS MATERIALIZED (
  SELECT release, brickid, objid, ref_id FROM decals_dr9.main
  WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius}) AND ref_cat = 'G2'),
n AS MATERIALIZED (
  SELECT DISTINCT ON (c.ref_id) c.release, c.brickid, c.objid, nb.dr3_source_id, nb.angular_distance, nb.magnitude_difference
  FROM c JOIN gaia_edr3.dr2_neighbourhood nb ON nb.dr2_source_id = c.ref_id
  ORDER BY c.ref_id, nb.angular_distance)
SELECT n.release, n.brickid, n.objid, n.dr3_source_id, g.pmra, g.pmdec, g.pmra_error, g.pmdec_error,
       g.pmra_pmdec_corr, g.parallax, g.parallax_error, g.phot_g_mean_mag, g.ruwe, g.astrometric_params_solved,
       n.angular_distance, n.magnitude_difference
FROM n JOIN gaia_dr3.gaia_source g ON g.source_id = n.dr3_source_id
"""
NAMES = ('release', 'brickid', 'objid', 'dr3_source_id', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr',
         'parallax', 'parallax_error', 'phot_g_mean_mag', 'ruwe', 'astrometric_params_solved', 'dr2_dr3_sep_mas', 'dr2_dr3_dmag')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('models/multisurvey_psf/work/stellar_preparation/2d43d81d0b5f2af0'))
    a = p.parse_args()
    import sqlutilpy as sqlutil
    prov = json.loads((a.root/'provenance.json').read_text())['config']; radius = prov['cone_radius_deg'] + .01
    regions = {r['cone']: r for r in json.loads((a.root/'spatial_roles.json').read_text())['regions']}
    out = a.root/'gaia_dr3'; out.mkdir(exist_ok=True)
    conn = sqlutil.getConnection(db='wsdb', driver='psycopg'); start = time.time(); summary = []
    for cone in sorted(regions):
        path = out/f'cone_{cone:03d}.npz'
        if path.exists():
            continue
        clean = np.load(a.root/'clean'/f'cone_{cone:03d}.npz'); n = len(clean['ra'])
        cols = sqlutil.get(SQL.format(ra=regions[cone]['ra'], dec=regions[cone]['dec'], radius=radius), conn=conn,
                           preamb="SET jit=off; SET statement_timeout='600s'")
        res = dict(zip(NAMES, cols))
        key = lambda r, b, o: (np.asarray(r, np.int64) << 40) | (np.asarray(b, np.int64) << 20) | np.asarray(o, np.int64)
        kc = key(clean['release'], clean['brickid'], clean['objid']); kq = key(res['release'], res['brickid'], res['objid'])
        order = np.argsort(kq); pos = np.searchsorted(kq[order], kc); pos = np.clip(pos, 0, max(len(kq) - 1, 0))
        hit = (len(kq) > 0) & (kq[order][pos] == kc) if len(kq) else np.zeros(n, bool); idx = order[pos]
        arrays = {'dr3_source_id': np.where(hit, res['dr3_source_id'][idx] if len(kq) else -1, -1).astype(np.int64)}
        for name in NAMES[4:]:
            v = np.full(n, np.nan)
            if len(kq):
                v[hit] = np.asarray(res[name], float)[idx[hit]]
            arrays[name] = v
        np.savez_compressed(path, **arrays)
        summary.append(dict(cone=cone, rows=n, gaia=int(hit.sum())))
        print(f'cone {cone:3d}: {n} rows, {hit.sum()} with DR3 ({time.time() - start:.0f} s)', flush=True)
    (out/'summary.json').write_text(json.dumps(dict(sql=SQL, radius_deg=radius, cones=summary), indent=1))


if __name__ == '__main__':
    main()
