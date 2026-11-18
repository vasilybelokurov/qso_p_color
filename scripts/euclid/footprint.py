#!/usr/bin/env python
"""Euclid DR1 wide coverage at |b| >= 24 deg from indexed cone probes (no table scan).

Probe = centre of every Galactic NESTED HEALPix nside-128 pixel (0.21 deg^2) with |b| >= 24; count of MER sources
(vis_det = 1, spurious_flag = 0) within 0.008 deg (~100 expected where covered), both wide tables.
Output ~/data/qso_p_color/catalogues/euclid_dr1/footprint/probes_nside128.npz (pix, l, b, ra, dec, count).
"""
from pathlib import Path
import time
import healpy as hp
import numpy as np
import sqlutilpy as sqlutil
from astropy.coordinates import SkyCoord
import astropy.units as u
from qso_pcolor.euclid_data import TABLES

OUT = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/footprint'
NSIDE, RADIUS = 128, 0.008

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    pix = np.arange(hp.nside2npix(NSIDE)); l, b = hp.pix2ang(NSIDE, pix, nest=True, lonlat=True)
    keep = np.abs(b) >= 24; pix, l, b = pix[keep], l[keep], b[keep]
    c = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic').icrs; ra, dec = c.ra.deg, c.dec.deg
    order = np.argsort(hp.ang2pix(1024, ra, dec, lonlat=True, nest=True))
    count = np.zeros(len(pix), int)
    for t in TABLES:
        t0 = time.monotonic()
        r = sqlutil.local_join(f"""select m.i, (select count(*) from {t} e where q3c_radial_query(e.right_ascension,
            e.declination, m.ra, m.dec, {RADIUS}) and e.vis_det = 1 and e.spurious_flag = 0) n from mytab m""",
            'mytab', (order, ra[order], dec[order]), ('i', 'ra', 'dec'), asDict=True,
            preamb="SET statement_timeout = '3600s'")
        count[np.asarray(r['i'], int)] += np.asarray(r['n'], int)
        print(t, f'{time.monotonic()-t0:.0f} s', flush=True)
    np.savez_compressed(OUT/f'probes_nside{NSIDE}.npz', pix=pix, l=l, b=b, ra=ra, dec=dec, count=count,
                        nside=NSIDE, radius_deg=RADIUS)
    print('DONE covered probes', int((count > 0).sum()), 'of', len(pix), flush=True)
