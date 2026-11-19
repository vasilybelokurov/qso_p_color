#!/usr/bin/env python
"""Phase E step 3 data: tile a Euclid Deep Field with deep-MER cones (D6: faint-galaxy colours from all of EDF-North).

Tiling: a hexagonal grid of cone centres with spacing S = 0.5 deg and cone radius R = 0.3 deg. Each Voronoi cell of a
hex grid has circumradius S/sqrt(3) = 0.289 deg < R - 2" (the isolation margin), so assigning each source to its
NEAREST tile centre (``owner`` mask, applied after the deployment selection) counts every source once with complete
isolation neighbours. Tiles are kept where the nearest covered coverage probe (deep_coverage_probes.py, 0.5-deg grid,
r = 0.03 deg probes with > 0 rows) is within 0.4 deg; edge tiles may be partly empty (area is measured separately).

Per tile, the same columns and Legacy DR9 / DESI-galaxy companions as scripts/euclid/fetch_cones.py, from the DEEP MER
table only (wide and deep rows are never mixed in one file: duplicates would count as isolation neighbours). Cached as
~/data/qso_p_color/catalogues/euclid_dr1/deep_tiles/<field>/tile_<id>.npz (centre and radius checked on reuse);
resumable. Network-bound, one process.

    nice -n 19 python scripts/score_all/fetch_deep_tiles.py --field N [--survey deep|wide] [--limit N] [--list]
"""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/euclid'))
DATA = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
PROBES = DATA/'deep_coverage/probes.json'
DEEP = 'euclid_dr1.mer_catalogue_deep_survey'
SPACING, RADIUS, PROBE_REACH = 0.5, 0.3, 0.4
CENTRES = {'N': (269.73, 66.02), 'F': (52.93, -28.09), 'S': (61.24, -48.42)}


def tangent(ra, dec, ra0, dec0):
    """Gnomonic projection (deg) about (ra0, dec0)."""
    a, d, a0, d0 = map(np.deg2rad, (np.asarray(ra, float), np.asarray(dec, float), ra0, dec0))
    c = np.sin(d0)*np.sin(d) + np.cos(d0)*np.cos(d)*np.cos(a - a0)
    x = np.cos(d)*np.sin(a - a0)/c; y = (np.cos(d0)*np.sin(d) - np.sin(d0)*np.cos(d)*np.cos(a - a0))/c
    return np.rad2deg(x), np.rad2deg(y)


def untangent(x, y, ra0, dec0):
    x, y, a0, d0 = np.deg2rad(x), np.deg2rad(y), np.deg2rad(ra0), np.deg2rad(dec0)
    rho = np.hypot(x, y); c = np.arctan(rho)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.arcsin(np.cos(c)*np.sin(d0) + np.where(rho > 0, y*np.sin(c)*np.cos(d0)/rho, 0))
    a = a0 + np.arctan2(x*np.sin(c), rho*np.cos(d0)*np.cos(c) - y*np.sin(d0)*np.sin(c))
    return np.rad2deg(a) % 360, np.rad2deg(d)


def tiles(field):
    """Hex-grid tile centres (ra, dec) for a deep field, kept near covered probes."""
    ra0, dec0 = CENTRES[field]
    p = json.loads(PROBES.read_text())[field]
    pts = np.asarray(p['pts'], float)[np.asarray(p['n']) > 0]
    px, py = tangent(pts[:, 0], pts[:, 1], ra0, dec0)
    h = SPACING*np.sqrt(3)/2; out = []
    for j, y in enumerate(np.arange(-4.5, 4.51, h)):
        for x in np.arange(-4.5, 4.51, SPACING) + (SPACING/2 if j % 2 else 0):
            if np.min(np.hypot(px - x, py - y)) <= PROBE_REACH:
                out.append((x, y))
    out = np.asarray(out)
    ra, dec = untangent(out[:, 0], out[:, 1], ra0, dec0)
    return np.column_stack([ra, dec])


def owner_mask(ra, dec, centre, all_centres):
    """True for sources whose nearest tile centre is ``centre`` (Voronoi ownership on the sphere)."""
    from cones import unit
    xyz = unit(ra, dec); cs = unit(all_centres[:, 0], all_centres[:, 1])
    from scipy.spatial import cKDTree
    _, i = cKDTree(cs).query(xyz)
    me = np.argmin(np.sum((cs - unit([centre[0]], [centre[1]]))**2, 1))
    return i == me


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--field', default='N', choices=list(CENTRES))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--list', action='store_true', help='print the tile count and exit')
    ap.add_argument('--survey', default='deep', choices=('deep', 'wide'),
                    help="deep: deep MER table -> deep_tiles/; wide: both DR1 wide tables at the same tile centres -> "
                         "wide_tiles/ (step 4 wide-vs-deep completeness and the wide-classification mimic)")
    a = ap.parse_args()
    t = tiles(a.field)
    out = DATA/f'{a.survey}_tiles'/a.field; out.mkdir(parents=True, exist_ok=True)
    (out/'tiles.json').write_text(json.dumps(dict(spacing=SPACING, radius=RADIUS, probe_reach=PROBE_REACH,
                                                  centres=t.round(6).tolist())))
    print(a.field, 'tiles', len(t), flush=True)
    if a.list:
        return
    import fetch_cones
    if a.survey == 'deep':
        fetch_cones.TABLES = (DEEP,)                             # wide: fetch_cones default = both DR1 wide tables
    done = 0
    for i, (ra, dec) in enumerate(t):
        cone = dict(ra=float(round(ra, 6)), dec=float(round(dec, 6)), radius_deg=RADIUS)
        path = out/f'tile_{i:03d}.npz'
        if fetch_cones.cached_matches(path, cone):
            continue
        if a.limit is not None and done >= a.limit:
            break
        t0 = time.monotonic(); d = fetch_cones.fetch(cone)
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **d, centre=np.array([cone['ra'], cone['dec']]), radius_deg=RADIUS); tmp.rename(path)
        done += 1
        print(f'{path.name} MER {len(d["mer_e_ra"])} Legacy {len(d["legacy_ra"])} {time.monotonic()-t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
