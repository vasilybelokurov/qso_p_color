#!/usr/bin/env python
"""Euclid sky roles (docs/euclid/PLAN.md T1): Galactic NESTED nside-16 cells in DR1 wide at |b| >= 25 deg.

Coverage of a cell = fraction of its 64 nside-128 probe pixels with >= MIN_COUNT MER sources inside the probe
(scripts/euclid/footprint.py). Cells with coverage < 0.5 go to fit. The others are assigned fit/select/calib/test
= 55/15/10/20% by a seeded shuffle within strata of (|b| tercile, matched-QSO-count tercile), so every role spans
latitude and QSO density (review D2.5). Cones (r = 0.3 deg) for select/calib/test: centres at covered probe pixels
whose whole cone is covered, >= 1 deg from every parent background cone and from each other, at most
MAX_PER_CELL per cell, chosen greedily in seeded random order. Frozen: the output records seed and inputs.
"""
import glob
import hashlib
import json
from pathlib import Path
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

SEED = 20261118
FRACTIONS = dict(fit=.55, select=.15, calib=.10, test=.20)
MIN_COUNT = 20                       # probe (0.008 deg radius) holds ~100 sources where covered
CONE_R, SEP_DEG, MAX_PER_CELL = 0.3, 1.0, 2
TARGET = dict(test=40, select=12, calib=12)
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
PREP = Path('models/multisurvey_psf/work/stellar_preparation/2d43d81d0b5f2af0')
QROOT = Path('models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff')


def sep(l1, b1, l2, b2):
    v1 = hp.ang2vec(l1, b1, lonlat=True); v2 = hp.ang2vec(l2, b2, lonlat=True)
    return np.rad2deg(np.arccos(np.clip(v1 @ v2.T, -1, 1)))


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    pr = np.load(BASE/'footprint/probes_nside128.npz')
    covered = pr['count'] >= MIN_COUNT
    cell_of_probe = pr['pix'] >> (2*3)                       # nested: nside 128 -> 16 drops 3 levels
    cells, inv = np.unique(cell_of_probe, return_inverse=True)
    cov = np.bincount(inv, weights=covered, minlength=len(cells))/64.
    lc, bc = hp.pix2ang(16, cells, nest=True, lonlat=True)
    use = (cov > 0) & (np.abs(bc) >= 25)
    cells, cov, lc, bc = cells[use], cov[use], lc[use], bc[use]
    # matched QSO targets per cell
    m = np.load(glob.glob(str(BASE/'qso_targets/*/match.npz'))[0]); t = np.load(QROOT/'targets.npz')
    g = SkyCoord(ra=t['ra'][m['matched']]*u.deg, dec=t['dec'][m['matched']]*u.deg).galactic
    qcell = hp.ang2pix(16, g.l.deg, g.b.deg, lonlat=True, nest=True)
    nq = np.array([(qcell == c).sum() for c in cells])
    role = np.full(len(cells), 'fit', dtype='<U6')
    full = cov >= 0.5
    tb = np.digitize(np.abs(bc), np.quantile(np.abs(bc[full]), [1/3, 2/3]))
    tq = np.digitize(nq, np.quantile(nq[full], [1/3, 2/3]))
    names = list(FRACTIONS); cum = np.cumsum([FRACTIONS[k] for k in names])
    for s in np.unique(tb*3 + tq):
        idx = np.flatnonzero(full & (tb*3 + tq == s)); rng.shuffle(idx)
        # systematic assignment: quantile position in the shuffled stratum
        pos = (np.arange(len(idx)) + rng.random())/len(idx)
        role[idx] = np.array(names)[np.searchsorted(cum, pos, side='right').clip(0, len(names)-1)]
    # cones
    regions = json.loads((PREP/'spatial_roles.json').read_text())['regions']
    pl = np.array([r['l'] for r in regions]); pb = np.array([r['b'] for r in regions])
    probe_l, probe_b = pr['l'], pr['b']
    cones = []; picked_l, picked_b = [], []
    for rname in ('test', 'select', 'calib'):
        rc = cells[role == rname]
        cand = np.flatnonzero(np.isin(cell_of_probe, rc) & covered)
        rng.shuffle(cand)
        per_cell = {}
        for i in cand:
            if sum(1 for c in cones if c['role'] == rname) >= TARGET[rname]:
                break
            c = int(cell_of_probe[i])
            if per_cell.get(c, 0) >= MAX_PER_CELL:
                continue
            l0, b0 = probe_l[i], probe_b[i]
            if np.abs(b0) < 25 + CONE_R:
                continue
            near = hp.query_disc(128, hp.ang2vec(l0, b0, lonlat=True), np.deg2rad(CONE_R + 0.05), nest=True)
            near_cov = covered[np.searchsorted(pr['pix'], near).clip(0, len(pr['pix'])-1)] & np.isin(near, pr['pix'])
            if not near_cov.all():
                continue
            if sep(np.array([l0]), np.array([b0]), pl, pb).min() < SEP_DEG:
                continue
            if picked_l and sep(np.array([l0]), np.array([b0]), np.array(picked_l), np.array(picked_b)).min() < SEP_DEG:
                continue
            e = SkyCoord(l=l0*u.deg, b=b0*u.deg, frame='galactic').icrs
            cones.append(dict(id=len(cones), role=rname, cell=c, l=float(l0), b=float(b0), ra=float(e.ra.deg),
                              dec=float(e.dec.deg), radius_deg=CONE_R))
            picked_l.append(l0); picked_b.append(b0); per_cell[c] = per_cell.get(c, 0) + 1
    out = dict(seed=SEED, nside=16, frame='Galactic', ordering='NESTED', fractions=FRACTIONS, min_count=MIN_COUNT,
               footprint=str(BASE/'footprint/probes_nside128.npz'),
               footprint_sha256=hashlib.sha256((BASE/'footprint/probes_nside128.npz').read_bytes()).hexdigest(),
               cells=[dict(cell=int(c), role=str(r), coverage=float(v), l=float(a), b=float(bb), n_qso_matched=int(n))
                      for c, r, v, a, bb, n in zip(cells, role, cov, lc, bc, nq)],
               cones=cones, rule=__doc__)
    Path('docs/euclid/sky_roles.json').write_text(json.dumps(out, indent=1))
    for rname in names:
        s = role == rname
        print(rname, 'cells', int(s.sum()), 'QSOs', int(nq[s].sum()), 'area deg2 %.0f' % (cov[s].sum()*hp.nside2pixarea(16, degrees=True)),
              'cones', sum(1 for c in cones if c['role'] == rname))
