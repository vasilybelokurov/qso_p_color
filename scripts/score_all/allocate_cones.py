#!/usr/bin/env python
"""Allocate the score-all fit and evaluation cones (PI decision D3, 19 Nov 2026; docs/euclid/PLAN_score_all.md).

docs/euclid/sky_roles.json is frozen and is not modified. New cones go to docs/score_all/new_cones.json:
  eval  N_EVAL cones in the cells with no recorded use and coverage >= 0.5 (docs/score_all/manifest.json): the
        background half of the new evaluation (quasars come from EDF-North, decision D3). Never used for fitting.
  fit   N_FIT cones (~30 deg^2) in fit-role cells outside the eval pool: the background fit of the VIS-referenced
        faint model (design 2B, decision D2).
Placement as scripts/euclid/sky_roles.py: cone r = 0.3 deg centred on a covered nside-128 probe pixel with the whole
cone covered, |b| >= 25 deg + r, >= 1 deg from every parent background cone, from every existing sky_roles cone and
from each other, at most MAX_PER_CELL per cell, seeded random order (eval first). Frozen: records seed and inputs.

    python scripts/score_all/allocate_cones.py
"""
import hashlib
import json
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
import healpy as hp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
PREP = ROOT/'models/multisurvey_psf/work/stellar_preparation/2d43d81d0b5f2af0'
OUT = ROOT/'docs/score_all/new_cones.json'
SEED = 20261119
MIN_COUNT, CONE_R, SEP_DEG = 20, 0.3, 1.0
N_EVAL, N_FIT, MAX_PER_CELL = 40, 106, 4


def sep(l1, b1, l2, b2):
    v1 = hp.ang2vec(l1, b1, lonlat=True); v2 = hp.ang2vec(l2, b2, lonlat=True)
    return np.rad2deg(np.arccos(np.clip(np.atleast_2d(v1) @ np.atleast_2d(v2).T, -1, 1)))


def main():
    rng = np.random.default_rng(SEED)
    roles = json.loads((ROOT/'docs/euclid/sky_roles.json').read_text())
    manifest = json.loads((ROOT/'docs/score_all/manifest.json').read_text())
    unused = set(manifest['cells_with_no_recorded_use']['cells'])
    cov = {c['cell']: c['coverage'] for c in roles['cells']}; role = {c['cell']: c['role'] for c in roles['cells']}
    eval_cells = sorted(c for c in unused if cov[c] >= 0.5)
    fit_cells = sorted(c for c in role if role[c] == 'fit' and c not in unused)
    pr = np.load(BASE/'footprint/probes_nside128.npz')
    covered = pr['count'] >= MIN_COUNT; cell_of_probe = pr['pix'] >> 6
    regions = json.loads((PREP/'spatial_roles.json').read_text())['regions']
    avoid_l = [r['l'] for r in regions] + [c['l'] for c in roles['cones']]
    avoid_b = [r['b'] for r in regions] + [c['b'] for c in roles['cones']]
    picked_l, picked_b, cones = list(avoid_l), list(avoid_b), []
    for rname, pool, target in (('eval', eval_cells, N_EVAL), ('fit', fit_cells, N_FIT)):
        cand = np.flatnonzero(np.isin(cell_of_probe, pool) & covered); rng.shuffle(cand); per_cell = {}; n = 0
        for i in cand:
            if n >= target:
                break
            c = int(cell_of_probe[i]); l0, b0 = float(pr['l'][i]), float(pr['b'][i])
            if per_cell.get(c, 0) >= MAX_PER_CELL or abs(b0) < 25 + CONE_R:
                continue
            near = hp.query_disc(128, hp.ang2vec(l0, b0, lonlat=True), np.deg2rad(CONE_R + 0.05), nest=True)
            j = np.searchsorted(pr['pix'], near).clip(0, len(pr['pix']) - 1)
            if not (covered[j] & (pr['pix'][j] == near)).all():
                continue
            if sep(l0, b0, np.array(picked_l), np.array(picked_b)).min() < SEP_DEG:
                continue
            e = SkyCoord(l=l0*u.deg, b=b0*u.deg, frame='galactic').icrs
            cones.append(dict(id=n, role=rname, cell=c, l=l0, b=b0, ra=float(e.ra.deg), dec=float(e.dec.deg),
                              radius_deg=CONE_R))
            picked_l.append(l0); picked_b.append(b0); per_cell[c] = per_cell.get(c, 0) + 1; n += 1
    area = lambda r: sum(np.pi*c['radius_deg']**2 for c in cones if c['role'] == r)
    out = dict(seed=SEED, rule=__doc__, min_count=MIN_COUNT, max_per_cell=MAX_PER_CELL, sep_deg=SEP_DEG,
               footprint_sha256=hashlib.sha256((BASE/'footprint/probes_nside128.npz').read_bytes()).hexdigest(),
               sky_roles_sha256=hashlib.sha256((ROOT/'docs/euclid/sky_roles.json').read_bytes()).hexdigest(),
               manifest_sha256=manifest['sha256_of_content'], eval_cells=eval_cells, fit_cells=fit_cells,
               summary={r: dict(cones=sum(c['role'] == r for c in cones), area_deg2=round(area(r), 2),
                                cells=len({c['cell'] for c in cones if c['role'] == r})) for r in ('eval', 'fit')},
               cones=cones)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out['summary'], indent=1))
    assert not ({c['cell'] for c in cones if c['role'] == 'eval'} & set(fit_cells)), 'eval and fit cells overlap'


if __name__ == '__main__':
    main()
