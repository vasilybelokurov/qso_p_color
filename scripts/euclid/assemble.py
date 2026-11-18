#!/usr/bin/env python
"""Training/validation tables for the Euclid extension (docs/euclid/PLAN.md 3): one row per parent row with a MER match.

QSOs: the frozen parent QSO arrays (1,657,444 rows, aligned with QROOT targets.npz) joined to the target match.
Background: parent background rows of the 33 cones overlapping DR1 wide, joined to the positional match (Gaia
proper-motion rows: the match from propagated positions). Euclid fluxes are native and NOT dereddened; parent
Legacy luptitudes/noise are the dereddened training values. Output: ~/data/qso_p_color/catalogues/euclid_dr1/tables/.
"""
import glob
import hashlib
import json
from pathlib import Path
import numpy as np
import healpy as hp

from qso_pcolor.euclid_bundle import photometry

RUN = Path('models/multisurvey_psf/work/unified_full/20261001/5f4002492dbb849c')
QROOT = Path('models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff')
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
LEGACY = [f'decals_dr9_{h}:{b}' for h in ('south', 'north') for b in ('g', 'r', 'z', 'w1', 'w2')]
KEEP = ('object_id', 'table', 'tile_index', 'sep_arcsec', 'n_within', 'vis_det', 'spurious_flag', 'det_quality_flag',
        'flag_vis', 'flag_y', 'flag_j', 'flag_h', 'point_like_prob', 'blended_prob', 'mumax_minus_mag',
        'flux_detection_total', 'fluxerr_detection_total', 'flux_vis_psf', 'gal_ebv', 'e_ra', 'e_dec')


def one(path):
    files = glob.glob(str(path))
    if len(files) != 1:
        raise RuntimeError(f'expected one file for {path}, got {files}')
    return Path(files[0])


def table(kind, rows, m, nb, extra):
    lay = json.loads((RUN/'layout.json').read_text()); cols = [lay['native_labels'].index(b) for b in LEGACY]
    arr = {k: np.load(RUN/kind/f'{k}.npy', mmap_mode='r') for k in ('y', 'noise', 'ra', 'dec', 'l', 'b', 'ebv', 'role', 'cell', 'zspec')}
    p = photometry(m)
    out = dict(parent_row=rows, legacy_y=np.asarray(arr['y'][rows][:, cols]), legacy_noise=np.asarray(arr['noise'][rows][:, cols]),
               euclid_flux=p.flux, euclid_variance=p.variance, neighbours=nb, **extra)
    for k in ('ra', 'dec', 'l', 'b', 'ebv', 'role', 'cell', 'zspec'):
        out[k] = np.asarray(arr[k][rows])
    for k in KEEP:
        out[k] = np.asarray(m[k])
    out['euclid_cell'] = hp.ang2pix(16, out['l'], out['b'], lonlat=True, nest=True)
    return out


if __name__ == '__main__':
    (BASE/'tables').mkdir(exist_ok=True)
    # QSOs
    t = np.load(QROOT/'targets.npz'); ra = np.load(RUN/'qso'/'ra.npy', mmap_mode='r')
    assert np.allclose(np.asarray(ra), t['ra']), 'parent QSO arrays not aligned with targets'
    mpath = one(BASE/'qso_targets/*/match.npz'); m = dict(np.load(mpath)); nb = np.load(mpath.with_name('neighbours.npz'))['neighbours']
    rows = np.flatnonzero(m['matched'])
    q = table('qso', rows, {k: v[rows] for k, v in m.items()}, nb[rows], dict(object_name=t['object_id'][rows]))
    np.savez_compressed(BASE/'tables/qso.npz', **q)
    print('qso rows', len(rows), flush=True)
    # background
    r = np.load(BASE/'background/rows.npz'); prow = r['rows']
    pos_path = one(BASE/'background/positional/*/match.npz'); pro_path = one(BASE/'background/propagated/*/match.npz')
    pos = dict(np.load(pos_path)); pro = dict(np.load(pro_path))
    npos = np.load(pos_path.with_name('neighbours.npz'))['neighbours']; npro = np.load(pro_path.with_name('neighbours.npz'))['neighbours']
    has = np.isfinite(r['pmra']) & np.isfinite(r['pmdec']); hi = np.flatnonzero(has)
    for k in pos:
        pos[k] = np.asarray(pos[k]).copy(); pos[k][hi] = pro[k]
    npos = npos.copy(); npos[hi] = npro
    path = np.where(has, 'propagated', 'positional')
    keep = np.flatnonzero(pos['matched'])
    b = table('stars', prow[keep], {k: v[keep] for k, v in pos.items()}, npos[keep],
              dict(match_path=path[keep], field=r['field'][keep], gaia_id=r['gaia_id'][keep]))
    np.savez_compressed(BASE/'tables/background.npz', **b)
    (BASE/'tables/provenance.json').write_text(json.dumps(dict(
        qso_match=str(mpath), background_matches=[str(pos_path), str(pro_path)], parent_run=str(RUN),
        legacy_columns=LEGACY, n_qso=int(len(rows)), n_background=int(len(keep)),
        n_background_rows_in_cones=int(len(prow)),
        rule='Euclid fluxes native, not dereddened; Legacy luptitudes are the parent dereddened training values'), indent=1))
    print('background rows', len(keep), 'of', len(prow), flush=True)
