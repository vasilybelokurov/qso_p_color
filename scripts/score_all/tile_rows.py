"""Per-tile row tables for Phase E step 3 (docs/score_all/PHASE_AB.md, "step 3: design and tests").

Deep tiles -> candidate pool: deep MER rows owned by the tile (nearest tile centre), vis_det = 1, spurious_flag = 0,
det_quality_flag free of bad_bits, VIS brighter-cut as the deployment selection, VIS <= VIS_MAX; NO isolation and NO
point-like cut (the mimic classifier learns what the wide selection does). Kept per row: classifier features, the
deep isolation-neighbour count, dereddened Euclid luptitudes (VIS, Y, J, H) and their diagonal variances (dereddened
flux variances through the luptitude transform; colour covariances follow from the shared VIS term), the known-QSO
flag (DESI DR1 / DR16Q master within 1", as cones.deployment).

Wide tiles -> the full wide deployment selection (scripts/euclid/cones.py: cleaning, isolation, point-like), owned by
the tile, positions only (for matching) plus VIS.

Field tables are concatenated over tiles and cached as
~/data/qso_p_color/score_all/phase_e/rows_<survey><field>.npz (float32), keyed by the tile list. One tile in memory
at a time (~0.5 GB peak).
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/euclid')); sys.path.insert(0, str(ROOT/'scripts/score_all'))
DATA = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
OUT = Path.home()/'data/qso_p_color/score_all/phase_e'
VIS_MAX = 26.75
FEATURES = ('vis', 'point_like_prob', 'fwhm', 'mumax_minus_mag', 'snr_vis', 'neighbours')


def _owner(ra, dec, centre, centres):
    from cones import unit
    _, i = cKDTree(unit(centres[:, 0], centres[:, 1])).query(unit(ra, dec))
    me = int(np.argmin(np.sum((centres - np.asarray(centre))**2, 1)))
    return i == me


def _neighbours(m):
    from cones import unit
    from qso_pcolor.euclid_data import ISOLATION_DMAG, ISOLATION_RADIUS_ARCSEC
    xyz = unit(m['e_ra'], m['e_dec']); tree = cKDTree(xyz)
    pairs = tree.query_pairs(2*np.sin(np.deg2rad(ISOLATION_RADIUS_ARCSEC/3600)/2), output_type='ndarray')
    f = np.nan_to_num(np.asarray(m['flux_vis_to_y_templfit'], float))
    nb = np.zeros(len(f), int); ratio = 10**(-0.4*ISOLATION_DMAG)
    for a, b in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
        np.add.at(nb, a, f[b] > ratio*f[a])
    return nb


def deep_candidates(path, settings, centres):
    from cones import externals, nearest
    from qso_pcolor.euclid_bundle import clean, dereddened, luptitudes, photometry
    c = dict(np.load(path, allow_pickle=False)); centre = c['centre']
    m = {k[4:]: v for k, v in c.items() if k.startswith('mer_')}
    m['matched'] = np.ones(len(m['e_ra']), bool)
    nb = _neighbours(m)
    keep = clean(m, neighbours=None, bad_bits=settings['bad_bits'], vis_bright=settings['vis_bright'])
    keep &= _owner(m['e_ra'], m['e_dec'], centre, centres)
    p = dereddened(photometry({k: np.asarray(v)[keep] for k, v in m.items()}), np.asarray(m['gal_ebv'], float)[keep])
    soft = np.asarray(settings['softening'], float)
    y = np.empty_like(p.flux); n = np.empty_like(p.flux)
    for j in range(4):
        y[:, j], n[:, j] = luptitudes(p.flux[:, j], p.variance[:, j], soft[j])
    with np.errstate(invalid='ignore', divide='ignore'):
        snr = p.flux[:, 0]/np.sqrt(p.variance[:, 0])
    ok = np.isfinite(y[:, 0]) & (y[:, 0] <= VIS_MAX)
    idx = np.flatnonzero(keep)[ok]
    out = dict(ra=np.asarray(m['e_ra'], float)[idx], dec=np.asarray(m['e_dec'], float)[idx],
               vis=y[ok, 0], point_like_prob=np.asarray(m['point_like_prob'], float)[idx],
               fwhm=np.asarray(m['fwhm'], float)[idx], mumax_minus_mag=np.asarray(m['mumax_minus_mag'], float)[idx],
               snr_vis=snr[ok], neighbours=nb[idx].astype(float), lup=y[ok], lupvar=n[ok])
    q = externals()['qso']
    out['known_qso'] = nearest(out['ra'], out['dec'], q[0], q[1], 1.0) >= 0
    return out


def wide_selected(path, settings, centres):
    from cones import deployment
    rows, _ = deployment(path, settings)
    c = np.load(path); own = _owner(rows['e_ra'], rows['e_dec'], c['centre'], centres) if len(rows['e_ra']) else []
    return dict(ra=np.asarray(rows['e_ra'], float)[own], dec=np.asarray(rows['e_dec'], float)[own],
                vis_flux=np.asarray(rows['flux_vis_to_y_templfit'], float)[own])


def field_table(survey, field, settings, limit=None):
    """Concatenated row table for one field (cached). survey: 'deep' (candidates) or 'wide' (deployment)."""
    tdir = DATA/f'{survey}_tiles'/field
    t = json.loads((tdir/'tiles.json').read_text()); centres = np.asarray(t['centres'])
    paths = sorted(tdir.glob('tile_*.npz'))[:limit]
    key = hashlib.sha256(json.dumps([survey, field, [p.name for p in paths], t, VIS_MAX, settings['bad_bits'],
                                     settings['vis_bright'], settings['p_min'], settings['softening']]).encode()).hexdigest()[:12]
    cache = OUT/f'rows_{survey}{field}_{key}.npz'
    if cache.exists():
        return dict(np.load(cache))
    fn = deep_candidates if survey == 'deep' else wide_selected
    parts = []
    for p in paths:
        r = fn(p, settings, centres); r['tile'] = np.full(len(r['ra']), int(p.stem.split('_')[1]))
        parts.append({k: (v.astype(np.float32) if v.dtype.kind == 'f' and k not in ('ra', 'dec') else v) for k, v in r.items()})
        print(survey, field, p.name, len(r['ra']), flush=True)
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    OUT.mkdir(parents=True, exist_ok=True); np.savez(cache, **out)
    return out


NSIDE = 2048            # HEALPix pixel ~ 8.2e-4 deg^2 (1.7' on a side)


def coverage(survey, field, limit=None):
    """Covered HEALPix pixels (nside NSIDE, nested) of a survey's tiles in a field and the covered area (deg^2).

    All MER rows of the tiles count (no selection). A pixel is covered if it holds >= half the median row count of
    occupied pixels, which drops pixels cut by the coverage edge; cached next to the row tables."""
    import healpy as hp
    tdir = DATA/f'{survey}_tiles'/field
    paths = sorted(tdir.glob('tile_*.npz'))[:limit]
    key = hashlib.sha256(json.dumps([survey, field, [p.name for p in paths], NSIDE]).encode()).hexdigest()[:12]
    cache = OUT/f'coverage_{survey}{field}_{key}.npz'
    if cache.exists():
        d = np.load(cache); return d['pix'], float(d['area'])
    counts = {}
    for p in paths:
        with np.load(p) as c:
            ra, dec = c['mer_e_ra'], c['mer_e_dec']
        if not len(ra):
            continue
        u, n = np.unique(hp.ang2pix(NSIDE, ra, dec, lonlat=True, nest=True), return_counts=True)
        for a, b in zip(u, n):
            counts[a] = max(counts.get(a, 0), b)       # tiles overlap: a pixel's rows appear in several tiles
    pix = np.array(list(counts), np.int64); n = np.array(list(counts.values()))
    pix = np.sort(pix[n >= 0.5*np.median(n)]) if len(n) else pix
    area = len(pix)*hp.nside2pixarea(NSIDE, degrees=True)
    OUT.mkdir(parents=True, exist_ok=True); np.savez(cache, pix=pix, area=area)
    return pix, area


def in_pixels(ra, dec, pix):
    import healpy as hp
    p = hp.ang2pix(NSIDE, np.asarray(ra, float), np.asarray(dec, float), lonlat=True, nest=True)
    return np.isin(p, pix)
