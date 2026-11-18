"""Deployment rows of the Euclid select/calib/test cones (docs/euclid/PLAN.md 3.3), with labels.

Deployment selection = clean + isolated + point-like (docs/euclid/settings.json), within the cone radius.
Labels: known QSO = DESI DR1 / DR16Q master object within 1" (with zspec); likely QSO = Quaia (G < 20) within 1" or
Legacy W1 - W2 > 0.8 Vega at S/N >= 5 in both (unless r - z > 2), the parent background-cleaning rule;
DESI galaxy = DESI DR1 spectroscopic GALAXY within 1". Legacy DR9 counterpart = nearest within 1" (fluxes kept for
the Legacy-only baseline).
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from qso_pcolor.euclid_bundle import clean, point_like
from qso_pcolor.euclid_data import ISOLATION_DMAG, ISOLATION_RADIUS_ARCSEC

BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
WISE_AB = 0.8 - (3.339 - 2.699)


def unit(ra, dec):
    r, d = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])


def nearest(ra1, dec1, ra2, dec2, radius_arcsec):
    """Index into set 2 of the nearest source within radius for each source of set 1 (-1 if none)."""
    if not len(ra2):
        return np.full(len(ra1), -1)
    d, i = cKDTree(unit(ra2, dec2)).query(unit(ra1, dec1), distance_upper_bound=2*np.sin(np.deg2rad(radius_arcsec/3600)/2))
    return np.where(np.isfinite(d), i, -1)


_EXT = {}


def externals():
    if not _EXT:
        from astropy.io import fits
        q = fits.getdata(Path('~/data/catalogues/quaia_G20.0.fits').expanduser(), 1)
        o = np.load('/Users/vasilybelokurov/data/qso_p_color/catalogues/qso_sdss_desi/77d7aa514e9e3e47/objects.npz')
        _EXT.update(quaia=(np.asarray(q['ra'], float), np.asarray(q['dec'], float)),
                    qso=(o['ra'], o['dec'], o['zspec']))
    return _EXT


def deployment(path, settings):
    c = dict(np.load(path, allow_pickle=False)); ra0, dec0 = c['centre']; R = float(c['radius_deg'])
    m = {k[4:]: v for k, v in c.items() if k.startswith('mer_')}
    m['matched'] = np.ones(len(m['e_ra']), bool)
    # isolation neighbours from the cone's own MER rows (fetched with a 2" margin)
    xyz = unit(m['e_ra'], m['e_dec']); tree = cKDTree(xyz)
    pairs = tree.query_pairs(2*np.sin(np.deg2rad(ISOLATION_RADIUS_ARCSEC/3600)/2), output_type='ndarray')
    f = np.nan_to_num(np.asarray(m['flux_vis_to_y_templfit'], float))
    nb = np.zeros(len(f), int); ratio = 10**(-0.4*ISOLATION_DMAG)
    for a, b in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
        np.add.at(nb, a, f[b] > ratio*f[a])
    inside = (unit([ra0], [dec0]) @ xyz.T)[0] >= np.cos(np.deg2rad(R))
    keep = inside & clean(m, neighbours=nb, bad_bits=settings['bad_bits'], vis_bright=settings['vis_bright']) \
        & point_like(m, settings['p_min'])
    rows = {k: np.asarray(v)[keep] for k, v in m.items()}
    rows['neighbours'] = nb[keep]
    ext = externals()
    iq = nearest(rows['e_ra'], rows['e_dec'], *ext['qso'][:2], 1.0)
    rows['zspec'] = np.where(iq >= 0, ext['qso'][2][np.maximum(iq, 0)], np.nan)
    rows['known_qso'] = iq >= 0
    rows['quaia'] = nearest(rows['e_ra'], rows['e_dec'], *ext['quaia'], 1.0) >= 0
    leg = {k[7:]: v for k, v in c.items() if k.startswith('legacy_')}
    il = nearest(rows['e_ra'], rows['e_dec'], leg['ra'], leg['dec'], 1.0); rows['legacy_index'] = il
    for k in ('type', 'release', 'maskbits') + tuple(f'{p}_{b}' for b in ('g', 'r', 'z', 'w1', 'w2')
                                                       for p in ('flux', 'flux_ivar', 'nobs')):
        v = np.asarray(leg[k]); fill = '' if v.dtype.kind in 'US' else np.nan if v.dtype.kind == 'f' else -1
        rows[f'legacy_{k}'] = np.where(il >= 0, v[np.maximum(il, 0)], fill)
    with np.errstate(invalid='ignore', divide='ignore'):
        sn = lambda b: rows[f'legacy_flux_{b}']*np.sqrt(rows[f'legacy_flux_ivar_{b}'])
        mag = lambda b: 22.5 - 2.5*np.log10(rows[f'legacy_flux_{b}'])
        wise = (sn('w1') >= 5) & (sn('w2') >= 5) & (mag('w1') - mag('w2') > WISE_AB) & ~((mag('r') - mag('z')) > 2)
    rows['likely_qso'] = rows['quaia'] | np.nan_to_num(wise).astype(bool)
    rows['desi_galaxy'] = nearest(rows['e_ra'], rows['e_dec'], c['desigal_ra'], c['desigal_dec'], 1.0) >= 0 \
        if len(c['desigal_ra']) else np.zeros(len(rows['e_ra']), bool)
    all_gal = dict(n=int(len(c['desigal_ra'])))
    if len(c['desigal_ra']):
        # T-D1: DESI galaxies with a clean isolated MER counterpart, and their point_like_prob
        ig = nearest(c['desigal_ra'], c['desigal_dec'], m['e_ra'], m['e_dec'], 1.0)
        okg = (ig >= 0)
        okg[okg] = clean({k: np.asarray(v)[ig[okg]] for k, v in m.items()}, neighbours=nb[ig[okg]],
                         bad_bits=settings['bad_bits'], vis_bright=settings['vis_bright'])
        all_gal.update(point_like_prob=np.asarray(m['point_like_prob'])[ig[okg]].tolist(),
                       vis_flux=np.asarray(m['flux_vis_to_y_templfit'])[ig[okg]].tolist())
    return rows, all_gal
