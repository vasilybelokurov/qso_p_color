"""Euclid DR1 MER photometry: positional matching against WSDB, cached on disk.

Two disjoint MER tables hold the DR1 wide survey (``mer_catalogue_wide_survey`` and
``mer_catalogue_wide_mode``; measured 18 November 2026: no object of a 50,000-quasar sample is in both).
Every target gets the nearest MER source within ``radius_arcsec`` from either table, the number of MER
sources inside that radius, and a broad set of flux, flag and morphology columns, so that column choices
(PSF vs template-fit vs aperture) can be decided later from the cached rows without re-querying.

Fluxes are stored exactly as catalogued (micro-Jansky, AB); conversion to native units happens in
:func:`native_flux` (magnitude 22.5 <-> flux 1: native = uJy * 10**((22.5 - 23.9)/2.5)).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

TABLES = ('euclid_dr1.mer_catalogue_wide_survey', 'euclid_dr1.mer_catalogue_wide_mode')
UJY_TO_NATIVE = 10**((22.5 - 23.9)/2.5)

FLUX_COLUMNS = (
    'flux_vis_psf', 'flux_vis_1fwhm_aper', 'flux_vis_2fwhm_aper', 'flux_vis_sersic',
    'flux_y_templfit', 'flux_j_templfit', 'flux_h_templfit',
    'flux_y_2fwhm_aper', 'flux_j_2fwhm_aper', 'flux_h_2fwhm_aper',
    'flux_vis_to_y_templfit', 'flux_vis_to_j_templfit', 'flux_vis_to_h_templfit',
    'flux_detection_total',
)
ERROR_COLUMNS = tuple(c.replace('flux_', 'fluxerr_', 1) for c in FLUX_COLUMNS)
OTHER_COLUMNS = (
    'object_id', 'tile_index', 'vis_det', 'flag_vis', 'flag_y', 'flag_j', 'flag_h', 'det_quality_flag',
    'spurious_flag', 'spurious_prob', 'point_like_flag', 'point_like_prob', 'extended_flag', 'extended_prob',
    'mumax_minus_mag', 'mu_max', 'blended_prob', 'deblended_flag', 'parent_id', 'binary_flag', 'she_flag',
    'variable_flag', 'gal_ebv', 'fwhm', 'semimajor_axis', 'ellipticity', 'kron_radius', 'segmentation_area',
    'gaia_id', 'gaia_match_quality',
)
COLUMNS = OTHER_COLUMNS + FLUX_COLUMNS + ERROR_COLUMNS


def _order(ra, dec):
    """Spatially coherent probe order (HEALPix nested, nside 1024) so index pages are reused."""
    import healpy
    return np.argsort(healpy.ang2pix(1024, ra, dec, lonlat=True, nest=True), kind='stable')


def _query(table: str, radius_arcsec: float) -> str:
    cols = ', '.join(f'e.{c}' for c in COLUMNS)
    r = radius_arcsec/3600.
    return f"""
    SELECT m.i, e.right_ascension AS e_ra, e.declination AS e_dec,
           q3c_dist(m.ra, m.dec, e.right_ascension, e.declination)*3600. AS sep_arcsec,
           (SELECT count(*) FROM {table} c
             WHERE q3c_join(m.ra, m.dec, c.right_ascension, c.declination, {r})) AS n_within, {cols}
    FROM mytab m
    LEFT JOIN LATERAL (SELECT * FROM {table} e
                       WHERE q3c_join(m.ra, m.dec, e.right_ascension, e.declination, {r})
                       ORDER BY q3c_dist(m.ra, m.dec, e.right_ascension, e.declination), e.object_id
                       LIMIT 1) e ON true"""


def match(ra, dec, cache: Path, *, radius_arcsec: float = 1.0, chunk: int = 100_000, log=print) -> dict:
    """Nearest MER source within ``radius_arcsec`` (either table) for every position; resumable chunks.

    Returns a dict of arrays aligned with the input (``matched`` boolean; ``table`` 0 = wide_survey,
    1 = wide_mode, -1 = none). ``n_within`` counts MER sources of both tables inside the radius.
    """
    import sqlutilpy as sqlutil
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    if ra.shape != dec.shape or ra.ndim != 1 or not np.isfinite(ra + dec).all():
        raise ValueError('finite 1-d ra/dec required')
    key = hashlib.sha256(ra.tobytes() + dec.tobytes() + json.dumps([TABLES, COLUMNS, radius_arcsec]).encode()).hexdigest()[:16]
    root = Path(cache)/key; root.mkdir(parents=True, exist_ok=True)
    final = root/'match.npz'
    if final.exists():
        return dict(np.load(final, allow_pickle=False))
    order = _order(ra, dec)
    parts = []
    for lo in range(0, len(order), chunk):
        path = root/f'chunk_{lo:09d}.npz'
        if not path.exists():
            ii = order[lo:lo+chunk]; t = time.monotonic(); res = []
            for code, table in enumerate(TABLES):
                r = sqlutil.local_join(_query(table, radius_arcsec), 'mytab', (ii, ra[ii], dec[ii]),
                                       ('i', 'ra', 'dec'), asDict=True)
                r['table'] = np.where(np.isfinite(np.asarray(r['sep_arcsec'], float)), code, -1)
                res.append(r)
            a, b = res
            ia = np.argsort(a['i']); ib = np.argsort(b['i'])
            a = {k: np.asarray(v)[ia] for k, v in a.items()}; b = {k: np.asarray(v)[ib] for k, v in b.items()}
            sa = np.where(a['table'] >= 0, a['sep_arcsec'], np.inf); sb = np.where(b['table'] >= 0, b['sep_arcsec'], np.inf)
            take_b = sb < sa
            out = {k: np.where(take_b, b[k], a[k]) for k in a if k not in ('n_within',)}
            out['n_within'] = np.asarray(a['n_within'], int) + np.asarray(b['n_within'], int)
            np.savez_compressed(path, **out)
            log(f'euclid match chunk {lo}: {len(ii)} rows, {int((out["table"] >= 0).sum())} matched, '
                f'{time.monotonic()-t:.0f} s')
        parts.append(dict(np.load(path, allow_pickle=False)))
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    back = np.empty(len(order), int); back[merged['i'].astype(int)] = np.arange(len(order))
    result = {k: v[back] for k, v in merged.items() if k != 'i'}
    result['matched'] = result['table'] >= 0
    np.savez_compressed(final, **result)
    (root/'provenance.json').write_text(json.dumps(dict(tables=TABLES, columns=COLUMNS, radius_arcsec=radius_arcsec,
        n=len(ra), n_matched=int(result['matched'].sum()), created=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        rule='nearest MER source within radius from either table; ties broken by object_id'), indent=1))
    for p in root.glob('chunk_*.npz'):
        p.unlink()
    return result


GAIA_NONE = 9223372036854775807     # MER gaia_id sentinel for "no Gaia association" (measured)


ISOLATION_RADIUS_ARCSEC = 1.5
ISOLATION_DMAG = 2.5


def isolation_neighbours(object_id, ra, dec, vis_flux, *, radius_arcsec: float = ISOLATION_RADIUS_ARCSEC,
                         dmag: float = ISOLATION_DMAG, chunk: int = 100_000) -> np.ndarray:
    """Number of isolation neighbours of each MER object (docs/euclid/PLAN.md 2.1).

    A neighbour is another MER source (either table) within ``radius_arcsec`` of the object's MER position with
    vis_det = 1, spurious_flag = 0 and a VIS template-fit flux brighter than the object's divided by 10**(0.4 dmag)
    (a NaN object flux counts every such source). Chance faint or spurious detections are not neighbours (they
    removed 15% of quasars under a plain count; review D2.1). Positions and fluxes come from the cached match, so
    the server only runs indexed cone counts (a join on object_id was ~1 h for 17k rows).
    """
    import sqlutilpy as sqlutil
    oid = np.asarray(object_id, np.int64); ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    f = np.nan_to_num(np.asarray(vis_flux, float), nan=0.)*10**(-0.4*dmag)
    out = np.zeros(len(oid), int); order = _order(ra, dec); r = radius_arcsec/3600.
    for lo in range(0, len(order), chunk):
        ii = order[lo:lo + chunk]
        parts = [f"""(SELECT count(*) FROM {t} n WHERE q3c_radial_query(n.right_ascension, n.declination, m.ra, m.dec, {r})
                 AND n.object_id <> m.oid AND n.vis_det = 1 AND n.spurious_flag = 0
                 AND n.flux_vis_to_y_templfit > m.fmin)""" for t in TABLES]
        res = sqlutil.local_join(f"SELECT m.i, {' + '.join(parts)} AS n FROM mytab m", 'mytab',
                                 (ii, oid[ii], ra[ii], dec[ii], f[ii]), ('i', 'oid', 'ra', 'dec', 'fmin'), asDict=True,
                                 preamb="SET statement_timeout = '3600s'")
        out[np.asarray(res['i'], int)] = np.asarray(res['n'], int)
    return out


def native_flux(flux_ujy, err_ujy):
    """Catalogue micro-Jansky (AB) to native flux and variance; non-finite or non-positive errors are unusable."""
    f = np.asarray(flux_ujy, float)*UJY_TO_NATIVE; e = np.asarray(err_ujy, float)*UJY_TO_NATIVE
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    return np.where(ok, f, np.nan), np.where(ok, e**2, np.inf)
