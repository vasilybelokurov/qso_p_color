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


def match_gaia(gaia_ids, cache: Path, *, log=print) -> dict:
    """MER source carrying each Gaia DR3 ``source_id`` (MER's own association), from either table.

    Proper-motion-safe alternative to a positional match: MER positions are at the Euclid epoch, so stars with
    tens of mas/yr move by > 0.3" from their Gaia DR3 (2016.0) positions. ``gaia_ids <= 0`` are not looked up.
    If several MER rows carry one id (tile overlaps), the one with the highest ``gaia_match_quality``, then the
    lowest object_id, is kept; ``n_gaia_rows`` records the multiplicity.
    """
    import sqlutilpy as sqlutil
    ids = np.asarray(gaia_ids, np.int64)
    key = hashlib.sha256(ids.tobytes() + json.dumps([TABLES, COLUMNS, 'gaia']).encode()).hexdigest()[:16]
    root = Path(cache)/key; root.mkdir(parents=True, exist_ok=True); final = root/'match.npz'
    if final.exists():
        return dict(np.load(final, allow_pickle=False))
    u = np.unique(ids[ids > 0])
    cols = ', '.join(f'e.{c}' for c in COLUMNS)
    parts = []
    for code, table in enumerate(TABLES):
        t = time.monotonic()
        r = sqlutil.local_join(f"""SELECT m.g, e.right_ascension AS e_ra, e.declination AS e_dec, {cols}
            FROM mytab m JOIN {table} e ON e.gaia_id = m.g""", 'mytab', (u,), ('g',), asDict=True)
        r = {k: np.asarray(v) for k, v in r.items()}; r['table'] = np.full(len(r['g']), code)
        parts.append(r)
        log(f'gaia-id lookup {table}: {len(r["g"])} MER rows for {len(u)} ids, {time.monotonic()-t:.0f} s')
    r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    q = np.where(np.isfinite(r['gaia_match_quality'].astype(float)), r['gaia_match_quality'], -np.inf)
    order = np.lexsort((r['object_id'], -q, r['g']))          # best candidate first within each id
    g_sorted = r['g'][order]
    first = np.ones(len(order), bool); first[1:] = g_sorted[1:] != g_sorted[:-1]
    pick = order[first]; counts = np.diff(np.append(np.flatnonzero(first), len(order)))
    where = np.searchsorted(r['g'][pick], ids)                 # r['g'][pick] is sorted ascending
    hit = (ids > 0) & (where < len(pick))
    hit[hit] = r['g'][pick][where[hit]] == ids[hit]
    out = {}
    for k in ['e_ra', 'e_dec', *COLUMNS, 'table']:
        v = r[k][pick]
        fill = np.nan if v.dtype.kind == 'f' else -1 if k == 'table' else -9999
        o = np.full(len(ids), fill, dtype=v.dtype if v.dtype.kind == 'f' else np.int64)
        o[hit] = v[where[hit]]; out[k] = o
    out['n_gaia_rows'] = np.where(hit, counts[np.minimum(where, len(pick)-1)], 0)
    out['matched'] = hit
    np.savez_compressed(final, **out)
    (root/'provenance.json').write_text(json.dumps(dict(tables=TABLES, rule='MER gaia_id == Gaia DR3 source_id',
        n=len(ids), n_looked_up=int(len(u)), n_matched=int(hit.sum()),
        created=time.strftime('%Y-%m-%d %H:%M:%S %Z')), indent=1))
    return out


def native_flux(flux_ujy, err_ujy):
    """Catalogue micro-Jansky (AB) to native flux and variance; non-finite or non-positive errors are unusable."""
    f = np.asarray(flux_ujy, float)*UJY_TO_NATIVE; e = np.asarray(err_ujy, float)*UJY_TO_NATIVE
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    return np.where(ok, f, np.nan), np.where(ok, e**2, np.inf)
