"""WSDB queries for the training and candidate samples.

Photometric systems available on this server (verified 2026-09-18)
------------------------------------------------------------------
``ls_dr9_grzw``
    ``desi_dr1.photometry`` for quasars and ``decals_dr9.main`` for the
    background.  The quasar side needs **no positional crossmatch at all**: the
    DESI photometry table carries the Legacy Surveys DR9 measurement for each
    ``targetid``, so quasar photometry is joined to redshift by identifier.  That
    removes a whole class of mismatch error, and it is the reason this is the
    recommended v1 system.  Bands: g, r, z, W1, W2.  1,645,842 DESI DR1 quasars
    have ``spectype='QSO'``, ``zwarn=0`` and ``zcat_primary``.

``ls_dr11_grizw``
    ``decals_dr11.main`` for everything, with a positional crossmatch to the
    spectroscopic quasars.  Adds the i band in the south.  ``release`` is 11010
    (south, DECam) or 11011 (north, BASS/MzLS) — these are **different
    photometric systems** and must be modelled separately.

``sdss_ugriz``
    ``sdssdr16qso.main`` PSF fluxes for quasars (750,414 rows) and an SDSS
    photometric table for the background.  Native SDSS system, u band included.

Never mix them.  Every model records its system identifier and the scorer
refuses a mismatch.

Costs measured on this server
-----------------------------
A full pull of 1.6M DESI quasar positions plus photometry takes minutes and
should be cached to disk; a 1-degree cone on ``decals_dr9.main`` returns in
seconds.  Every function here writes an ``.npz`` cache and reloads from it, so
that a server outage stalls nothing already fetched.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "SYSTEMS",
    "cached_query",
    "fetch_desi_qso_training",
    "fetch_ls_background",
    "fetch_dr16q_training",
    "fetch_candidate_photometry",
    "galactic_from_equatorial",
]


SYSTEMS = {
    "ls_dr9_grzw": {
        "bands": ("g", "r", "z", "w1", "w2"),
        "optical_bands": ("g", "r", "z"),
        "reference_band": "r",
        "qso_table": "desi_dr1.photometry",
        "background_table": "decals_dr9.main",
        "release_codes": {9010: "south", 9011: "north"},
    },
    "ls_dr11_grizw": {
        "bands": ("g", "r", "i", "z", "w1", "w2"),
        "optical_bands": ("g", "r", "i", "z"),
        "reference_band": "r",
        "qso_table": "decals_dr11.main",
        "background_table": "decals_dr11.main",
        "release_codes": {11010: "south", 11011: "north"},
    },
    "sdss_ugriz": {
        "bands": ("u", "g", "r", "i", "z"),
        "optical_bands": ("u", "g", "r", "i", "z"),
        "reference_band": "r",
        "qso_table": "sdssdr16qso.main",
        "background_table": None,
        "release_codes": {},
    },
}


def _sqlutil():
    import sqlutilpy as sqlutil  # imported lazily so tests run without a server

    return sqlutil


def cached_query(query: str, cache: str | Path, *, refresh: bool = False, **kw) -> dict:
    """Run a WSDB query, caching the result as a compressed ``.npz``.

    The file name is the caller's name plus a short hash of the query text, so a
    changed query gets a fresh cache automatically and can never be served stale
    rows.  The query itself is stored in the archive under ``_query``, so any
    cache can be traced back to what produced it.
    """
    cache = Path(cache)
    # Key the file on the query text.  Keying on the caller's chosen name alone
    # would silently return the wrong rows the moment a cone radius or a cut
    # changed, which is the kind of error that survives all the way into a plot.
    digest = hashlib.sha1(query.encode()).hexdigest()[:10]
    if cache.suffix == ".npz":
        cache = cache.with_name(f"{cache.stem}_{digest}.npz")
    else:
        cache = cache.with_name(f"{cache.name}_{digest}.npz")

    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=True) as z:
            return {k: z[k] for k in z.files if not k.startswith("_")}

    t0 = time.time()
    res = _sqlutil().get(query, asDict=True, **kw)
    log.info("query returned %d rows in %.1f s", len(next(iter(res.values()))), time.time() - t0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, _query=np.array(query), **res)
    return res


def galactic_from_equatorial(ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Galactic (l, b) in degrees.  Computed locally, not in SQL."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg).galactic
    return c.l.deg, c.b.deg


# -- quasar training samples ----------------------------------------------

_DESI_QSO_QUERY = """
SELECT z.targetid,
       z.z            AS zspec,
       z.zerr         AS zspec_err,
       p.ra, p.dec, p.release, p.morphtype, p.maskbits,
       p.flux_g,  p.flux_r,  p.flux_z,  p.flux_w1,  p.flux_w2,
       p.flux_ivar_g, p.flux_ivar_r, p.flux_ivar_z,
       p.flux_ivar_w1, p.flux_ivar_w2,
       p.mw_transmission_g, p.mw_transmission_r, p.mw_transmission_z,
       p.mw_transmission_w1, p.mw_transmission_w2,
       p.fracflux_g, p.fracflux_r, p.fracflux_z,
       p.nobs_g, p.nobs_r, p.nobs_z,
       p.psfdepth_g, p.psfdepth_r, p.psfdepth_z,
       p.parallax, p.parallax_ivar, p.pmra, p.pmdec
FROM desi_dr1.zpix z
JOIN desi_dr1.photometry p ON p.targetid = z.targetid
WHERE z.spectype = 'QSO'
  AND z.zwarn = 0
  AND z.zcat_primary
  AND z.z > %(zmin)s AND z.z < %(zmax)s
"""


def fetch_desi_qso_training(
    cache: str | Path,
    *,
    zmin: float = 0.05,
    zmax: float = 5.0,
    refresh: bool = False,
) -> dict:
    """DESI DR1 quasars with their Legacy Surveys DR9 photometry.

    Joined on ``targetid``, so the photometry belongs to the object that was
    spectroscopically observed — no crossmatch radius to choose and no ambiguity
    in crowded fields.

    Selection applied: ``spectype='QSO'``, ``zwarn=0``, ``zcat_primary`` (one row
    per astrophysical object, which is the de-duplication the plan requires), and
    a redshift range.  Photometric quality cuts are deliberately **not** applied
    here; they belong in the sample-building step where they are recorded.

    Caveat that must travel with this sample: DESI quasar targeting uses a
    random forest on exactly these Legacy Surveys colours, so this is
    :math:`p(\\mathbf{c}\\mid Q, z, \\text{selected by DESI})`.  Combine with
    :func:`fetch_dr16q_training`, whose selection channels differ, and report
    held-out likelihood per channel.
    """
    return cached_query(
        _DESI_QSO_QUERY % {"zmin": zmin, "zmax": zmax}, cache, refresh=refresh
    )


_DR16Q_QUERY = """
SELECT sdss_name, ra, dec, z, zwarning, sdss_morpho,
       psfflux, psfflux_ivar, extinction,
       boss_target1, eboss_target0, ancillary_target1,
       w1_flux, w1_flux_ivar, w2_flux, w2_flux_ivar
FROM sdssdr16qso.main
WHERE z > %(zmin)s AND z < %(zmax)s
"""


def fetch_dr16q_training(
    cache: str | Path,
    *,
    zmin: float = 0.05,
    zmax: float = 5.0,
    refresh: bool = False,
) -> dict:
    """SDSS DR16Q quasars with native SDSS PSF fluxes (u, g, r, i, z arrays).

    ``psfflux`` and ``psfflux_ivar`` are five-element arrays in nanomaggies;
    ``extinction`` holds the magnitudes of Galactic extinction per band, which
    must be converted to a transmission before use:
    ``T = 10^(-0.4 * extinction)``.

    Use this for the ``sdss_ugriz`` system, or as an independently selected
    validation sample for a Legacy model after a positional crossmatch.
    """
    return cached_query(
        _DR16Q_QUERY % {"zmin": zmin, "zmax": zmax}, cache, refresh=refresh
    )


# -- background sample ----------------------------------------------------

_LS_BACKGROUND_QUERY = """
SELECT ra, dec, release, type, maskbits,
       flux_g, flux_r, flux_z, flux_w1, flux_w2,
       flux_ivar_g, flux_ivar_r, flux_ivar_z, flux_ivar_w1, flux_ivar_w2,
       mw_transmission_g, mw_transmission_r, mw_transmission_z,
       mw_transmission_w1, mw_transmission_w2,
       fracflux_g, fracflux_r, fracflux_z,
       nobs_g, nobs_r, nobs_z,
       psfdepth_g, psfdepth_r, psfdepth_z
FROM {table}
WHERE q3c_radial_query(ra, dec, %(ra)s, %(dec)s, %(radius)s)
  AND flux_ivar_r > 0
  AND maskbits = 0
"""


def fetch_ls_background(
    cache: str | Path,
    *,
    ra: float,
    dec: float,
    radius_deg: float,
    table: str = "decals_dr9.main",
    refresh: bool = False,
) -> dict:
    """All usable Legacy Surveys sources in a cone: the background population.

    This is *everything* the imaging contains at these positions, which is
    exactly what a chance projection beside the primary draws from.  No
    star/galaxy split is imposed, because imposing one would require a selection
    function we do not have; the mixture model learns whatever structure is
    there.

    ``maskbits = 0`` removes sources near bright stars and other flagged
    regions.  Apply the *same* cut to the candidates, or the background density
    will not describe them.  Note also that this cut removes sky area, so the
    effective area used for :class:`~qso_pcolor.priors.BackgroundSurfaceDensity`
    must be the masked-out area, not the nominal cone.

    Parameters
    ----------
    ra, dec, radius_deg : float
        Cone centre and radius in degrees.  The q3c index is on the table's own
        coordinates, so the column names come first in the query — reversing
        them silently disables the index.
    table : str
        ``decals_dr9.main`` (matches ``ls_dr9_grzw``) or ``decals_dr11.main``.
    """
    q = _LS_BACKGROUND_QUERY.format(table=table) % {
        "ra": ra,
        "dec": dec,
        "radius": radius_deg,
    }
    return cached_query(q, cache, refresh=refresh)


_CANDIDATE_QUERY = """
SELECT m.target_idx,
       c.ra, c.dec, c.release, c.type, c.maskbits,
       q3c_dist(m.ra, m.dec, c.ra, c.dec) * 3600 AS sep_arcsec,
       c.flux_g, c.flux_r, c.flux_z, c.flux_w1, c.flux_w2,
       c.flux_ivar_g, c.flux_ivar_r, c.flux_ivar_z,
       c.flux_ivar_w1, c.flux_ivar_w2,
       c.mw_transmission_g, c.mw_transmission_r, c.mw_transmission_z,
       c.mw_transmission_w1, c.mw_transmission_w2,
       c.fracflux_g, c.fracflux_r, c.fracflux_z,
       c.fracin_g, c.fracin_r, c.fracin_z,
       c.fracmasked_g, c.fracmasked_r, c.fracmasked_z,
       c.rchisq_g, c.rchisq_r, c.rchisq_z,
       c.nobs_g, c.nobs_r, c.nobs_z,
       c.psfsize_g, c.psfsize_r, c.psfsize_z,
       c.parallax, c.parallax_ivar, c.pmra, c.pmra_ivar, c.pmdec, c.pmdec_ivar,
       c.gaia_phot_g_mean_mag
FROM mytmptable AS m
JOIN {table} AS c
  ON q3c_join(m.ra, m.dec, c.ra, c.dec, %(radius)s)
ORDER BY m.target_idx, sep_arcsec
"""


def fetch_candidate_photometry(
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    radius_arcsec: float = 30.0,
    table: str = "decals_dr9.main",
) -> dict:
    """All catalogue sources within ``radius_arcsec`` of each primary quasar.

    The local list goes first in ``q3c_join`` and the survey table second, so the
    index is used on the large table.  ``target_idx`` is carried through because
    a positional join returns zero or many rows per input and the result order is
    not the input order.

    The blending diagnostics (``fracflux``, ``fracin``, ``fracmasked``,
    ``rchisq``, ``psfsize``) come back with every row.  For a companion a few
    arcseconds from a bright quasar these are not decoration: they are the
    difference between a measurement and an artefact of the deblender, and the
    scorer's quality flags are built from them.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    idx = np.arange(ra.size, dtype=np.int64)
    q = _CANDIDATE_QUERY.format(table=table) % {"radius": radius_arcsec / 3600.0}
    return _sqlutil().local_join(
        q, "mytmptable", (idx, ra, dec), ("target_idx", "ra", "dec"),
        asDict=True, intNullVal=-1,
    )
