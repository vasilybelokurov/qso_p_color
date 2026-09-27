"""Legacy Surveys DR9: the one selection shared by training, counts and candidates.

The baseline model is built from, counted over and applied to exactly one
population, defined here and nowhere else:

* **hemisphere** from position, not from the catalogue row: north if
  dec >= 32.375 deg and b > 0 (the DESI/Legacy DR9 convention), south
  otherwise. WSDB's ``decals_dr9.main`` holds both north (release 9011) and
  south (9010, 9012) rows in the overlap stripe; a row whose release does not
  match its position's hemisphere is not the object's photometry of record.
* **reference band** r usable: flux, positive inverse variance, nobs_r > 0.
* **clean**: ``maskbits == 0``. The catalogue value is the maskbits image at
  the source position, so the same condition evaluated on the images at random
  positions gives the usable area (:func:`cone_usable_fraction`).
* **morphology** from Tractor ``TYPE``: PSF is ``point``; REX, EXP, DEV, SER
  are ``extended``; DUP (Gaia-only duplicates), blank or anything else is
  ``unknown``. ``unknown`` is never accepted as point, and is not accepted
  by the ``all`` selection either.

Photometry of an accepted object: native Legacy fluxes (nanomaggies, not
dereddened) in the five bands of its hemisphere, a band usable if its inverse
variance is positive and nobs > 0. Negative fluxes are measurements and kept.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .multisurvey_data import Photometry

BANDS = ("g", "r", "z", "w1", "w2")
NORTH_DEC_MIN = 32.375
RELEASES = {"south": (9010, 9012), "north": (9011,)}
POINT_TYPES = ("PSF",)
EXTENDED_TYPES = ("REX", "EXP", "DEV", "SER")


def hemisphere_labels(hemisphere: str) -> tuple[str, ...]:
    if hemisphere not in RELEASES:
        raise ValueError(f"unknown hemisphere {hemisphere!r}")
    return tuple(f"decals_dr9_{hemisphere}:{b}" for b in BANDS)


def is_north(ra: np.ndarray, dec: np.ndarray, b_deg: np.ndarray | None = None) -> np.ndarray:
    """DR9 photometric north: dec >= 32.375 deg in the north Galactic cap."""
    dec = np.asarray(dec, float)
    if b_deg is None:
        from .data import galactic_from_equatorial
        _, b_deg = galactic_from_equatorial(np.asarray(ra, float), dec)
    return (dec >= NORTH_DEC_MIN) & (np.asarray(b_deg, float) > 0)


def hemisphere_of(ra, dec, b_deg=None) -> np.ndarray:
    return np.where(is_north(ra, dec, b_deg), "north", "south")


def morphology_status(tractor_type: np.ndarray) -> np.ndarray:
    t = np.char.strip(np.asarray(tractor_type).astype(str))
    out = np.full(t.shape, "unknown", dtype="<U8")
    out[np.isin(t, POINT_TYPES)] = "point"
    out[np.isin(t, EXTENDED_TYPES)] = "extended"
    return out


@dataclass(frozen=True)
class LegacySelection:
    """A declared population. ``morphology`` is ``"point"`` or ``"all"``."""
    morphology: str = "point"
    maskbits_zero: bool = True
    reference: str = "r"
    version: int = 1

    def __post_init__(self):
        if self.morphology not in ("point", "all"):
            raise ValueError("morphology must be 'point' or 'all'")
        if self.reference not in BANDS:
            raise ValueError(f"reference must be one of {BANDS}")

    def to_dict(self) -> dict:
        return dict(kind="legacy_dr9_selection", **asdict(self),
                    hemisphere_rule=f"north if dec >= {NORTH_DEC_MIN} and b > 0",
                    releases={k: list(v) for k, v in RELEASES.items()},
                    point_types=list(POINT_TYPES), extended_types=list(EXTENDED_TYPES))

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    def decide(self, rows: dict, *, b_deg: np.ndarray | None = None) -> dict:
        """Per-object decision from catalogue columns.

        ``rows`` needs ra, dec, release, type, maskbits, and for the reference
        band flux_<r>, flux_ivar_<r>, nobs_<r>. Returns ``accepted`` (bool),
        ``hemisphere``, ``morphology`` (point/extended/unknown) and ``reason``
        (first failed condition, "" if accepted).
        """
        ra, dec = np.asarray(rows["ra"], float), np.asarray(rows["dec"], float)
        n = ra.size
        hemi = hemisphere_of(ra, dec, b_deg)
        release = np.asarray(rows["release"], int)
        morph = morphology_status(rows["type"])
        a = self.reference
        flux = np.asarray(rows[f"flux_{a}"], float)
        ivar = np.asarray(rows[f"flux_ivar_{a}"], float)
        nobs = np.asarray(rows[f"nobs_{a}"], float)
        maskbits = np.asarray(rows["maskbits"], float)
        checks = [
            ("no_match", np.isfinite(ra) & np.isfinite(dec) & (release > 0)),
            ("wrong_hemisphere_release",
             np.where(hemi == "north", np.isin(release, RELEASES["north"]),
                      np.isin(release, RELEASES["south"]))),
            ("reference_unusable", np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (nobs > 0)),
            ("masked", (maskbits == 0) if self.maskbits_zero else np.ones(n, bool)),
            ("morphology", (morph == "point") if self.morphology == "point"
             else np.isin(morph, ("point", "extended"))),
        ]
        reason = np.full(n, "", dtype="<U24")
        accepted = np.ones(n, bool)
        for name, ok in checks:
            ok = np.asarray(ok, bool)
            reason[accepted & ~ok] = name
            accepted &= ok
        return dict(accepted=accepted, hemisphere=hemi, morphology=morph, reason=reason)


def legacy_photometry(rows: dict, hemisphere: str, *, maskbits_zero: bool = True) -> Photometry:
    """Native five-band photometry of one hemisphere from catalogue columns."""
    n = len(np.asarray(rows["ra"]))
    f = np.full((n, len(BANDS)), np.nan)
    v = np.full((n, len(BANDS)), np.inf)
    mask_ok = (np.asarray(rows["maskbits"], float) == 0) if maskbits_zero else np.ones(n, bool)
    for j, b in enumerate(BANDS):
        flux = np.asarray(rows[f"flux_{b}"], float)
        ivar = np.asarray(rows[f"flux_ivar_{b}"], float)
        ok = (np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
              & (np.asarray(rows[f"nobs_{b}"], float) > 0) & mask_ok)
        f[ok, j] = flux[ok]
        v[ok, j] = 1.0 / ivar[ok]
    return Photometry(f, v, hemisphere_labels(hemisphere))


# -- usable area from the brick images ----------------------------------------

BRICK_URL = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{hemi}/coadd/{d}/{b}/legacysurvey-{b}-{kind}.fits.fz"


def random_in_cone(ra: float, dec: float, radius_deg: float, n: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Points uniform in area inside a spherical cap."""
    cos_r = np.cos(np.deg2rad(radius_deg))
    ct = rng.uniform(cos_r, 1.0, n)
    st = np.sqrt(1 - ct ** 2)
    ph = rng.uniform(0, 2 * np.pi, n)
    # cap around +z, rotated to (ra, dec)
    v = np.column_stack([st * np.cos(ph), st * np.sin(ph), ct])
    a, d = np.deg2rad(ra), np.deg2rad(dec)
    ry = np.array([[np.sin(d), 0, np.cos(d)], [0, 1, 0], [-np.cos(d), 0, np.sin(d)]])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    w = v @ (rz @ ry).T
    return (np.rad2deg(np.arctan2(w[:, 1], w[:, 0])) % 360.0,
            np.rad2deg(np.arcsin(np.clip(w[:, 2], -1, 1))))


def assign_bricks(bricks: dict, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Index into ``bricks`` of the brick whose primary area holds each point.

    Bricks tile the sky in rows of constant [dec1, dec2) with ascending ra1;
    a point outside every brick gets -1.
    """
    ra, dec = np.asarray(ra, float) % 360.0, np.asarray(dec, float)
    index = bricks.get("_index")
    if index is None:
        order = np.lexsort((bricks["ra1"], bricks["dec1"]))
        d1 = bricks["dec1"][order]
        starts = np.flatnonzero(np.r_[True, d1[1:] != d1[:-1]])
        index = (order, d1[starts], np.r_[starts, order.size])
        bricks["_index"] = index
    order, row_dec1, bounds = index
    out = np.full(ra.size, -1)
    row = np.searchsorted(row_dec1, dec, side="right") - 1
    for r in np.unique(row[row >= 0]):
        pts = np.flatnonzero(row == r)
        members = order[bounds[r]:bounds[r + 1]]
        j = np.searchsorted(bricks["ra1"][members], ra[pts], side="right") - 1
        ok = j >= 0
        k = members[np.clip(j, 0, None)]
        ok &= (ra[pts] < bricks["ra2"][k]) & (dec[pts] < bricks["dec2"][k])
        out[pts[ok]] = k[ok]
    return out


def load_bricks(path: str | Path) -> dict:
    from astropy.io import fits
    t = fits.getdata(path)
    out = {k.lower(): np.asarray(t[k], float) for k in ("RA1", "RA2", "DEC1", "DEC2")}
    out["brickname"] = np.char.strip(np.asarray(t["BRICKNAME"]).astype(str))
    return out


def brick_image_fetcher(cache: str | Path) -> Callable:
    """fetch(brick, hemisphere, kind) -> (image, header) or None if not observed."""
    cache = Path(cache)

    def fetch(brick: str, hemisphere: str, kind: str):
        import urllib.error
        import urllib.request
        from astropy.io import fits
        path = cache / hemisphere / f"{brick}-{kind}.fits.fz"
        missing = path.with_suffix(".missing")
        if missing.exists():
            return None
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = BRICK_URL.format(hemi=hemisphere, d=brick[:3], b=brick, kind=kind)
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    data = r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    missing.write_text(url)
                    return None
                raise
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        with fits.open(path) as h:
            hdu = h[1]
            return np.asarray(hdu.data), hdu.header.copy()

    return fetch


def cone_usable_fraction(ra: float, dec: float, radius_deg: float, hemisphere: str,
                         bricks: dict, fetch: Callable, *, n_points: int = 50000,
                         seed: int = 0) -> dict:
    """Fraction of a cone where a clean source with usable r could be catalogued.

    Usable = inside some brick's primary area, ``MASKBITS == 0`` (which also
    excludes the non-primary margins) and nexp_r > 0, evaluated at uniform
    random positions. A brick with no images in this hemisphere is unusable.
    """
    from astropy.wcs import WCS
    rng = np.random.default_rng(seed)
    pr, pd = random_in_cone(ra, dec, radius_deg, n_points, rng)
    which = assign_bricks(bricks, pr, pd)
    usable = np.zeros(n_points, bool)
    per_brick = {}
    for k in np.unique(which[which >= 0]):
        name = str(bricks["brickname"][k])
        sel = which == k
        mb = fetch(name, hemisphere, "maskbits")
        nx = fetch(name, hemisphere, "nexp-r")
        if mb is None or nx is None:
            per_brick[name] = dict(n=int(sel.sum()), usable=0, observed=False)
            continue
        wcs = WCS(mb[1])
        x, y = wcs.all_world2pix(pr[sel], pd[sel], 0)
        ix, iy = np.round(x).astype(int), np.round(y).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < mb[0].shape[1]) & (iy < mb[0].shape[0])
        ok = np.zeros(sel.sum(), bool)
        ok[inside] = (mb[0][iy[inside], ix[inside]] == 0) & (nx[0][iy[inside], ix[inside]] > 0)
        usable[np.flatnonzero(sel)] = ok
        per_brick[name] = dict(n=int(sel.sum()), usable=int(ok.sum()), observed=True)
    frac = float(usable.mean())
    return dict(fraction=frac, fraction_err=float(np.sqrt(frac * (1 - frac) / n_points)),
                n_points=n_points, n_bricks=len(per_brick), bricks=per_brick,
                area_deg2=frac * 2 * np.pi * (1 - np.cos(np.deg2rad(radius_deg))) * (180 / np.pi) ** 2)


# -- catalogue access ------------------------------------------------------------

CATALOGUE_COLUMNS = ("ra", "dec", "release", "brickid", "objid", "type", "maskbits",
                     "fracflux_r") + tuple(f"{p}_{b}" for b in BANDS
                                          for p in ("flux", "flux_ivar", "nobs"))


def _columns(prefix: str = "c.") -> str:
    return ", ".join(f"{prefix}{c}" for c in CATALOGUE_COLUMNS)


def legacy_cone(ra: float, dec: float, radius_deg: float, cache: str | Path, *,
                min_flux_r: float) -> dict:
    """Every DR9 row (both releases, every type) in a cone with flux_r > min_flux_r.

    Rows are selected afterwards by :meth:`LegacySelection.decide`, so the
    same code path decides for field sources, quasars and candidates.
    """
    from .data import cached_query
    q = (f"SELECT {_columns()} FROM decals_dr9.main c "
         f"WHERE q3c_radial_query(c.ra, c.dec, {ra:.8f}, {dec:.8f}, {radius_deg}) "
         f"AND c.flux_r > {min_flux_r}")
    return cached_query(q, Path(cache))


def legacy_match(ra: np.ndarray, dec: np.ndarray, cache: str | Path, *,
                 radius_arcsec: float) -> dict:
    """Nearest DR9 row of the position's own hemisphere, one row per input.

    The release filter is the hemisphere rule, so a north/south duplicate in
    the overlap stripe is never chosen. Unmatched inputs have release -1.
    """
    import sqlutilpy as sqlutil
    import time
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    north = is_north(ra, dec).astype(int)
    q = f"""SELECT m.idx, x.* FROM mytmptable m LEFT JOIN LATERAL (
        SELECT {_columns()}, q3c_dist(m.ra, m.dec, c.ra, c.dec)*3600 AS match_sep_arcsec
        FROM decals_dr9.main c
        WHERE q3c_join(m.ra, m.dec, c.ra, c.dec, {radius_arcsec}/3600.)
          AND ((m.north = 1 AND c.release = 9011) OR (m.north = 0 AND c.release IN (9010, 9012)))
        ORDER BY q3c_dist(m.ra, m.dec, c.ra, c.dec) LIMIT 1) x ON TRUE"""
    digest = hashlib.sha256(q.encode() + ra.tobytes() + dec.tobytes()).hexdigest()[:16]
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"legacy_match_{digest}.npz"
    from .data import _load_npz, _save_npz
    if path.exists():
        return _load_npz(path)
    t0 = time.monotonic()
    # Upload in HEALPix order: nearby rows then share index and table pages.
    # RA-ordered input made each row a random disk read (7 ms/row vs 0.2 ms).
    import healpy as hp
    order = np.argsort(hp.ang2pix(1024, ra, dec, nest=True, lonlat=True), kind="stable")
    res = sqlutil.local_join(q, "mytmptable", (order, ra[order], dec[order], north[order]),
                             ("idx", "ra", "dec", "north"), asDict=True, intNullVal=-1,
                             preamb="SET jit=off; SET statement_timeout='7200s'")
    o = np.argsort(res["idx"])
    res = {k: (v.astype(str) if v.dtype == object else v)[o] for k, v in res.items()}
    if not np.array_equal(res["idx"], np.arange(ra.size)):
        raise RuntimeError("positional join did not preserve one row per input")
    res["input_ra"], res["input_dec"] = ra, dec
    _save_npz(path, **res)
    path.with_suffix(".json").write_text(json.dumps(
        {"query": q, "n": int(ra.size), "elapsed_s": time.monotonic() - t0}, indent=2))
    return res
