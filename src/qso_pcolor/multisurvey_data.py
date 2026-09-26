"""Survey-labelled photometry and cached WSDB catalogue queries.

Fluxes use each catalogue's native magnitude system, scaled so that a native
magnitude of 22.5 has flux one. These are NOT uniformly AB nanomaggies: AllWISE
and VHS retain their Vega zero points. Survey-qualified labels and the saved
schema prevent treating different filters or calibrations as interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import _load_npz, _save_npz, cached_query


@dataclass(frozen=True)
class Survey:
    name: str
    table: str
    bands: tuple[str, ...]
    values: tuple[str, ...]
    errors: tuple[str, ...]
    kind: str
    ra: str = "ra"
    dec: str = "dec"
    where: str = "TRUE"
    extra: tuple[str, ...] = ()
    magnitude_system: str = "AB"

    @property
    def columns(self) -> str:
        cols = [f"c.{self.ra} AS ra", f"c.{self.dec} AS dec"]
        for b, v, e in zip(self.bands, self.values, self.errors):
            cols += [f"c.{v} AS value_{b}", f"c.{e} AS error_{b}"]
        cols += [f"c.{k}" for k in self.extra]
        return ", ".join(cols)


def _survey(name, table, bands, value, error, kind="mag", **kw):
    return Survey(name, table, tuple(bands), tuple(value(b) for b in bands),
                  tuple(error(b) for b in bands), kind, **kw)


SURVEYS = {
    "sdss": _survey("sdss", "sdssdr14.photoobjall", "ugriz",
                    lambda b: f"psfflux_{b}", lambda b: f"psffluxivar_{b}",
                    "ivar", where="c.mode=1", extra=("clean",)),
    # w1, w2 are the Legacy forced unWISE fluxes at the optical position (AB
    # nanomaggies): a different system from the AllWISE catalogue fluxes, and
    # the single strongest quasar/star discriminant these catalogues offer.
    "decals": _survey("decals", "decals_dr9.main", ("g", "r", "z", "w1", "w2"),
                      lambda b: f"flux_{b}", lambda b: f"flux_ivar_{b}",
                      "ivar", extra=("release", "maskbits", "nobs_g", "nobs_r", "nobs_z",
                                     "nobs_w1", "nobs_w2")),
    "allwise": _survey("allwise", "allwise.main", ("w1", "w2", "w3", "w4"),
                       lambda b: f"{b}flux", lambda b: f"{b}sigflux", "dn",
                       extra=("cc_flags",), magnitude_system="Vega"),
    "ps1": _survey("ps1", "panstarrs_dr1.stackobjectthin", "grizy",
                   lambda b: f"{b}psfmag", lambda b: f"{b}psfmagerr",
                   where="c.primarydetection=1", extra=tuple(f"{b}nframes" for b in "grizy")),
    "nsc": _survey("nsc", "nsc_dr2.object", ("u", "g", "r", "i", "z", "y", "vr"),
                   lambda b: f"{b}mag", lambda b: f"{b}err", extra=("flags",)),
    "skymapper": _survey("skymapper", "skymapper_dr4.main", "uvgriz",
                         lambda b: f"{b}_psf", lambda b: f"e_{b}_psf",
                         ra="raj2000", dec="dej2000",
                         extra=tuple(f"{b}_{suffix}" for b in "uvgriz"
                                     for suffix in ("ngood", "flags"))),
    "vhs": _survey("vhs", "vhs_dr5.main", ("y", "j", "h", "ks"),
                   lambda b: f"{b}apermag3", lambda b: f"{b}apermag3err",
                   ra="ra2000", dec="dec2000", magnitude_system="Vega",
                   extra=tuple(f"{b}pperrbits" for b in ("y", "j", "h", "ks"))),
}

# AllWISE catalogue DN zero points; IRSA Explanatory Supplement IV.3.a, Table 1.
WISE_DN_ZEROPOINT = dict(w1=20.5, w2=19.5, w3=18.0, w4=13.0)


def band_labels(surveys: tuple[str, ...] = tuple(SURVEYS)) -> tuple[str, ...]:
    """Canonical labels; northern LS measurements are separate from DECaLS."""
    labels = []
    for name in surveys:
        spec = SURVEYS[name]
        systems = ("decals_dr9_south", "decals_dr9_north") if name == "decals" else (name,)
        labels.extend(f"{system}:{b}" for system in systems for b in spec.bands)
    return tuple(labels)


def survey_of(label: str) -> str:
    system = label.split(":")[0]
    return "decals" if system.startswith("decals_dr9_") else system


@dataclass
class Photometry:
    """Native calibrated fluxes, variances, and explicit measurement masks.

    Arrays have shape (objects, bands). Variances have squared native-flux
    units. Missing entries are NaN/inf, never invented measurements.
    """
    flux: np.ndarray
    variance: np.ndarray
    bands: tuple[str, ...]

    def __post_init__(self):
        self.flux = np.atleast_2d(np.asarray(self.flux, float))
        self.variance = np.atleast_2d(np.asarray(self.variance, float))
        self.bands = tuple(self.bands)
        if self.flux.shape != self.variance.shape or self.flux.shape[1] != len(self.bands):
            raise ValueError("flux, variance, and band labels must have matching shapes")
        if len(set(self.bands)) != len(self.bands):
            raise ValueError("duplicate survey-band labels")
        unknown = set(self.bands) - set(band_labels())
        if unknown:
            raise ValueError(f"unknown survey-band labels: {sorted(unknown)}")

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.flux) & np.isfinite(self.variance) & (self.variance > 0)

    def subset(self, rows: np.ndarray) -> Photometry:
        return Photometry(self.flux[rows], self.variance[rows], self.bands)

    def align(self, bands: tuple[str, ...]) -> Photometry:
        """Reorder and pad missing bands, refusing unrecognised extra data."""
        if set(self.bands) - set(bands):
            raise ValueError("input contains bands absent from the fitted model")
        f = np.full((len(self.flux), len(bands)), np.nan)
        v = np.full_like(f, np.inf)
        for j, b in enumerate(self.bands):
            k = bands.index(b)
            f[:, k], v[:, k] = self.flux[:, j], self.variance[:, j]
        return Photometry(f, v, bands)

    def keep_surveys(self, surveys: tuple[str, ...]) -> Photometry:
        unknown = set(surveys) - set(SURVEYS)
        if unknown:
            raise ValueError(f"unknown surveys: {sorted(unknown)}")
        idx = [i for i, b in enumerate(self.bands) if survey_of(b) in surveys]
        return Photometry(self.flux[:, idx], self.variance[:, idx], tuple(self.bands[i] for i in idx))


def catalogue_photometry(name: str, rows: dict, *, clean: bool, vhs_bad_bits: int) -> Photometry:
    """Convert a WSDB result, preserving negative raw flux and its variance.

    Magnitude-only catalogues cannot recover unpublished flux measurements.
    Their missing/sentinel magnitudes are masked, not interpreted as limits.
    Quality choices are explicit and must be shared by training and candidates.
    No signal-to-noise or flux-sign cut is made.
    """
    spec = SURVEYS[name]
    n = len(rows["ra"])
    f, v = np.full((n, len(spec.bands)), np.nan), np.full((n, len(spec.bands)), np.inf)
    matched = np.isfinite(rows["ra"]) & np.isfinite(rows["dec"])
    for j, b in enumerate(spec.bands):
        value, error = np.asarray(rows[f"value_{b}"], float), np.asarray(rows[f"error_{b}"], float)
        ok = matched & np.isfinite(value) & np.isfinite(error) & (error > 0)
        if spec.kind == "mag":
            ok &= (value > -50) & (value < 90) & (error < 90)
            f[ok, j] = 10 ** ((22.5 - value[ok]) / 2.5)
            v[ok, j] = (np.log(10) / 2.5 * f[ok, j] * error[ok]) ** 2
        else:
            scale = 10 ** (0.4 * (22.5 - WISE_DN_ZEROPOINT[b])) if spec.kind == "dn" else 1.0
            f[ok, j] = value[ok] * scale
            v[ok, j] = (1 / error[ok] if spec.kind == "ivar" else error[ok] ** 2) * scale**2
        if name == "decals":
            ok &= np.asarray(rows[f"nobs_{b}"]) > 0
            if clean:
                ok &= np.asarray(rows["maskbits"]) == 0
        elif name == "sdss" and clean:
            ok &= np.asarray(rows["clean"]) == 1
        elif name == "ps1":
            ok &= np.asarray(rows[f"{b}nframes"]) > 0
        elif name == "nsc" and clean:
            ok &= np.asarray(rows["flags"]) == 0
        elif name == "skymapper":
            ok &= np.asarray(rows[f"{b}_ngood"]) > 0
            if clean:
                ok &= np.asarray(rows[f"{b}_flags"]) == 0
        elif name == "vhs" and clean:
            bits = np.asarray(rows[f"{b}pperrbits"], np.int64)
            ok &= (bits >= 0) & ((bits & vhs_bad_bits) == 0)
        elif name == "allwise" and clean:
            flags = np.asarray(rows["cc_flags"]).astype(str)
            ok &= np.array([len(s) == 4 and s[j] == "0" for s in flags])
        f[~ok, j], v[~ok, j] = np.nan, np.inf
    if name == "decals":
        nb = len(spec.bands)
        ff, vv = np.full((n, 2 * nb), np.nan), np.full((n, 2 * nb), np.inf)
        for offset, releases in ((0, (9010, 9012)), (nb, (9011,))):
            sel = np.isin(rows["release"], releases)
            ff[sel, offset:offset+nb], vv[sel, offset:offset+nb] = f[sel], v[sel]
        f, v = ff, vv
    return Photometry(f, v, band_labels((name,)))


def match_catalogue(name: str, ra: np.ndarray, dec: np.ndarray, cache: Path,
                    *, radius_arcsec: float, refresh: bool = False) -> dict:
    """Nearest indexed WSDB match; cache identity includes positions and SQL."""
    import sqlutilpy as sqlutil

    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError("match radius must be positive and finite")
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    if ra.ndim != 1 or ra.shape != dec.shape or not np.isfinite(ra + dec).all():
        raise ValueError("finite one-dimensional coordinates are required")
    spec = SURVEYS[name]
    query = f"""SELECT m.idx, x.* FROM mytmptable m LEFT JOIN LATERAL (
        SELECT {spec.columns},
          q3c_dist(m.ra,m.dec,c.{spec.ra},c.{spec.dec})*3600 AS match_sep_arcsec
        FROM {spec.table} c
        WHERE q3c_join(m.ra,m.dec,c.{spec.ra},c.{spec.dec},{radius_arcsec}/3600.)
          AND ({spec.where})
        ORDER BY q3c_dist(m.ra,m.dec,c.{spec.ra},c.{spec.dec}) LIMIT 1
        ) x ON TRUE"""
    digest = hashlib.sha256(query.encode() + ra.tobytes() + dec.tobytes()).hexdigest()[:16]
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}_{digest}.npz"
    if path.exists() and not refresh:
        return _load_npz(path)
    started=time.monotonic()
    result = sqlutil.local_join(query, "mytmptable", (np.arange(len(ra)), ra, dec),
                               ("idx", "ra", "dec"), asDict=True, intNullVal=-1,
                               preamb="SET jit=off; SET statement_timeout='7200s'")
    order = np.argsort(result["idx"])
    result = {k: (v.astype(str) if v.dtype == object else v)[order] for k, v in result.items()}
    if not np.array_equal(result["idx"], np.arange(len(ra))):
        raise RuntimeError("positional join did not preserve one row per target")
    _save_npz(path, **result)
    path.with_suffix(".json").write_text(json.dumps(
        {"query": query, "n": len(ra), "elapsed_s": time.monotonic()-started}, indent=2))
    return result


def cone_catalogue(name: str, ra: float, dec: float, radius_deg: float, cache: Path) -> dict:
    """All catalogue source types in a cone, cached by the exact SQL."""
    if not np.isfinite([ra, dec, radius_deg]).all() or radius_deg <= 0:
        raise ValueError("finite coordinates and positive radius are required")
    spec = SURVEYS[name]
    query = f"""SELECT {spec.columns} FROM {spec.table} c
        WHERE q3c_radial_query(c.{spec.ra},c.{spec.dec},{ra},{dec},{radius_deg})
          AND ({spec.where})"""
    Path(cache).mkdir(parents=True, exist_ok=True)
    return cached_query(query, Path(cache) / f"cone_{name}.npz")
