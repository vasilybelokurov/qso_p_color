"""Surface densities: how many of each class are actually out there.

The colour likelihoods answer "does this look like a quasar at :math:`z_0`?".
They cannot answer "is it one?", because that depends on how many quasars and
how many background sources there are at this brightness and sky position.  This
module supplies the two intensities

.. math::
    \\Sigma_B(m, l, b) \\quad [\\mathrm{deg^{-2}\\,mag^{-1}}], \\qquad
    \\Sigma_Q(z, m) \\quad [\\mathrm{deg^{-2}\\,mag^{-1}}\\,\\Delta z^{-1}],

so that the products :math:`\\Sigma\\,p(\\mathbf{c}\\mid\\cdot)` are comparable
intensities per unit (colour, magnitude) volume per square degree.

Honesty rule
------------
:class:`EmpiricalQSOPrior` divides observed spectroscopic counts by a
completeness :math:`C(z,m) \\le 1`.  The default ``C = 1`` *understates* the true
quasar density, so the resulting ``p_sameq`` is a **lower bound** under this
model.  That is a defensible default; inventing a completeness curve is not.  If
no defensible prior exists at all, the scorer is expected to return Bayes factors
and leave the posterior fields null rather than fabricate one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .background import _parent_pix, galactic_healpix

__all__ = ["BackgroundSurfaceDensity", "EmpiricalQSOPrior", "GridQSOPrior"]


@dataclass
class BackgroundSurfaceDensity:
    """:math:`\\Sigma_B(m, l, b)` from area-corrected catalogue counts.

    Built from the *imaging catalogue itself*, over the same footprint and with
    the same quality definition used for candidates.  This is the only estimate
    that needs no selection function: the objects being counted are exactly the
    objects a chance projection would draw from.

    Attributes
    ----------
    nside, nside_parent : int
    mag_edges : ndarray, shape (M+1,)
    counts : dict
        ``(ipix, imag) -> raw count``.
    area : dict
        ``ipix -> effective area in deg^2``.  This must be the *usable* area
        after masking, not the nominal pixel area, or the density is biased high
        in heavily masked cells.
    meta : dict
    """

    nside: int
    nside_parent: int
    mag_edges: np.ndarray
    counts: dict[tuple[int, int], float]
    area: dict[int, float]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mag_edges = np.asarray(self.mag_edges, dtype=float)

    def __call__(
        self, ref_mag: np.ndarray, l_deg: np.ndarray, b_deg: np.ndarray
    ) -> np.ndarray:
        """Surface density in deg^-2 mag^-1, shape (n,).

        Falls back to the parent cell and then to the global mean when a cell is
        empty or has no recorded area, so that a candidate in a sparsely
        surveyed region still gets a number — with the fallback recorded by
        :meth:`level`.
        """
        dens, _ = self._evaluate(ref_mag, l_deg, b_deg)
        return dens

    def level(
        self, ref_mag: np.ndarray, l_deg: np.ndarray, b_deg: np.ndarray
    ) -> np.ndarray:
        """Which hierarchy level supplied each density: 0 local, 1 parent, 2 global."""
        _, lvl = self._evaluate(ref_mag, l_deg, b_deg)
        return lvl

    def _evaluate(self, ref_mag, l_deg, b_deg):
        m = np.atleast_1d(np.asarray(ref_mag, dtype=float))
        imag = np.clip(
            np.digitize(m, self.mag_edges) - 1, 0, self.mag_edges.size - 2
        )
        ipix = galactic_healpix(l_deg, b_deg, self.nside)
        ppix = _parent_pix(ipix, self.nside, self.nside_parent)
        dm = np.diff(self.mag_edges)

        out = np.zeros(m.size)
        lvl = np.full(m.size, 2)
        for i in range(m.size):
            mb = int(imag[i])
            c = self.counts.get((int(ipix[i]), mb))
            a = self.area.get(int(ipix[i]), 0.0)
            if c is not None and a > 0:
                out[i] = c / a / dm[mb]
                lvl[i] = 0
                continue
            cp, ap = self._parent_totals(int(ppix[i]), mb)
            if ap > 0:
                out[i] = cp / ap / dm[mb]
                lvl[i] = 1
                continue
            out[i] = self._global(mb)
        return out, lvl

    def _parent_totals(self, ppix: int, mb: int) -> tuple[float, float]:
        tot_c, tot_a = 0.0, 0.0
        for (pix, b), c in self.counts.items():
            if b != mb:
                continue
            if _parent_pix(np.array([pix]), self.nside, self.nside_parent)[0] == ppix:
                tot_c += c
                tot_a += self.area.get(pix, 0.0)
        return tot_c, tot_a

    def _global(self, mb: int) -> float:
        tot_c = sum(c for (p, b), c in self.counts.items() if b == mb)
        tot_a = sum(
            self.area.get(p, 0.0) for (p, b) in self.counts if b == mb
        )
        dm = float(np.diff(self.mag_edges)[mb])
        return tot_c / tot_a / dm if tot_a > 0 else 0.0

    @classmethod
    def from_catalogue(
        cls,
        ref_mag: np.ndarray,
        l_deg: np.ndarray,
        b_deg: np.ndarray,
        *,
        mag_edges: np.ndarray,
        nside: int = 8,
        nside_parent: int = 2,
        area_per_pixel: dict[int, float] | None = None,
        meta: dict | None = None,
    ) -> "BackgroundSurfaceDensity":
        """Count a catalogue into (cell, magnitude) bins.

        ``area_per_pixel`` should give the *usable* area of each cell.  When it
        is omitted the nominal HEALPix pixel area is used for every cell that
        contains at least one object, which assumes complete, unmasked coverage
        — an assumption that is recorded in ``meta`` and is wrong wherever the
        footprint has holes.
        """
        import healpy as hp

        mag_edges = np.asarray(mag_edges, dtype=float)
        imag = np.digitize(ref_mag, mag_edges) - 1
        inside = (imag >= 0) & (imag < mag_edges.size - 1)
        ipix = galactic_healpix(l_deg, b_deg, nside)

        counts: dict[tuple[int, int], float] = {}
        for p, b in zip(ipix[inside], imag[inside]):
            counts[(int(p), int(b))] = counts.get((int(p), int(b)), 0.0) + 1.0

        if area_per_pixel is None:
            nominal = hp.nside2pixarea(nside, degrees=True)
            area = {int(p): nominal for p in np.unique(ipix[inside])}
            assumed = True
        else:
            area = {int(k): float(v) for k, v in area_per_pixel.items()}
            assumed = False

        return cls(
            nside,
            nside_parent,
            mag_edges,
            counts,
            area,
            meta={"area_assumed_full_pixel": assumed, **(meta or {})},
        )

    def to_dict(self) -> dict:
        return {
            "kind": "background_surface_density",
            "nside": self.nside,
            "nside_parent": self.nside_parent,
            "mag_edges": self.mag_edges.tolist(),
            "counts": {f"{k[0]}|{k[1]}": v for k, v in self.counts.items()},
            "area": {str(k): v for k, v in self.area.items()},
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BackgroundSurfaceDensity":
        counts = {
            tuple(int(p) for p in k.split("|")): float(v) for k, v in d["counts"].items()
        }
        return cls(
            d["nside"],
            d["nside_parent"],
            np.asarray(d["mag_edges"], float),
            counts,
            {int(k): float(v) for k, v in d["area"].items()},
            d.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "BackgroundSurfaceDensity":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class GridQSOPrior:
    """:math:`\\Sigma_Q(z, m)` tabulated on a grid, with bilinear interpolation.

    Units: deg^-2 mag^-1 per unit redshift.  The grid is over bin *centres*;
    values outside the grid are zero, which is a statement that the model knows
    of no quasars there, and is reported through :meth:`in_support`.
    """

    z_centres: np.ndarray
    mag_centres: np.ndarray
    sigma: np.ndarray  # shape (n_z, n_mag)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.z_centres = np.asarray(self.z_centres, dtype=float)
        self.mag_centres = np.asarray(self.mag_centres, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.sigma.shape != (self.z_centres.size, self.mag_centres.size):
            raise ValueError("sigma must have shape (n_z, n_mag)")
        if (self.sigma < 0).any():
            raise ValueError("surface densities must be non-negative")

    def __call__(self, z: np.ndarray, ref_mag: float) -> np.ndarray:
        """Density at redshifts ``z`` and a single reference magnitude."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        col = np.array(
            [np.interp(ref_mag, self.mag_centres, self.sigma[j]) for j in range(
                self.z_centres.size)]
        )
        out = np.interp(z, self.z_centres, col, left=0.0, right=0.0)
        return out

    def in_support(self, z: np.ndarray, ref_mag: float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return (
            (z >= self.z_centres[0])
            & (z <= self.z_centres[-1])
            & (ref_mag >= self.mag_centres[0])
            & (ref_mag <= self.mag_centres[-1])
        )

    def to_dict(self) -> dict:
        return {
            "kind": "grid_qso_prior",
            "z_centres": self.z_centres.tolist(),
            "mag_centres": self.mag_centres.tolist(),
            "sigma": self.sigma.tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridQSOPrior":
        return cls(
            np.asarray(d["z_centres"], float),
            np.asarray(d["mag_centres"], float),
            np.asarray(d["sigma"], float),
            d.get("meta", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "GridQSOPrior":
        return cls.from_dict(json.loads(Path(path).read_text()))


class EmpiricalQSOPrior:
    """Build :class:`GridQSOPrior` from a spectroscopic quasar catalogue.

    .. math::
        \\Sigma_Q(z, m) = \\frac{N(z, m)}{A\\,\\Delta z\\,\\Delta m\\,C(z, m)}

    with ``A`` the survey area in deg^2 and ``C`` the spectroscopic completeness
    (targeting times redshift success).  ``C`` defaults to 1, which makes the
    density a lower bound and hence ``p_sameq`` a lower bound.

    This estimate is only as good as ``A`` and ``C``.  A quasar catalogue's raw
    redshift histogram is *not* the quasar redshift distribution: it is the
    targeting selection.  Whenever the resulting posterior matters, the
    completeness must be supplied.
    """

    @staticmethod
    def build(
        z: np.ndarray,
        ref_mag: np.ndarray,
        *,
        area_deg2: float,
        z_edges: np.ndarray,
        mag_edges: np.ndarray,
        completeness: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> GridQSOPrior:
        """Histogram a quasar catalogue into a surface-density grid.

        Parameters
        ----------
        z, ref_mag : ndarray, shape (n,)
            Spectroscopic redshift and reference magnitude of each quasar.
        area_deg2 : float
            Area of the footprint the catalogue covers, in square degrees.
        z_edges, mag_edges : ndarray
            Bin edges.
        completeness : ndarray, shape (n_z, n_mag), optional
            ``C(z, m)`` in (0, 1].  ``None`` means 1 everywhere.
        """
        if area_deg2 <= 0:
            raise ValueError("area_deg2 must be positive")
        z_edges = np.asarray(z_edges, dtype=float)
        mag_edges = np.asarray(mag_edges, dtype=float)
        counts, _, _ = np.histogram2d(
            np.asarray(z, float), np.asarray(ref_mag, float), bins=[z_edges, mag_edges]
        )
        dz = np.diff(z_edges)[:, None]
        dm = np.diff(mag_edges)[None, :]
        sigma = counts / (area_deg2 * dz * dm)
        if completeness is not None:
            c = np.asarray(completeness, dtype=float)
            if c.shape != sigma.shape:
                raise ValueError("completeness must have shape (n_z, n_mag)")
            if (c <= 0).any() or (c > 1).any():
                raise ValueError("completeness must lie in (0, 1]")
            sigma = sigma / c
        return GridQSOPrior(
            0.5 * (z_edges[:-1] + z_edges[1:]),
            0.5 * (mag_edges[:-1] + mag_edges[1:]),
            sigma,
            meta={
                "area_deg2": area_deg2,
                "completeness_supplied": completeness is not None,
                "n_quasars": int(np.asarray(z).size),
                **(meta or {}),
            },
        )
