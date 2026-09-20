"""Colour distribution of the local non-quasar background.

What this models
----------------
:math:`p(\\mathbf{c}\\mid B, m, l, b)`: the colour distribution of ordinary
catalogue sources at the candidate's brightness and sky position.  Two sample
definitions are supported behind one interface:

``mode='background'`` (default)
    Every catalogue source passing the same photometric quality definition as
    the candidates.  This is what a chance projection next to the primary
    actually draws from — stars, compact galaxies, and whatever else the imaging
    contains — and it needs no selection function, because it *is* the
    population being counted.

    **Known approximation.**  :func:`qso_pcolor.data.fetch_ls_background` does
    *not* remove spectroscopic quasars, so ``field_q`` and ``bkg`` are not
    strictly disjoint and quasars are counted in both.  Quasars are a tiny
    fraction of all sources, but that does not bound the contamination *in
    quasar-like colour space*, which is the only place it matters.  Removing
    them needs a crossmatch against the quasar catalogue; until that is done and
    the size of the effect measured, treat the background as slightly
    over-dense at quasar colours, which makes ``p_sameq`` conservative there.

``mode='star'``
    An explicitly purified stellar sample (Gaia astrometry, or spectroscopic
    classifications).  Purer, but its selection is not the selection of the
    objects we are competing against, and it is not representative at faint
    magnitudes where Gaia runs out.  Useful as a diagnostic and for the
    two-class quasar-versus-star number the user asked for, but not the
    recommended denominator for a real candidate ranking.

Sky and magnitude structure
---------------------------
Density in colour space varies with Galactic position (halo versus disc stars,
reddening) and with apparent magnitude (which populations are reachable, and how
much galaxy contamination there is).  We fit a mixture per (HEALPix cell,
magnitude bin) and fall back through a hierarchy when a cell is sparse:

.. math::
    p = a_\\ell p_{\\rm local} + (1-a_\\ell)
        \\left[a_p p_{\\rm parent} + (1-a_p) p_{\\rm global}\\right],

with shrinkage :math:`a = n/(n + n_0)`.  The pooling constant :math:`n_0` is
tuned on held-out sky blocks by :func:`tune_shrinkage`, not fixed in the source.
HEALPix is used in the NESTED scheme so the parent cell is a bit shift, and in
*Galactic* coordinates because that is the frame the structure lives in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import healpy as hp
import numpy as np

from .gaussmix import GaussianMixture
from .xd import fit_xd

log = logging.getLogger(__name__)

__all__ = [
    "BackgroundColourModel",
    "fit_background_model",
    "fit_local_background",
    "tune_shrinkage",
    "galactic_healpix",
]


# Legacy Surveys ``release`` codes by photometric system. 9010 is DECaLS
# (south), 9011 BASS/MzLS (north); they are different systems, not two halves of
# one, which is why a model carries its system and the scorer refuses to mix.
_RELEASE_FOR_SYSTEM = {
    "ls_dr9_south_grzw": 9010,
    "ls_dr9_north_grzw": 9011,
}


def galactic_healpix(l_deg: np.ndarray, b_deg: np.ndarray, nside: int) -> np.ndarray:
    """NESTED HEALPix index from Galactic coordinates, in degrees."""
    return hp.ang2pix(nside, np.asarray(l_deg), np.asarray(b_deg), nest=True, lonlat=True)


def _parent_pix(ipix: np.ndarray, nside: int, nside_parent: int) -> np.ndarray:
    """Parent cell in the NESTED scheme: a right shift by 2 orders."""
    if nside_parent > nside:
        raise ValueError("parent nside must not exceed the local nside")
    shift = 2 * (int(np.log2(nside)) - int(np.log2(nside_parent)))
    return np.asarray(ipix) >> shift


@dataclass
class BackgroundColourModel:
    """Hierarchical (sky cell, magnitude bin) mixtures for the background colours.

    Attributes
    ----------
    nside, nside_parent : int
        HEALPix resolutions of the local and parent levels, powers of two.
    mag_edges : ndarray, shape (M+1,)
        Reference-magnitude bin edges.  Objects outside the range use the
        nearest bin and are flagged.
    local, parent : dict
        ``(ipix, imag) -> GaussianMixture``.
    global_ : list of GaussianMixture
        One per magnitude bin; the last resort.
    counts_local, counts_parent : dict
        Training counts per cell, used for the shrinkage weights.
    n0 : float
        Pooling constant.  A cell carries weight ``n / (n + n0)``.
    mode : {'background', 'star'}
    system : str
        Photometric system identifier; must match the candidates.
    labels : tuple of str
    meta : dict
    """

    nside: int
    nside_parent: int
    mag_edges: np.ndarray
    local: dict[tuple[int, int], GaussianMixture]
    parent: dict[tuple[int, int], GaussianMixture]
    global_: list[GaussianMixture]
    counts_local: dict[tuple[int, int], float]
    counts_parent: dict[tuple[int, int], float]
    n0: float
    mode: str
    system: str
    labels: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mag_edges = np.asarray(self.mag_edges, dtype=float)
        if (np.diff(self.mag_edges) <= 0).any():
            raise ValueError("mag_edges must be strictly ascending")

    @property
    def n_mag_bins(self) -> int:
        return self.mag_edges.size - 1

    def check_system(self, system: str) -> None:
        if system != self.system:
            raise ValueError(
                f"photometric system mismatch: model is {self.system!r}, "
                f"candidate is {system!r}."
            )

    def mag_bin(self, ref_mag: np.ndarray) -> np.ndarray:
        """Index of the magnitude bin, clipped to the trained range."""
        m = np.asarray(ref_mag, dtype=float)
        idx = np.digitize(m, self.mag_edges) - 1
        return np.clip(idx, 0, self.n_mag_bins - 1)

    def out_of_mag_range(self, ref_mag: np.ndarray) -> np.ndarray:
        m = np.asarray(ref_mag, dtype=float)
        return ~np.isfinite(m) | (m < self.mag_edges[0]) | (m > self.mag_edges[-1])

    def log_prob(
        self,
        x: np.ndarray,
        cov: np.ndarray | None,
        ref_mag: np.ndarray,
        l_deg: np.ndarray,
        b_deg: np.ndarray,
        *,
        observed: np.ndarray | None = None,
        return_level: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """log p(c_obs | B, m, l, b), shape (n,).

        Parameters
        ----------
        return_level : bool
            Also return, per object, the weight the local cell received.  A
            value near zero means the score came from the pooled model, which is
            a caveat on that object's number, not a failure.

        Returns
        -------
        logp : ndarray, shape (n,)
        local_weight : ndarray, shape (n,), only if ``return_level``
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = x.shape[0]
        imag = self.mag_bin(ref_mag)
        ipix = galactic_healpix(l_deg, b_deg, self.nside)
        ppix = _parent_pix(ipix, self.nside, self.nside_parent)

        out = np.full(n, -np.inf)
        w_local_out = np.zeros(n)
        for i in range(n):
            mb = int(imag[i])
            xi = x[i : i + 1]
            ci = None if cov is None else np.asarray(cov)[i : i + 1]
            oi = None if observed is None else np.asarray(observed)[i : i + 1]

            mix_g = self.global_[mb]
            p_g = float(mix_g.log_prob(xi, ci, observed=oi)[0])

            key_p = (int(ppix[i]), mb)
            mix_p = self.parent.get(key_p)
            if mix_p is None:
                log_mid, w_p = p_g, 0.0
            else:
                n_p = self.counts_parent.get(key_p, 0.0)
                w_p = n_p / (n_p + self.n0)
                log_mid = np.logaddexp(
                    np.log(w_p) + float(mix_p.log_prob(xi, ci, observed=oi)[0]),
                    np.log1p(-w_p) + p_g,
                )

            key_l = (int(ipix[i]), mb)
            mix_l = self.local.get(key_l)
            if mix_l is None:
                out[i], w_local_out[i] = log_mid, 0.0
            else:
                n_l = self.counts_local.get(key_l, 0.0)
                w_l = n_l / (n_l + self.n0)
                out[i] = np.logaddexp(
                    np.log(w_l) + float(mix_l.log_prob(xi, ci, observed=oi)[0]),
                    np.log1p(-w_l) + log_mid,
                )
                w_local_out[i] = w_l
        if return_level:
            return out, w_local_out
        return out

    def to_dict(self) -> dict:
        def pack(d):
            return {f"{k[0]}|{k[1]}": v.to_dict() for k, v in d.items()}

        return {
            "kind": "background_colour",
            "nside": self.nside,
            "nside_parent": self.nside_parent,
            "mag_edges": self.mag_edges.tolist(),
            "local": pack(self.local),
            "parent": pack(self.parent),
            "global": [m.to_dict() for m in self.global_],
            "counts_local": {f"{k[0]}|{k[1]}": v for k, v in self.counts_local.items()},
            "counts_parent": {f"{k[0]}|{k[1]}": v for k, v in self.counts_parent.items()},
            "n0": self.n0,
            "mode": self.mode,
            "system": self.system,
            "labels": list(self.labels),
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, d: dict) -> "BackgroundColourModel":
        def unpack(dd):
            out = {}
            for k, v in dd.items():
                a, b = k.split("|")
                out[(int(a), int(b))] = GaussianMixture.from_dict(v)
            return out

        def unpack_counts(dd):
            return {tuple(int(p) for p in k.split("|")): float(v) for k, v in dd.items()}

        return cls(
            d["nside"],
            d["nside_parent"],
            np.asarray(d["mag_edges"], float),
            unpack(d["local"]),
            unpack(d["parent"]),
            [GaussianMixture.from_dict(m) for m in d["global"]],
            unpack_counts(d["counts_local"]),
            unpack_counts(d["counts_parent"]),
            float(d["n0"]),
            d["mode"],
            d["system"],
            tuple(d["labels"]),
            d.get("meta", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BackgroundColourModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_background_model(
    x: np.ndarray,
    cov: np.ndarray | None,
    ref_mag: np.ndarray,
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    *,
    mag_edges: np.ndarray,
    nside: int = 8,
    nside_parent: int = 2,
    observed: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    n_components: int = 10,
    n_components_global: int | None = None,
    min_per_cell: int = 500,
    n0: float = 500.0,
    mode: str = "background",
    system: str = "unspecified",
    labels: tuple[str, ...] = (),
    meta: dict | None = None,
    **fit_kwargs,
) -> BackgroundColourModel:
    """Fit the hierarchy of background colour mixtures.

    A cell is fitted only when it holds at least ``min_per_cell`` objects; the
    shrinkage weight then decides how much it is trusted relative to its parent.
    Both thresholds are model choices and are stored in ``meta``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    mag_edges = np.asarray(mag_edges, dtype=float)
    imag = np.clip(np.digitize(ref_mag, mag_edges) - 1, 0, mag_edges.size - 2)
    ipix = galactic_healpix(l_deg, b_deg, nside)
    ppix = _parent_pix(ipix, nside, nside_parent)

    def _fit(sel: np.ndarray, k: int) -> GaussianMixture:
        return fit_xd(
            x[sel],
            None if cov is None else np.asarray(cov)[sel],
            n_components=k,
            observed=None if observed is None else observed[sel],
            weights=None if weights is None else np.asarray(weights)[sel],
            labels=labels,
            **fit_kwargs,
        ).mixture

    n_mag = mag_edges.size - 1
    global_ = []
    for mb in range(n_mag):
        sel = imag == mb
        if sel.sum() < 2:
            raise ValueError(f"magnitude bin {mb} has {int(sel.sum())} training objects")
        k = n_components_global or n_components
        global_.append(_fit(sel, min(k, max(1, int(sel.sum()) // 50))))

    local: dict[tuple[int, int], GaussianMixture] = {}
    parent: dict[tuple[int, int], GaussianMixture] = {}
    counts_local: dict[tuple[int, int], float] = {}
    counts_parent: dict[tuple[int, int], float] = {}

    for cell_ids, store, counts, res_name in (
        (ppix, parent, counts_parent, "parent"),
        (ipix, local, counts_local, "local"),
    ):
        for (cid, mb), sel in _group(cell_ids, imag):
            if sel.sum() < min_per_cell:
                continue
            k = min(n_components, max(1, int(sel.sum()) // 50))
            store[(int(cid), int(mb))] = _fit(sel, k)
            counts[(int(cid), int(mb))] = float(sel.sum())

    return BackgroundColourModel(
        nside,
        nside_parent,
        mag_edges,
        local,
        parent,
        global_,
        counts_local,
        counts_parent,
        n0,
        mode,
        system,
        labels,
        meta={
            "min_per_cell": min_per_cell,
            "n_components": n_components,
            **(meta or {}),
        },
    )


def _group(cell_ids: np.ndarray, imag: np.ndarray):
    """Yield ``((cell, mag_bin), boolean_selection)`` for every populated pair."""
    key = np.stack([np.asarray(cell_ids), np.asarray(imag)], axis=1)
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    for u in range(uniq.shape[0]):
        yield (uniq[u, 0], uniq[u, 1]), inverse == u


def fit_local_background(
    ra: float,
    dec: float,
    radius_deg: float,
    *,
    transform,
    bands: tuple[str, ...],
    mag_edges: np.ndarray,
    system: str,
    table: str = "decals_dr9.main",
    exclude_quasars: bool = True,
    mask_fraction: float | None = None,
    n_components: int = 8,
    max_ref_mag: float | None = None,
    cache: "str | Path | None" = None,
    **fit_kwargs,
):
    r"""Fit a background model from the candidate's own neighbourhood.

    The hierarchical model in :func:`fit_background_model` amortises one fit
    over a whole footprint, which is what a survey-wide scan needs.  For a
    handful of candidates the better answer is to fit where the candidate
    actually is: the stellar population, the survey depth and the reddening are
    then the candidate's own rather than a cell average, and the surveyed area
    is exactly :math:`\pi R^2`, which removes the commonest way of getting
    :math:`\Sigma_B` wrong.

    Choose the radius for the number of sources needed.  At Legacy Surveys depth
    away from the plane the density is of order $2\times10^4$ per square degree,
    so $0.5^\circ$ gives some 15,000 sources --- ample for a mixture of this
    size --- while staying local.

    Returns
    -------
    model : BackgroundColourModel
        Carrying a single cell, so every candidate in the cone uses this fit.
    density : BackgroundSurfaceDensity
        With the cone area, not a HEALPix pixel area.
    info : dict
        Counts, the fraction removed as known quasars, and the area, for the
        provenance record.
    """
    from pathlib import Path

    from .data import (
        drop_known_quasars,
        fetch_known_quasars,
        fetch_ls_background,
        galactic_from_equatorial,
    )
    from .features import deredden
    from .priors import BackgroundSurfaceDensity

    cache = Path(cache) if cache is not None else Path("data") / "local_bkg.npz"
    r = fetch_ls_background(cache, ra=ra, dec=dec, radius_deg=radius_deg, table=table)
    n_raw = int(np.size(r["ra"]))

    # ``system`` is a label the caller chose; the scorer compares labels, so a
    # wrong one silently certifies data it does not describe. North (BASS/MzLS)
    # and south (DECaLS) are different photometric systems, and a cone near the
    # boundary returns both. Check the label against what the query returned,
    # here, where the data is -- not downstream, where only the label survives.
    rel = np.asarray(r.get("release", []), int)
    if rel.size:
        want = _RELEASE_FOR_SYSTEM.get(system)
        present = sorted(int(x) for x in np.unique(rel))
        if want is None:
            log.warning(
                "system %r has no known release; cone returned releases %s, "
                "which are not being checked", system, present,
            )
        elif present != [want]:
            raise ValueError(
                f"photometric system mismatch: system={system!r} implies "
                f"release {want}, but this cone at ({ra}, {dec}) returned "
                f"release(s) {present}. A mixed or wrong-hemisphere cone cannot "
                f"be labelled with one system; fit north and south separately."
            )

    maskbits = np.asarray(r.get("maskbits", np.zeros(n_raw)))
    unmasked = maskbits == 0
    keep = unmasked.copy()
    n_qso = 0
    if exclude_quasars:
        q = fetch_known_quasars(ra, dec, radius_deg,
                                cache=cache.with_name("local_qso.npz"))
        not_qso = drop_known_quasars(r["ra"], r["dec"], q["ra"], q["dec"])
        n_qso = int((unmasked & ~not_qso).sum())
        keep = unmasked & not_qso

    flux = np.stack([np.asarray(r[f"flux_{b}"], float) for b in bands], axis=1)[keep]
    ivar = np.stack([np.asarray(r[f"flux_ivar_{b}"], float) for b in bands], axis=1)[keep]
    tran = np.stack([np.asarray(r[f"mw_transmission_{b}"], float) for b in bands],
                    axis=1)[keep]
    f, v = deredden(flux, ivar, tran)
    fs = transform(f, v, bands)

    ok = fs.usable(min_dims=3) & np.isfinite(fs.ref_mag)
    if max_ref_mag is not None:
        ok &= fs.ref_mag < max_ref_mag
    l, b = galactic_from_equatorial(np.asarray(r["ra"])[keep][ok],
                                    np.asarray(r["dec"])[keep][ok])

    mag_edges = np.asarray(mag_edges, dtype=float)
    imag = np.clip(np.digitize(fs.ref_mag[ok], mag_edges) - 1, 0, mag_edges.size - 2)

    # One mixture per magnitude bin, no sky subdivision: the cone IS the cell.
    global_ = []
    for mb in range(mag_edges.size - 1):
        sel = imag == mb
        if sel.sum() < 50:
            raise ValueError(
                f"magnitude bin {mb} has only {int(sel.sum())} local sources; "
                f"widen radius_deg or coarsen mag_edges"
            )
        k = min(n_components, max(1, int(sel.sum()) // 50))
        global_.append(
            fit_xd(fs.x[ok][sel], fs.cov[ok][sel], n_components=k,
                   observed=fs.observed[ok][sel], labels=fs.labels,
                   **fit_kwargs).mixture
        )

    # The cone query keeps only ``maskbits = 0`` sources, so the *counts* are
    # masked but the nominal cone area is not. Dividing one by the other
    # understates the background density and inflates every quasar posterior --
    # the same class of error that once made Sigma_B 17x too small.
    if mask_fraction is None:
        # Estimate it from the data: what fraction of catalogue sources in this
        # cone carry a mask bit. This is a proxy for area, and a biased one --
        # masked regions sit around bright stars where the detection density is
        # higher than average, so it tends to OVERstate the masked area and
        # hence overstate Sigma_B. Supply a measured value where it matters.
        mask_fraction = float(np.mean(~unmasked)) if n_raw else 0.0
        mask_estimated = True
    else:
        mask_estimated = False
    area = float(np.pi * radius_deg**2 * (1.0 - mask_fraction))
    info = {
        "ra": ra, "dec": dec, "radius_deg": radius_deg, "area_deg2": area,
        "n_raw": n_raw, "n_known_quasars_removed": n_qso,
        "quasar_fraction_removed": n_qso / n_raw if n_raw else 0.0,
        "n_fitted": int(ok.sum()), "table": table,
        "mask_fraction": mask_fraction,
        "mask_fraction_estimated_from_counts": mask_estimated,
    }
    model = BackgroundColourModel(
        1, 1, mag_edges, {}, {}, global_, {}, {}, 1.0,
        "background_local", system, fs.labels, meta=info,
    )
    density = BackgroundSurfaceDensity.from_catalogue(
        fs.ref_mag[ok], l, b, mag_edges=mag_edges, nside=1, nside_parent=1,
        total_area_deg2=area, meta=info,
    )
    return model, density, info


def tune_shrinkage(
    model_builder,
    x: np.ndarray,
    cov: np.ndarray | None,
    ref_mag: np.ndarray,
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    candidates: list[float],
    *,
    observed: np.ndarray | None = None,
    n_folds: int = 3,
    seed: int = 0,
) -> tuple[float, dict[float, float]]:
    """Choose the pooling constant ``n0`` by held-out log density on sky blocks.

    Folds are whole parent HEALPix cells, so a validation object is never scored
    by a local model fitted to its own neighbours.  Returns the best ``n0`` and
    the score for each candidate.

    ``model_builder(train_selection)`` must return a fitted
    :class:`BackgroundColourModel` for that subset; the caller supplies it so
    that this function stays independent of the fit settings.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    rng = np.random.default_rng(seed)
    # Block on a coarse cell so that adjacent sky does not straddle the split.
    block = galactic_healpix(l_deg, b_deg, 2)
    uniq = np.unique(block)
    assign = rng.permutation(uniq.size) % n_folds
    fold_of = assign[np.searchsorted(uniq, block)]

    scores: dict[float, float] = {}
    for n0 in candidates:
        total, count = 0.0, 0
        for f in range(n_folds):
            tr, va = fold_of != f, fold_of == f
            if tr.sum() == 0 or va.sum() == 0:
                continue
            model = model_builder(tr)
            model.n0 = n0
            lp = model.log_prob(
                x[va],
                None if cov is None else np.asarray(cov)[va],
                np.asarray(ref_mag)[va],
                np.asarray(l_deg)[va],
                np.asarray(b_deg)[va],
                observed=None if observed is None else observed[va],
            )
            total += float(np.sum(lp))
            count += int(va.sum())
        scores[n0] = total / count if count else -np.inf
    return max(scores, key=scores.get), scores
