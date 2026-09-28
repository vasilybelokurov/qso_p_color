"""Spatial PSF backgrounds with shared colour components and local refits.

Colour densities are normalised over the observed relative-flux coordinates.
Surface densities are per square degree per magnitude. Sky cells use Galactic
NESTED HEALPix. This module changes field weights and counts, not quasar fits.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.special import logsumexp

from .background import BackgroundColourModel, galactic_healpix, _parent_pix
from .gaussmix import GaussianMixture
from .priors import BackgroundSurfaceDensity


def cone_within_cell(l: float, b: float, radius_deg: float, *, nside: int,
                     boundary_factor: int) -> bool:
    """Whether an entire cone fits in one Galactic NESTED sky cell.

    Angles are degrees. HEALPix's inclusive overlap query is conservative:
    uncertain boundary cones are rejected, preserving the measured whole-cone
    area and keeping every source in the same training/validation partition.
    ``boundary_factor`` controls the overlap calculation, not the model scale.
    """
    if (not np.isfinite([l, b, radius_deg]).all() or abs(b) > 90
            or not 0 < radius_deg < 180 or not hp.isnsideok(nside, nest=True)
            or not hp.isnsideok(boundary_factor, nest=True)):
        raise ValueError("invalid cone geometry or boundary resolution")
    cells = hp.query_disc(nside, hp.ang2vec(l, b, lonlat=True),
                         np.deg2rad(radius_deg), inclusive=True,
                         fact=boundary_factor, nest=True)
    return len(cells) == 1


def component_log_prob(mixture: GaussianMixture, x: np.ndarray, cov: np.ndarray,
                       observed: np.ndarray) -> np.ndarray:
    """Noise-convolved component log densities, excluding component weights."""
    return np.column_stack([
        GaussianMixture(np.ones(1), mu[None], v[None], mixture.labels).log_prob(
            x, cov, observed=observed)
        for mu, v in zip(mixture.means, mixture.covs)])


def fit_component_weights(log_prob: np.ndarray, initial: np.ndarray,
                          sample_weight: np.ndarray, *, max_iter: int,
                          tol: float) -> tuple[np.ndarray, dict]:
    """Fit only mixture proportions; keep component shapes and noise fixed."""
    lp = np.asarray(log_prob, float)
    sw = np.asarray(sample_weight, float)
    if (lp.ndim != 2 or sw.shape != (len(lp),) or not np.isfinite(lp).all()
            or not np.isfinite(sw).all() or (sw < 0).any()
            or max_iter < 1 or not np.isfinite(tol) or tol <= 0):
        raise ValueError("invalid component-weight fitting data or controls")
    w = np.asarray(initial, float).copy()
    if w.shape != (lp.shape[1],) or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError("initial weights must be a probability vector")
    if not sw.sum():
        return w, dict(n_iter=0, converged=True, effective_count=0.)
    scaled = np.exp(lp - lp.max(axis=1, keepdims=True))
    converged = False
    for it in range(max_iter):
        denominator = scaled @ w
        update = w * (scaled.T @ (sw / np.maximum(denominator, np.finfo(float).tiny)))
        update /= update.sum()
        change = np.max(np.abs(update - w))
        w = update
        if change < tol:
            converged = True
            break
    return w, dict(n_iter=it + 1, converged=converged, effective_count=float(sw.sum()))


@dataclass
class SpatialSurfaceDensity:
    """Measured cell rates pooled towards their parent and a fixed global rate.

    Counts are fractional non-quasar counts; areas are usable deg^2. Zero
    counts in an observed cell are measurements, not missing coverage. The
    pooling strength is a count of expected objects, separately selected from
    the colour pooling strength. ``level`` records the finest contributing
    level: 0 cell, 1 parent, 2 pooled hemisphere, 3 candidate-local cone.
    """
    nside: int
    nside_parent: int
    mag_edges: np.ndarray
    counts: dict
    area: dict
    global_density: np.ndarray
    n0: float
    meta: dict

    def __post_init__(self):
        self.mag_edges = np.asarray(self.mag_edges, float)
        self.global_density = np.asarray(self.global_density, float)
        if (not hp.isnsideok(self.nside, nest=True) or not hp.isnsideok(self.nside_parent, nest=True)
                or self.nside_parent > self.nside or self.n0 < 0 or not np.isfinite(self.n0)
                or (np.diff(self.mag_edges) <= 0).any()
                or self.global_density.shape != (len(self.mag_edges) - 1,)
                or not np.isfinite(self.global_density).all() or (self.global_density <= 0).any()
                or any(not np.isfinite(a) or a <= 0 for a in self.area.values())
                or any(not np.isfinite(c) or c < 0 for c in self.counts.values())):
            raise ValueError("invalid spatial density configuration")

    def _pooled(self, count, area, parent, dm):
        expected = area * dm * parent
        weight = expected / (expected + self.n0) if expected + self.n0 > 0 else 0.
        return weight * count / (area * dm) + (1 - weight) * parent

    def _evaluate(self, ref_mag, l_deg, b_deg):
        m, l, b = np.broadcast_arrays(np.atleast_1d(ref_mag), l_deg, b_deg)
        mb = np.clip(np.digitize(m, self.mag_edges) - 1, 0, len(self.mag_edges) - 2)
        pix = galactic_healpix(l, b, self.nside)
        ppix = _parent_pix(pix, self.nside, self.nside_parent)
        parent = {}
        for cell, area in self.area.items():
            p = int(_parent_pix(np.array([cell]), self.nside, self.nside_parent)[0])
            c, a = parent.setdefault(p, (np.zeros(len(self.global_density)), 0.))
            c += [self.counts.get((cell, j), 0.) for j in range(len(c))]
            parent[p] = c, a + area
        out = self.global_density[mb].copy()
        level = np.full(m.shape, 2, int)
        for i in range(len(m)):
            j, p, cell = int(mb[i]), int(ppix[i]), int(pix[i])
            dm = self.mag_edges[j + 1] - self.mag_edges[j]
            if self.nside_parent != self.nside and p in parent:
                c, a = parent[p]
                out[i] = self._pooled(c[j], a, out[i], dm)
                level[i] = 1
            if cell in self.area:
                out[i] = self._pooled(self.counts.get((cell, j), 0.), self.area[cell], out[i], dm)
                level[i] = 0
        if self.meta.get("mode") == "local":
            out = self.global_density[mb].copy()
            level[:] = 3
        return out, level

    def __call__(self, ref_mag, l_deg, b_deg):
        return self._evaluate(ref_mag, l_deg, b_deg)[0]

    def level(self, ref_mag, l_deg, b_deg):
        return self._evaluate(ref_mag, l_deg, b_deg)[1]

    def to_dict(self) -> dict:
        return dict(kind="spatial_surface_density", nside=self.nside, nside_parent=self.nside_parent,
            mag_edges=self.mag_edges.tolist(), counts={f"{p}|{j}": c for (p, j), c in self.counts.items()},
            area={str(p): a for p, a in self.area.items()}, global_density=self.global_density.tolist(),
            n0=self.n0, meta=self.meta)

    @classmethod
    def from_dict(cls, d: dict) -> "SpatialSurfaceDensity":
        return cls(d["nside"], d["nside_parent"], d["mag_edges"],
                   {tuple(map(int, k.split("|"))): v for k, v in d["counts"].items()},
                   {int(k): v for k, v in d["area"].items()}, d["global_density"], d["n0"], d["meta"])


def fit_spatial_background(base: BackgroundColourModel, data: dict, cones: list[dict], *,
                           nside: int, nside_parent: int, colour_n0: float,
                           density_n0: float, density_edges: np.ndarray,
                           global_density: np.ndarray, max_iter: int, tol: float,
                           meta: dict) -> tuple[BackgroundColourModel, SpatialSurfaceDensity]:
    """Fit cell/parent component proportions and area-corrected count rates.

    ``data`` holds magnitude, l, b, cone IDs, non-quasar probabilities,
    component log densities, and sampling weights. Each cone supplies its
    usable area and non-quasar count vector in ``density_edges``. The caller
    must remove validation cones before entering this function.
    """
    if colour_n0 < 0 or not np.isfinite(colour_n0):
        raise ValueError("colour_n0 must be finite and non-negative")
    model = copy(base)
    model.nside, model.nside_parent, model.n0 = nside, nside_parent, colour_n0
    model.local, model.parent, model.counts_local, model.counts_parent = {}, {}, {}, {}
    model.meta = dict(base.meta, **meta)
    pix = galactic_healpix(data["l"], data["b"], nside)
    parent = _parent_pix(pix, nside, nside_parent)
    imag = base.mag_bin(data["m"])
    records = []
    for cells, store, counts in ((parent, model.parent, model.counts_parent),
                                  (pix, model.local, model.counts_local)):
        if store is model.parent and nside == nside_parent:
            continue
        for cell in np.unique(cells):
            for j, mix in enumerate(base.global_):
                rows = (cells == cell) & (imag == j)
                if not rows.any():
                    continue
                sampling = data["sampling_weight"][rows]
                weight = data["nonq"][rows] * sampling / sampling.mean()
                if not weight.sum():
                    continue
                w, info = fit_component_weights(data["log_components"][rows, :mix.n_components],
                    mix.weights, weight, max_iter=max_iter, tol=tol)
                store[int(cell), j] = GaussianMixture(w, mix.means, mix.covs, mix.labels)
                counts[int(cell), j] = info["effective_count"]
                records.append(dict(cell=int(cell), bin=j, level="parent" if store is model.parent else "cell", **info))
    model.meta["weight_fits"] = records
    # Area expansion weights account for unequal numbers of cones per sampling
    # stratum. Rescale within each fitted cell to its actual sampled area, so
    # pooling depends on measured information, not the expanded footprint area.
    areas, count_table = {}, {}
    cpix = galactic_healpix(np.array([c["l"] for c in cones]), np.array([c["b"] for c in cones]), nside)
    for cell in np.unique(cpix):
        cc = [c for c, p in zip(cones, cpix) if p == cell]
        actual_area = sum(c["area"] for c in cc)
        expanded_area = sum(c["area"] * c["sampling_weight"] for c in cc)
        cnt = sum(np.asarray(c["nonq_counts"]) * c["sampling_weight"] for c in cc)
        cnt *= actual_area / expanded_area
        areas[int(cell)] = actual_area
        count_table.update({(int(cell), j): float(v) for j, v in enumerate(cnt)})
    density = SpatialSurfaceDensity(nside, nside_parent, density_edges, count_table,
                                    areas, global_density, density_n0, dict(meta))
    return model, density


def effective_weights(model: BackgroundColourModel, magnitude: float, l: float, b: float) -> np.ndarray:
    """Mixture proportions at one sky position for shared-shape components."""
    j = int(model.mag_bin(np.array([magnitude]))[0])
    cell = int(galactic_healpix(np.array([l]), np.array([b]), model.nside)[0])
    parent = int(_parent_pix(np.array([cell]), model.nside, model.nside_parent)[0])
    w = model.global_[j].weights.copy()
    for key, table, counts in (((parent, j), model.parent, model.counts_parent),
                               ((cell, j), model.local, model.counts_local)):
        if key in table:
            n = counts[key]
            a = n / (n + model.n0) if n + model.n0 > 0 else 0.
            w = a * table[key].weights + (1 - a) * w
    return w


@dataclass
class BackgroundAdaptation:
    """Portable field replacement tied to one immutable baseline bundle."""
    source_bundle_id: str
    selection_id: str
    parts: dict
    meta: dict

    def to_dict(self) -> dict:
        return dict(kind="legacy_psf_background_adaptation", source_bundle_id=self.source_bundle_id,
            selection_id=self.selection_id, meta=self.meta, parts={h: dict(
                background=p["background"].to_dict(), density=p["density"].to_dict())
                for h, p in self.parts.items()})

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict(); d["identity"] = self.identity
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(d, sort_keys=True, allow_nan=False))
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> "BackgroundAdaptation":
        d = json.loads(Path(path).read_text())
        if d["kind"] != "legacy_psf_background_adaptation":
            raise ValueError("not a PSF background adaptation")
        obj = cls(d["source_bundle_id"], d["selection_id"], {h: dict(
            background=BackgroundColourModel.from_dict(p["background"]),
            density=SpatialSurfaceDensity.from_dict(p["density"])) for h, p in d["parts"].items()}, d["meta"])
        if obj.identity != d["identity"]:
            raise ValueError("background adaptation content hash mismatch")
        return obj

    def apply(self, baseline):
        """Return a scorer using this field, preserving all quasar components."""
        if (baseline.manifest.get("base_bundle_id", baseline.bundle_id) != self.source_bundle_id
                or baseline.selection.identity != self.selection_id):
            raise ValueError("background belongs to another baseline or selection")
        result = copy(baseline)
        result.parts = {h: dict(p) for h, p in baseline.parts.items()}
        for h, p in self.parts.items():
            if h not in result.parts or p["background"].system != result.parts[h]["background"].system:
                raise ValueError("background photometric system mismatch")
            original = result.parts[h]["background"]
            modified = p["background"]
            if (not np.array_equal(modified.mag_edges, original.mag_edges)
                    or modified.labels != original.labels
                    or len(modified.global_) != len(original.global_)):
                raise ValueError("adaptation feature or magnitude layout mismatch")
            # Fixed component shapes preserve the original broad envelope and
            # noise model. Arbitrary replacements require a new outlier fit.
            all_mixtures = list(enumerate(modified.global_))
            all_mixtures += [(j, m) for (cell, j), m in modified.local.items()]
            all_mixtures += [(j, m) for (cell, j), m in modified.parent.items()]
            for j, a in all_mixtures:
                b = original.global_[j]
                if not np.array_equal(a.means, b.means) or not np.array_equal(a.covs, b.covs):
                    raise ValueError("adaptation must preserve component shapes")
            result.parts[h].update(background=p["background"], background_density=p["density"])
        result.manifest = dict(baseline.manifest, base_bundle_id=self.source_bundle_id,
                               bundle_id=f"{self.source_bundle_id}+{self.identity}",
                               background_adaptation=self.meta)
        return result


def fit_local_background(baseline, rows: dict, *, hemisphere: str, centre_ra: float,
                         centre_dec: float, radius_deg: float, usable_area_deg2: float,
                         exclude_ra: np.ndarray, exclude_dec: np.ndarray,
                         exclusion_arcsec: float, colour_n0: float, density_n0: float,
                         max_iter: int, tol: float, min_density_fraction: float) -> BackgroundAdaptation:
    """Refit component proportions and counts in a candidate's own cone.

    Pass the primary and companion positions in the required exclusion list.
    The usable area (deg^2) must account for survey masks and excluded holes;
    it is never inferred from detection counts. Input rows include a boolean
    ``known_quasar`` crossmatch. Component means/covariances stay fixed, as in
    the spatial fit. This is the local A, w refit specified in the field plan.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from .data import galactic_from_equatorial
    from .legacy import dereddened_relative_fluxes
    if (not np.isfinite(usable_area_deg2) or usable_area_deg2 <= 0
            or not np.isfinite(radius_deg) or not 0 < radius_deg < 180
            or usable_area_deg2 > 2 * np.pi * (1 - np.cos(np.deg2rad(radius_deg))) * (180 / np.pi)**2
            or not 0 < min_density_fraction <= 1
            or not np.isfinite(exclusion_arcsec) or exclusion_arcsec <= 0
            or colour_n0 < 0 or density_n0 < 0):
        raise ValueError("invalid cone area, radius, exclusions, or pooling")
    er, ed = np.asarray(exclude_ra, float), np.asarray(exclude_dec, float)
    if er.size < 2 or er.shape != ed.shape or not np.isfinite(er).all() or not np.isfinite(ed).all():
        raise ValueError("supply primary and companion exclusion positions")
    pos = SkyCoord(rows["ra"] * u.deg, rows["dec"] * u.deg)
    centre = SkyCoord(centre_ra * u.deg, centre_dec * u.deg)
    use = pos.separation(centre).deg <= radius_deg
    for ra, dec in zip(er, ed):
        use &= pos.separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec > exclusion_arcsec
    selection = baseline.selection.decide(rows)
    fs, ok = dereddened_relative_fluxes(rows, hemisphere,
        min_dims=baseline.manifest["domain"]["min_dims"],
        min_ref_snr=baseline.manifest["domain"]["min_ref_snr"])
    lo, hi = baseline.manifest["domain"]["ref_mag"]
    use &= selection["accepted"] & (selection["hemisphere"] == hemisphere) & ok
    use &= (fs.ref_mag >= lo) & (fs.ref_mag < hi)
    known = np.asarray(rows["known_quasar"], bool)
    part = baseline.parts[hemisphere]
    prior, original = part["qso_prior"], part["background"]
    sq_bins = np.sum(prior.sigma * np.diff(prior.z_edges)[:, None], axis=0)
    expected_known = usable_area_deg2 * np.sum(sq_bins * np.diff(prior.mag_edges))
    kappa = min(1., int((use & known).sum()) / expected_known)
    idx = np.flatnonzero(use & ~known)
    if not len(idx):
        raise ValueError("no eligible non-quasar catalogue rows in the local cone")
    f = fs.subset(idx)
    l, b = galactic_from_equatorial(np.array([centre_ra]), np.array([centre_dec]))
    l, b = float(l[0]), float(b[0])
    wz = np.array([np.interp(f.ref_mag, prior.mag_centres, s) for s in prior.sigma]).T * np.diff(prior.z_edges)
    lpq = part["qso"]._log_p_slices(f.x, f.cov, f.observed)
    with np.errstate(divide="ignore"):
        logq = logsumexp(lpq + np.log(wz), axis=1) + np.log(1 - kappa)
    lb = original.log_prob(f.x, f.cov, f.ref_mag, np.full(len(idx), l), np.full(len(idx), b), observed=f.observed)
    sb = part["background_density"](f.ref_mag, np.full(len(idx), l), np.full(len(idx), b))
    logb = np.log(sb) + lb
    nonq = np.exp(logb - np.logaddexp(logb, logq))
    background = copy(original)
    background.local, background.parent = {}, {}
    background.counts_local, background.counts_parent = {}, {}
    background.global_ = []
    fits = []
    for j, mix in enumerate(original.global_):
        m = (original.mag_edges[j] + original.mag_edges[j + 1]) / 2
        w0 = effective_weights(original, m, l, b)
        which = original.mag_bin(f.ref_mag) == j
        lp = component_log_prob(mix, f.x[which], f.cov[which], f.observed[which])
        w, info = fit_component_weights(lp, w0, nonq[which], max_iter=max_iter, tol=tol)
        n = info["effective_count"]
        a = n / (n + colour_n0) if n + colour_n0 > 0 else 0.
        background.global_.append(GaussianMixture(a * w + (1 - a) * w0, mix.means, mix.covs, mix.labels))
        fits.append(info)
    counted = np.histogram(f.ref_mag, prior.mag_edges)[0] / usable_area_deg2 / np.diff(prior.mag_edges)
    corrected = np.maximum(counted - (1 - kappa) * sq_bins, min_density_fraction * counted)
    parent_rate = part["background_density"](prior.mag_centres, np.full(len(sq_bins), l), np.full(len(sq_bins), b))
    expected = usable_area_deg2 * np.diff(prior.mag_edges) * parent_rate
    a = expected / (expected + density_n0)
    rates = a * corrected + (1 - a) * parent_rate
    meta = dict(mode="local", hemisphere=hemisphere, centre_ra=float(centre_ra), centre_dec=float(centre_dec),
        radius_deg=float(radius_deg), usable_area_deg2=float(usable_area_deg2),
        exclusion_arcsec=float(exclusion_arcsec), excluded_positions=list(zip(er.tolist(), ed.tolist())),
        n_used=len(idx), n_known=int((use & known).sum()), recognised_fraction=kappa,
        colour_n0=float(colour_n0), density_n0=float(density_n0), weight_fits=fits,
        source_adaptation=baseline.manifest.get("background_adaptation"),
        data_sha256=hashlib.sha256(b"".join(np.ascontiguousarray(rows[k]).tobytes()
            for k in sorted(rows))).hexdigest())
    background.meta = dict(original.meta, local_refit=meta)
    density = SpatialSurfaceDensity(1, 1, prior.mag_edges, {}, {}, rates, density_n0, meta)
    return BackgroundAdaptation(baseline.manifest.get("base_bundle_id", baseline.bundle_id),
                               baseline.selection.identity, {hemisphere: dict(background=background, density=density)}, meta)
