"""HEALPix mixture proportions for a joint, arbitrary-band field model.

All components retain the same means and covariances. The spatial mixture is
chosen before marginalising or conditioning, so a missing band never changes
the fitted population. Brightness dependence comes from the joint components.
"""
from dataclasses import dataclass, field

import healpy as hp
import numpy as np

from .background import galactic_healpix, _parent_pix
from .gaussmix import GaussianMixture
from .spatial import component_log_prob, fit_component_weights


@dataclass
class JointSpatialWeights:
    """Cell -> parent -> global pooling of joint mixture proportions.

    ``n0`` is the configured pooling count. Counts are effective training
    counts; coordinates and optional local-cone scope are Galactic degrees.
    The component shapes always belong to the enclosing joint model.
    """
    nside: int
    nside_parent: int
    n0: float
    global_weights: np.ndarray
    cells: dict = field(default_factory=dict)
    parents: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    parent_counts: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if (not hp.isnsideok(self.nside, nest=True) or
                not hp.isnsideok(self.nside_parent, nest=True) or
                self.nside_parent > self.nside or not np.isfinite(self.n0) or self.n0 < 0):
            raise ValueError("invalid joint spatial resolution or pooling count")
        self.global_weights = np.asarray(self.global_weights, float)
        self.cells = {int(k): np.asarray(v, float) for k, v in self.cells.items()}
        self.parents = {int(k): np.asarray(v, float) for k, v in self.parents.items()}
        self.counts = {int(k): float(v) for k, v in self.counts.items()}
        self.parent_counts = {int(k): float(v) for k, v in self.parent_counts.items()}
        for w in [self.global_weights, *self.cells.values(), *self.parents.values()]:
            if (w.ndim != 1 or w.shape != self.global_weights.shape or
                    not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1.)):
                raise ValueError("spatial weights must be finite non-negative proportions")
        for weights, counts, nside in ((self.cells, self.counts, self.nside),
                                       (self.parents, self.parent_counts, self.nside_parent)):
            if set(weights) != set(counts) or any(
                    not 0 <= k < hp.nside2npix(nside) or not np.isfinite(v) or v <= 0
                    for k, v in counts.items()):
                raise ValueError("each fitted cell needs a positive effective count")

    def evaluate(self, l_deg, b_deg):
        """Return row-wise component weights and the total non-global share."""
        l, b = np.broadcast_arrays(np.atleast_1d(l_deg), np.atleast_1d(b_deg))
        if l.ndim != 1 or not np.isfinite(l + b).all() or (np.abs(b) > 90).any():
            raise ValueError("finite one-dimensional Galactic coordinates are required")
        if "local_scope" in self.meta:
            scope = self.meta["local_scope"]
            vectors = hp.ang2vec(l, b, lonlat=True)
            centre = hp.ang2vec(scope["l_deg"], scope["b_deg"], lonlat=True)
            separation = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(vectors, centre), axis=1),
                                               vectors @ centre))
            if (separation > scope["radius_deg"]).any():
                raise ValueError("candidate lies outside the fitted local background cone")
            return np.tile(self.global_weights, (len(l), 1)), np.ones(len(l))
        pix = galactic_healpix(l, b, self.nside)
        parent = _parent_pix(pix, self.nside, self.nside_parent)
        weights = np.tile(self.global_weights, (len(l), 1))
        global_share = np.ones(len(l))
        for groups, fitted, counts in ((parent, self.parents, self.parent_counts),
                                        (pix, self.cells, self.counts)):
            for cell in np.unique(groups):
                if cell not in fitted:
                    continue
                use = groups == cell
                fraction = counts[cell] / (counts[cell] + self.n0)
                weights[use] = fraction * fitted[cell] + (1 - fraction) * weights[use]
                global_share[use] *= 1 - fraction
        return weights, 1 - global_share

    def groups(self, mixture: GaussianMixture, l_deg, b_deg):
        """Yield row indices, shared-shape mixture and adaptation share."""
        if len(self.global_weights) != mixture.n_components:
            raise ValueError("spatial weights and joint components disagree")
        weights, share = self.evaluate(l_deg, b_deg)
        unique, inverse = np.unique(weights, axis=0, return_inverse=True)
        for i, w in enumerate(unique):
            rows = np.flatnonzero(inverse == i)
            yield rows, GaussianMixture(w, mixture.means, mixture.covs, mixture.labels), share[rows]

    def to_dict(self):
        return dict(nside=self.nside, nside_parent=self.nside_parent, n0=self.n0,
                    global_weights=self.global_weights.tolist(),
                    cells={str(k): v.tolist() for k, v in self.cells.items()},
                    parents={str(k): v.tolist() for k, v in self.parents.items()},
                    counts=self.counts, parent_counts=self.parent_counts, meta=self.meta)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def fit_joint_spatial_weights(mixture: GaussianMixture, x: np.ndarray, cov: np.ndarray,
                              observed: np.ndarray, l_deg: np.ndarray, b_deg: np.ndarray,
                              sample_weight: np.ndarray, *, nside: int, nside_parent: int,
                              n0: float, max_iter: int, tol: float, meta: dict) -> JointSpatialWeights:
    """Fit proportions using observed-coordinate likelihoods, with fixed shapes.

    The caller supplies only eligible training rows, excluding held-out rows
    and known quasars. ``sample_weight`` encodes sampling and population
    membership, never an imputed measurement in an absent band.
    """
    result = JointSpatialWeights(nside, nside_parent, n0, mixture.weights, meta=dict(meta))
    l, b, sw = np.asarray(l_deg), np.asarray(b_deg), np.asarray(sample_weight, float)
    if l.shape != (len(x),) or b.shape != l.shape or sw.shape != l.shape:
        raise ValueError("training coordinates and weights must have one entry per row")
    result.evaluate(l, b)  # validate coordinates before HEALPix indexing
    if not np.asarray(observed).any(axis=1).all():
        raise ValueError("a training row must have at least one measurement")
    lp = component_log_prob(mixture, x, cov, observed)
    result.global_weights, info = fit_component_weights(lp, mixture.weights, sw,
                                                        max_iter=max_iter, tol=tol)
    records = [dict(level="global", **info)]
    pix = galactic_healpix(l, b, nside)
    parents = _parent_pix(pix, nside, nside_parent)
    for groups, store, counts, level in ((parents, result.parents, result.parent_counts, "parent"),
                                          (pix, result.cells, result.counts, "cell")):
        if level == "parent" and nside == nside_parent:
            continue
        for cell in np.unique(groups):
            use = groups == cell
            if sw[use].sum() <= 0:
                continue
            store[int(cell)], info = fit_component_weights(lp[use], result.global_weights, sw[use],
                                                            max_iter=max_iter, tol=tol)
            counts[int(cell)] = info["effective_count"]
            records.append(dict(level=level, cell=int(cell), **info))
    result.meta["weight_fits"] = records
    result.__post_init__()
    return result


def fit_local_joint_weights(mixture: GaussianMixture, spatial: JointSpatialWeights,
                            x: np.ndarray, cov: np.ndarray, observed: np.ndarray,
                            sample_weight: np.ndarray, *, l_deg: float, b_deg: float,
                            radius_deg: float, n0: float, max_iter: int, tol: float,
                            meta: dict) -> JointSpatialWeights:
    """Refit joint proportions in a candidate-local cone, pooled to its sky fit.

    Eligible cone rows and their exclusions are the caller's responsibility;
    the resulting model refuses evaluation outside the declared cone. Surface
    densities must separately be fitted with the measured usable cone area.
    """
    if not np.isfinite([radius_deg, n0]).all() or not 0 < radius_deg < 180 or n0 < 0:
        raise ValueError("positive cone radius and non-negative pooling are required")
    initial = spatial.evaluate([l_deg], [b_deg])[0][0]
    lp = component_log_prob(mixture, x, cov, observed)
    weights, info = fit_component_weights(lp, initial, sample_weight, max_iter=max_iter, tol=tol)
    count = info["effective_count"]
    fraction = count / (count + n0) if count + n0 else 0.
    return JointSpatialWeights(spatial.nside, spatial.nside_parent, n0,
        fraction * weights + (1 - fraction) * initial,
        meta=dict(meta, weight_fits=[info], local_scope=dict(l_deg=l_deg, b_deg=b_deg,
                                                          radius_deg=radius_deg)))
