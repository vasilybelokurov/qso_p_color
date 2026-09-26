"""Joint photometry models with exact missing-band marginalisation.

Each mixture describes survey-labelled luptitudes, including brightness. At
scoring time it is conditioned on one available band: p(other bands | anchor).
At fixed anchor this is exactly the density of colours relative to that anchor
(unit Jacobian). Thus the existing colour scorer and its three hypotheses can
be reused without requiring the same reference band for every candidate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .features import FeatureSet
from .gaussmix import GaussianMixture
from .multisurvey_data import Photometry, survey_of
from .qso_model import RedshiftMatch, SlicedColourRedshiftModel
from .score import PairScore, score_candidates

_MAG_FACTOR = 2.5 / np.log(10.)


@dataclass
class BandLuptitudeTransform:
    """Per-band luptitudes with a fixed, saved softening in native flux units.

    The output covariance is in magnitude squared. Missing bands are masked;
    negative fluxes remain finite measurements. A supplied flux covariance is
    propagated in full, including off-diagonal measurement correlations.
    """
    bands: tuple[str, ...]
    softening: np.ndarray

    def __post_init__(self):
        self.bands = tuple(self.bands)
        self.softening = np.asarray(self.softening, float)
        if (self.softening.shape != (len(self.bands),) or
                not np.isfinite(self.softening).all() or (self.softening <= 0).any()):
            raise ValueError("one positive finite softening is required per band")
        if len(set(self.bands)) != len(self.bands):
            raise ValueError("duplicate band labels")

    def __call__(self, photometry: Photometry, *, flux_covariance: np.ndarray | None = None) -> FeatureSet:
        """Transform native flux; full covariance, if supplied, follows input order."""
        p = photometry.align(self.bands)
        obs = p.observed
        f = np.where(obs, p.flux, 0.)
        b = self.softening
        x = 22.5 - _MAG_FACTOR * (np.arcsinh(f / (2*b)) + np.log(b))
        derivative = -_MAG_FACTOR / np.hypot(f, 2*b)
        n, d = f.shape
        if flux_covariance is None:
            cov = np.zeros((n, d, d))
            cov[:, np.arange(d), np.arange(d)] = np.where(obs, p.variance, 0.) * derivative**2
        else:
            original = np.asarray(flux_covariance, float)
            dp = len(photometry.bands)
            if original.shape != (n, dp, dp):
                raise ValueError("flux covariance must have shape (objects, input bands, input bands)")
            pair = photometry.observed[:, :, None] & photometry.observed[:, None, :]
            usable = np.where(pair, original, 0.)
            if not np.isfinite(usable).all() or not np.allclose(usable, usable.swapaxes(1, 2)):
                raise ValueError("observed covariance entries must be finite and symmetric")
            scale=np.maximum(1.,np.max(np.abs(usable),axis=(1,2)))
            if (np.linalg.eigvalsh(usable)[:,0] < -1e-10*scale).any():
                raise ValueError("flux covariance must be positive semidefinite")
            diag = np.diagonal(original, axis1=1, axis2=2)
            if not np.allclose(diag[photometry.observed], photometry.variance[photometry.observed]):
                raise ValueError("covariance diagonal disagrees with the supplied variances")
            cov = np.zeros((n, d, d))
            idx = [self.bands.index(label) for label in photometry.bands]
            cov[:, np.array(idx)[:, None], np.array(idx)[None, :]] = usable
            cov *= derivative[:, :, None] * derivative[:, None, :]
        cov = np.where(obs[:, :, None] & obs[:, None, :], cov, 0.)
        nan = np.full(n, np.nan)
        return FeatureSet(np.where(obs, x, np.nan), cov, obs, nan.copy(), nan.copy(),
                          nan.copy(), self.bands, {})

    def to_dict(self) -> dict:
        return dict(kind="native_band_luptitudes", bands=list(self.bands),
                    softening=self.softening.tolist(), dereddened=False)


def conditional_log_prob(mix: GaussianMixture, x: np.ndarray, cov: np.ndarray | None,
                         observed: np.ndarray, anchor: int) -> np.ndarray:
    """Log p(other observed luptitudes | measured anchor), per mag^(N-1).

    Noise is convolved with the JOINT distribution before conditioning. This
    includes noisy-anchor correlations and updates mixture weights correctly.
    The difference of two normalised log densities is exact for the mixture.
    """
    ref_obs = np.zeros_like(observed, dtype=bool)
    ref_obs[:, anchor] = observed[:, anchor]
    return (mix.log_prob(x, cov, observed=observed)
            - mix.log_prob(x, cov, observed=ref_obs))


class _ConditionalQSO(SlicedColourRedshiftModel):
    def __init__(self, model: SlicedColourRedshiftModel, anchor: int):
        super().__init__(model.z_centres, model.mixtures, model.n_train,
                         model.system, model.labels, model.meta)
        self.anchor = anchor

    def _log_p_slices(self, x, cov, observed):
        obs = np.ones_like(x, bool) if observed is None else observed
        return np.column_stack([conditional_log_prob(m, x, cov, obs, self.anchor)
                                for m in self.mixtures])

    def ood_score(self, x, cov, z, *, observed=None):
        mix = self.mixtures[int(np.argmin(abs(self.z_centres-z)))]
        return _conditional_min_mahalanobis([mix], x, cov, observed, self.anchor)

    def ood_score_any_z(self, x, cov, *, observed=None):
        return _conditional_min_mahalanobis(self.mixtures, x, cov, observed, self.anchor)


def _conditional_min_mahalanobis(mixtures, x, cov, observed, anchor, chunk=2048):
    """Nearest-component distance of the non-reference bands given the reference.

    Same conditioning as :func:`conditional_log_prob`: noise is added to the
    joint covariance first, then each component is conditioned on the anchor.
    Vectorised over objects sharing an observed-band pattern.
    """
    x = np.atleast_2d(np.asarray(x, float))
    n, d = x.shape
    obs = np.ones_like(x, bool) if observed is None else np.asarray(observed, bool)
    s = np.zeros((n, d, d)) if cov is None else np.asarray(cov, float)
    mus = np.concatenate([m.means for m in mixtures])
    vs = np.concatenate([m.covs for m in mixtures])
    out = np.full(n, np.nan)
    pattern = obs & (np.arange(d) != anchor)
    ok = obs[:, anchor] & pattern.any(axis=1)
    packed = np.packbits(pattern, axis=1)
    _, inverse = np.unique(packed, axis=0, return_inverse=True)
    for g in np.unique(inverse[ok]):
        rows = np.flatnonzero(ok & (inverse == g))
        dims = np.flatnonzero(pattern[rows[0]])
        j = np.concatenate([[anchor], dims])
        vj = vs[np.ix_(np.arange(len(vs)), j, j)]                    # (K, J, J)
        for lo in range(0, rows.size, chunk):
            r = rows[lo:lo + chunk]
            t = vj[None] + s[np.ix_(r, j, j)][:, None]                # (m, K, J, J)
            va = t[..., 0, 0]
            cross = t[..., 1:, 0]
            mean = mus[None][..., dims] + cross / va[..., None] * (
                x[r, anchor][:, None, None] - mus[None][..., anchor][..., None])
            cond = t[..., 1:, 1:] - cross[..., :, None] * cross[..., None, :] / va[..., None, None]
            y = np.linalg.solve(np.linalg.cholesky(cond),
                                (x[r][:, dims][:, None, :] - mean)[..., None])[..., 0]
            out[r] = np.sqrt(np.einsum("mki,mki->mk", y, y).min(axis=1))
    return out


def _background_log_prob(model, marginals, x, cov, observed, anchor):
    """Conditional field density, using a declared marginal when it covers input.

    Routing depends only on observed band labels. A marginal is never applied
    when an observed band or the reference lies outside its fitted coordinates.
    The smallest covering model takes precedence; ties keep saved order.
    """
    lp = np.empty(len(x)); remaining = np.ones(len(x), bool)
    for marginal in sorted(marginals, key=lambda m: m.n_dim):
        indices = np.array([model.labels.index(label) for label in marginal.labels])
        if anchor not in indices:
            continue
        outside = np.ones(len(model.labels), bool); outside[indices] = False
        use = remaining & ~observed[:, outside].any(axis=1)
        if not use.any():
            continue
        subcov = None if cov is None else cov[use][:, indices, :][:, :, indices]
        lp[use] = conditional_log_prob(marginal, x[use][:, indices], subcov,
            observed[use][:, indices], int(np.flatnonzero(indices == anchor)[0]))
        remaining[use] = False
    if remaining.any():
        lp[remaining] = conditional_log_prob(model, x[remaining],
            None if cov is None else cov[remaining], observed[remaining], anchor)
    return lp


class _ConditionalBackground:
    def __init__(self, model: GaussianMixture, system: str, anchor: int, bounds: np.ndarray,
                 marginals=()):
        self.model, self.system, self.anchor = model, system, anchor
        self.labels, self.bounds = model.labels, bounds
        self.marginals = marginals

    def check_system(self, system):
        if system != self.system:
            raise ValueError("multisurvey background system mismatch")

    def log_prob(self, x, cov, ref_mag, l_deg, b_deg, *, observed=None, return_level=False):
        obs = np.ones_like(x, bool) if observed is None else observed
        lp = _background_log_prob(self.model, self.marginals, x, cov, obs, self.anchor)
        return (lp, np.zeros(len(x))) if return_level else lp

    def ood_score(self, x, cov, ref_mag, l_deg, b_deg, *, observed=None):
        """Nearest joint-background component, conditioned on the reference.

        Uses the joint fit only; a declared marginal, where it applies, is a
        refit of the same population in fewer bands.
        """
        return _conditional_min_mahalanobis([self.model], x, cov, observed, self.anchor)

    def out_of_mag_range(self, ref_mag):
        lo, hi = self.bounds[self.anchor]
        return ~np.isfinite(ref_mag) | (ref_mag < lo) | (ref_mag > hi)


@dataclass
class MultiSurveyScore(PairScore):
    """The existing score contract plus the exact measurements used."""
    reference_band: str = ""
    bands_used: tuple[str, ...] = ()
    surveys_used: tuple[str, ...] = ()


@dataclass
class MultiSurveyModel:
    """A joint quasar model with declared field densities for arbitrary bands.

    Field likelihoods use the joint background unless a saved marginal fit
    contains every observed band. Each is a normalised conditional density.

    Priors are optional, as in the original scorer. A prior pair must explicitly
    identify this transform and reference band; an optical prior is never used
    for an infrared-only object. The default supplies colour evidence and the
    quasar-conditional redshift probability, leaving population posteriors NaN.
    """
    qso: SlicedColourRedshiftModel
    background: GaussianMixture
    transform: BandLuptitudeTransform
    reference_priority: tuple[str, ...]
    background_bounds: np.ndarray
    meta: dict = field(default_factory=dict)
    background_marginals: tuple[GaussianMixture, ...] = ()

    def __post_init__(self):
        self.reference_priority = tuple(self.reference_priority)
        self.background_bounds = np.asarray(self.background_bounds, float)
        if not (tuple(self.qso.labels) == self.background.labels == self.transform.bands):
            raise ValueError("quasar, background and transform band layouts disagree")
        if set(self.reference_priority) != set(self.transform.bands):
            raise ValueError("reference priority must contain every band exactly once")
        if len(self.reference_priority) != len(self.transform.bands):
            raise ValueError("duplicate reference priority entries")
        if self.background_bounds.shape != (len(self.transform.bands), 2):
            raise ValueError("background bounds must have shape (bands, 2)")
        self.background_marginals = tuple(self.background_marginals)
        for marginal in self.background_marginals:
            if (not marginal.labels or len(set(marginal.labels)) != marginal.n_dim or
                    not set(marginal.labels) <= set(self.transform.bands)):
                raise ValueError("background marginal must have unique model band labels")

    def background_log_prob(self, x: np.ndarray, cov: np.ndarray | None,
                            observed: np.ndarray, anchor: int) -> np.ndarray:
        """Field log density per observed non-reference luptitude volume.

        Inputs follow the saved full band schema. Declared marginal fits apply
        only when they contain every observed band, including the reference.
        """
        return _background_log_prob(self.background, self.background_marginals,
                                    x, cov, observed, anchor)

    @property
    def transform_id(self) -> str:
        return hashlib.sha256(json.dumps(self.transform.to_dict(), sort_keys=True).encode()).hexdigest()

    def score(self, photometry: Photometry, *, z_primary: np.ndarray,
              match: RedshiftMatch, min_bands: int,
              l_deg: np.ndarray, b_deg: np.ndarray,
              priors: dict | None = None, flux_covariance: np.ndarray | None = None,
              outlier: "MultiSurveyOutlier | None" = None,
              **kwargs) -> list[MultiSurveyScore]:
        """Score arbitrary survey subsets; ``min_bands`` counts measured bands.

        With two measured bands there is one colour. The caller supplies the
        minimum explicitly rather than silently rejecting infrared-only input.
        Candidate blend policy and measurements are forwarded to the original
        scorer. Priors map an anchor label to (GridQSOPrior, BackgroundDensity).
        """
        if min_bands < 2:
            raise ValueError("colour evidence requires at least two measured bands")
        if kwargs.get("outlier_model") is not None:
            raise ValueError(
                "an unconditional outlier model has the wrong units for the conditional "
                "multi-survey densities; pass outlier=MultiSurveyOutlier instead")
        if outlier is not None and outlier.transform_id != self.transform_id:
            raise ValueError("outlier model was built for a different luptitude transform")
        features = self.transform(photometry, flux_covariance=flux_covariance)
        n = features.n_obs
        if not n:
            return []
        order = np.array([features.labels.index(b) for b in self.reference_priority])
        anchor = order[np.argmax(features.observed[:, order], axis=1)]
        result = [None] * n
        for a in np.unique(anchor):
            rows = np.flatnonzero(anchor == a)
            fs = features.subset(rows)
            label = features.labels[a]
            fs.ref_mag = fs.x[:, a].copy()
            aligned = photometry.align(features.labels)
            fs.ref_flux = aligned.flux[rows, a]
            usable_reference=fs.observed[:,a]
            safe_variance=np.where(usable_reference,aligned.variance[rows,a],np.inf)
            fs.ref_snr = np.divide(fs.ref_flux, np.sqrt(safe_variance),
                                   out=np.full(len(rows), np.nan),
                                   where=usable_reference)
            primary=np.broadcast_to(np.asarray(z_primary),(n,))[rows]
            fs.flags["primary_z_outside_model_support"]=~self.qso.in_support(primary)
            per_slice=self.qso.meta.get("per_slice",[])
            minimum=self.meta.get("settings",{}).get("config",{}).get("min_band_training")
            if minimum is not None and per_slice and "band_counts" in per_slice[0]:
                counts=np.array([s["band_counts"] for s in per_slice])
                nearest=np.argmin(abs(primary[:,None]-self.qso.z_centres),axis=1)
                fs.flags["sparse_qso_training_bands"]=np.any((counts[nearest]<minimum)&fs.observed,axis=1)
            qprior, density = (None, None) if priors is None else priors.get(label, (None, None))
            for prior in (qprior, density):
                if prior is not None and (prior.meta.get("reference_band") != label or
                                          prior.meta.get("transform_id") != self.transform_id):
                    raise ValueError("prior reference band or luptitude transform does not match")
            local = dict(kwargs)
            local.setdefault("manifest_id",self.meta.get("run_id",self.qso.system))
            if outlier is not None and outlier.has(label):
                local["outlier_model"] = outlier.conditional(int(a), label, self.qso.system)
            for key in ("candidate_id", "primary_id", "separation_arcsec", "fracflux"):
                if key in local and local[key] is not None:
                    local[key] = np.broadcast_to(np.asarray(local[key]), (n,))[rows]
            if "candidate_id" not in local:
                local["candidate_id"] = np.array([f"cand{i}" for i in rows])
            if "primary_id" not in local:
                local["primary_id"] = np.array([f"prim{i}" for i in rows])
            scores = score_candidates(fs,
                z_primary=np.broadcast_to(np.asarray(z_primary), (n,))[rows],
                l_deg=np.broadcast_to(np.asarray(l_deg), (n,))[rows],
                b_deg=np.broadcast_to(np.asarray(b_deg), (n,))[rows],
                qso_model=_ConditionalQSO(self.qso, int(a)),
                background_model=_ConditionalBackground(self.background, self.qso.system,
                                                        int(a), self.background_bounds,
                                                        self.background_marginals),
                match=match, min_bands=min_bands, qso_prior=qprior, background_density=density,
                **local)
            for i, score in zip(rows, scores):
                used = tuple(b for b, ok in zip(features.labels, features.observed[i]) if ok)
                result[i] = MultiSurveyScore(**asdict(score), reference_band=label,
                    bands_used=used, surveys_used=tuple(sorted({survey_of(b) for b in used})))
        return result

    def to_dict(self) -> dict:
        return dict(kind="multisurvey_conditional_photometry", version=2 if self.background_marginals else 1,
                    qso=self.qso.to_dict(), background=self.background.to_dict(),
                    transform=self.transform.to_dict(),
                    reference_priority=list(self.reference_priority),
                    background_bounds=self.background_bounds.tolist(), meta=self.meta,
                    background_marginals=[m.to_dict() for m in self.background_marginals])

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(self.to_dict(), allow_nan=False))
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @classmethod
    def load(cls, path: str | Path) -> MultiSurveyModel:
        d = json.loads(Path(path).read_text())
        if d.get("kind") != "multisurvey_conditional_photometry" or d.get("version") not in (1, 2):
            raise ValueError("unsupported multisurvey model format")
        tr = d["transform"]
        if tr.get("kind")!="native_band_luptitudes" or tr.get("dereddened") is not False:
            raise ValueError("unsupported photometric calibration convention")
        return cls(SlicedColourRedshiftModel.from_dict(d["qso"]),
                   GaussianMixture.from_dict(d["background"]),
                   BandLuptitudeTransform(tuple(tr["bands"]), np.array(tr["softening"])),
                   tuple(d["reference_priority"]), np.array(d["background_bounds"]), d["meta"],
                   tuple(GaussianMixture.from_dict(m) for m in d.get("background_marginals", [])))


def load_priors(path: str | Path, model: MultiSurveyModel) -> dict:
    """Reference-band surface densities for ``model``, ready for ``score(priors=...)``.

    The file (``scripts/build_multisurvey_priors.py``) holds one
    (Sigma_Q(z, u_a), Sigma_B(u_a)) pair per reference band, both per unit
    native luptitude, for objects with that band measured. A file built for a
    different luptitude transform is refused, since its densities would be per
    unit of a different coordinate.

    Returns
    -------
    dict
        ``reference band label -> (GridQSOPrior, BackgroundSurfaceDensity)``.
        Bands without enough data to build a pair are absent; candidates whose
        reference is one of them get the no-prior status.
    """
    from .priors import BackgroundSurfaceDensity, GridQSOPrior

    d = json.loads(Path(path).read_text())
    if d.get("kind") != "multisurvey_priors":
        raise ValueError(f"not a multi-survey prior file: kind={d.get('kind')!r}")
    if d["transform_id"] != model.transform_id:
        raise ValueError("priors were built for a different luptitude transform")
    return {label: (GridQSOPrior.from_dict(e["qso_prior"]),
                    BackgroundSurfaceDensity.from_dict(e["background_density"]))
            for label, e in d["anchors"].items()}


@dataclass
class MultiSurveyOutlier:
    """The unmodelled hypothesis for the conditional multi-survey densities.

    One joint Gaussian :math:`\\mathcal N(\\bar{\\mathbf u}, \\kappa^2\\mathbf E)` over
    all luptitudes, conditioned on the reference band exactly as the quasar and
    background mixtures are.  If :math:`\\kappa^2\\mathbf E \\succeq \\mathbf V_k` for
    every joint component, the same holds for the conditional covariances
    (Schur complements are monotone in the Loewner order), so the conditional
    density dominates every conditional tail.  :math:`\\eta` is fitted per
    reference band and magnitude bin; a band with too few calibration objects
    uses the pooled fraction stored under ``"*"``.
    """
    mean: np.ndarray
    cov: np.ndarray
    kappa: float
    kappa_min: float
    labels: tuple[str, ...]
    transform_id: str
    fractions: dict                   # label -> (mag_edges, fraction)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mean = np.asarray(self.mean, float)
        self.cov = np.asarray(self.cov, float)
        self.labels = tuple(self.labels)
        if not self.kappa > self.kappa_min:
            raise ValueError("kappa does not exceed kappa_min: the density would not "
                             "dominate every component's tail")
        self.fractions = {k: (np.asarray(e, float), np.asarray(f, float))
                          for k, (e, f) in self.fractions.items()}
        for k, (e, f) in self.fractions.items():
            if f.size != e.size - 1 or ((f < 0) | (f >= 1)).any():
                raise ValueError(f"{k}: one fraction in [0, 1) per magnitude bin")
        self._mix = GaussianMixture(np.ones(1), self.mean[None], self.cov[None],
                                    labels=self.labels)

    def has(self, label: str) -> bool:
        return label in self.fractions or "*" in self.fractions

    def conditional(self, anchor: int, label: str, system: str) -> "_ConditionalOutlier":
        edges, frac = self.fractions.get(label, self.fractions.get("*"))
        return _ConditionalOutlier(self._mix, anchor, edges, frac, system, self.labels)

    def to_dict(self) -> dict:
        return {"kind": "multisurvey_outlier", "mean": self.mean.tolist(),
                "cov": self.cov.tolist(), "kappa": self.kappa, "kappa_min": self.kappa_min,
                "labels": list(self.labels), "transform_id": self.transform_id,
                "fractions": {k: [e.tolist(), f.tolist()] for k, (e, f) in self.fractions.items()},
                "meta": self.meta}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "MultiSurveyOutlier":
        d = json.loads(Path(path).read_text())
        if d.get("kind") != "multisurvey_outlier":
            raise ValueError(f"not a multi-survey outlier model: kind={d.get('kind')!r}")
        return cls(d["mean"], d["cov"], d["kappa"], d["kappa_min"], tuple(d["labels"]),
                   d["transform_id"], {k: tuple(v) for k, v in d["fractions"].items()},
                   d.get("meta", {}))


class _ConditionalOutlier:
    """What ``score_candidates`` needs: a conditional log density and eta(m)."""

    def __init__(self, mix, anchor, edges, fraction, system, labels):
        self.mix, self.anchor, self.edges, self.fraction = mix, anchor, edges, fraction
        self.system, self.labels = system, labels

    def check_system(self, system):
        if system != self.system:
            raise ValueError("multisurvey outlier system mismatch")

    def log_prob(self, x, cov=None, *, observed=None):
        obs = np.ones_like(x, bool) if observed is None else observed
        return conditional_log_prob(self.mix, x, cov, obs, self.anchor)

    def fraction_at(self, ref_mag):
        idx = np.clip(np.digitize(np.asarray(ref_mag, float), self.edges) - 1, 0,
                      self.fraction.size - 1)
        return self.fraction[idx]
