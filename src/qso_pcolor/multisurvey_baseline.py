"""PSF science interface for one spatial, arbitrary-band joint model."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .legacy import morphology_status
from .multisurvey import MultiSurveyModel, MultiSurveyOutlier, load_priors
from .multisurvey_data import Photometry
from .qso_model import RedshiftMatch
from .score import BlendPolicy


@dataclass
class PSFMultiSurveyBaseline:
    """A consistent PSF population, spatial field, priors and outlier bundle.

    Morphology is supplied separately from the photometry. It does not require
    any particular band to be passed to the likelihood. All accepted rows use
    the same joint model and marginalisation, regardless of band subset.
    """
    model: MultiSurveyModel
    priors: dict
    outlier: MultiSurveyOutlier
    manifest: dict

    def __post_init__(self):
        if self.model.meta.get("population") != "psf" or self.model.legacy_field_fits:
            raise ValueError("the science interface requires one PSF joint model")
        if self.model.spatial_background is None:
            raise ValueError("the PSF bundle must include the spatial background")
        run_id = self.model.meta["run_id"]
        if (self.outlier.transform_id != self.model.transform_id or
                self.outlier.labels != self.model.transform.bands or
                self.outlier.meta.get("model_run_id") != run_id):
            raise ValueError("outlier calibration belongs to a different PSF model")
        for pair in self.priors.values():
            for prior in pair:
                if prior.meta.get("population") != "psf" or prior.meta.get("model_run_id") != run_id:
                    raise ValueError("population priors belong to a different PSF model")

    @classmethod
    def load(cls, path: str | Path) -> "PSFMultiSurveyBaseline":
        """Load and verify every file in a recorded bundle or its current pointer."""
        root = Path(path)
        if root.is_file():
            pointer = json.loads(root.read_text())
            root = root.parent / pointer["bundle"]
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("kind") != "multisurvey_psf_bundle":
            raise ValueError("not a multi-survey PSF bundle")
        if set(manifest["files"]) != {"model.json", "priors.json", "outlier.json"}:
            raise ValueError("manifest must cover the model, population priors and outlier")
        for name, expected in manifest["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"{name}: content does not match the bundle manifest")
        model = MultiSurveyModel.load(root / "model.json")
        return cls(model, load_priors(root / "priors.json", model),
                   MultiSurveyOutlier.load(root / "outlier.json"), manifest)

    def score(self, photometry: Photometry, *, morphology: np.ndarray,
              z_primary: np.ndarray | float, l_deg: np.ndarray, b_deg: np.ndarray,
              match: RedshiftMatch, blend_policy: BlendPolicy, ood_flag_sigma: float,
              flux_covariance: np.ndarray | None = None, **kwargs) -> tuple[list, dict]:
        """Score a chosen band subset, returning rows and explicit eligibility.

        ``morphology`` contains Tractor-compatible type labels (PSF, EXP, ...),
        independent of which bands the caller supplies. Unknown morphology,
        blends, unsupported positions and objects outside both density models
        cannot enter science ranking. Retained likelihoods remain diagnostics.
        """
        if (blend_policy is None or blend_policy.action != "exclude" or
                blend_policy.min_separation_arcsec is None or blend_policy.max_fracflux is None or
                not np.isfinite([blend_policy.min_separation_arcsec, blend_policy.max_fracflux]).all()):
            raise ValueError("an excluding blend policy with finite limits is required")
        if not np.isfinite(ood_flag_sigma) or ood_flag_sigma <= 0:
            raise ValueError("positive finite ood_flag_sigma is required")
        forbidden = {"priors", "outlier", "min_bands", "manifest_id"} & kwargs.keys()
        if forbidden:
            raise ValueError(f"bundle scoring controls cannot be overridden: {sorted(forbidden)}")
        n = len(photometry.flux)
        l = np.broadcast_to(np.asarray(l_deg, float), (n,))
        b = np.broadcast_to(np.asarray(b_deg, float), (n,))
        if not np.isfinite(l + b).all() or (np.abs(b) > 90).any():
            raise ValueError("finite Galactic coordinates are required")
        # A bundle fitted to extinction-corrected photometry corrects catalogue
        # input here, from each object's position, before anything else sees it.
        photometry, flux_covariance = self._corrected(photometry, l, b, flux_covariance)
        morph = morphology_status(np.broadcast_to(np.asarray(morphology), (n,)))
        accepted = (morph == "point") & (np.abs(b) >= self.manifest["min_abs_b_deg"])
        reason = np.full(n, "", dtype="<U64")
        reason[morph != "point"] = "non_psf_not_scored"
        reason[(morph == "point") & ~accepted] = "outside_latitude"
        limit = self.model.faint_limit_status(photometry)
        if limit is not None:
            has, bright = limit[0], limit[1]
            reason[accepted & ~has] = "no_faint_limit_band"
            reason[accepted & has & ~bright] = "too_faint"
            accepted &= has & bright
        indices = np.flatnonzero(accepted)
        output = [None] * n
        eligible = np.zeros(n, bool)
        if not len(indices):
            return output, dict(accepted=accepted, eligible=eligible, reason=reason, morphology=morph)
        local = dict(kwargs)
        for key in ("candidate_id", "primary_id", "separation_arcsec", "fracflux"):
            if key in local and local[key] is not None:
                local[key] = np.broadcast_to(np.asarray(local[key]), (n,))[indices]
        local.setdefault("candidate_id", np.array([f"cand{i}" for i in indices]))
        local.setdefault("primary_id", np.array([f"prim{i}" for i in indices]))
        config = json.dumps(dict(match=match.__dict__, blend=blend_policy.describe(),
            ood_flag_sigma=float(ood_flag_sigma), band_policy="any nonempty subset; exact marginalisation",
            extinction=self.model.meta["extinction"]["map"] if "extinction" in self.model.meta else "none"),
            sort_keys=True)
        config_hash = hashlib.sha256(config.encode()).hexdigest()[:16]
        scores = self.model.score(photometry.subset(indices),
            flux_covariance=None if flux_covariance is None else np.asarray(flux_covariance)[indices],
            z_primary=np.broadcast_to(np.asarray(z_primary), (n,))[indices],
            l_deg=l[indices], b_deg=b[indices], match=match,
            priors=self.priors, outlier=self.outlier, blend_policy=blend_policy,
            ood_flag_sigma=ood_flag_sigma, manifest_id=self.manifest["bundle_id"], **local)
        for i, score in zip(indices, scores):
            rejected = ("outside_both_models" if "outside_both_models" in score.quality_flags else
                        "background_out_of_mag_range" if "background_out_of_mag_range" in score.quality_flags else "")
            if rejected:
                score = replace(score, status=rejected, log_r_per_unit_z=np.nan,
                    p_sameq=np.nan, p_sameq_vs_bkg=np.nan, p_outlier=np.nan,
                    p_zmatch_given_qso=np.nan)
            score = replace(score, scoring_config_json=config, config_hash=config_hash)
            output[i] = score
            eligible[i] = score.status == "ok" and np.isfinite(score.log_r_per_unit_z)
            if not eligible[i]:
                reason[i] = score.status
        return output, dict(accepted=accepted, eligible=eligible, reason=reason, morphology=morph)

    def _corrected(self, photometry, l, b, flux_covariance=None):
        """Apply the bundle's declared Galactic extinction correction (identity if none)."""
        record = self.model.meta.get("extinction")
        if record is None:
            return photometry, flux_covariance
        from .extinction import deredden
        corrected, cov, _ = deredden(photometry, l, b, record, flux_covariance=flux_covariance)
        return corrected, cov

    def fit_local(self, photometry: Photometry, *, morphology: np.ndarray,
                  known_quasar: np.ndarray, l_deg: np.ndarray, b_deg: np.ndarray,
                  centre_l_deg: float, centre_b_deg: float, radius_deg: float,
                  excluded_positions: np.ndarray, exclusion_arcsec: float,
                  usable_area_deg2: dict[str, float], colour_n0: float,
                  density_n0: float, max_iter: int, tol: float,
                  flux_covariance: np.ndarray | None = None) -> "PSFMultiSurveyBaseline":
        """Refit the same joint weights and band-specific counts in a local cone.

        Input is a clean field catalogue with known-quasar flags. Positions and
        cone radius are Galactic degrees; exclusions (at least the primary and
        candidate) use an explicit arcsecond radius. ``usable_area_deg2`` maps
        reference bands to measured usable survey areas. Missing area entries
        retain the survey-wide density for that band, recorded in metadata.
        """
        import healpy as hp
        from .joint_spatial import fit_local_joint_weights
        from .spatial import SpatialSurfaceDensity

        n = len(photometry.flux)
        l, b = np.asarray(l_deg, float), np.asarray(b_deg, float)
        known = np.asarray(known_quasar)
        excluded = np.asarray(excluded_positions, float)
        if (l.shape != (n,) or b.shape != (n,) or known.shape != (n,) or known.dtype != bool or
                not np.isfinite(l + b).all() or (np.abs(b) > 90).any() or
                excluded.ndim != 2 or excluded.shape[1] != 2 or len(excluded) < 2 or
                not np.isfinite(excluded).all() or (np.abs(excluded[:, 1]) > 90).any() or
                not np.isfinite(exclusion_arcsec) or exclusion_arcsec <= 0 or
                not np.isfinite(density_n0) or density_n0 < 0):
            raise ValueError("local fitting needs valid coordinates, known-Q flags, exclusions and pooling")
        if not usable_area_deg2 or any(
                label not in self.model.transform.bands or not np.isfinite(area) or area <= 0
                for label, area in usable_area_deg2.items()):
            raise ValueError("positive measured usable areas are required for declared model bands")
        photometry, flux_covariance = self._corrected(photometry, l, b, flux_covariance)
        vectors = hp.ang2vec(l, b, lonlat=True)

        def separation(l0, b0):
            centre = hp.ang2vec(l0, b0, lonlat=True)
            return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(vectors, centre), axis=1),
                                        vectors @ centre))

        use = (morphology_status(np.broadcast_to(morphology, (n,))) == "point") & ~known
        use &= (np.abs(b) >= self.manifest["min_abs_b_deg"]) & (separation(centre_l_deg, centre_b_deg) <= radius_deg)
        for l0, b0 in excluded:
            use &= separation(l0, b0) * 3600 > exclusion_arcsec
        aligned = photometry.align(self.model.transform.bands)
        use &= aligned.observed.any(axis=1)
        if not use.any():
            raise ValueError("no eligible local PSF field measurements remain")
        rows = np.flatnonzero(use)
        features = self.model.transform(photometry.subset(rows),
            flux_covariance=None if flux_covariance is None else np.asarray(flux_covariance)[rows])
        meta = dict(population="psf", model_run_id=self.model.meta["run_id"],
                    n_input=n, n_used=int(use.sum()), known_quasars_removed=int(known.sum()),
                    excluded_positions=excluded.tolist(), exclusion_arcsec=exclusion_arcsec,
                    usable_area_deg2=usable_area_deg2)
        sky = fit_local_joint_weights(self.model.background, self.model.spatial_background,
            features.x, features.cov, features.observed, np.ones(use.sum()),
            l_deg=centre_l_deg, b_deg=centre_b_deg, radius_deg=radius_deg,
            n0=colour_n0, max_iter=max_iter, tol=tol, meta=meta)
        priors = dict(self.priors)
        fitted_labels = []
        for label, area in usable_area_deg2.items():
            if label not in priors:
                continue
            qp, base = priors[label]
            a = features.labels.index(label)
            edges = base.mag_edges
            mid = (edges[1:] + edges[:-1]) / 2
            density = base(mid, np.full(len(mid), centre_l_deg), np.full(len(mid), centre_b_deg))
            count = np.histogram(features.x[features.observed[:, a], a], edges)[0]
            expected = area * np.diff(edges) * density
            fraction = np.divide(expected, expected + density_n0,
                                 out=np.zeros_like(expected), where=expected + density_n0 > 0)
            pooled = fraction * count / (area * np.diff(edges)) + (1 - fraction) * density
            if (pooled <= 0).any():
                raise ValueError("local density has unsupported empty bins; use positive pooling")
            priors[label] = qp, SpatialSurfaceDensity(base.nside, base.nside_parent, edges,
                {}, {}, pooled, density_n0, dict(base.meta, mode="local", local_fit=meta))
            fitted_labels.append(label)
        sky.meta["density_bands_refitted"] = fitted_labels
        model = replace(self.model, spatial_background=sky,
                        meta=dict(self.model.meta, local_background=meta))
        identity = json.dumps(dict(base=self.manifest["bundle_id"], spatial=sky.to_dict(),
            densities={label: priors[label][1].to_dict() for label in fitted_labels}), sort_keys=True)
        manifest = dict(self.manifest, bundle_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                        base_bundle_id=self.manifest["bundle_id"], local_background=meta)
        return PSFMultiSurveyBaseline(model, priors, self.outlier, manifest)
