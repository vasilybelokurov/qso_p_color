"""The Legacy-PSF baseline: one bundle, one selection, two hemispheres.

A bundle is a directory with ``manifest.json`` and, per hemisphere, a model,
its priors and its unmodelled term. The manifest records the declared
selection, the domain, the sky partition and the sample identities, plus the
SHA-256 of every file. :meth:`LegacyBaseline.load` refuses a bundle whose files
do not match their recorded hashes or whose artifacts carry a different
bundle id, so a model can never be scored with another run's priors even when
the luptitude transforms happen to agree.

Candidates enter as Legacy DR9 catalogue rows (the columns of
:data:`qso_pcolor.legacy.CATALOGUE_COLUMNS`, e.g. from
:func:`qso_pcolor.legacy.legacy_match`). The bundle's own
:class:`~qso_pcolor.legacy.LegacySelection` decides eligibility, the same code
that selected the training data; ineligible rows are not scored and get a
reason instead.

    from qso_pcolor.baseline import LegacyBaseline
    bl = LegacyBaseline.load("models/legacy_psf/current")
    scores, decision = bl.score_rows(rows, z_primary=z0, match=RedshiftMatch(2000.0))
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .legacy import BANDS, LegacySelection, hemisphere_labels, legacy_photometry
from .qso_model import RedshiftMatch
from .score import BlendPolicy, PairScore


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_bundle(path: str | Path) -> Path:
    """A bundle directory, or a pointer file ``{"bundle": "<dir name>"}``."""
    path = Path(path)
    if path.is_file():
        target = json.loads(path.read_text())["bundle"]
        return (path.parent / target).resolve()
    return path


@dataclass
class LegacyBaseline:
    manifest: dict
    selection: LegacySelection
    models: dict
    priors: dict
    outliers: dict

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]

    @property
    def candidate_range(self) -> tuple[float, float]:
        lo, hi = self.manifest["domain"]["candidate_ref_range"]
        return float(lo), float(hi)

    @classmethod
    def load(cls, path: str | Path) -> "LegacyBaseline":
        from .multisurvey import MultiSurveyModel, MultiSurveyOutlier, load_priors
        root = resolve_bundle(path)
        man = json.loads((root / "manifest.json").read_text())
        if man.get("kind") != "legacy_baseline_bundle":
            raise ValueError("not a Legacy baseline bundle")
        for name, digest in man["files"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"{name}: content does not match the manifest")
        sel = LegacySelection(**{k: man["selection"][k] for k in
                                 ("morphology", "maskbits_zero", "reference", "version")})
        if sel.identity != man["selection_id"]:
            raise ValueError("manifest selection does not reproduce its recorded identity")
        models, priors, outliers = {}, {}, {}
        for h, files in man["hemispheres"].items():
            m = MultiSurveyModel.load(root / files["model"])
            o = MultiSurveyOutlier.load(root / files["outlier"])
            pr = json.loads((root / files["priors"]).read_text())
            for what, meta in (("model", m.meta), ("outlier", o.meta), ("priors", pr)):
                if meta.get("bundle_id") != man["bundle_id"]:
                    raise ValueError(f"{h} {what} belongs to a different bundle")
                if meta.get("selection_id") != man["selection_id"]:
                    raise ValueError(f"{h} {what} was built for a different selection")
            if m.transform.bands != hemisphere_labels(h):
                raise ValueError(f"{h} model bands are not the {h} Legacy bands")
            if o.transform_id != m.transform_id:
                raise ValueError(f"{h} outlier transform differs from its model")
            models[h], outliers[h] = m, o
            priors[h] = load_priors(root / files["priors"], m)
        return cls(man, sel, models, priors, outliers)

    def score_rows(self, rows: dict, *, z_primary, match, min_bands: int = 2,
                   separation_arcsec=None, fracflux=None, blend_policy=None,
                   candidate_id=None, primary_id=None):
        """Score catalogue rows; returns (scores, decision).

        ``scores[i]`` is a :class:`MultiSurveyScore`, or None when row i was
        not eligible; ``decision["reason"][i]`` then says why: a selection
        failure (see :meth:`LegacySelection.decide`), ``outside_latitude``,
        ``outside_domain`` (reference luptitude outside the candidate range) or
        ``status:<...>`` (scored, but the scorer's status is not ok, e.g. the
        primary redshift is outside the model's support).
        """
        from .data import galactic_from_equatorial
        n = len(np.asarray(rows["ra"]))
        dec = self.selection.decide(rows)
        reason = dec["reason"].astype("<U24")
        z_primary = np.broadcast_to(np.asarray(z_primary, float), (n,))
        l, b = galactic_from_equatorial(np.asarray(rows["ra"], float), np.asarray(rows["dec"], float))
        out = [None] * n
        lo, hi = self.candidate_range
        b_min = self.manifest["domain"].get("min_abs_b_deg")
        if b_min is not None:
            low_b = dec["accepted"] & (np.abs(b) < b_min)
            reason[low_b] = "outside_latitude"
            dec = dict(dec, accepted=dec["accepted"] & ~low_b)
        extra = dict(separation_arcsec=separation_arcsec, fracflux=fracflux,
                     candidate_id=candidate_id, primary_id=primary_id)
        for h, model in self.models.items():
            use = np.flatnonzero(dec["accepted"] & (dec["hemisphere"] == h))
            if not use.size:
                continue
            sub = {k: np.asarray(v)[use] for k, v in rows.items() if np.ndim(v) and len(v) == n}
            phot = legacy_photometry(sub, h, maskbits_zero=self.selection.maskbits_zero)
            u_ref = model.transform(phot).x[:, BANDS.index(self.selection.reference)]
            inside = (u_ref >= lo) & (u_ref < hi)
            reason[use[~inside]] = "outside_domain"
            use, phot = use[inside], phot.subset(np.flatnonzero(inside))
            if not use.size:
                continue
            kw = {k: np.broadcast_to(np.asarray(v), (n,))[use] for k, v in extra.items()
                  if v is not None}
            if blend_policy is not None:
                kw["blend_policy"] = blend_policy
            scores = model.score(phot, z_primary=z_primary[use], match=match, min_bands=min_bands,
                                 l_deg=l[use], b_deg=b[use], priors=self.priors[h],
                                 outlier=self.outliers[h], **kw)
            ref = hemisphere_labels(h)[BANDS.index(self.selection.reference)]
            for i, s in zip(use, scores):
                if s.reference_band != ref:
                    raise RuntimeError("baseline scored with a reference other than the declared one")
                out[i] = s
        for i, s_ in enumerate(out):
            if s_ is not None and s_.status != "ok":
                reason[i] = f"status:{s_.status}"[:24]
        dec = dict(dec, reason=reason,
                   eligible=np.array([s_ is not None and s_.status == "ok" for s_ in out]))
        return out, dec


@dataclass
class XDQSOBaseline:
    """The Legacy-PSF baseline in the original XDQSO-style design.

    Per hemisphere: quasar slices, field mixtures per r bin, Sigma_Q, Sigma_B
    and the broad unmodelled term (``scripts/fit_legacy_psf_xdqso.py``).
    Candidates are Legacy DR9 catalogue rows with the ``mw_transmission_*``
    columns; they are selected, dereddened and scored exactly as the training
    data were.
    """
    manifest: dict
    selection: LegacySelection
    parts: dict

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]

    @classmethod
    def load(cls, path: str | Path) -> "XDQSOBaseline":
        from .background import BackgroundColourModel
        from .outlier import OutlierModel
        from .priors import BackgroundSurfaceDensity, GridQSOPrior
        from .qso_model import SlicedColourRedshiftModel
        root = resolve_bundle(path)
        man = json.loads((root / "manifest.json").read_text())
        if man.get("kind") != "legacy_xdqso_bundle":
            raise ValueError("not a Legacy XDQSO baseline bundle")
        for name, digest in man["files"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"{name}: content does not match the manifest")
        sel = LegacySelection(**{k: man["selection"][k] for k in
                                 ("morphology", "maskbits_zero", "reference", "version")})
        if sel.identity != man["selection_id"]:
            raise ValueError("manifest selection does not reproduce its recorded identity")
        loaders = dict(qso=SlicedColourRedshiftModel.load, background=BackgroundColourModel.load,
                       qso_prior=GridQSOPrior.load, background_density=BackgroundSurfaceDensity.load,
                       outlier=OutlierModel.load)
        parts = {}
        for h, files in man["hemispheres"].items():
            parts[h] = {k: loaders[k](root / f) for k, f in files.items()}
            for k, obj in parts[h].items():
                if obj.meta.get("bundle_id") != man["bundle_id"]:
                    raise ValueError(f"{h} {k} belongs to a different bundle")
                if obj.meta.get("selection_id") != man["selection_id"]:
                    raise ValueError(f"{h} {k} was built for a different selection")
            systems = {parts[h]["qso"].system, parts[h]["background"].system, parts[h]["outlier"].system}
            if systems != {f"ls_dr9_{h}_grzw_psf"}:
                raise ValueError(f"{h}: photometric systems differ: {systems}")
        return cls(man, sel, parts)

    def score_rows(self, rows: dict, *, z_primary: np.ndarray | float,
                   match: RedshiftMatch, blend_policy: BlendPolicy, ood_flag_sigma: float,
                   separation_arcsec=None, fracflux=None, candidate_id=None,
                   primary_id=None) -> tuple[list[PairScore | None], dict]:
        """Score clean Legacy companions with explicit blend and support policies.

        Separation is in arcseconds; ``fracflux`` is dimensionless. Both must
        be measured. The blend policy must exclude failures and supply a finite
        non-negative fracflux limit. ``ood_flag_sigma`` is a positive distance
        to the nearest component, measured in noise-convolved standard deviations.

        Outside both fitted populations, retain model likelihoods, intensities,
        and distances for diagnosis, but withhold all posterior probabilities
        and the ranking statistic. ``eligible`` is true only for status ``ok``.
        The underlying Gaussian outlier fit is not a calibrated tail guarantee.
        """
        from .data import galactic_from_equatorial
        from .legacy import dereddened_relative_fluxes
        from .score import score_candidates
        if (not isinstance(blend_policy, BlendPolicy) or blend_policy.action != "exclude"
                or blend_policy.max_fracflux is None
                or not np.isfinite(blend_policy.max_fracflux) or blend_policy.max_fracflux < 0
                or not np.isfinite(blend_policy.min_separation_arcsec)):
            raise ValueError("science scoring requires an excluding BlendPolicy with finite limits")
        if not np.isfinite(ood_flag_sigma) or ood_flag_sigma <= 0:
            raise ValueError("ood_flag_sigma must be positive and finite")
        n = len(np.asarray(rows["ra"]))
        dec = self.selection.decide(rows)
        reason = dec["reason"].astype("<U64")
        l, b = galactic_from_equatorial(np.asarray(rows["ra"], float), np.asarray(rows["dec"], float))
        dom = self.manifest["domain"]
        low_b = dec["accepted"] & (np.abs(b) < dom["min_abs_b_deg"])
        reason[low_b] = "outside_latitude"
        accepted = dec["accepted"] & ~low_b
        z_primary = np.broadcast_to(np.asarray(z_primary, float), (n,))
        lo, hi = dom["ref_mag"]
        out = [None] * n
        candidate_id = np.arange(n).astype(str) if candidate_id is None else candidate_id
        primary_id = np.arange(n).astype(str) if primary_id is None else primary_id
        extra = dict(separation_arcsec=separation_arcsec, fracflux=fracflux,
                     candidate_id=candidate_id, primary_id=primary_id)
        for h, p in self.parts.items():
            use = np.flatnonzero(accepted & (dec["hemisphere"] == h))
            if not use.size:
                continue
            sub = {k: np.asarray(v)[use] for k, v in rows.items() if np.ndim(v) and len(v) == n}
            fs, ok = dereddened_relative_fluxes(sub, h, min_ref_snr=dom["min_ref_snr"],
                                                min_dims=dom["min_dims"])
            inside = ok & (fs.ref_mag >= lo) & (fs.ref_mag < hi)
            reason[use[~inside]] = "outside_domain"
            keep = np.flatnonzero(inside)
            use = use[keep]
            if not use.size:
                continue
            kw = {k: np.broadcast_to(np.asarray(v), (n,))[use] for k, v in extra.items() if v is not None}
            if blend_policy is not None:
                kw["blend_policy"] = blend_policy
            scores = score_candidates(fs.subset(keep), z_primary=z_primary[use], l_deg=l[use],
                                      b_deg=b[use], qso_model=p["qso"], background_model=p["background"],
                                      match=match, qso_prior=p["qso_prior"],
                                      background_density=p["background_density"],
                                      outlier_model=p["outlier"], min_bands=dom["min_dims"],
                                      ood_flag_sigma=ood_flag_sigma, manifest_id=self.bundle_id, **kw)
            for i, s_ in zip(use, scores):
                if s_.status == "ok" and "outside_both_models" in s_.quality_flags:
                    s_ = replace(s_, status="outside_both_models", log_r_per_unit_z=np.nan,
                                 p_sameq=np.nan, p_sameq_vs_bkg=np.nan,
                                 p_zmatch_given_qso=np.nan, p_outlier=np.nan)
                out[i] = s_
        for i, s_ in enumerate(out):
            if s_ is not None and s_.status != "ok":
                reason[i] = f"status:{s_.status}"
        return out, dict(dec, accepted=accepted, reason=reason,
                         eligible=np.array([s_ is not None and s_.status == "ok" for s_ in out]))
