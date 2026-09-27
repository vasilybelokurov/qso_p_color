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

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .legacy import BANDS, LegacySelection, hemisphere_labels, legacy_photometry


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
        failure (see :meth:`LegacySelection.decide`) or ``outside_domain``
        (reference luptitude outside the candidate range).
        """
        from .data import galactic_from_equatorial
        n = len(np.asarray(rows["ra"]))
        dec = self.selection.decide(rows)
        reason = dec["reason"].astype("<U24")
        z_primary = np.broadcast_to(np.asarray(z_primary, float), (n,))
        l, b = galactic_from_equatorial(np.asarray(rows["ra"], float), np.asarray(rows["dec"], float))
        out = [None] * n
        lo, hi = self.candidate_range
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
        dec = dict(dec, reason=reason, eligible=np.array([s is not None for s in out]))
        return out, dec
