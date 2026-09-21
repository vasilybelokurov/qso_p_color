"""Combine the colour likelihoods and the surface densities into a score.

Three hypotheses
----------------
``same_q``
    A quasar whose redshift matches the primary, per :class:`RedshiftMatch`.
``field_q``
    A quasar, but at some other redshift.
``bkg``
    Anything else in the imaging catalogue at this brightness and sky position:
    stars, compact galaxies, and unclassified sources.

The intensities are

.. math::
    \\lambda_{\\rm sameQ} &= \\int W(z\\mid z_0)\\,\\Sigma_Q(z,m)\\,
        p(\\mathbf{c}\\mid Q,z)\\,\\mathrm{d}z \\\\
    \\lambda_{\\rm fieldQ} &= \\int [1 - W(z\\mid z_0)]\\,\\Sigma_Q(z,m)\\,
        p(\\mathbf{c}\\mid Q,z)\\,\\mathrm{d}z \\\\
    \\lambda_{\\rm bkg} &= \\Sigma_B(m,l,b)\\,p(\\mathbf{c}\\mid B,m,l,b)

all in deg^-2 mag^-1 per unit colour volume, so their ratios are meaningful.

Why ``field_q`` has to be there
-------------------------------
Without it, a real quasar at a completely different redshift is scored as
evidence *for* a same-redshift companion, because it beats the stellar locus
easily.  For a binary-quasar search that is the dominant failure mode, not a
refinement: the two-class number ``p_sameq_vs_bkg`` that the brief asks for is
reported, but ``log_r_per_unit_z`` is the one to rank on.

What the scorer refuses to do
-----------------------------
If no defensible quasar prior is available, the posterior fields are ``NaN``
with a status code, and only the prior-independent Bayes factor is returned.
An arbitrary class fraction is never substituted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .background import BackgroundColourModel
from .features import FeatureSet
from .priors import BackgroundSurfaceDensity, GridQSOPrior
from .qso_model import (RedshiftMatch, SlicedColourRedshiftModel,
                        _integration_grid, _window_subgrid, _window_integral)

__all__ = [
    "PairScore",
    "BlendPolicy",
    "score_candidates",
    "compare_backgrounds",
    "DEFAULT_Z_GRID",
]

DEFAULT_Z_GRID = np.linspace(0.05, 5.0, 496)


@dataclass(frozen=True)
class BlendPolicy:
    """Which companions are close enough to the primary to be untrustworthy.

    Scope decision, recorded here so it is enforced rather than remembered: the
    pipeline is being completed for **cleanly deblended** companions first.
    Below a few arcseconds the Legacy Surveys model fit divides flux between
    overlapping sources, so the two colour vectors are neither independent nor
    individually reliable, and a probability computed from them is a statement
    about the deblender rather than about the sky.  Blended pairs need
    image-level forced photometry, which is a separate piece of work.

    Both limits must be stated explicitly; there is no default, because a
    separation floor changes which objects the reported probabilities apply to.

    Parameters
    ----------
    min_separation_arcsec : float
        Companions closer than this to the primary are not scored.
    max_fracflux : float, optional
        Upper limit on the reference-band ``fracflux`` — the fraction of the
        source's flux contributed by *other* sources.  ``None`` disables the
        check, which is appropriate only when the value is unavailable.
    action : {'exclude', 'flag'}
        ``'exclude'`` returns NaN scores with ``status='blended_not_scored'``.
        ``'flag'`` scores the object anyway and records ``blended`` in
        ``quality_flags`` — use it to study the blend regime, never to rank.
    """

    min_separation_arcsec: float
    max_fracflux: float | None = None
    action: str = "exclude"

    def __post_init__(self) -> None:
        if self.action not in ("exclude", "flag"):
            raise ValueError("action must be 'exclude' or 'flag'")
        if self.min_separation_arcsec <= 0:
            raise ValueError("min_separation_arcsec must be positive")

    def violations(
        self,
        separation_arcsec: np.ndarray | None,
        fracflux: np.ndarray | None,
        n: int,
    ) -> np.ndarray:
        """Boolean mask of companions this policy considers blended.

        A missing measurement counts as a violation: we cannot certify that an
        object is cleanly deblended without the numbers that would show it.
        """
        bad = np.zeros(n, dtype=bool)
        if separation_arcsec is None:
            return np.ones(n, dtype=bool)
        sep = np.broadcast_to(np.asarray(separation_arcsec, float), (n,))
        bad |= ~np.isfinite(sep) | (sep < self.min_separation_arcsec)
        if self.max_fracflux is not None:
            if fracflux is None:
                return np.ones(n, dtype=bool)
            ff = np.broadcast_to(np.asarray(fracflux, float), (n,))
            bad |= ~np.isfinite(ff) | (ff > self.max_fracflux)
        return bad

    def describe(self) -> dict:
        return {
            "min_separation_arcsec": self.min_separation_arcsec,
            "max_fracflux": self.max_fracflux,
            "action": self.action,
        }


@dataclass
class PairScore:
    """Everything one candidate produced, evidence and posterior kept separate."""

    # -- identification
    candidate_id: str
    primary_id: str
    z_primary: float
    ref_mag: float
    photometric_system: str

    # -- prior-independent evidence
    loglike_qso_zprimary: float
    loglike_bkg: float
    log_bayes_factor_qz_bkg: float
    p_zmatch_given_qso: float
    z_phot_mode: float

    # -- intensities and posteriors (NaN when no defensible prior)
    # Intensities are stored as natural logs, in log(deg^-2 mag^-1 per unit
    # colour volume); -inf means an intensity of exactly zero.
    log_lambda_sameq: float
    log_lambda_fieldq: float
    log_lambda_bkg: float
    p_sameq_vs_bkg: float
    p_sameq: float

    # Window-averaged same-redshift intensity per unit redshift divided by the
    # total intensity. It tends to the window-free point value for narrow
    # windows; p_sameq = exp(log_r_per_unit_z) * dz_match_eff exactly.
    log_r_per_unit_z: float
    dz_match_eff: float

    # -- diagnostics
    # Share of the redshift normalisation that the integration grid would have
    # taken from outside the model's trained range, where log_p_colour_given_z
    # clamps and replays the edge slice.  The scorer excludes that region, so
    # this is what was DISCARDED, not what was used: a large value means this
    # candidate's colours are best explained at a redshift the model has never
    # seen, and p_zmatch_given_qso is conditional on a range that may exclude
    # the truth.
    frac_norm_outside_support: float
    qso_ood_sigma: float
    background_local_weight: float
    background_density_level: int
    n_bands_used: int
    status: str
    quality_flags: tuple[str, ...] = ()
    model_manifest_id: str = ""

    def as_row(self) -> dict:
        d = asdict(self)
        d["quality_flags"] = ",".join(self.quality_flags)
        return d


def _trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.trapezoid(y, x, axis=-1)


def score_candidates(
    features: FeatureSet,
    *,
    z_primary: np.ndarray,
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    qso_model: SlicedColourRedshiftModel,
    background_model: BackgroundColourModel,
    match: RedshiftMatch,
    qso_prior: GridQSOPrior | None = None,
    background_density: BackgroundSurfaceDensity | None = None,
    z_grid: np.ndarray = DEFAULT_Z_GRID,
    candidate_id: np.ndarray | None = None,
    primary_id: np.ndarray | None = None,
    system: str | None = None,
    manifest_id: str = "",
    min_bands: int = 2,
    n_window_sub: int = 129,
    blend_policy: "BlendPolicy | None" = None,
    separation_arcsec: np.ndarray | None = None,
    fracflux: np.ndarray | None = None,
) -> list[PairScore]:
    """Score a batch of companions against a batch of primary redshifts.

    Parameters
    ----------
    features : FeatureSet
        Output of a :class:`~qso_pcolor.features.FeatureTransform`.
    z_primary : ndarray, shape (n,)
        Spectroscopic redshift of each candidate's primary.
    l_deg, b_deg : ndarray, shape (n,)
        Galactic coordinates of the candidate.
    qso_model, background_model
        Fitted colour models.  Both are checked against ``system``.
    match : RedshiftMatch
        Definition of a redshift match; recorded in every row.
    qso_prior, background_density
        Surface-density models.  If either is ``None`` the posterior fields are
        ``NaN`` and ``status`` says so; the Bayes factor is still returned.
    z_grid : ndarray
        Quadrature grid. Exact model/prior boundaries, interpolation knots and
        window nodes are inserted; all integrals use the same supported grid.
    min_bands : int
        Minimum number of usable feature dimensions.  Below this the object is
        returned with ``status='insufficient_photometry'`` and NaN scores,
        rather than being scored from one colour.
    n_window_sub : int
        Number of points used to integrate the match window on its own dense
        sub-grid.  The default resolves any window well; see
        :func:`_window_subgrid`.
    blend_policy : BlendPolicy, optional
        Separation and ``fracflux`` limits below which catalogue photometry is
        not trusted.  ``None`` scores everything and records nothing, which is
        appropriate only for training or diagnostics — for real candidates,
        state the policy.
    separation_arcsec, fracflux : ndarray, shape (n,), optional
        Angular separation from the primary and reference-band ``fracflux``.
        Required by ``blend_policy``; a missing value counts as a violation,
        because a companion cannot be certified clean without them.

    Returns
    -------
    list of PairScore, one per input row, in input order.
    """
    sysname = system or qso_model.system
    qso_model.check_system(sysname)
    background_model.check_system(sysname)
    for name, model in (("quasar", qso_model), ("background", background_model)):
        if tuple(features.labels) != tuple(model.labels):
            raise ValueError(
                f"feature layout mismatch against the {name} model: it expects "
                f"{model.labels}, candidate features are {features.labels}. "
                f"Checking only one of the two models would let a reversed "
                f"feature order corrupt the Bayes factor silently."
            )

    n = features.n_obs
    z_primary = np.atleast_1d(np.asarray(z_primary, dtype=float))
    if z_primary.size == 1 and n > 1:
        z_primary = np.repeat(z_primary, n)
    l_deg = np.broadcast_to(np.atleast_1d(np.asarray(l_deg, float)), (n,))
    b_deg = np.broadcast_to(np.atleast_1d(np.asarray(b_deg, float)), (n,))
    cid = (
        np.asarray(candidate_id).astype(str)
        if candidate_id is not None
        else np.array([f"cand{i}" for i in range(n)])
    )
    pid = (
        np.asarray(primary_id).astype(str)
        if primary_id is not None
        else np.array([f"prim{i}" for i in range(n)])
    )

    z_grid = np.asarray(z_grid, dtype=float)
    n_bands = features.observed.sum(axis=1)

    if blend_policy is not None:
        blended = blend_policy.violations(separation_arcsec, fracflux, n)
    else:
        blended = np.zeros(n, dtype=bool)

    # log p(c | Q, z) on the whole grid, one pass over the slice models.
    log_slices = qso_model._log_p_slices(features.x, features.cov, features.observed)
    log_pq_grid = qso_model.log_p_colour_given_z(
        features.x, features.cov, z_grid, _log_slices=log_slices
    )

    log_pb, local_w = background_model.log_prob(
        features.x,
        features.cov,
        features.ref_mag,
        l_deg,
        b_deg,
        observed=features.observed,
        return_level=True,
    )

    if background_density is not None:
        sigma_b = background_density(features.ref_mag, l_deg, b_deg)
        dens_level = background_density.level(features.ref_mag, l_deg, b_deg)
    else:
        sigma_b = np.full(n, np.nan)
        dens_level = np.full(n, -1)

    out_of_mag = background_model.out_of_mag_range(features.ref_mag)
    out: list[PairScore] = []
    for i in range(n):
        flags = [k for k, v in features.flags.items() if bool(np.atleast_1d(v)[i])]
        status = "ok"
        if out_of_mag[i]:
            flags.append("background_out_of_mag_range")

        if blended[i]:
            flags = flags + ["blended"]
            if blend_policy.action == "exclude":
                out.append(
                    _null_score(
                        cid[i], pid[i], z_primary[i], features, sysname, i,
                        "blended_not_scored", flags, manifest_id, int(n_bands[i]),
                    )
                )
                continue

        if n_bands[i] < min_bands:
            out.append(
                _null_score(
                    cid[i], pid[i], z_primary[i], features, sysname, i,
                    "insufficient_photometry", flags, manifest_id, int(n_bands[i]),
                )
            )
            continue

        # -- prior-independent evidence -----------------------------------
        # Reuse the slice densities computed above rather than re-evaluating
        # every slice mixture for this one object.
        log_lq = float(
            qso_model.log_p_colour_given_z(
                features.x[i : i + 1],
                features.cov[i : i + 1],
                np.array([z_primary[i]]),
                observed=features.observed[i : i + 1],
                _log_slices=log_slices[i : i + 1],
            )[0, 0]
        )
        log_bf = log_lq - float(log_pb[i])

        z_sub = _window_subgrid(
            match, float(z_primary[i]), qso_model.support, n_window_sub
        )
        if z_sub.size == 0:
            out.append(
                _null_score(
                    cid[i], pid[i], z_primary[i], features, sysname, i,
                    "window_outside_model_support", flags, manifest_id,
                    int(n_bands[i]),
                )
            )
            continue
        w_sub = match.weight(z_sub, float(z_primary[i]))
        dz_eff = float(_trapz(w_sub, z_sub))      # window area actually usable
        # One supported grid for the total and the window: a narrow prior
        # peak resolved by the numerator must also enter the denominator.
        support = qso_model.support
        knots = qso_model.z_centres
        if qso_prior is not None:
            support = (max(support[0], qso_prior.z_edges[0]),
                       min(support[1], qso_prior.z_edges[-1]))
            knots = np.concatenate([knots, qso_prior.z_centres])
        nodes = _integration_grid(z_grid, support, knots, z_sub)
        lp = qso_model.log_p_colour_given_z(
            features.x[i:i+1], features.cov[i:i+1], nodes,
            _log_slices=log_slices[i:i+1],
        )[0] if nodes.size else np.empty(0)
        if qso_prior is not None:
            with np.errstate(divide="ignore"):
                lp += np.log(qso_prior(nodes, float(features.ref_mag[i])))
        if not np.isfinite(lp).any():
            row = _null_score(
                cid[i], pid[i], z_primary[i], features, sysname, i,
                "qso_prior_empty_at_this_magnitude", flags, manifest_id, int(n_bands[i]),
            )
            row.loglike_qso_zprimary = log_lq
            row.loglike_bkg = float(log_pb[i])
            row.log_bayes_factor_qz_bkg = log_bf
            row.dz_match_eff = dz_eff
            row.background_local_weight = float(local_w[i])
            row.background_density_level = int(dens_level[i])
            out.append(row)
            continue
        shift = float(lp[np.isfinite(lp)].max())
        post = np.exp(lp - shift)
        norm = float(_trapz(post, nodes))
        same = float(_window_integral(post, nodes, z_sub, match, float(z_primary[i])))
        p_zmatch = same / norm
        z_mode = float(nodes[int(np.argmax(post))])

        # Diagnostic only: how much mass would edge-clamping invent on the
        # requested grid outside the trained support? Never use it in a score.
        raw_lp = log_pq_grid[i].copy()
        if qso_prior is not None:
            with np.errstate(divide="ignore"):
                raw_lp += np.log(qso_prior(z_grid, float(features.ref_mag[i])))
        raw_shift = max(shift, float(np.max(raw_lp)))
        # Integrate exterior pieces separately, with exact boundary endpoints.
        outside = 0.0
        for lo, hi in ((z_grid[0], qso_model.support[0]),
                       (qso_model.support[1], z_grid[-1])):
            if hi > lo:
                ext = _integration_grid(z_grid, (lo, hi))
                ext_lp = qso_model.log_p_colour_given_z(
                    features.x[i:i+1], features.cov[i:i+1], ext,
                    _log_slices=log_slices[i:i+1],
                )[0]
                if qso_prior is not None:
                    with np.errstate(divide="ignore"):
                        ext_lp += np.log(qso_prior(ext, float(features.ref_mag[i])))
                outside += float(_trapz(np.exp(ext_lp - raw_shift), ext))
        supported = norm * np.exp(shift - raw_shift)
        frac_out = outside / (supported + outside)

        ood = float(
            qso_model.ood_score(
                features.x[i : i + 1],
                features.cov[i : i + 1],
                float(z_primary[i]),
                observed=features.observed[i : i + 1],
            )[0]
        )

        # -- intensities ---------------------------------------------------
        if qso_prior is None or background_density is None:
            log_lam_s = log_lam_f = log_lam_b = np.nan
            log_r = np.nan
            p_vs_bkg = p_full = np.nan
            status = "no_prior_posterior_unavailable"
        else:
            log_b = float(log_pb[i]) + (
                np.log(sigma_b[i]) if sigma_b[i] > 0 else -np.inf
            )
            scale = max(shift, log_b)
            total_q = norm * np.exp(shift - scale)
            s_rel = same * np.exp(shift - scale)
            f_rel = max(total_q - s_rel, 0.0)  # protect round-off only
            b_rel = float(np.exp(log_b - scale))

            denom = s_rel + f_rel + b_rel
            p_vs_bkg = s_rel / (s_rel + b_rel) if (s_rel + b_rel) > 0 else np.nan
            p_full = s_rel / denom if denom > 0 else np.nan

            # R is the window-AVERAGED same-redshift intensity per unit
            # redshift, divided by the total intensity of every explanation:
            #
            #     R = [ (1/dZ) \int W Sigma_Q p dz ] / (Lambda_Q,tot + lambda_bkg)
            #
            # so p_sameq = R * dz_match_eff by construction.  R tends to the
            # point value Sigma_Q(z0,m) p(c|Q,z0) / (...) as the window narrows,
            # and is window-independent only in that limit -- for a velocity
            # window it is, to well under a per cent.  For a wide window R is a
            # genuine average and does depend on the width.
            with np.errstate(divide="ignore"):
                log_r = (
                    float(np.log(s_rel) - np.log(dz_eff) - np.log(denom))
                    if denom > 0 and s_rel > 0 and dz_eff > 0
                    else -np.inf
                )
            with np.errstate(divide="ignore"):
                log_lam_s = float(np.log(s_rel) + scale)
                log_lam_f = float(np.log(f_rel) + scale)
                log_lam_b = float(np.log(b_rel) + scale)
            if not qso_model.in_support(np.array([z_primary[i]]))[0]:
                status = "primary_z_outside_model_support"

        out.append(
            PairScore(
                candidate_id=str(cid[i]),
                primary_id=str(pid[i]),
                z_primary=float(z_primary[i]),
                ref_mag=float(features.ref_mag[i]),
                photometric_system=sysname,
                loglike_qso_zprimary=log_lq,
                loglike_bkg=float(log_pb[i]),
                log_bayes_factor_qz_bkg=log_bf,
                p_zmatch_given_qso=p_zmatch,
                z_phot_mode=z_mode,
                frac_norm_outside_support=frac_out,
                log_lambda_sameq=log_lam_s,
                log_lambda_fieldq=log_lam_f,
                log_lambda_bkg=log_lam_b,
                log_r_per_unit_z=log_r,
                dz_match_eff=dz_eff,
                p_sameq_vs_bkg=p_vs_bkg,
                p_sameq=p_full,
                qso_ood_sigma=ood,
                background_local_weight=float(local_w[i]),
                background_density_level=int(dens_level[i]),
                n_bands_used=int(n_bands[i]),
                status=status,
                quality_flags=tuple(flags),
                model_manifest_id=manifest_id,
            )
        )
    return out


def compare_backgrounds(
    features: FeatureSet,
    *,
    models: dict,
    densities: dict,
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    **score_kwargs,
) -> dict:
    """Score the same candidates under several background models and compare.

    The hierarchical model and a locally fitted one answer the same question
    with different amounts of locality: the first averages over a HEALPix cell,
    the second uses only the candidate's own neighbourhood. They should agree
    where the field is smooth and the cell is well populated, and disagree where
    the local stellar density, depth or reddening departs from the cell average.
    Disagreement is therefore a diagnostic, not a failure: it says the answer
    depends on which background was assumed, and by how much.

    Parameters
    ----------
    models, densities : dict
        ``name -> BackgroundColourModel`` and ``name -> BackgroundSurfaceDensity``,
        with matching keys. Every model must carry the candidates' feature
        layout and photometric system.
    **score_kwargs
        Passed to :func:`score_candidates` unchanged (``qso_model``, ``match``,
        ``qso_prior``, ``z_primary`` and the rest).

    Returns
    -------
    dict
        ``rows``   : ``name -> list[PairScore]``;
        ``deltas`` : per-candidate differences of the quantities that matter,
        each as ``max - min`` across the models, so a single number says how
        much the choice of background moved the answer.
    """
    if set(models) != set(densities):
        raise ValueError("models and densities must have the same keys")

    rows = {}
    for name, model in models.items():
        rows[name] = score_candidates(
            features, l_deg=l_deg, b_deg=b_deg, background_model=model,
            background_density=densities[name], **score_kwargs,
        )

    names = list(models)
    n = features.n_obs
    fields = ("loglike_bkg", "log_bayes_factor_qz_bkg", "log_r_per_unit_z",
              "p_sameq", "p_sameq_vs_bkg")
    deltas = {}
    for f in fields:
        vals = np.array([[getattr(r, f) for r in rows[nm]] for nm in names], float)
        with np.errstate(invalid="ignore"):
            deltas[f] = np.nanmax(vals, axis=0) - np.nanmin(vals, axis=0)
    deltas["log_sigma_bkg"] = np.array([
        np.log(max(densities[names[0]](features.ref_mag[i:i+1], l_deg[i:i+1],
                                       b_deg[i:i+1])[0], 1e-300))
        - np.log(max(densities[names[-1]](features.ref_mag[i:i+1], l_deg[i:i+1],
                                          b_deg[i:i+1])[0], 1e-300))
        for i in range(n)
    ])
    return {"rows": rows, "deltas": deltas, "names": names}


def _null_score(cid, pid, zp, features, sysname, i, status, flags, manifest, nb):
    nan = float("nan")
    return PairScore(
        candidate_id=str(cid),
        primary_id=str(pid),
        z_primary=float(zp),
        ref_mag=float(features.ref_mag[i]),
        photometric_system=sysname,
        loglike_qso_zprimary=nan,
        loglike_bkg=nan,
        log_bayes_factor_qz_bkg=nan,
        p_zmatch_given_qso=nan,
        frac_norm_outside_support=nan,
        z_phot_mode=nan,
        log_lambda_sameq=nan,
        log_lambda_fieldq=nan,
        log_lambda_bkg=nan,
        log_r_per_unit_z=nan,
        dz_match_eff=nan,
        p_sameq_vs_bkg=nan,
        p_sameq=nan,
        qso_ood_sigma=nan,
        background_local_weight=nan,
        background_density_level=-1,
        n_bands_used=nb,
        status=status,
        quality_flags=tuple(flags),
        model_manifest_id=manifest,
    )
