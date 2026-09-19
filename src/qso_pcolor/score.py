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
reported, but ``p_sameq`` is the one to rank on.

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
from .qso_model import RedshiftMatch, SlicedColourRedshiftModel

__all__ = ["PairScore", "score_candidates", "DEFAULT_Z_GRID"]

DEFAULT_Z_GRID = np.linspace(0.05, 5.0, 496)


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

    # -- diagnostics
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
        Grid for the redshift integrals.  It must span the quasar model's
        trained range; the integrals are trapezoidal on this grid.
    min_bands : int
        Minimum number of usable feature dimensions.  Below this the object is
        returned with ``status='insufficient_photometry'`` and NaN scores,
        rather than being scored from one colour.

    Returns
    -------
    list of PairScore, one per input row, in input order.
    """
    sysname = system or qso_model.system
    qso_model.check_system(sysname)
    background_model.check_system(sysname)
    if tuple(features.labels) != tuple(qso_model.labels):
        raise ValueError(
            f"feature layout mismatch: model expects {qso_model.labels}, "
            f"candidate features are {features.labels}"
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

    out: list[PairScore] = []
    for i in range(n):
        flags = [k for k, v in features.flags.items() if bool(np.atleast_1d(v)[i])]
        status = "ok"

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

        w_match = match.weight(z_grid, float(z_primary[i]))

        # p(z | c, Q) uses the quasar prior as its redshift prior when one is
        # available, and states a flat prior otherwise.
        if qso_prior is not None:
            sigma_q_grid = qso_prior(z_grid, float(features.ref_mag[i]))
            with np.errstate(divide="ignore"):
                log_zprior = np.log(sigma_q_grid)
        else:
            sigma_q_grid = None
            log_zprior = np.zeros_like(z_grid)

        log_post = log_pq_grid[i] + log_zprior
        if not np.isfinite(log_post).any():
            out.append(
                _null_score(
                    cid[i], pid[i], z_primary[i], features, sysname, i,
                    "qso_prior_empty_at_this_magnitude", flags, manifest_id,
                    int(n_bands[i]),
                )
            )
            continue
        post = np.exp(log_post - np.nanmax(log_post[np.isfinite(log_post)]))
        post = np.where(np.isfinite(post), post, 0.0)
        norm = _trapz(post, z_grid)
        post = post / norm if norm > 0 else post
        p_zmatch = float(_trapz(post * w_match, z_grid))
        z_mode = float(z_grid[int(np.argmax(post))])

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
            p_vs_bkg = p_full = np.nan
            status = "no_prior_posterior_unavailable"
        else:
            # Work relative to a common scale so that the three intensities can
            # be added without overflowing; only the ratios are ever needed.
            # The scale must dominate *both* hypotheses: an object far off the
            # quasar locus has a tiny p(c|Q,z) everywhere, and rescaling by that
            # alone would overflow the background term.
            scale = max(float(log_pq_grid[i].max()), float(log_pb[i]))
            pq = np.exp(log_pq_grid[i] - scale)
            s_rel = float(_trapz(w_match * sigma_q_grid * pq, z_grid))
            f_rel = float(_trapz((1.0 - w_match) * sigma_q_grid * pq, z_grid))
            b_rel = float(sigma_b[i]) * float(np.exp(float(log_pb[i]) - scale))

            denom = s_rel + f_rel + b_rel
            p_vs_bkg = s_rel / (s_rel + b_rel) if (s_rel + b_rel) > 0 else np.nan
            p_full = s_rel / denom if denom > 0 else np.nan

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
                log_lambda_sameq=log_lam_s,
                log_lambda_fieldq=log_lam_f,
                log_lambda_bkg=log_lam_b,
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
        z_phot_mode=nan,
        log_lambda_sameq=nan,
        log_lambda_fieldq=nan,
        log_lambda_bkg=nan,
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
