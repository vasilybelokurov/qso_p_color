"""Scoring sources without Legacy r by integrating over the unobserved Legacy r.

The promoted bundles define every population prior in Legacy r: the quasar abundance Sigma_Q(z, r) (eBOSS QLF),
the background abundance Sigma_B(r, l, b), and the magnitude bins of the background colour model. A source with
no usable Legacy r (outside the DR9 footprint, or masked) is therefore flagged and not scored. Here its Legacy r
is treated as an unobserved latent quantity m (noise-free Legacy South r luptitude, the canonical latent
``legacy:r``) and integrated out:

    Lambda_H = \\int lambda_H(m) dm,        lambda_H(m) = Sigma_H(m) p_H(observed bands | m),

for each scorer intensity H in (same-z quasar, other-z quasar, background, unmodelled). lambda_H(m) is exactly
what the scorer computes for the same object with a noise-free Legacy South r measurement equal to m, so the
integral reuses the scorer unchanged: the object is replicated on a grid of synthetic noise-free Legacy r values
and the returned intensities are integrated with the trapezoid rule. The joint colour models were fitted to
objects observed in Legacy and the other surveys at once, so p_H(observed | m) carries the learned relation
between Legacy r and the observed bands (e.g. PS1 r - Legacy r); the integral is narrow when a fallback band is
measured well.

Eligibility mirrors the faint limit: the source must have a fallback band (PS1 r, else SkyMapper r) at
S/N >= ``min_snr`` (10, as for Legacy r). The calibrated outside-both-models test and the QSO-support cut are
evaluated at the posterior-mode grid value of m. Prior-independent evidences in the returned record
(loglike_*, p_zmatch_given_qso, z_phot_mode) are also those at the mode; the intensities and posteriors
(log_lambda_*, p_sameq, p_sameq_vs_bkg, p_outlier, log_r_per_unit_z) are the integrated ones, and ``ref_mag`` is
the posterior mean of Legacy r.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import logsumexp

from .multisurvey_data import Photometry

LEGACY_R = ('decals_dr9_south:r', 'decals_dr9_north:r')
LATENT_BAND = 'decals_dr9_south:r'           # native band that is the latent legacy:r with an identity operator
FALLBACK_BANDS = ('ps1:r', 'skymapper:r')
A = 2.5/np.log(10.)
SYNTHETIC_REL_ERROR = 1e-6                   # noise-free synthetic Legacy r (relative flux error)


def luptitude(f, s):
    return 22.5 - A*(np.arcsinh(f/(2*s)) + np.log(s))


def flux_of(l, s):
    return 2*s*np.sinh((22.5 - l)/A - np.log(s))


def fallback_status(photometry: Photometry, bands, min_snr: float):
    """(needs_fallback, eligible, band index or -1, S/N) for rows without usable Legacy r."""
    p = photometry.align(bands)
    legacy = p.observed[:, [bands.index(b) for b in LEGACY_R]].any(axis=1)
    order = [bands.index(b) for b in FALLBACK_BANDS]
    avail = p.observed[:, order]
    has = avail.any(axis=1)
    band = np.where(has, np.array(order)[np.argmax(avail, axis=1)], -1)
    snr = np.full(len(has), np.nan)
    rows = np.flatnonzero(has)
    with np.errstate(invalid='ignore', divide='ignore'):
        snr[rows] = p.flux[rows, band[rows]]/np.sqrt(p.variance[rows, band[rows]])
    need = ~legacy
    return need, need & has & (snr >= min_snr), band, snr


def legacy_r_grid(centre, sigma_mag, n: int, halfwidth: float = 1.2, n_sigma: float = 5.):
    """Uniform grid of Legacy r luptitudes around ``centre``.

    The half-width covers the models' scatter of the fallback band about Legacy r (QSOs: ~0.2 mag in PS1 r,
    mostly variability between epochs; stars ~0.1 mag) plus ``n_sigma`` times the measurement error.
    """
    w = halfwidth + n_sigma*sigma_mag
    return np.linspace(centre - w, centre + w, n)


def _integrate(log_lam, h):
    """Trapezoid integral of exp(log_lam) over a uniform grid with step h; log result."""
    w = np.full(log_lam.shape[-1], np.log(h)); w[[0, -1]] -= np.log(2.)
    with np.errstate(invalid='ignore'):
        return logsumexp(log_lam + w, axis=-1)


def _posteriors(ls, lf, lb, lo, dz_eff):
    """Posteriors from (integrated) log intensities, as in qso_pcolor.score."""
    with np.errstate(divide='ignore', invalid='ignore'):
        lo = -np.inf if not np.isfinite(lo) else lo
        tot = logsumexp([ls, lf, lb, lo]); nonq = logsumexp([ls, lb, lo])
        return dict(p_sameq=float(np.exp(ls - tot)), p_sameq_vs_bkg=float(np.exp(ls - nonq)),
                    p_outlier=float(np.exp(lo - tot)) if np.isfinite(lo) else np.nan,
                    log_r_per_unit_z=float(ls - np.log(dz_eff) - tot) if dz_eff > 0 else np.nan)


def score_with_fallback(model, photometry: Photometry, scores: list, decision: dict, *, ra_deg, dec_deg,
                        z_primary, morphology, min_snr: float = 10., n_grid: int = 41, flux_covariance=None,
                        **kwargs):
    """Rescore rows flagged ``no_faint_limit_band`` by integrating over Legacy r (see module docstring).

    ``model`` is a :class:`qso_pcolor.unified.UnifiedPSFModel`; ``scores``/``decision`` its normal output for the
    same rows, updated in place and returned. Rows without an eligible fallback band get reason
    ``no_fallback_band`` or ``fallback_too_faint``.
    """
    from .unified import galactic_from_equatorial
    bands = tuple(model.base.model.transform.bands)
    p = photometry.align(bands)
    n = len(p.flux)
    need, ok, band, snr = fallback_status(p, bands, min_snr)
    target = np.flatnonzero(need & (decision['reason'] == 'no_faint_limit_band'))
    if not len(target):
        return scores, decision
    if flux_covariance is not None:
        raise NotImplementedError('the Legacy r fallback takes diagonal photometry only')
    has_band = band >= 0
    decision['reason'][target[~has_band[target]]] = 'no_fallback_band'
    decision['reason'][target[has_band[target] & ~ok[target]]] = 'fallback_too_faint'
    rows = target[ok[target]]
    if not len(rows):
        return scores, decision
    ra = np.broadcast_to(np.asarray(ra_deg, float), (n,)); dec = np.broadcast_to(np.asarray(dec_deg, float), (n,))
    z = np.broadcast_to(np.asarray(z_primary, float), (n,)); morph = np.broadcast_to(np.asarray(morphology), (n,))
    l, b = galactic_from_equatorial(ra, dec)
    soft = np.asarray(model.base.model.transform.softening, float); ref = bands.index(LATENT_BAND)
    # the scorer dereddens its input: synthetic Legacy r enters as an observed (reddened) flux
    corrected, factor = p, np.ones(n)
    if 'extinction' in model.base.model.meta:
        from .extinction import deredden
        corrected, _, _ = deredden(p, l, b, model.base.model.meta['extinction'])
        unit = Photometry(np.ones((n, len(bands))), np.ones((n, len(bands))), bands)
        factor = deredden(unit, l, b, model.base.model.meta['extinction'])[0].flux[:, ref]
    fc = corrected.flux[rows, band[rows]]; sc = soft[band[rows]]
    centre = luptitude(fc, sc); sig = A*np.sqrt(corrected.variance[rows, band[rows]])/np.hypot(fc, 2*sc)
    grids = np.array([legacy_r_grid(c, s, n_grid) for c, s in zip(centre, sig)])          # (R, K)
    ff = flux_of(grids, soft[ref])/factor[rows, None]                                      # observed flux
    F = np.repeat(p.flux[rows], n_grid, axis=0); V = np.repeat(p.variance[rows], n_grid, axis=0)
    F[:, ref] = ff.ravel(); V[:, ref] = (SYNTHETIC_REL_ERROR*ff.ravel())**2
    rep = lambda x: np.repeat(np.asarray(x)[rows], n_grid)
    grid_kw = {k: (rep(v) if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in kwargs.items()}
    grid_kw.pop('ood_calibration', None); grid_kw.pop('apply_support', None)
    gs, _ = model.base.score(Photometry(F, V, bands), morphology=rep(morph), z_primary=rep(z), l_deg=rep(l),
                             b_deg=rep(b), ood_calibration=None, **grid_kw)
    get = lambda k: np.array([np.nan if s is None else getattr(s, k) for s in gs], float).reshape(len(rows), n_grid)
    lam = {k: get('log_lambda_'+k) for k in ('sameq', 'fieldq', 'bkg', 'out')}
    lam['out'] = np.where(np.isfinite(lam['out']), lam['out'], -np.inf)
    h = grids[:, 1] - grids[:, 0]
    with np.errstate(invalid='ignore'):
        total = logsumexp(np.stack([lam[k] for k in lam]), axis=0)                         # (R, K)
    total = np.where(np.isfinite(total), total, -np.inf)
    usable = np.isfinite(total).any(axis=1)                  # no finite intensity: keep the mode-row result
    post = np.zeros_like(total)
    post[usable] = np.exp(total[usable] - logsumexp(total[usable], axis=1, keepdims=True))
    mode = np.where(usable, np.argmax(total, axis=1), n_grid//2)
    # the outside-both test and the support cut, at the posterior mode (a full score of one synthetic row)
    Fm = p.flux[rows].copy(); Vm = p.variance[rows].copy()
    fm = ff[np.arange(len(rows)), mode]; Fm[:, ref] = fm; Vm[:, ref] = (SYNTHETIC_REL_ERROR*fm)**2
    sub = {k: (np.asarray(v)[rows] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in kwargs.items()}
    ms, md = model.score(Photometry(Fm, Vm, bands), ra_deg=ra[rows], dec_deg=dec[rows], z_primary=z[rows],
                         morphology=morph[rows], legacy_r_fallback=False, **sub)
    for j, i in enumerate(rows):
        rec = ms[j]
        if rec is None:
            decision['reason'][i] = md['reason'][j]; continue
        if not usable[j]:
            scores[i] = rec; decision['reason'][i] = md['reason'][j] or rec.status; continue
        integ = {k: float(_integrate(lam[k][j], h[j])) for k in lam}
        edge = float(post[j, 0] + post[j, -1])
        flags = tuple(rec.quality_flags) + (f'legacy_r_marginalised:{bands[band[i]]}',) + \
            (('legacy_r_grid_truncated',) if edge > 1e-3 else ())
        new = replace(rec, ref_mag=float((post[j]*grids[j]).sum()), quality_flags=flags,
                      log_lambda_sameq=integ['sameq'], log_lambda_fieldq=integ['fieldq'],
                      log_lambda_bkg=integ['bkg'], log_lambda_out=integ['out'] if np.isfinite(integ['out']) else np.nan)
        if rec.status == 'ok' and md['eligible'][j]:
            new = replace(new, **_posteriors(integ['sameq'], integ['fieldq'], integ['bkg'], integ['out'], rec.dz_match_eff))
        scores[i] = new
        decision['eligible'][i] = bool(md['eligible'][j]) and np.isfinite(new.log_r_per_unit_z)
        decision['accepted'][i] = bool(md['accepted'][j])
        decision['reason'][i] = '' if decision['eligible'][i] else md['reason'][j]
        for key in ('percentile', 'log_density'):
            decision['qso_support'][key][i] = md['qso_support'][key][j]
        decision['qso_support']['status'][i] = md['qso_support']['status'][j]
        decision['support_rejected'][i] = md['support_rejected'][j]
    decision.setdefault('fallback_reference', np.full(n, '', dtype='<U16'))
    decision['fallback_reference'][rows] = np.array([bands[k] for k in band[rows]])
    return scores, decision
