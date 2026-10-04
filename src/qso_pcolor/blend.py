"""Blend mode: is the unresolved companion of a known quasar a second quasar at the same redshift, or a star?

A blended source is a spectroscopic quasar at known redshift ``z0`` plus an unresolved companion. Only the
combined flux is measured (any subset of the bundle's bands; best extended photometry, e.g. Legacy model
flux, SDSS cModel, PS1 Kron). Two hypotheses are compared:

* ``QQ``: the companion is a quasar at ``z0`` (same redshift);
* ``QS``: the companion is a star (a member of the background population).

For hypothesis ``H`` with companion population ``C`` (quasar at ``z0``, or background at the source's
sky position), and labelled components (1 = the known quasar, 2 = the companion),

.. math::

    L_H(y) = \\iint p(m_1\\mid Q, z_0)\\, p(m_2\\mid C)\\, p(y\\mid m_1, m_2, H)\\, dm_1\\, dm_2,

where :math:`m_i` are the components' reference-band luptitudes and the magnitude distributions are the
bundle's surface densities normalised over their tabulated range (:math:`\\Sigma_Q(z_0, m)` for quasars,
:math:`\\Sigma_B(m, \\ell, b)` for the background), both per unit reference luptitude.

The reference band is Legacy r (South where observed, else North), as for the faint limit of the single
object scorer. Fluxes add, so the integral is taken over the companion's share ``alpha`` of the
reference-band flux ``T`` (``F1 = (1-alpha) T``, ``F2 = alpha T``) with ``T`` fixed at the measured blend
flux (the reference noise is not integrated; the faint limit keeps it below 0.1 mag), and with the
Jacobian of ``(m1, m2) -> (u, alpha)``. At each ``alpha`` each component's native mixture is conditioned on
its own noise-free reference luptitude; for every pair of components the blend luptitude in each other
band is linearised: mean ``lup(g(mu1) + g(mu2))`` with ``g`` the inverse luptitude, covariance
``W1 C1 W1 + W2 C2 W2 + noise`` with ``W_i = diag(hypot(F_i, 2s) / hypot(F1 + F2, 2s))``, the exact
first-order derivative of the blend luptitude with respect to each component's luptitude. Components are
conditioned with their intrinsic covariance (the bundle's instrumental covariance removed), which is added
back once for the single blend measurement.

Base rules: any morphology; extinction corrected on the blend (both components lie behind the same dust);
both components must carry at least ``faint_snr`` (10) times the blend's Legacy r flux error, so the
blend itself needs Legacy r at S/N >= 2 * faint_snr. Likelihoods therefore describe companions above that
limit. No catch-all and no support cut are applied (two hypotheses only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .multisurvey_data import Photometry

A = 2.5/np.log(10.)
REFERENCE = ('decals_dr9_south:r', 'decals_dr9_north:r')


def luptitude(f, s):
    """Luptitude of flux ``f`` with softening ``s`` (native units)."""
    return 22.5 - A*(np.arcsinh(f/(2*s)) + np.log(s))


def flux_of(l, s):
    """Inverse luptitude."""
    return 2*s*np.sinh((22.5 - l)/A - np.log(s))


def condition_mixture(log_w, means, covs, anchor: int, value: float, keep: np.ndarray, prune: float = 30.):
    """Condition a Gaussian mixture on coordinate ``anchor`` = ``value`` (noise-free).

    Returns log weights (normalised), conditional means and covariances of coordinates ``keep``.
    Components whose log weight falls ``prune`` nats below the largest are dropped.
    """
    va = covs[:, anchor, anchor]; da = value - means[:, anchor]
    lw = log_w - .5*np.log(2*np.pi*va) - .5*da**2/va
    lw = lw - logsumexp(lw)
    use = lw > lw.max() - prune
    lw, va, da = lw[use], va[use], da[use]
    cross = covs[use][:, keep, anchor]                                   # (K, d)
    mu = means[use][:, keep] + cross*(da/va)[:, None]
    c = covs[use][:, keep][:, :, keep] - cross[:, :, None]*cross[:, None, :]/va[:, None, None]
    return lw, mu, .5*(c + c.swapaxes(1, 2))


def blend_colour_loglike(y, noise, s, comp1, comp2, extra=None):
    """log p(y | component conditionals) for a blend, summed over component pairs (linearised).

    ``y`` (d,) observed blend luptitudes of the non-reference bands, ``noise`` (d, d) their measurement
    covariance, ``s`` (d,) softenings; ``comp1``/``comp2`` = (log_w, mu, cov) from condition_mixture.
    ``comp1``/``comp2`` must be INTRINSIC (the instrumental covariance removed before conditioning on the
    reference); ``extra`` (d, d), the bundle's per-measurement instrumental covariance, is added once, since
    it belongs to the single blend measurement.
    """
    lw1, mu1, c1 = comp1; lw2, mu2, c2 = comp2
    if len(y) == 0:
        return 0.
    f1, f2 = flux_of(mu1, s), flux_of(mu2, s)                          # (K1, d), (K2, d)
    ftot = f1[:, None, :] + f2[None, :, :]                              # (K1, K2, d)
    h = np.hypot(ftot, 2*s)
    w1 = np.hypot(f1, 2*s)[:, None, :]/h; w2 = np.hypot(f2, 2*s)[None, :, :]/h
    mean = luptitude(ftot, s)
    cov = (w1[..., :, None]*c1[:, None]*w1[..., None, :] + w2[..., :, None]*c2[None]*w2[..., None, :] + noise)
    if extra is not None:
        cov = cov + extra
    k1, k2, d = mean.shape
    cov = cov.reshape(-1, d, d); r = (y - mean).reshape(-1, d)
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, r[..., None])[..., 0]
    logdet = 2*np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    ln = -.5*(z**2).sum(axis=1) - .5*logdet - .5*d*np.log(2*np.pi)
    lw = (lw1[:, None] + lw2[None, :]).reshape(-1)
    return float(logsumexp(lw + ln))


@dataclass
class BlendScore:
    """Result for one blended source. Log likelihoods are per unit of the observed luptitudes."""
    status: str
    reference_band: str = ''
    bands_used: tuple = ()
    log_like_qq: float = np.nan
    log_like_qs: float = np.nan
    log_bf_qq_qs: float = np.nan
    p_qq: float = np.nan
    prior_odds_qq: float = np.nan
    alpha_mean_qq: float = np.nan      # posterior mean share of the fainter quasar (QQ is symmetric)
    alpha_mean_qs: float = np.nan      # posterior mean share of the star (QS)
    alpha_range: tuple = ()
    notes: tuple = field(default_factory=tuple)


class BlendModel:
    """Blend-mode scorer built on a loaded :class:`UnifiedPSFModel` (any promoted bundle)."""

    def __init__(self, unified, *, faint_snr: float = 10., n_alpha: int = 48, mag_step: float = .01, extra_covariance=None):
        self.unified = unified; self.base = unified.base; self.model = unified.base.model
        d = len(self.model.transform.bands)
        self.extra = np.zeros((d, d)) if extra_covariance is None else np.asarray(extra_covariance, float)
        self.faint_snr, self.n_alpha, self.mag_step = float(faint_snr), int(n_alpha), float(mag_step)
        self.bands = tuple(self.model.transform.bands); self.soft = np.asarray(self.model.transform.softening, float)

    @classmethod
    def load(cls, path, **kwargs):
        """Load a bundle or pointer; the instrumental covariance is read from the bundle's latent layout."""
        import json
        from pathlib import Path
        from .unified import UnifiedPSFModel
        root = Path(path)
        if root.is_file():
            root = root.parent/json.loads(root.read_text())['bundle']
        layout = json.loads((root/'latent.json').read_text())['layout']
        extra = np.asarray(layout['operators']['qso']['covariance'], float)
        if not np.allclose(extra, np.asarray(layout['operators']['stars']['covariance'], float)):
            raise ValueError('quasar and background instrumental covariances differ; blend mode assumes one')
        return cls(UnifiedPSFModel.load(root), extra_covariance=extra, **kwargs)

    # -- population pieces -------------------------------------------------------------------------
    def _intrinsic(self, covs):
        """Remove the per-measurement instrumental covariance that native components carry (C = H V H^T + T)."""
        c = covs - self.extra
        return .5*(c + c.swapaxes(-1, -2)) + 1e-12*np.eye(c.shape[-1])

    def _qso_mixture(self, z0):
        q = self.model.qso; zc = np.asarray(q.z_centres)
        j = int(np.clip(np.searchsorted(zc, z0) - 1, 0, len(zc) - 2)); frac = float(np.clip((z0 - zc[j])/(zc[j+1] - zc[j]), 0, 1))
        parts = [(m, w) for m, w in ((q.mixtures[j], 1 - frac), (q.mixtures[j + 1], frac)) if w > 0]
        lw = np.concatenate([np.log(w) + np.log(np.maximum(m.weights, 1e-300)) for m, w in parts])
        return lw, np.concatenate([m.means for m, _ in parts]), self._intrinsic(np.concatenate([m.covs for m, _ in parts]))

    def _background_mixture(self, l, b):
        mix = self.model.background
        w = (self.model.spatial_background.evaluate([l], [b])[0][0] if self.model.spatial_background is not None
             else mix.weights)
        return np.log(np.maximum(w, 1e-300)), mix.means, self._intrinsic(mix.covs)

    def _log_mag_density(self, density, edges, upper=None):
        """Normalised log p(m) on a fine grid over ``edges`` (cut at ``upper`` if given) -> callable."""
        hi = edges[-1] if upper is None else min(edges[-1], upper)
        if hi <= edges[0]:
            return None
        grid = np.arange(edges[0], hi + 1e-9, self.mag_step)
        val = np.array([density(m) for m in grid], float)
        norm = np.trapezoid(val, grid)
        if not np.isfinite(norm) or norm <= 0:
            return None
        with np.errstate(divide='ignore'):
            lv = np.log(val/norm)
        return lambda m: np.interp(m, grid, lv, left=-np.inf, right=-np.inf)

    # -- scoring -----------------------------------------------------------------------------------
    def score(self, photometry: Photometry, *, ra_deg, dec_deg, z_qso, prior_odds_qq: float = 1.,
              dereddened: bool = False, l_deg=None, b_deg=None) -> list:
        """Score blends. ``photometry`` holds catalogue fluxes of the whole blend (native units).

        ``dereddened=True`` declares that fluxes are already extinction-corrected (then Galactic
        ``l_deg``, ``b_deg`` may be passed directly). ``prior_odds_qq`` = P(QQ)/P(QS) before the photometry.
        """
        from .unified import galactic_from_equatorial
        n = len(photometry.flux)
        if l_deg is None:
            l_deg, b_deg = galactic_from_equatorial(np.broadcast_to(np.asarray(ra_deg, float), (n,)),
                                                    np.broadcast_to(np.asarray(dec_deg, float), (n,)))
        l_deg = np.broadcast_to(np.asarray(l_deg, float), (n,)); b_deg = np.broadcast_to(np.asarray(b_deg, float), (n,))
        z = np.broadcast_to(np.asarray(z_qso, float), (n,))
        p = photometry.align(self.bands)
        if not dereddened and 'extinction' in self.model.meta:
            from .extinction import deredden
            p, _, _ = deredden(p, l_deg, b_deg, self.model.meta['extinction'])
        return [self._score_one(p.subset([i]), z[i], l_deg[i], b_deg[i], float(prior_odds_qq)) for i in range(n)]

    def _score_one(self, p, z0, l, b, prior_odds):
        obs = p.observed[0]; flux = np.where(obs, p.flux[0], 0.); var = np.where(obs, p.variance[0], np.inf)
        ref = next((self.bands.index(r) for r in REFERENCE if obs[self.bands.index(r)]), None)
        if ref is None:
            return BlendScore('no_reference_band')
        label = self.bands[ref]
        if not self.model.qso.in_support(np.array([z0]))[0]:
            return BlendScore('redshift_outside_support', label)
        T, sig = flux[ref], np.sqrt(var[ref])
        if not (np.isfinite(sig) and T > 0) or T/sig < 2*self.faint_snr:
            return BlendScore('too_faint_for_two_components', label)
        if label not in self.base.priors:
            return BlendScore('no_prior_for_reference_band', label)
        qprior, bdens = self.base.priors[label]
        # The known quasar's magnitude prior is the full Sigma_Q(z0, m) shape. The companion's prior is
        # conditioned on the companion being detectable (above the faint limit, F2 >= faint_snr * sigma):
        # the result is P(QQ | blend, detectable companion), and how often a detectable companion is a
        # quasar rather than a star belongs to the prior odds. Without this, the far larger share of
        # stars below the limit would penalise QS for companions the mode excludes by construction.
        m_lim = float(luptitude(self.faint_snr*sig, self.soft[ref]))
        lpq = self._log_mag_density(lambda m: qprior(np.array([z0]), m)[0], qprior.mag_edges)
        lpq2 = self._log_mag_density(lambda m: qprior(np.array([z0]), m)[0], qprior.mag_edges, upper=m_lim)
        lpb = self._log_mag_density(lambda m: bdens(np.array([m]), l, b)[0], bdens.mag_edges, upper=m_lim)
        if lpq is None or lpq2 is None or lpb is None:
            return BlendScore('empty_magnitude_prior', label)
        others = np.flatnonzero(obs & (np.arange(len(obs)) != ref))
        f = self.model.transform(Photometry(flux[None], np.where(obs, var, np.inf)[None], self.bands))
        y = f.x[0, others]; noise = f.cov[0][np.ix_(others, others)]
        s_o, s_r = self.soft[others], self.soft[ref]
        amin = self.faint_snr*sig/T
        t = np.linspace(np.log(amin/(1 - amin)), np.log((1 - amin)/amin), self.n_alpha)
        alpha = 1/(1 + np.exp(-t))
        q_mix = self._qso_mixture(z0); b_mix = self._background_mixture(l, b)
        out = {}
        for hyp, comp_mix, lp2 in (('qq', q_mix, lpq2), ('qs', b_mix, lpb)):
            lg = np.full(len(alpha), -np.inf)
            for i, a in enumerate(alpha):
                f1, f2 = (1 - a)*T, a*T
                m1, m2 = luptitude(f1, s_r), luptitude(f2, s_r)
                prior = lpq(m1) + lp2(m2)
                if not np.isfinite(prior):
                    continue
                logj = np.log(A*T*np.hypot(T, 2*s_r)) - np.log(np.hypot(f1, 2*s_r)) - np.log(np.hypot(f2, 2*s_r))
                c1 = condition_mixture(*q_mix, ref, m1, others); c2 = condition_mixture(*comp_mix, ref, m2, others)
                lg[i] = prior + logj + blend_colour_loglike(y, noise, s_o, c1, c2, self.extra[np.ix_(others, others)]) + np.log(a*(1 - a))
            if not np.isfinite(lg).any():
                out[hyp] = (-np.inf, np.nan); continue
            dt = t[1] - t[0]; wts = np.full(len(t), dt); wts[[0, -1]] = dt/2
            total = logsumexp(lg + np.log(wts))
            post = np.exp(lg + np.log(wts) - total)
            # QQ is symmetric in the two quasars, so report the minor component's share; QS: the star's share.
            share = np.minimum(alpha, 1 - alpha) if hyp == 'qq' else alpha
            out[hyp] = (float(total), float((post*share).sum()))
        lqq, lqs = out['qq'][0], out['qs'][0]
        if not (np.isfinite(lqq) or np.isfinite(lqs)):
            return BlendScore('empty_hypotheses', label)
        bf = lqq - lqs
        pqq = 1/(1 + np.exp(-(bf + np.log(prior_odds)))) if np.isfinite(bf) else float(np.isfinite(lqq))
        notes = () if len(others) else ('no_colour_information',)
        return BlendScore('ok', label, tuple(self.bands[j] for j in [ref, *others]), lqq, lqs, bf, float(pqq), prior_odds,
                          out['qq'][1], out['qs'][1], (float(alpha[0]), float(alpha[-1])), notes)
