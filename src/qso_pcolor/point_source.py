"""Point-source test for resolved mode outside the Legacy Surveys footprint.

Resolved mode scores only point sources: inside Legacy, the caller passes the Legacy Tractor ``type``
(``PSF``). A source with no Legacy counterpart needs another test. This module returns a Tractor-like label,
``'PSF'`` (point), ``'EXT'`` (extended) or ``'unknown'`` (no test possible, not scored), from

1. PS1-PSC (Tachibana & Miller 2018, PASP 130, 128001; arXiv:1902.01935), where the source has a PS1 detection:
   the random-forest point-source score ``ps_score`` (WSDB ``panstarrs_dr1.hlsp``); point source if
   ``ps_score >= 0.83``, the threshold that maximises their figure of merit (their Table 3: 0.829).
2. otherwise Gaia DR3, when G, BP and RP are measured: the corrected BP/RP flux excess factor
   C* = C - f(G_BP - G_RP) (Riello et al. 2021, A&A 649, A3, Table 2), compared with its 1-sigma scatter for
   well-behaved isolated point sources, sigma_C*(G) = c0 + c1 G^m (their Eq. 18). Extended sources have more
   flux in the BP and RP windows than in G, so C* > 0. Point source if |C*| < ``n_sigma`` sigma_C*(G).
3. otherwise ``'unknown'``.

PS1-PSC comes first because it is the better test against Legacy ``type`` (docs/point_source/validation.json):
for Legacy-PSF field sources 98-99% complete at r < 20 and 80% at 21-21.5, with < 1% of extended sources called
point; Gaia C* calls only 60% (r < 17) to 83% (17-18) of Legacy-PSF quasars point (variability, hosts) and
lets 10-15% of extended sources through at r = 20-21. Gaia is the test only where PS1 is absent (Dec < -30).

Riello et al. note that a C* filter also removes variable sources; quasars are variable, so the test's
completeness for quasars is measured, not assumed (scripts/point_source/validate_point_source.py).
"""
from __future__ import annotations

import numpy as np

# Riello et al. 2021, Table 2: f(x) = sum a_i x^i, x = G_BP - G_RP
_CSTAR_POLY = (((-np.inf, 0.5), (1.154360, 0.033772, 0.032277)),
               ((0.5, 4.0), (1.162004, 0.011464, 0.049255, -0.005879)),
               ((4.0, np.inf), (1.057572, 0.140537)))
# Riello et al. 2021, Eq. 18
_SIGMA_C0, _SIGMA_C1, _SIGMA_M = 0.0059898, 8.817481e-12, 7.618399
PSC_THRESHOLD = 0.83
GAIA_N_SIGMA = 3.


def cstar(excess_factor, bp_rp):
    """Corrected BP/RP flux excess factor C* (Riello et al. 2021, Eq. 6 and Table 2)."""
    c, x = np.asarray(excess_factor, float), np.asarray(bp_rp, float)
    f = np.full(np.broadcast(c, x).shape, np.nan)
    for (lo, hi), a in _CSTAR_POLY:
        m = (x >= lo) & (x < hi)
        f = np.where(m, sum(ai * x**i for i, ai in enumerate(a)), f)
    return c - f


def sigma_cstar(g):
    """1-sigma scatter of C* for well-behaved isolated point sources (Riello et al. 2021, Eq. 18)."""
    return _SIGMA_C0 + _SIGMA_C1 * np.asarray(g, float)**_SIGMA_M


def point_source_morphology(*, gaia_g=None, gaia_bp_rp=None, gaia_excess_factor=None, ps_score=None,
                            n_sigma: float = GAIA_N_SIGMA, psc_threshold: float = PSC_THRESHOLD):
    """Tractor-like morphology label ('PSF', 'EXT', 'unknown') and the test used ('ps1_psc', 'gaia', '').

    Inputs are arrays (missing values NaN); any of them may be None. PS1-PSC is used where ``ps_score`` is
    finite; Gaia where it is not and G, BP-RP and the excess factor are all finite.
    """
    arrays = [a for a in (gaia_g, gaia_bp_rp, gaia_excess_factor, ps_score) if a is not None]
    if not arrays:
        raise ValueError('give Gaia DR3 (G, BP-RP, excess factor) and/or PS1-PSC ps_score')
    n = len(np.atleast_1d(arrays[0]))
    nan = np.full(n, np.nan)
    g = nan if gaia_g is None else np.asarray(gaia_g, float)
    x = nan if gaia_bp_rp is None else np.asarray(gaia_bp_rp, float)
    c = nan if gaia_excess_factor is None else np.asarray(gaia_excess_factor, float)
    ps = nan if ps_score is None else np.asarray(ps_score, float)
    label = np.full(n, 'unknown', dtype='<U7'); used = np.full(n, '', dtype='<U7')
    psc = np.isfinite(ps)
    label[psc] = np.where(ps[psc] >= psc_threshold, 'PSF', 'EXT'); used[psc] = 'ps1_psc'
    gaia = ~psc & np.isfinite(g) & np.isfinite(x) & np.isfinite(c)
    with np.errstate(invalid='ignore'):
        point = np.abs(cstar(c, x)) < n_sigma * sigma_cstar(g)
    label[gaia] = np.where(point[gaia], 'PSF', 'EXT'); used[gaia] = 'gaia'
    return label, used
