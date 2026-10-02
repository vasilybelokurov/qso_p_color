"""Shared helpers for the unified method write-up figures.

Models are the promoted bundles: ``independent`` (``models/multisurvey_psf/current``,
QSO colours independent of magnitude) and ``dependent`` (``current_magdep``).
Data are the prepared, extinction-corrected arrays of the baseline run: native
luptitudes ``y`` (mag), their variances ``noise`` (mag^2), observed masks, roles,
redshifts and Galactic coordinates (deg). Only role 3 (test) rows are plotted as data.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.plotting import SERIES, save_figure, use_paper_style  # noqa: F401
from run_unified_pilot import arrays

ROOT = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')
POINTERS = {'independent': 'models/multisurvey_psf/current', 'dependent': 'models/multisurvey_psf/current_magdep'}
COLOURS = {'independent': SERIES['same_z'], 'dependent': SERIES['field_q'], 'background': SERIES['background']}
LABELS = {'independent': 'QSO model, magnitude-independent', 'dependent': 'QSO model, magnitude-dependent',
          'background': 'background model'}


@lru_cache(maxsize=2)
def model(name):
    from qso_pcolor.unified import UnifiedPSFModel
    return UnifiedPSFModel.load(POINTERS[name]).base.model


@lru_cache(maxsize=2)
def data(kind):
    return arrays(ROOT, kind)


def bands():
    return list(json.loads((ROOT/'layout.json').read_text())['native_labels'])


def idx(label):
    return bands().index(label)


def snr(kind, rows, labels):
    """Signal-to-noise of corrected fluxes, shape (rows, labels)."""
    d = data(kind); cols = [idx(b) for b in labels]
    f = np.asarray(d['flux_dered'][rows][:, cols]); v = np.asarray(d['variance_dered'][rows][:, cols])
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(np.isfinite(f) & (v > 0), f/np.sqrt(v), np.nan)


def colour_matrix(pairs):
    """Rows of A such that A @ y gives colours (band_a - band_b) in mag."""
    a = np.zeros((len(pairs), len(bands())))
    for i, (p, q) in enumerate(pairs):
        a[i, idx(p)] += 1; a[i, idx(q)] -= 1
    return a


def project(mix: GaussianMixture, a: np.ndarray, offset=None) -> GaussianMixture:
    """Linear projection of a native mixture: weights, A mu (+offset), A V A^T."""
    means = mix.means @ a.T + (0 if offset is None else offset)
    return GaussianMixture(mix.weights, means, a @ mix.covs @ a.T)


def condition_on(mix: GaussianMixture, band: str, value: float, variance: float = 0.) -> GaussianMixture:
    """Condition a native mixture on one band's luptitude (mag); returns the full-dimension mixture.

    Component weights become w_k N(value | mu_k, V_kk + variance); means and covariances
    follow the standard Gaussian conditioning, with the conditioned coordinate fixed.
    """
    j = idx(band); m, v, w = mix.means, mix.covs, mix.weights
    s = v[:, j, j] + variance
    lw = np.log(np.maximum(w, 1e-300)) - .5*np.log(2*np.pi*s) - .5*(value - m[:, j])**2/s
    w_new = np.exp(lw - lw.max()); w_new /= w_new.sum()
    gain = v[:, :, j]/s[:, None]
    means = m + gain*(value - m[:, j])[:, None]
    covs = v - gain[:, :, None]*v[:, j, :][:, None, :]
    covs = .5*(covs + covs.swapaxes(1, 2)) + np.eye(v.shape[1])*1e-12
    return GaussianMixture(w_new, means, covs)


def combine(mixtures, weights) -> GaussianMixture:
    """Weighted union of mixtures (weights need not be normalised)."""
    weights = np.asarray(weights, float)/np.sum(weights)
    return GaussianMixture(np.concatenate([w*m.weights for w, m in zip(weights, mixtures)]),
                           np.concatenate([m.means for m in mixtures]), np.concatenate([m.covs for m in mixtures]))


def density_2d(mix: GaussianMixture, xx, yy, noise=None):
    """Mixture density (per mag^2) on a grid; ``noise`` (2x2, mag^2) is added to every component."""
    pts = np.stack([xx.ravel(), yy.ravel()], 1); out = np.zeros(len(pts))
    for w, mu, c in zip(mix.weights, mix.means, mix.covs):
        c = c + (0 if noise is None else noise); inv = np.linalg.inv(c); d = pts - mu
        out += w*np.exp(-.5*np.einsum('ni,ij,nj->n', d, inv, d))/(2*np.pi*np.sqrt(np.linalg.det(c)))
    return out.reshape(xx.shape)


def quantiles_1d(mix: GaussianMixture, qs=(.16, .5, .84)):
    """Quantiles of a 1-D mixture by root finding on its CDF."""
    from scipy.optimize import brentq
    from scipy.stats import norm
    m, s = mix.means[:, 0], np.sqrt(mix.covs[:, 0, 0])
    cdf = lambda x: np.sum(mix.weights*norm.cdf((x - m)/s))
    lo, hi = (m - 8*s).min(), (m + 8*s).max()
    return [brentq(lambda x: cdf(x) - q, lo, hi) for q in qs]


def background_weights(name, l_deg, b_deg):
    """Mean spatial component weights of the background over a set of positions."""
    mod = model(name); w, _ = mod.spatial_background.evaluate(l_deg, b_deg)
    return GaussianMixture(w.mean(axis=0), mod.background.means, mod.background.covs)


def contour_levels(z, fractions=(.5, .9)):
    """Density levels enclosing the given probability fractions of a gridded density."""
    flat = np.sort(z.ravel())[::-1]; cum = np.cumsum(flat)/flat.sum()
    return sorted(flat[np.searchsorted(cum, f)] for f in fractions)
