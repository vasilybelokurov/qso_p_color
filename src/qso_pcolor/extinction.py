"""Galactic extinction correction for native survey fluxes.

Coefficients R are A_band / E(B-V)_SFD98 (magnitudes per unit SFD E(B-V)).
A corrected flux is ``flux * 10**(0.4 R E)`` and its variance is multiplied by
the square of that factor, so signal-to-noise and negative fluxes are kept.

Shared latent coordinates carry the coefficient of the system they represent
(``legacy:g`` is DECam g). Native coefficients follow from the linear
observation operator, ``R_native = H @ R_latent``: identity rows return their
own value, while the northern Legacy rows give the DECam coefficients passed
through the DESI North->South relation. This applies the relation to
uncorrected magnitudes and removes extinction in the DECam system.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .multisurvey_data import Photometry


def latent_coefficients(latent_labels, coefficients: dict) -> np.ndarray:
    """Coefficient for each latent coordinate; ``legacy:*`` uses the DECam system."""
    out = []
    for label in latent_labels:
        system, band = label.split(':')
        key = f'decals_dr9_south:{band}' if system == 'legacy' else label
        if key not in coefficients:
            raise ValueError(f'no extinction coefficient for latent coordinate {label}')
        out.append(float(coefficients[key]))
    return np.array(out)


def native_coefficients(layout: dict, coefficients: dict, kind: str = 'qso') -> np.ndarray:
    """R for every native label through the observation operator, per unit SFD E(B-V)."""
    h = np.asarray(layout['operators'][kind]['matrix'], float)
    other = [k for k in layout['operators'] if k != kind]
    for k in other:
        if not np.allclose(np.asarray(layout['operators'][k]['matrix']), h):
            raise ValueError('observation matrices differ between populations')
    return h @ latent_coefficients(layout['latent_labels'], coefficients)


def extinction_record(layout: dict, path: str | Path) -> dict:
    """Serialisable declaration stored in a model: map, native coefficients, provenance."""
    source = json.loads(Path(path).read_text())
    r = native_coefficients(layout, source['coefficients'])
    return dict(map=source['map'], labels=list(layout['native_labels']), coefficients=r.tolist(),
                source=str(path), source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                rule='flux * 10**(0.4 R E), variance * 10**(0.8 R E); R = H @ R_latent')


@lru_cache(maxsize=1)
def _sfd():
    from dustmaps.sfd import SFDQuery
    return SFDQuery()


def sfd_ebv(l_deg, b_deg) -> np.ndarray:
    """SFD98 E(B-V) (mag) at Galactic coordinates in degrees."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    l = np.atleast_1d(np.asarray(l_deg, float)); b = np.atleast_1d(np.asarray(b_deg, float))
    if l.shape != b.shape or not np.isfinite(l + b).all():
        raise ValueError('finite matching Galactic coordinates required')
    out = np.empty(len(l))
    for lo in range(0, len(l), 500_000):
        sl = slice(lo, lo + 500_000)
        out[sl] = _sfd()(SkyCoord(l=l[sl]*u.deg, b=b[sl]*u.deg, frame='galactic'))
    if not np.isfinite(out).all() or (out < 0).any():
        raise ValueError('invalid SFD reddening')
    return out


def correction_factors(ebv: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Multiplicative flux factors 10**(0.4 R E), shape (objects, bands)."""
    ebv = np.asarray(ebv, float); r = np.asarray(coefficients, float)
    if ebv.ndim != 1 or r.ndim != 1 or (ebv < 0).any() or not np.isfinite(r).all():
        raise ValueError('non-negative E(B-V) vector and finite coefficients required')
    return 10**(0.4*np.outer(ebv, r))


def deredden_arrays(flux, variance, ebv, coefficients):
    """Corrected flux and variance arrays; missing (NaN/inf) entries stay missing."""
    f = correction_factors(ebv, coefficients)
    return np.asarray(flux, float)*f, np.asarray(variance, float)*f**2


def deredden(photometry: Photometry, l_deg, b_deg, record: dict,
             flux_covariance: np.ndarray | None = None):
    """Correct photometry (and an optional full flux covariance) declared by a model record."""
    labels = list(record['labels']); r_all = np.asarray(record['coefficients'], float)
    r = np.array([r_all[labels.index(b)] for b in photometry.bands])
    ebv = sfd_ebv(np.broadcast_to(l_deg, (len(photometry.flux),)), np.broadcast_to(b_deg, (len(photometry.flux),)))
    f = correction_factors(ebv, r)
    corrected = Photometry(photometry.flux*f, photometry.variance*f**2, photometry.bands)
    if flux_covariance is None:
        return corrected, None, ebv
    cov = np.asarray(flux_covariance, float)
    return corrected, cov*f[:, :, None]*f[:, None, :], ebv
