"""Band extinction coefficients R = A_band / E(B-V)_SFD from filter curves (Schlafly & Finkbeiner 2011 recipe).

The Fitzpatrick (1999) R_V-dependent curve, optical/infrared branch (lambda > 2700 A), is a natural cubic spline in
inverse wavelength through the anchor points of Fitzpatrick (1999, PASP 111, 63; https://doi.org/10.1086/316293) as
implemented in the IDL routine ``fm_unred`` (W. Landsman). A band's A_b is the change in photon-counting synthetic
magnitude of a source spectrum (default: 7000 K blackbody, as SF11) reddened by a small E(B-V); SF11 then scale the
F99 E(B-V) to the SFD98 one (their N = 0.78 times a normalisation; here a single factor fitted to their Table 6).
Coefficients of the existing bands (configs/extinction_coefficients.json) were made by the same recipe with
scale 0.8828; :func:`fit_scale` reproduces that fit from filter curves.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import CubicSpline

H_C_OVER_K = 1.4387769e8          # h c / k in Angstrom Kelvin


def f99_a_over_ebv(wave_aa, r_v: float = 3.1) -> np.ndarray:
    """A(lambda)/E(B-V) of Fitzpatrick (1999), optical/IR spline; valid for 2700 A < lambda < 10 micron."""
    wave = np.asarray(wave_aa, float)
    if (wave < 2700).any():
        raise ValueError('only the optical/IR branch (lambda >= 2700 A) is implemented')
    x = 1e4/wave
    xk = np.array([0., 1e4/26500, 1e4/12200, 1e4/6000, 1e4/5470, 1e4/4670, 1e4/4110, 1e4/2700, 1e4/2600])
    ir = np.array([0., 0.26469, 0.82925])*r_v/3.1
    op = [np.polyval(np.array(c)[::-1], r_v) for c in (
        [-4.22809e-01, 1.00270, 2.13572e-04], [-5.13540e-02, 1.00216, -7.35778e-05],
        [7.00127e-01, 1.00184, -3.32598e-05], [1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05])]
    # UV anchors at 2700 and 2600 A from the FM90 parametrisation with F99's R_V-dependent c1, c2
    c2 = -0.824 + 4.717/r_v; c1 = 2.030 - 3.007*c2
    x0, gamma, c3, c4 = 4.596, 0.99, 3.23, 0.41
    def uv(xx):
        d = xx**2/((xx**2 - x0**2)**2 + (xx*gamma)**2)
        return c1 + c2*xx + c3*d + r_v       # F > 5.9 term absent below x = 5.9
    yk = np.concatenate([ir, op, [uv(xk[-2]), uv(xk[-1])]])
    return CubicSpline(xk, yk, bc_type='natural')(x)


def read_svo(path) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength (A) and transmission from an SVO Filter Profile Service VOTable."""
    root = ET.parse(path).getroot()
    rows = [[float(td.text) for td in tr] for tr in root.iter() if tr.tag.endswith('TR')
            for tr in [tr] if all(td.tag.endswith('TD') for td in tr)]
    a = np.array(rows)
    order = np.argsort(a[:, 0])
    return a[order, 0], np.clip(a[order, 1], 0, None)


def blackbody(wave_aa, t_k: float = 7000.) -> np.ndarray:
    """B_lambda shape (arbitrary units)."""
    w = np.asarray(wave_aa, float)
    return w**-5/np.expm1(H_C_OVER_K/(w*t_k))


def band_a(wave, trans, sed, ebv: float = 0.01, r_v: float = 3.1) -> float:
    """A_band / E(B-V)_F99 for a photon-counting band (small-E linear regime)."""
    w = np.asarray(wave, float); s = np.asarray(sed, float)*np.asarray(trans, float)*w
    att = 10**(-0.4*ebv*f99_a_over_ebv(w, r_v))
    return float(-2.5*np.log10(np.trapezoid(s*att, w)/np.trapezoid(s, w))/ebv)


def fit_scale(curves: dict, targets: dict, sed=blackbody) -> tuple[float, dict]:
    """Least-squares factor N with R_b = N A_b/E_F99 matching published R (SF11 Table 6); relative residuals."""
    a = np.array([band_a(*curves[k], sed(curves[k][0])) for k in targets])
    t = np.array([targets[k] for k in targets])
    n = float(np.sum(a*t/t**2)/np.sum(a**2/t**2))          # minimise sum((N a - t)/t)^2
    return n, {k: float(n*ai/ti - 1) for k, ai, ti in zip(targets, a, t)}
