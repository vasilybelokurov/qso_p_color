#!/usr/bin/env python
"""Phase C: is the scorer's Gaussian-in-luptitude measurement model calibrated at low S/N? (PLAN_score_all.md, T8)

The scorer turns each Euclid flux f (nanomaggies) with variance s^2 into a luptitude
    y = 22.5 - A [asinh(f / 2b) + ln b],  A = 2.5 / ln 10,
with the first-order variance  var_y = (A s / hypot(f, 2b))^2  (euclid_bundle.luptitudes, multisurvey.py:50), and
treats y_obs as Gaussian about the true luptitude. That is exact only at high S/N.

Benchmark. Labelled bright objects (DESI/SDSS QSOs and parent-background objects of the Euclid tables, S/N > 50 in
the band) are dimmed so that the band has a target S/N against the measured faint-end flux error of the deployment
selection (median error of select-cone sources at VIS 24-25 in that band: sky-limited, so independent of the
source). For true flux f_t = SN * sigma_band, draw f_obs = f_t + N(0, sigma_band^2) (Gaussian flux noise, the
catalogue's own error model), and compare two likelihoods for the true value:
  luptitude:  y_t under N(y_obs, var_y(f_obs))    (the scorer)
  flux:       f_t under N(f_obs, sigma_band^2)     (exact under the stated noise model; the reference)
Reported per band and S/N bin (1, 2, 3, 5, 7, 10, 15, 20): coverage of the central 68% and 95% intervals and the
KS distance of the PIT from uniform. Pass (T8, fixed before running): coverage within +-3 pp of nominal.
The test checks the measurement model only; it does not use the population models.

    python scripts/score_all/phase_c_lowsnr.py
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kstest, norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/euclid'))
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'
OUT = ROOT/'docs/score_all/phase_c_lowsnr.json'
BANDS = ('vis', 'y', 'j', 'h')
SNR = (1, 2, 3, 5, 7, 10, 15, 20)
A = 2.5/np.log(10)
SEED = 20261119


def faint_sigma(settings):
    """Median flux error (nanomaggies) per band of selected select-cone sources at 24 <= VIS < 25."""
    from cones import deployment
    from qso_pcolor.euclid_bundle import photometry
    errs = []
    for p in sorted(glob.glob(str(BASE/'cones/select_*.npz'))):
        rows, _ = deployment(p, settings)
        ph = photometry(dict(rows, matched=np.ones(len(rows['e_ra']), bool)))
        with np.errstate(invalid='ignore', divide='ignore'):
            vis = 22.5 - 2.5*np.log10(ph.flux[:, 0])
        sel = (vis >= 24) & (vis < 25)
        errs.append(np.sqrt(ph.variance[sel]))
    e = np.concatenate(errs)
    return np.array([np.nanmedian(np.where(np.isfinite(e[:, j]), e[:, j], np.nan)) for j in range(4)])


def bright_sources(settings):
    """True-flux templates: band fluxes of labelled objects with S/N > 50 (only the count matters per band)."""
    from qso_pcolor.euclid_bundle import photometry
    n = {}
    for name in ('qso', 'background'):
        d = dict(np.load(BASE/'tables'/f'{name}.npz', allow_pickle=False))
        f = d['euclid_flux']; v = d['euclid_variance']
        with np.errstate(invalid='ignore', divide='ignore'):
            sn = f/np.sqrt(v)
        n[name] = (sn > 50).sum(axis=0)
    return n


def coverage(z):
    """z: standardised residual (true - centre)/sd under the tested Gaussian."""
    z = z[np.isfinite(z)]
    return dict(n=int(len(z)), cov68=float(np.mean(np.abs(z) < norm.ppf(.84))), cov95=float(np.mean(np.abs(z) < norm.ppf(.975))),
                ks=float(kstest(norm.cdf(z), 'uniform').statistic), median_z=float(np.median(z)))


def main():
    settings = json.loads((ROOT/'docs/euclid/settings.json').read_text())
    b = np.asarray(settings['softening'], float)
    sig = faint_sigma(settings); counts = bright_sources(settings)
    rng = np.random.default_rng(SEED); res = {}
    for j, band in enumerate(BANDS):
        res[band] = {}
        n_obj = int(min(20000, counts['qso'][j] + counts['background'][j]))
        for sn in SNR:
            ft = np.full(n_obj, sn*sig[j]); fo = ft + rng.normal(0, sig[j], n_obj)
            yt = 22.5 - A*(np.arcsinh(ft/(2*b[j])) + np.log(b[j])); yo = 22.5 - A*(np.arcsinh(fo/(2*b[j])) + np.log(b[j]))
            sy = A*sig[j]/np.hypot(fo, 2*b[j])
            res[band][str(sn)] = dict(luptitude=coverage((yt - yo)/sy), flux=coverage((ft - fo)/sig[j]))
    passes = {band: {sn: bool(abs(r['luptitude']['cov68'] - .6827) <= .03 and abs(r['luptitude']['cov95'] - .9545) <= .03)
                     for sn, r in v.items()} for band, v in res.items()}
    out = dict(softening=b.tolist(), sigma_faint_nanomaggies=sig.tolist(), softening_over_sigma=(b/sig).tolist(),
               bright_counts={k: v.tolist() for k, v in counts.items()}, results=res, luptitude_passes_T8=passes,
               note='flux-space coverage is exact by construction under Gaussian flux noise; it is the reference')
    OUT.write_text(json.dumps(out, indent=1))
    print('sigma_faint', np.round(sig, 4), 'b/sigma', np.round(b/sig, 2))
    for band in BANDS:
        print(band, ' '.join(f"SN{sn}:{res[band][str(sn)]['luptitude']['cov68']:.3f}/{res[band][str(sn)]['luptitude']['cov95']:.3f}"
                             f"{'' if passes[band][str(sn)] else '*'}" for sn in SNR))
    plot(res)


def plot(res):
    sys.path.insert(0, str(ROOT/'scripts/method_unified'))
    import matplotlib.pyplot as plt
    from common import save_figure, use_paper_style
    use_paper_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for band, c in zip(BANDS, ('k', '#1b6ca8', '#e08a2c', '#b5342c')):
        for k, (key, nom) in enumerate((('cov68', .6827), ('cov95', .9545))):
            ax[k].plot(SNR, [res[band][str(s)]['luptitude'][key] for s in SNR], 'o-', color=c, label=f'{band.upper()} luptitude', ms=4)
            ax[k].plot(SNR, [res[band][str(s)]['flux'][key] for s in SNR], ':', color=c, lw=1)
    for k, nom in enumerate((.6827, .9545)):
        ax[k].axhspan(nom - .03, nom + .03, color='#ddd', zorder=0); ax[k].axhline(nom, color='#888', lw=.8)
        ax[k].set(xscale='log', xlabel='S/N of the true flux', ylabel=f'coverage of the {68 if k == 0 else 95}% interval')
    ax[0].legend(frameon=False, fontsize=7); ax[1].set_title('dotted: flux-space likelihood (reference)', fontsize=8)
    fig.tight_layout(); print(save_figure(fig, 'score_all/phase_c_lowsnr'))


if __name__ == '__main__':
    main()
