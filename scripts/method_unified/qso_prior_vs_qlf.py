#!/usr/bin/env python
"""Compare the current quasar abundance prior Sigma_Q(z, r) with the eBOSS quasar luminosity function.

Current prior: the promoted bundle's GridQSOPrior for Legacy r (South), spectroscopic DESI DR1 + DR16Q counts
with completeness 1, on extinction-corrected magnitudes (deg^-2 mag^-1 per unit z).
Reference: Palanque-Delabrouille et al. 2016, A&A 587, A41 (arXiv:1509.05607), Table 7: predicted differential
quasar counts in r for 10,000 deg^2, bins of 0.5 mag and Delta z = 1 (centres 0.5 ... 5.5), for the best-fit
PLE and PLE+LEDE models (fitted on 0.68 < z < 4, g_dered < 22.5; brighter/fainter and z < 0.68 are
extrapolations). Their r is SDSS r; Legacy r differs by a few hundredths of a magnitude for quasars (ignored).
Both are integrated over the same redshift bins; our prior covers 0.15 < z < 4.35.

Writes plots/qso_prior/Q1_qso_prior_vs_qlf.png and docs/qso_prior/qso_prior_vs_qlf.json.
Usage: python scripts/method_unified/qso_prior_vs_qlf.py
"""
import json
from pathlib import Path

import numpy as np

R = np.arange(15.75, 23.76, .5)
ZC = np.array([.5, 1.5, 2.5, 3.5, 4.5])
# Table 7, PLE+LEDE model (columns z = 0.5 ... 4.5; z = 5.5 omitted, < 0.1% of the counts)
LEDE = np.array([
    [74, 12, 2, 0, 0], [173, 46, 8, 1, 0], [425, 173, 29, 3, 0], [1063, 645, 109, 11, 1], [2637, 2318, 407, 37, 3],
    [6171, 7534, 1476, 119, 10], [12760, 20293, 4934, 380, 27], [22256, 42853, 14012, 1146, 71],
    [33079, 72019, 31432, 3068, 182], [44377, 102993, 55189, 6846, 432], [56787, 135480, 81153, 12487, 915],
    [71385, 172381, 108661, 19287, 1689], [89178, 216865, 139912, 26781, 2731], [111154, 271898, 177682, 35182, 3976],
    [138390, 340627, 224720, 45094, 5393], [172133, 426753, 283998, 57240, 7015], [213829, 534834, 359048, 72418, 8921]], float)
# Table 7, PLE model
PLE = np.array([
    [54, 5, 14, 0, 0], [127, 23, 41, 3, 0], [323, 104, 108, 9, 0], [860, 469, 290, 25, 1], [2316, 2009, 793, 64, 2],
    [5854, 7439, 2204, 167, 5], [12619, 21145, 5994, 432, 12], [22090, 44412, 14675, 1100, 31],
    [32538, 72781, 29834, 2700, 80], [43787, 102680, 50203, 6208, 201], [57009, 136093, 74188, 12952, 502],
    [73477, 176857, 101326, 24039, 1217], [94414, 228620, 132051, 39635, 2811], [121186, 295216, 167766, 58919, 6038],
    [155438, 381240, 210616, 80786, 11825], [199196, 492527, 263236, 104700, 20944], [254945, 636610, 328743, 131042, 33694]], float)
TO_DENSITY = 1/(10000*.5)          # counts per 10,000 deg^2 per 0.5 mag -> deg^-2 mag^-1 (per Delta z = 1 bin)


def current_prior(pointer='models/multisurvey_psf/current'):
    from qso_pcolor.unified import UnifiedPSFModel
    qp, _ = UnifiedPSFModel.load(pointer).base.priors['decals_dr9_south:r']
    zf = np.arange(.151, 4.35, .01)
    out = np.zeros((len(R), len(ZC)))
    for i, r in enumerate(R):
        mg = np.linspace(r - .25, r + .25, 11)                         # average over the 0.5-mag bin
        dens = np.array([[qp(np.array([z]), m)[0] for z in zf] for m in mg]).mean(axis=0)
        for j, zc in enumerate(ZC):
            sel = (zf >= zc - .5) & (zf < zc + .5)
            out[i, j] = np.trapezoid(dens[sel], zf[sel]) if sel.sum() > 1 else 0.
    return out, qp


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ours, qp = current_prior(); led, ple = LEDE*TO_DENSITY, PLE*TO_DENSITY
    Path('plots/qso_prior').mkdir(parents=True, exist_ok=True); Path('docs/qso_prior').mkdir(parents=True, exist_ok=True)
    tot_o, tot_l, tot_p = ours.sum(1), led.sum(1), ple.sum(1)
    k = int(np.flatnonzero(R == 18.25)[0]); eu = tot_l[k]*10**(.6*(R - R[k]))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5)); ax = ax.ravel()
    x = ax[0]
    x.semilogy(R, tot_o, 'o-', color='#e8663d', label='current prior (spectroscopic counts, completeness 1)')
    x.fill_between(R, np.minimum(tot_l, tot_p), np.maximum(tot_l, tot_p), color='#2a78d6', alpha=.2, lw=0)
    x.semilogy(R, tot_l, 's-', color='#2a78d6', label='eBOSS QLF (PLE+LEDE; band: PLE)')
    x.semilogy(R, eu, ':', color='0.4', label='$10^{0.6m}$, normalised at r = 18.25')
    x.set(xlabel='Legacy r (dereddened)', ylabel='quasars deg$^{-2}$ mag$^{-1}$', title='all redshifts (our prior: 0.15 < z < 4.35)', ylim=(.05, 3e3))
    x.legend(frameon=False, fontsize=7)
    for j, (x, zc) in enumerate(zip(ax[1:5], ZC[:4])):
        x.semilogy(R, ours[:, j], 'o-', color='#e8663d', label='current prior')
        x.fill_between(R, np.minimum(led[:, j], ple[:, j]), np.maximum(led[:, j], ple[:, j]), color='#2a78d6', alpha=.2, lw=0)
        x.semilogy(R, led[:, j], 's-', color='#2a78d6', label='eBOSS QLF')
        lo = max(zc - .5, .15)
        x.set(xlabel='Legacy r', ylabel='deg$^{-2}$ mag$^{-1}$', title=f'{lo:.2f} < z < {zc + .5:.1f}' + (' (QLF extrapolated below 0.68)' if zc == .5 else ''),
              ylim=(.01, 2e3)); x.legend(frameon=False, fontsize=7)
    x = ax[5]
    for j, (zc, c) in enumerate(zip(ZC[:4], ('#7c3aed', '#1baa7d', '#e8a33d', '#c2410c'))):
        with np.errstate(divide='ignore', invalid='ignore'):
            x.plot(R, ours[:, j]/led[:, j], 'o-', color=c, ms=3, label=f'z {max(zc-.5, .15):.2f}-{zc+.5:.1f}')
    with np.errstate(divide='ignore', invalid='ignore'):
        x.plot(R, tot_o/tot_l, 'k-', lw=2, label='all z')
    x.axhline(1, ls=':', color='0.4'); x.set(xlabel='Legacy r', ylabel='current prior / QLF', yscale='log', ylim=(.01, 10), title='ratio (completeness of the current prior)')
    x.legend(frameon=False, fontsize=7)
    fig.suptitle('Q1. Quasar abundance: current prior against the eBOSS luminosity function (Palanque-Delabrouille et al. 2016, Table 7)', fontsize=10)
    fig.tight_layout(); fig.savefig('plots/qso_prior/Q1_qso_prior_vs_qlf.png', dpi=140); plt.close(fig)
    # redshift distributions at fixed magnitude
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    for x, r in zip(ax, (19.25, 20.75, 21.75, 22.75)):
        i = int(np.flatnonzero(R == r)[0])
        x.bar(ZC[:4] - .15, ours[i, :4]/ours[i].sum(), .3, color='#e8663d', label='current prior')
        x.bar(ZC[:4] + .15, led[i, :4]/led[i].sum(), .3, color='#2a78d6', label='eBOSS QLF')
        x.set(xlabel='redshift bin centre', ylabel='fraction', title=f'r = {r}: {tot_o[i]:.0f} vs {tot_l[i]:.0f} deg$^{{-2}}$ mag$^{{-1}}$')
        x.legend(frameon=False, fontsize=7)
    fig.suptitle('Q2. Redshift distribution of quasars at fixed Legacy r: current prior against the QLF', fontsize=10)
    fig.tight_layout(); fig.savefig('plots/qso_prior/Q2_redshift_distribution.png', dpi=140); plt.close(fig)
    # Q3: one panel, the three counts
    fig, x = plt.subplots(figsize=(6.4, 4.8)); meas = R <= 22.3          # eBOSS fit range g_dered < 22.5 ~ r < 22.3
    x.semilogy(R, tot_o, 'o-', color='#e8663d', label='our prior (both models): spectroscopic counts')
    x.fill_between(R, np.minimum(tot_l, tot_p), np.maximum(tot_l, tot_p), color='#2a78d6', alpha=.2, lw=0)
    x.semilogy(R[meas], tot_l[meas], 's-', color='#2a78d6', label='eBOSS QLF, completeness-corrected (PLE+LEDE; band: PLE)')
    x.semilogy(R[R >= 22.25], tot_l[R >= 22.25], 's--', color='#2a78d6', mfc='white', label='eBOSS QLF, extrapolated beyond its fit range')
    x.semilogy(R, eu, ':', color='0.3', lw=1.5, label='constant density: $\\propto 10^{0.6m}$ (normalised to the QLF at r = 18.25)')
    x.axvline(22.3, color='0.7', lw=.8)
    x.set(xlabel='Legacy r (extinction-corrected)', ylabel='quasars deg$^{-2}$ mag$^{-1}$ (all redshifts)', ylim=(.05, 3e3), xlim=(15.5, 24))
    x.legend(frameon=False, fontsize=7, loc='upper left')
    x.set_title('Quasar counts against magnitude', fontsize=10)
    fig.tight_layout(); fig.savefig('plots/qso_prior/Q3_counts_three_models.png', dpi=150); plt.close(fig)
    cum = lambda a, rmax: float(a[R < rmax].sum()*.5)
    rep = dict(definition=__doc__, r_bins=R.tolist(), z_bins=ZC.tolist(), current=ours.tolist(), qlf_ple_lede=led.tolist(), qlf_ple=ple.tolist(),
               ratio_total=(tot_o/tot_l).tolist(),
               cumulative_deg2={f'r<{m}': dict(current=cum(tot_o, m), qlf=cum(tot_l, m)) for m in (20, 21, 22, 22.5, 23, 24)})
    Path('docs/qso_prior/qso_prior_vs_qlf.json').write_text(json.dumps(rep, indent=1))
    for i, r in enumerate(R):
        print(f'r {r:5.2f}  current {tot_o[i]:8.2f}  QLF {tot_l[i]:8.2f} (PLE {tot_p[i]:8.2f})  ratio {tot_o[i]/tot_l[i]:6.3f}  by z: ' +
              ' '.join(f'{ours[i, j]/led[i, j]:5.2f}' if led[i, j] > 0 else '  -  ' for j in range(4)))
    print(json.dumps(rep['cumulative_deg2']))


if __name__ == '__main__':
    main()
