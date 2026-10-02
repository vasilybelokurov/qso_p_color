#!/usr/bin/env python
"""F6: QSO colour versus redshift, data and both model variants.

Data: role-3 test QSOs, extinction-corrected luptitudes, both bands observed at S/N > 10.
Grey: data density normalised within each redshift column (0.05 wide). Lines: per-slice
model median; shaded: 16-84% range of the noise-free (deconvolved) model colour.
The magnitude-dependent model is marginalised over magnitude.
"""
import numpy as np
import matplotlib.pyplot as plt
from common import (COLOURS, LABELS, colour_matrix, data, idx, model, project, quantiles_1d,
                    save_figure, snr, use_paper_style)

PANELS = [('decals_dr9_south:g', 'decals_dr9_south:r'), ('decals_dr9_south:r', 'decals_dr9_south:z'),
          ('decals_dr9_south:z', 'decals_dr9_south:w1'), ('decals_dr9_south:w1', 'decals_dr9_south:w2'),
          ('sdss:u', 'sdss:g'), ('ps1:i', 'ps1:z'), ('ps1:z', 'ps1:y'), ('vhs:j', 'vhs:ks')]


SURVEY = dict(decals_dr9_south='Legacy S', sdss='SDSS', ps1='PS1', vhs='VHS')


def band(label):
    b = label.split(':')[1]
    return b.upper() if b.startswith('w') else ('K$_s$' if b == 'ks' else (b.upper() if b in ('j', 'h') else b))


def main():
    use_paper_style(); d = data('qso'); role3 = np.flatnonzero(d['role'] == 3)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True)
    for ax, (p, q) in zip(axes.ravel(), PANELS):
        s = snr('qso', role3, [p, q]); use = role3[(s > 10).all(axis=1)]
        z = d['zspec'][use]; c = d['y'][use][:, idx(p)] - d['y'][use][:, idx(q)]
        lo, hi = np.nanpercentile(c, [.5, 99.5]); pad = .15*(hi - lo); lim = (lo - pad, hi + pad)
        h, xe, ye = np.histogram2d(z, c, bins=[np.arange(0, 4.45, .05), np.linspace(*lim, 70)])
        h = h/np.maximum(h.max(axis=1, keepdims=True), 1)
        ax.pcolormesh(xe, ye, h.T, cmap='Greys', vmin=0, vmax=1.3, rasterized=True)
        a = colour_matrix([(p, q)])
        for mname in ('independent', 'dependent'):
            mod = model(mname); zc = mod.qso.z_centres
            qs = np.array([quantiles_1d(project(m, a)) for m in mod.qso.mixtures])
            ax.plot(zc, qs[:, 1], color=COLOURS[mname], lw=1.6, label=LABELS[mname])
            ax.fill_between(zc, qs[:, 0], qs[:, 2], color=COLOURS[mname], alpha=.18, lw=0)
        ax.set_ylim(lim); ax.set_title(f"{SURVEY[p.split(':')[0]]}: {band(p)} $-$ {band(q)}   (n = {len(use):,})", fontsize=8.5)
        ax.set_ylabel('colour [mag]')
    for ax in axes[-1]:
        ax.set_xlabel('spectroscopic redshift $z$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, .95))
    print(save_figure(fig, 'method_unified/F06_colour_redshift'))


if __name__ == '__main__':
    main()
