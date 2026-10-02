#!/usr/bin/env python
"""F2-F5: data, roles, extinction and the North/South relation.

F2  Sky distribution of QSO rows by role (Galactic Mollweide, HEALPix nside 16 counts)
    and the fraction of fit/select rows with each of the 41 bands observed.
F3  Redshift and Legacy r distributions of fit/select QSOs by parent survey.
F4  Extinction check: our SFD98 x DR9 coefficients against catalogue mw_transmission in
    four reserved Legacy cones; E(B-V) distributions of the training samples.
F5  Same-object North minus South colours in overlap fields, before and after the DESI
    North->South relation (uncorrected photometry, as the relation is defined).
"""
import json
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from common import COLOURS, ROOT, bands, data, idx, save_figure, use_paper_style

ROLE_NAMES = {0: 'fit', 1: 'select', 2: 'calibration', 3: 'test'}


def f2():
    d = data('qso'); fig = plt.figure(figsize=(11, 6.2))
    for i, (roles, title) in enumerate((((0, 1), 'fit + select (training)'), ((2,), 'calibration'), ((3,), 'test'))):
        use = np.isin(d['role'], roles); pix = hp.ang2pix(16, d['l'][use], d['b'][use], lonlat=True)
        m = np.bincount(pix, minlength=hp.nside2npix(16)).astype(float); m[m == 0] = hp.UNSEEN
        hp.mollview(m, fig=fig.number, sub=(2, 3, i+1), title=f'QSOs, {title} (n = {use.sum():,})\nQSOs per nside-16 pixel, log scale', unit='',
                    cmap='Greys', norm='log', min=1, cbar=True, notext=False, badcolor='white')
        hp.graticule(dpar=30, dmer=60, alpha=.3)
    ax = fig.add_axes((.07, .08, .9, .36)); lab = bands(); x = np.arange(len(lab))
    for k, (kind, name) in enumerate((('qso', 'independent'), ('stars', 'background'))):
        dd = data(kind); fit = np.isin(dd['role'], (0, 1))
        frac = np.asarray(dd['observed'][fit]).mean(axis=0)
        ax.bar(x + (k - .5)*.38, frac, width=.36, color=COLOURS[name], label=('QSO' if kind == 'qso' else 'background')+' fit/select rows')
    ax.set_xticks(x); ax.set_xticklabels([l.replace('decals_dr9_', 'LS-').replace(':', ' ') for l in lab], rotation=90, fontsize=6.5)
    ax.set_ylabel('fraction observed'); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=8, loc='upper right')
    return fig


def f3():
    d = data('qso'); fit = np.flatnonzero(np.isin(d['role'], (0, 1)))
    inputs = Path(json.loads((ROOT/'config.json').read_text())['inputs'])
    master = np.load(inputs/'qso'/'master_index.npy')[d['source_row'][fit]]
    cat = json.loads((inputs/'manifest.json').read_text()).get('master') or 'models/none'
    objects = np.load(Path.home()/'data/qso_p_color/catalogues/qso_sdss_desi/77d7aa514e9e3e47/objects.npz')
    nd, ns = objects['n_desi'][master] > 0, objects['n_sdss'][master] > 0
    groups = {'DESI only': nd & ~ns, 'SDSS only': ns & ~nd, 'both': nd & ns}
    shades = {'DESI only': '#2a78d6', 'SDSS only': '#eb6834', 'both': '#1baf7a'}
    r = np.where(d['observed'][fit][:, idx('decals_dr9_south:r')], d['y'][fit][:, idx('decals_dr9_south:r')],
                 d['y'][fit][:, idx('decals_dr9_north:r')])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for g, m in groups.items():
        axes[0].hist(d['zspec'][fit][m], bins=np.arange(0, 4.5, .05), histtype='step', lw=1.4, color=shades[g], label=f'{g} (n = {m.sum():,})')
        axes[1].hist(r[m], bins=np.arange(16, 24.5, .1), histtype='step', lw=1.4, color=shades[g])
    axes[0].set_xlabel('spectroscopic redshift $z$'); axes[0].set_ylabel('QSOs per bin'); axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel('Legacy $r$ (corrected luptitude) [mag]'); axes[1].set_ylabel('QSOs per bin')
    fig.tight_layout(); return fig


def f4():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from qso_pcolor.extinction import sfd_ebv
    coeff = json.loads(Path('configs/extinction_coefficients.json').read_text())['coefficients']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4)); dm = []
    for name in ('example_bkg_05_1873011699', 'modes_cone_15_8d67dac1e5', 'modes_cone_00_08a2d50446', 'modes_cone_17_2ed1f3e573'):
        z = np.load(Path.home()/f'data/qso_p_color/project_data/{name}.npz')
        c = SkyCoord(ra=z['ra']*u.deg, dec=z['dec']*u.deg).galactic; e = sfd_ebv(c.l.deg, c.b.deg)
        for b in 'grz':
            ok = z[f'mw_transmission_{b}'] > 0
            dm.append(-2.5*np.log10(10**(-.4*coeff[f'decals_dr9_south:{b}']*e[ok])/z[f'mw_transmission_{b}'][ok]))
    dm = np.concatenate(dm)
    axes[0].hist(dm*1e3, bins=60, color='0.45'); axes[0].set_yscale('log')
    axes[0].set_xlabel('our minus catalogue extinction, $g,r,z$ [mmag]'); axes[0].set_ylabel('objects')
    axes[0].set_title(f'{len(dm):,} band measurements; max $|\\Delta|$ = {1e3*np.abs(dm).max():.3f} mmag', fontsize=8.5)
    for kind, name in (('qso', 'independent'), ('stars', 'background')):
        e = np.asarray(data(kind)['ebv'])
        axes[1].hist(e, bins=np.linspace(0, .3, 61), histtype='step', lw=1.4, color=COLOURS[name], density=True,
                     label=f"{'QSO' if kind == 'qso' else 'background'} rows: median {np.median(e):.3f}")
    axes[1].set_xlabel('SFD98 $E(B-V)$ [mag]'); axes[1].set_ylabel('probability density'); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); return fig


def f5():
    layout = json.loads((ROOT/'layout.json').read_text()); lab = layout['native_labels']; soft = np.array(layout['softening'])
    h = np.array(layout['operators']['stars']['matrix']); b0 = np.array(layout['operators']['stars']['offset'])
    o = np.load(json.loads((ROOT/'config.json').read_text())['overlap'])
    sel = o['observed'][:, :, :3].all(axis=(1, 2)); f = o['flux'][sel].astype(float); v = o['variance'][sel]
    a = 2.5/np.log(10)
    def lupt(flux, labels):
        s = np.array([soft[lab.index(x)] for x in labels]); return 22.5 - a*(np.arcsinh(flux/(2*s)) + np.log(s))
    north = lupt(f[:, 0, :3], [f'decals_dr9_north:{x}' for x in 'grz']); south = lupt(f[:, 1, :3], [f'decals_dr9_south:{x}' for x in 'grz'])
    snr = (f[:, :, :3]/np.sqrt(v[:, :, :3]) > 20).all(axis=(1, 2)); north, south = north[snr], south[snr]
    lat = [layout['latent_labels'].index(f'legacy:{x}') for x in 'grz']; rows = [lab.index(f'decals_dr9_north:{x}') for x in 'grz']
    pred = south @ h[np.ix_(rows, lat)].T + b0[rows]          # south values play the latent coordinates
    gr = south[:, 0] - south[:, 1]; fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (y, t) in zip(axes, ((north - south, 'before: North $-$ South'), (north - pred, 'after: North $-$ relation(South)'))):
        for k, (bnd, col) in enumerate(zip('grz', ('#2a78d6', '#eb6834', '#1baf7a'))):
            order = np.argsort(gr); x = gr[order]; yy = y[order, k]
            edges = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 16); cen = .5*(edges[1:] + edges[:-1])
            med = [np.median(yy[(x >= lo) & (x < hi)]) if ((x >= lo) & (x < hi)).sum() > 20 else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
            ax.scatter(gr, y[:, k], s=1, color=col, alpha=.08, rasterized=True)
            ax.plot(cen, med, color=col, lw=1.8, label=f'${bnd}$ (binned median)')
        ax.axhline(0, color='0.3', lw=.8); ax.set_title(t + f'  (n = {len(gr):,}, S/N > 20)', fontsize=8.5)
        ax.set_xlabel('South $g-r$ [mag]'); ax.set_ylim(-.15, .15)
    axes[0].set_ylabel('same-object difference [mag]'); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); return fig


def main():
    use_paper_style()
    for fn, name in ((f2, 'F02_data_roles_bands'), (f3, 'F03_qso_samples'), (f4, 'F04_extinction'), (f5, 'F05_north_south')):
        print(save_figure(fn(), 'method_unified/'+name))


if __name__ == '__main__':
    main()
