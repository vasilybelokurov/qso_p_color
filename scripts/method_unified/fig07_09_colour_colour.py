#!/usr/bin/env python
"""F7-F9: colour-colour planes of QSOs and background, data and models.

Planes (Legacy South, extinction-corrected luptitudes): P1 = (g-r, r-z), P2 = (r-z, z-W1).
Data: role-3 objects with Legacy South r observed and S/N > 5 in every plane band.
Grey: data density (log). Contours: model densities enclosing 50% and 90%, each
conditioned on the bin's median Legacy South r, QSO slices weighted by the data's
redshift mix, and convolved with the bin's median measurement covariance in the plane.
F7: apparent-magnitude bins. F8: Galactic (l, b) bins at 19 <= r < 21.5.
F9: redshift bins (QSOs; background shown at the slice's median r for reference).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import (faint_ok, COLOURS, LABELS, background_weights, colour_matrix, combine, condition_on, contour_levels,
                    data, density_2d, idx, model, project, save_figure, snr, use_paper_style)

S = 'decals_dr9_south:'
PLANES = {'P1': ([(S+'g', S+'r'), (S+'r', S+'z')], [S+'g', S+'r', S+'z'], ((-.5, 2.), (-.6, 2.))),
          'P2': ([(S+'r', S+'z'), (S+'z', S+'w1')], [S+'r', S+'z', S+'w1'], ((-.6, 2.), (-1.5, 3.)))}
AXIS = {'P1': ('$g-r$', '$r-z$'), 'P2': ('$r-z$', '$z-W1$')}
MAG_BINS = [(17, 19), (19, 20.5), (20.5, 21.5), (21.5, 22.5)]
SKY_BINS = [(lab, blo, bhi, centre) for blo, bhi in ((25, 35), (35, 55), (55, 90)) for centre, lab in ((True, 'centre'), (False, 'anticentre'))]
Z_TARGETS = [0.55, 0.95, 1.45, 2.25, 2.85, 3.45]
COVERAGE = []


def selection(kind, plane):
    """Role-3 rows with Legacy South r, the faint limit (and, for background objects, likely QSOs removed),
    and S/N > 5 in the plane's bands."""
    d = data(kind); rows = np.flatnonzero((d['role'] == 3) & d['observed'][:, idx(S+'r')])
    rows = rows[faint_ok(kind, rows)]
    s = snr(kind, rows, PLANES[plane][1]); return rows[(s > 5).all(axis=1)]


def plane_values(kind, rows, plane):
    pairs = PLANES[plane][0]; y = data(kind)['y'][rows]
    return np.stack([y[:, idx(a)] - y[:, idx(b)] for a, b in pairs], 1)


def plane_noise(kind, rows, plane):
    """Median measurement covariance (mag^2) of the two plane colours over rows."""
    a = colour_matrix(PLANES[plane][0]); v = np.asarray(data(kind)['noise'][rows])
    v = np.where(np.isfinite(v), v, 0.)
    return np.median(np.einsum('ij,nj,kj->nik', a, v, a), axis=0)


def qso_model(name, rows, r0):
    """Union of slice mixtures weighted by the data's redshift mix, conditioned on r0."""
    mod = model(name); zc = np.asarray(mod.qso.z_centres)
    k = np.abs(data('qso')['zspec'][rows][:, None] - zc[None]).argmin(axis=1)
    counts = np.bincount(k, minlength=len(zc)); use = np.flatnonzero(counts)
    return combine([condition_on(mod.qso.mixtures[j], S+'r', r0) for j in use], counts[use])


def bkg_model(rows, r0):
    d = data('stars'); return condition_on(background_weights('independent', d['l'][rows], d['b'][rows]), S+'r', r0)


def draw(ax, plane, kind, rows, mixtures, reference=None, title=''):
    (xl, yl) = PLANES[plane][2]; c = plane_values(kind, rows, plane)
    h, xe, ye = np.histogram2d(c[:, 0], c[:, 1], bins=[np.linspace(*xl, 90), np.linspace(*yl, 90)])
    ax.pcolormesh(xe, ye, np.log10(h.T + 1), cmap='Greys', rasterized=True)
    xx, yy = np.meshgrid(np.linspace(*xl, 160), np.linspace(*yl, 160)); a = colour_matrix(PLANES[plane][0])
    noise = plane_noise(kind, rows, plane)
    cover = []
    for name, mix in mixtures:
        m2 = project(mix, a); z = density_2d(m2, xx, yy, noise); lev = contour_levels(z)
        ax.contour(xx, yy, z, levels=lev, colors=COLOURS[name], linewidths=(1.4, .9))
        # Coverage: fraction of data inside the model's 50% / 90% regions (ideal 0.50 / 0.90).
        dz = density_2d(m2, c[:, 0][:, None], c[:, 1][:, None], noise).ravel()
        cover.append(f"{(dz >= lev[1]).mean():.2f}/{(dz >= lev[0]).mean():.2f}")
        COVERAGE.append(dict(title=title, model=name, n=len(rows), inside50=float((dz >= lev[1]).mean()), inside90=float((dz >= lev[0]).mean())))
    if reference is not None:
        name, mix, style = reference; z = density_2d(project(mix, a), xx, yy, noise)
        ax.contour(xx, yy, z, levels=contour_levels(z, (.9,)), colors=COLOURS[name], linewidths=.8, linestyles=style, alpha=.8)
    ax.set_xlim(xl); ax.set_ylim(yl); ax.set_title(title + f'\nn = {len(rows):,}; in 50%/90%:\n' + ', '.join(cover), fontsize=7)
    ax.set_xlabel(AXIS[plane][0])


def legend(fig):
    handles = [Line2D([], [], color=COLOURS[n], lw=1.4, label=LABELS[n]) for n in ('independent', 'dependent', 'background')]
    handles.append(Line2D([], [], color='0.4', lw=.9, label='inner/outer contour: 50% / 90% of model'))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=8)


def r_of(kind, rows):
    return data(kind)['y'][rows][:, idx(S+'r')]


def fig_mag(plane, q, b):
    fig, axes = plt.subplots(2, len(MAG_BINS), figsize=(11, 5.6))
    for j, (lo, hi) in enumerate(MAG_BINS):
        qr = q[(r_of('qso', q) >= lo) & (r_of('qso', q) < hi)]; br = b[(r_of('stars', b) >= lo) & (r_of('stars', b) < hi)]
        r0 = .5*(lo + hi); B = bkg_model(br, r0)
        draw(axes[0, j], plane, 'qso', qr, [(n, qso_model(n, qr, r0)) for n in ('independent', 'dependent')],
             ('background', B, 'dotted'), f'QSOs, {lo} $\\leq r <$ {hi}')
        draw(axes[1, j], plane, 'stars', br, [('background', B)], ('independent', qso_model('independent', qr, r0), 'dotted'),
             f'background, {lo} $\\leq r <$ {hi}')
    for ax in axes[:, 0]:
        ax.set_ylabel(AXIS[plane][1])
    legend(fig); fig.tight_layout(rect=(0, 0, 1, .94)); return fig


def fig_sky(plane, q, b):
    fig, axes = plt.subplots(2, len(SKY_BINS), figsize=(14, 5.6))
    def in_bin(kind, rows, blo, bhi, centre):
        d = data(kind); l, bb = d['l'][rows], np.abs(d['b'][rows]); towards = (l < 90) | (l > 270)
        r = r_of(kind, rows)
        return rows[(bb >= blo) & (bb < bhi) & (towards == centre) & (r >= 19) & (r < 21.5)]
    for j, (lab, blo, bhi, centre) in enumerate(SKY_BINS):
        qr = in_bin('qso', q, blo, bhi, centre); br = in_bin('stars', b, blo, bhi, centre)
        r0 = float(np.median(r_of('stars', br))) if len(br) else 20.25; B = bkg_model(br, r0)
        t = f'|b| {blo}-{bhi}$^\\circ$, {lab}'
        draw(axes[0, j], plane, 'qso', qr, [(n, qso_model(n, qr, r0)) for n in ('independent', 'dependent')], ('background', B, 'dotted'), 'QSO, '+t)
        draw(axes[1, j], plane, 'stars', br, [('background', B)], ('independent', qso_model('independent', qr, r0), 'dotted'), 'bkg, '+t)
    for ax in axes[:, 0]:
        ax.set_ylabel(AXIS[plane][1])
    legend(fig); fig.tight_layout(rect=(0, 0, 1, .93)); return fig


def fig_z(q, b):
    fig, axes = plt.subplots(2, len(Z_TARGETS), figsize=(14, 5.6))
    mod = model('independent'); zc = np.asarray(mod.qso.z_centres)
    for i, plane in enumerate(('P1', 'P2')):
        qq = selection('qso', plane); bb = selection('stars', plane)
        for j, zt in enumerate(Z_TARGETS):
            k = int(np.abs(zc - zt).argmin()); zk = zc[k]; z = data('qso')['zspec'][qq]
            qr = qq[(z >= zk - .1) & (z < zk + .1)]; r0 = float(np.median(r_of('qso', qr)))
            mixes = [(n, condition_on(model(n).qso.mixtures[k], S+'r', r0)) for n in ('independent', 'dependent')]
            br = bb[np.abs(r_of('stars', bb) - r0) < .5]
            draw(axes[i, j], plane, 'qso', qr, mixes, ('background', bkg_model(br, r0), 'dotted'),
                 f'QSOs, $z$ = {zk:.2f} $\\pm$ 0.1 (median $r$ = {r0:.1f})')
            if j == 0:
                axes[i, j].set_ylabel(AXIS[plane][1])
    legend(fig); fig.tight_layout(rect=(0, 0, 1, .93)); return fig


def main():
    use_paper_style()
    for plane, tag in (('P1', 'a'), ('P2', 'b')):
        q, b = selection('qso', plane), selection('stars', plane)
        print(save_figure(fig_mag(plane, q, b), f'method_unified/F07{tag}_colour_colour_magnitude'))
        print(save_figure(fig_sky(plane, q, b), f'method_unified/F08{tag}_colour_colour_sky'))
    print(save_figure(fig_z(None, None), 'method_unified/F09_colour_colour_redshift'))
    import json; Path('docs/method_unified').mkdir(parents=True, exist_ok=True)
    Path('docs/method_unified/coverage_F07_F09.json').write_text(json.dumps(COVERAGE, indent=1))


if __name__ == '__main__':
    main()
