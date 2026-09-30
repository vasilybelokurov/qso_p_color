#!/usr/bin/env python
"""Illustrate the saved tail audit and compare native Legacy photometric systems.

No scoring, fitting, catalogue acquisition or active-model changes. Uses every
fit/select object passing the stated display cuts, with no random subsampling.
Filter curves come from the installed speclite reference tables.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from speclite import filters

from qso_pcolor.full_sample import write_json
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.plotting import save_figure


NORTH, SOUTH = '#246db4', '#d55e00'
QSO, STAR = '#246db4', '#279469'
PREFIX = 'northern_review/'


def read_training(model, cfg, settings):
    """Return native luptitudes (mag) for usable grz fit/select rows."""
    result = {}
    for kind in ('qso', 'stars'):
        root = Path(cfg['inputs']) / kind
        d = {k: np.load(root / f'{k}.npy', mmap_mode='r')
             for k in ('flux', 'variance', 'role', 'eligible', 'zspec')}
        for hemi in ('north', 'south'):
            bands = tuple(f'decals_dr9_{hemi}:{b}' for b in ('g', 'r', 'z'))
            ix = [model.transform.bands.index(b) for b in bands]
            soft = model.transform.softening[ix]
            pieces, redshifts = [], []
            for lo in range(0, len(d['role']), settings['batch_size']):
                sl = slice(lo, lo + settings['batch_size'])
                phot = Photometry(d['flux'][sl][:, ix], d['variance'][sl][:, ix], bands)
                use = phot.observed.all(axis=1) & np.isin(d['role'][sl], [0, 1]) & d['eligible'][sl]
                f = phot.flux[use]
                u = 22.5 - 2.5 / np.log(10) * (np.arcsinh(f / (2 * soft)) + np.log(soft))
                pieces.append(u)
                redshifts.append(d['zspec'][sl][use])
            result[kind, hemi] = (np.concatenate(pieces), np.concatenate(redshifts))
            print(kind, hemi, len(result[kind, hemi][0]), 'usable grz training rows', flush=True)
    return result


def colours(u):
    return np.column_stack((u[:, 0] - u[:, 1], u[:, 1] - u[:, 2]))


def training_clouds(ax, data, hemi, settings, alpha=.16):
    counts = {}
    for kind, color in [('stars', STAR), ('qso', QSO)]:
        u, _ = data[kind, hemi]
        lo, hi = settings['bright_reference_window']
        c = colours(u[(u[:, 1] >= lo) & (u[:, 1] < hi)])
        ax.scatter(c[:, 0], c[:, 1], s=2, color=color, alpha=alpha,
                   edgecolors='none', rasterized=True, zorder=3)
        counts[kind] = len(c)
    return counts


def failure_maps(data, audit, settings, report):
    cache = Path(audit['cache'])
    cfg = audit['config']
    nodes = np.linspace(*cfg['grid']['grid_colour_range'], cfg['grid']['grid_size'])
    gx, gy = np.meshgrid(nodes, nodes)
    step = nodes[1] - nodes[0]
    extent = [nodes[0] - step/2, nodes[-1] + step/2] * 2
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#d9dde0')
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    fig.subplots_adjust(top=.83, bottom=.22, right=.85, hspace=.26, wspace=.13)
    mag = settings['grid_reference_magnitude']
    probe = np.array(settings['probe_colours'])
    pi = np.flatnonzero(np.isclose(gx.ravel(), probe[0]) & np.isclose(gy.ravel(), probe[1]))[0]
    report['grid'] = {}
    for row, hemi in enumerate(('north', 'south')):
        results = {name: dict(np.load(cache / f'grid_{hemi}_{mag}_{name}.npz'))
                   for name in ('baseline', 'candidate')}
        low = np.ones(gx.size, dtype=bool)
        for r in results.values():
            q = np.logaddexp(r['log_lambda_sameq'], r['log_lambda_fieldq'])
            b = r['log_lambda_bkg']
            low &= (q < np.nanmax(q) + np.log(settings['low_density_fraction']))
            low &= (b < np.nanmax(b) + np.log(settings['low_density_fraction']))
        for col, name in enumerate(('baseline', 'candidate')):
            ax = axes[row, col]
            r = results[name]
            value = np.ma.array(r['p_quasar'], mask=~r['eligible']).reshape(gx.shape)
            im = ax.imshow(value, origin='lower', extent=extent, vmin=0, vmax=1,
                           cmap=cmap, interpolation='nearest', aspect='equal')
            high = low & r['eligible'] & (r['p_quasar'] > cfg['high_qso_probability'])
            ax.scatter(gx.ravel()[high], gy.ravel()[high], s=23, facecolors='none',
                       edgecolors='white', linewidths=.7, zorder=4)
            counts = training_clouds(ax, data, hemi, settings, alpha=.32)
            ax.scatter(*probe, s=155, marker='*', color='#00e8ef', edgecolors='black', linewidths=.8, zorder=6)
            oldnew = 'Old: small training sample' if name == 'baseline' else 'New: full training sample'
            ax.set_title(f'{hemi.title()} | {oldnew}\n{high.sum()} high-score low-density grid points', fontsize=11)
            ax.text(.03, .97, f'Star marker: QSO score {r["p_quasar"][pi]:.1%}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
            ax.set_xticks([-6, -3, 0, 3, 6])
            ax.set_yticks([-6, -3, 0, 3, 6])
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.grid(False)
            if row == 1: ax.set_xlabel('g − r  [luptitude mag]')
            if col == 0: ax.set_ylabel('r − z  [luptitude mag]', labelpad=10)
            entry = dict(high_low_density=int(high.sum()), low_density_grid_count=int(low.sum()),
                         probe_p_quasar=float(r['p_quasar'][pi]), training_counts=counts)
            expected = audit['grids'][f'{hemi}_{mag}'][str(settings['low_density_fraction'])]
            assert entry['high_low_density'] == expected[name]['high_qso_after_guard']
            report['grid'][f'{hemi}_{name}'] = entry
    cbax = fig.add_axes([.88, .25, .02, .52])
    fig.colorbar(im, cax=cbax, label='Total-QSO model score (any redshift; uncalibrated)')
    fig.suptitle('The failure: high QSO scores far from the measured colour distribution', fontsize=15, y=.97)
    fig.text(.5, .92, f'Identical synthetic colour grids; r = {mag}, error = 0.03 mag per band; Galactic (l, b) = (180°, 45°)', ha='center', fontsize=10)
    fig.legend(handles=[Patch(facecolor='#d9dde0', label='Rejected by guard'),
                        Line2D([], [], color=QSO, marker='.', ls='', label='Measured QSOs'),
                        Line2D([], [], color=STAR, marker='.', ls='', label='Measured stellar background'),
                        Line2D([], [], color='black', marker='o', mfc='none', ls='', label='High score in low-density region')],
               loc='lower center', bbox_to_anchor=(.48, .09), ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .015, 'Measured points: all fit/select rows with 18 ≤ r < 19. Grid counts are NOT real-object error rates.\nLow density: below 1% of each model’s grid-peak QSO and stellar intensities. Same numbers in different native passbands are not the same source.', ha='center', fontsize=9)
    report['figures'].append(str(save_figure(fig, PREFIX + 'failure_maps')))


def measured_clouds(data, settings, report):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(top=.81, bottom=.27, wspace=.14)
    for ax, hemi in zip(axes, ('north', 'south')):
        counts = training_clouds(ax, data, hemi, settings)
        ax.scatter(*settings['probe_colours'], marker='*', s=210, c='#d55e00', edgecolors='black', zorder=5)
        ax.annotate('Artificial test point\n(g − r, r − z) = (3, −4)', xy=settings['probe_colours'], xytext=(.1, -3.5),
                    fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
        ax.set_title(f'{hemi.title()}: {counts["qso"]:,} QSOs; {counts["stars"]:,} background', fontsize=11)
        ax.set_xlim(-2, 5); ax.set_ylim(-5, 4); ax.set_xlabel('g − r  [luptitude mag]')
        ax.grid(alpha=.2); ax.set_aspect('equal', adjustable='box')
    axes[0].set_ylabel('r − z  [luptitude mag]', labelpad=10)
    fig.suptitle('Where the actual bright training objects lie', fontsize=15, y=.98)
    fig.text(.5, .9, 'All usable native g/r/z measurements with 18 ≤ r < 19; all QSO redshifts', ha='center')
    fig.legend(handles=[Line2D([], [], color=QSO, marker='o', ls='', label='Measured QSOs'),
                        Line2D([], [], color=STAR, marker='o', ls='', label='Measured stellar background')],
               loc='lower center', bbox_to_anchor=(.5, .08), ncol=2, frameon=False)
    fig.text(.5, .025, 'The test point has no nearby northern training examples: nearest QSO 3.91 mag, nearest background 4.67 mag away in this plane.', ha='center', fontsize=10)
    report['figures'].append(str(save_figure(fig, PREFIX + 'training_colours')))


def passbands(report):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    fig.subplots_adjust(top=.76, bottom=.32, wspace=.12)
    report['passbands'] = {}
    for ax, band, northern, lim in zip(axes, 'grz', ('BASS-g', 'BASS-r', 'MzLS-z'), ((350, 610), (520, 780), (770, 1110))):
        for name, color, ls, label in [(northern, NORTH, '-', 'North: ' + northern.split('-')[0]),
                                       ('decam2014-' + band, SOUTH, '--', 'South: DECam')]:
            f = filters.load_filter(name)
            wave = f.wavelength / 10
            response = f.response / f.response.max()
            ax.plot(wave, response, color=color, ls=ls, lw=2, label=label)
            report['passbands'][name] = dict(metadata=f.meta, wavelength_nm=wave.tolist(), peak_normalized_response=response.tolist())
        ax.set_title(band + ' band', fontsize=13); ax.set_xlim(*lim); ax.set_ylim(0, 1.08)
        ax.set_xlabel('Wavelength [nm]'); ax.grid(alpha=.2); ax.legend(fontsize=10, loc='upper right', frameon=False)
    axes[0].set_ylabel('Response / peak response', labelpad=10)
    fig.suptitle('North and South: overlapping, but different optical passbands', fontsize=15, y=.98)
    fig.text(.5, .87, 'North: BASS g/r + MzLS z     |     South: DECam g/r/z', ha='center')
    fig.text(.5, .10, 'Published reference system responses from speclite, each normalized to its own peak.\nReference atmospheres differ: North airmass 1.1; DECam 2014 airmass 1.3. These are not pure filter-only curves.', ha='center', fontsize=10)
    fig.text(.5, .015, 'Legacy W1/W2 come from WISE in both hemispheres: there is no analogous north–south filter difference for those bands.', ha='center', fontsize=10)
    report['figures'].append(str(save_figure(fig, PREFIX + 'optical_passbands')))


def colour_redshift(data, settings, report):
    edges = np.array(settings['redshift_edges'])
    centres = (edges[1:] + edges[:-1]) / 2
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    fig.subplots_adjust(top=.83, bottom=.16, hspace=.13, wspace=.2)
    report['colour_redshift'] = {}
    for hemi, color, ls in [('north', NORTH, '-'), ('south', SOUTH, '--')]:
        u, z = data['qso', hemi]
        lo, hi = settings['locus_reference_window']
        use = (u[:, 1] >= lo) & (u[:, 1] < hi)
        c, z = colours(u[use]), z[use]
        count = np.zeros(len(centres), int)
        stats = np.full((len(centres), 3, 2), np.nan)
        for j, (zl, zh) in enumerate(zip(edges[:-1], edges[1:])):
            keep = (z >= zl) & (z < zh)
            count[j] = keep.sum()
            if count[j] >= settings['minimum_locus_bin_rows']:
                stats[j] = np.percentile(c[keep], [16, 50, 84], axis=0)
        serial = [[[float(v) if np.isfinite(v) else None for v in pair] for pair in row] for row in stats]
        report['colour_redshift'][hemi] = dict(count=count.tolist(), percentiles=serial)
        for col in range(2):
            axes[0, col].plot(centres, stats[:, 1, col], color=color, ls=ls, label=hemi.title(), lw=2)
            axes[0, col].fill_between(centres, stats[:, 0, col], stats[:, 2, col], color=color, alpha=.12)
            axes[1, col].step(centres, count, where='mid', color=color, ls=ls)
    for col, label in enumerate(('g − r', 'r − z')):
        axes[0, col].set_ylabel(label + '  [luptitude mag]', labelpad=10)
        axes[0, col].legend(frameon=False); axes[0, col].grid(alpha=.2)
        axes[1, col].set_yscale('log'); axes[1, col].set_ylabel('QSOs / bin', labelpad=10)
        axes[1, col].set_xlabel('Spectroscopic redshift'); axes[1, col].grid(alpha=.2)
    fig.suptitle('Measured QSO colours in the two native systems', fontsize=15, y=.97)
    fig.text(.5, .9, f'{lo:g} ≤ r < {hi:g}; median and 16–84% population range in fixed redshift bins', ha='center')
    fig.text(.5, .025, 'Different objects in different sky regions, with the same magnitude cut; this is not a same-object filter transformation.\nDifferences can include population selection, extinction and measurement effects. Bins with fewer than 30 QSOs are omitted.', ha='center', fontsize=10)
    report['figures'].append(str(save_figure(fig, PREFIX + 'qso_colour_redshift')))


def main():
    settings = json.loads(Path('configs/northern_review_plots.json').read_text())
    audit = json.loads(Path(settings['audit']).read_text())
    model = MultiSurveyModel.load(Path(audit['config']['candidate']) / 'model.json')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    report = dict(settings=settings, figures=[], source_cache=audit['cache'],
                  probability_calibration=False, model_changed=False)
    data = read_training(model, audit['config'], settings)
    failure_maps(data, audit, settings, report)
    measured_clouds(data, settings, report)
    passbands(report)
    colour_redshift(data, settings, report)
    write_json(Path('docs/NORTHERN_REVIEW_PLOTS_2026-09-30.json'), report)
    print('\n'.join(report['figures']), flush=True)


if __name__ == '__main__':
    main()
