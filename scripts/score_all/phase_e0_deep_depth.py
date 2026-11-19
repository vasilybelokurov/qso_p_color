#!/usr/bin/env python
"""Phase E step 0: is Euclid deep-field photometry deep enough to train the faint background colours?

Coverage (checked 19 Nov 2026 by counting euclid_dr1.mer_catalogue_deep_survey rows in r = 0.03 deg probes on a
0.5-deg grid within 4 deg of each field centre): EDF-N ~28 deg^2, EDF-F ~13.5 deg^2, EDF-S ~33.5 deg^2 (number of
covered probes x 0.25 deg^2; coarse); scripts/score_all/deep_coverage_probes.py ->
~/data/qso_p_color/catalogues/euclid_dr1/deep_coverage/probes.json.

This script fetches CONES_PER_FIELD cones of r = RADIUS deg per deep field from the deep MER table, with the same
columns, Legacy DR9 and DESI-galaxy companions as scripts/euclid/fetch_cones.py, and caches them as
~/data/qso_p_color/catalogues/euclid_dr1/cones/deep<F>_<id>.npz. Cone centres: the probe grid points (from PROBES)
nearest to fixed offsets from each centre (listed in CENTRES; chosen inside the covered extent, not tuned).
It then applies the wide deployment selection (scripts/euclid/cones.py: cleaning, isolation, point-like) and
reports, per field and per 0.25-mag VIS bin, the median S/N in VIS, Y, J, H and the selected density, next to the
wide fit-cone census (docs/score_all/census_fit.json).

Acceptance (pre-stated): a deep field is usable for the faint-background colour fit to VIS_lim if the median S/N of the
deployment selection is >= 10 in all of Y, J, H at VIS_lim; we need VIS_lim >= 25.5 (the wide selected turnover is
VIS 25.4, docs/score_all/census_fit.json).

Output: docs/score_all/phase_e0_deep_depth.json. Light on CPU (one process, no fitting).

    python scripts/score_all/phase_e0_deep_depth.py            # fetch missing cones, then summarise
    python scripts/score_all/phase_e0_deep_depth.py --no-fetch # summarise cached cones only
"""
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/euclid')); sys.path.insert(0, str(ROOT/'scripts/score_all'))
BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/cones'
DEEP = 'euclid_dr1.mer_catalogue_deep_survey'
RADIUS = 0.2
# cone centres (deg): inside the covered extent of each field from the coverage probes
CENTRES = {'N': [(269.73, 66.02), (266.0, 65.5), (273.5, 67.0)],
           'F': [(52.93, -28.09), (51.8, -29.0), (54.0, -27.2)],
           'S': [(61.24, -48.42), (58.5, -47.0), (64.0, -50.0)]}
OUT = ROOT/'docs/score_all/phase_e0_deep_depth.json'


def fetch_missing():
    import fetch_cones
    fetch_cones.TABLES = (DEEP,)                                  # same columns, deep table only
    for f, cs in CENTRES.items():
        for i, (ra, dec) in enumerate(cs):
            cone = dict(ra=ra, dec=dec, radius_deg=RADIUS)
            path = BASE/f'deep{f}_{i:03d}.npz'
            if fetch_cones.cached_matches(path, cone):
                continue
            t = time.monotonic(); d = fetch_cones.fetch(cone)
            np.savez_compressed(path, **d, centre=np.array([ra, dec]), radius_deg=RADIUS)
            print(path.name, 'MER', len(d['mer_e_ra']), 'Legacy', len(d['legacy_ra']), f'{time.monotonic()-t:.0f} s',
                  flush=True)


def vis_at_snr(cen, snr, level):
    """Faintest bin centre with median S/N >= level, scanning from bright to faint until the first bin below it."""
    ok = np.isfinite(snr)
    last = np.nan
    for c, s in zip(cen[ok], snr[ok]):
        if c < 20:
            continue
        if s < level:
            break
        last = float(c)
    return last


def summarise():
    from census import EDGES, bin_stats, per_cone
    settings = json.loads((ROOT/'docs/euclid/settings.json').read_text())
    cen = 0.5*(EDGES[1:] + EDGES[:-1])
    wide = json.loads((ROOT/'docs/score_all/census_fit.json').read_text())['bins']
    rep = dict(radius_deg=RADIUS, centres=CENTRES, acceptance='median S/N >= 10 in Y, J, H at VIS >= 25.5', fields={})
    rows = {'wide_fit': {b: np.array(wide[f'snr_{b}'], float) for b in ('vis', 'y', 'j', 'h')}}
    rows['wide_fit']['density_sel'] = np.array(wide['density_sel'], float)
    for f in CENTRES:
        paths = sorted(glob.glob(str(BASE/f'deep{f}_*.npz')))
        if not paths:
            continue
        cones = [per_cone(p, settings) for p in paths]
        s = bin_stats(cones, range(len(cones)))
        rows[f'deep{f}'] = {b: s[f'snr_{b}'] for b in ('vis', 'y', 'j', 'h')}; rows[f'deep{f}']['density_sel'] = s['density_sel']
        per = [bin_stats(cones, [i]) for i in range(len(cones))]
        rep['fields'][f] = dict(
            cones=[Path(p).name for p in paths], n_selected=int(sum(np.isfinite(c['vis']).sum() for c in cones)),
            vis_turnover_selected=float(cen[np.argmax(s['density_sel'])]),
            **{f'vis_snr{lv}_{b}': vis_at_snr(cen, s[f'snr_{b}'], lv) for lv in (5, 10) for b in ('y', 'j', 'h')},
            vis_snr10_h_per_cone=[vis_at_snr(cen, q['snr_h'], 10) for q in per])
    rep['fields']['wide_fit'] = {f'vis_snr{lv}_{b}': vis_at_snr(cen, rows['wide_fit'][b], lv) for lv in (5, 10)
                                 for b in ('y', 'j', 'h')}
    sel = (cen >= 21) & (cen <= 27)
    rep['table'] = dict(vis=cen[sel].tolist(), **{k: {b: np.round(v[b][sel], 2).tolist() for b in v} for k, v in rows.items()})
    for f, v in rep['fields'].items():
        if f != 'wide_fit':
            lim = min(v['vis_snr10_y'], v['vis_snr10_j'], v['vis_snr10_h'])
            v['vis_lim_snr10_all_nisp'] = lim; v['pass'] = bool(lim >= 25.5)
    OUT.write_text(json.dumps(rep, indent=1, allow_nan=True))
    print(json.dumps(rep['fields'], indent=1))
    hdr = 'VIS   ' + '  '.join(f'{k:>22s}' for k in rows)
    print(hdr + '\n      ' + '  '.join(f'{"S/N Y  J  H | dens":>22s}' for _ in rows))
    for j in np.where(sel)[0]:
        print(f'{cen[j]:5.2f} ' + '  '.join(f'{v["y"][j]:5.1f}{v["j"][j]:5.1f}{v["h"][j]:5.1f}{v["density_sel"][j]:7.0f}' for v in rows.values()))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-fetch', action='store_true')
    if not ap.parse_args().no_fetch:
        fetch_missing()
    summarise()
