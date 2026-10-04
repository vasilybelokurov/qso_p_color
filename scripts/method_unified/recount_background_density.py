#!/usr/bin/env python
"""Recount a bundle's background surface density Sigma_B(m) without the flagged likely QSOs.

The binned background colour model is trained on the stellar sample with likely QSOs removed
(clean_stellar_qso_contamination.py: Quaia members, WISE AGN colours). Its count prior must describe
the same population, so the counts are redone exactly as complete_unified_pilot.build_priors does
(fit/select rows, per-cone counts in the bundle's existing magnitude bins of each reference band,
measured cone areas, HEALPix pooling count_prior with the run's nside / parent / n0) but excluding
flagged rows. No signal-to-noise cut is applied to the counts: like the QSO abundance, Sigma_B counts
every object at a magnitude. Bands whose density was inherited (no pilot support) are left unchanged.
Rewrites priors.json and the manifest hash in the bundle directory; reports old/new global density.

Usage: python scripts/method_unified/recount_background_density.py --bundle DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor.full_population import count_prior
from qso_pcolor.full_sample import write_json
from qso_pcolor.multisurvey import MultiSurveyModel
from qso_pcolor.multisurvey_data import Photometry
from complete_unified_pilot import fit_arrays

RUN = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bundle', type=Path, required=True); a = p.parse_args()
    cfg = json.loads((RUN/'config.json').read_text()); model = MultiSurveyModel.load(a.bundle/'model.json')
    priors = json.loads((a.bundle/'priors.json').read_text()); data = fit_arrays(RUN, 'stars')
    flag = np.load(RUN/'stars'/'qso_flag.npy')
    regions = {r['cone']: r for r in json.loads((Path(cfg['stellar_root'])/'spatial_roles.json').read_text())['regions']}
    train = np.isin(data['role'], cfg['fit_roles'])
    rows = np.flatnonzero(train & ~flag); fields = np.unique(data['field'][train]); bands = model.transform.bands
    ph = Photometry(data['flux'][rows], data['variance'][rows], bands); obs = ph.observed
    soft = model.transform.softening; k = 2.5/np.log(10)
    values = 22.5 - k*(np.arcsinh(ph.flux/(2*soft)) + np.log(soft))
    report = {}
    for j, label in enumerate(bands):
        pair = priors['anchors'][label]; old = pair['background_density']
        if old['meta'].get('inherited_no_pilot_counts'):
            report[label] = dict(origin='inherited; unchanged'); continue
        edges = np.asarray(old['mag_edges']); system, band = label.split(':')
        counts = np.zeros((len(fields), len(edges) - 1), int); area = np.zeros(len(fields)); cells = []
        for i, field in enumerate(fields):
            field = int(field); reg = regions[field]; cells.append(reg['cell']); use = data['field'][rows] == field
            info = json.loads((Path(cfg['stellar_root'])/'areas'/f'cone_{field:03d}.json').read_text())
            if system.startswith('decals_') and band in ('g', 'r', 'z'):
                if system.endswith(reg['hemisphere']):
                    area[i] = info['band_area_deg2'][band]
            else:
                cols = [c for c, b in enumerate(bands) if b.split(':')[0] == system]
                if obs[use][:, cols].any():
                    area[i] = info['area_deg2']
            counts[i] = np.histogram(values[use & obs[:, j], j], edges)[0]
        meta = dict(old['meta'], qso_flag_removed=True, recount='recount_background_density.py, likely QSOs excluded')
        dens = count_prior(counts, area, np.array(cells), edges, nside=cfg['nside'], nside_parent=cfg['spatial']['nside_parent'],
                           n0=cfg['density_n0'], meta=meta)
        new = dens.to_dict(); pair['background_density'] = new
        og, ng = np.asarray(old['global_density'], float), np.asarray(new['global_density'], float)
        ratio = np.where(og > 0, ng/np.where(og > 0, og, 1), np.nan)
        report[label] = dict(n_counted=int(counts.sum()), area_deg2=float(area.sum()),
                             ratio_new_old_by_bin=[None if not np.isfinite(x) else float(x) for x in ratio], mag_edges=edges.tolist())
    priors['report_recount'] = report
    write_json(a.bundle/'priors.json', priors)
    manifest = json.loads((a.bundle/'manifest.json').read_text())
    manifest['files']['priors.json'] = hashlib.sha256((a.bundle/'priors.json').read_bytes()).hexdigest()
    write_json(a.bundle/'manifest.json', manifest)
    for label in ('decals_dr9_south:r', 'decals_dr9_north:r'):
        r = report[label]; e = r['mag_edges']
        print(label, 'counted', r['n_counted'], 'new/old by bin:',
              ' '.join(f"{e[i]:.1f}:{x:.3f}" for i, x in enumerate(r['ratio_new_old_by_bin']) if x is not None and 16 <= e[i] <= 24.5))


if __name__ == '__main__':
    main()
