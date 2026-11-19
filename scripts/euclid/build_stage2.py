#!/usr/bin/env python
"""Assemble stage-2 bundles (docs/euclid/PLAN.md 2.5) from scripts/euclid/fit_stage2.py fits.

Two variants per model: selection 'euclid_point' (Euclid thinning + c_Q, for Euclid-only scoring) and the parent
selection 'legacy_psf' (for the 41-band non-inferiority check against the promoted parent bundles).

    python scripts/euclid/build_stage2.py --bundle 20261005_magdep_qlf
"""
import argparse
import json
from pathlib import Path
import numpy as np

from qso_pcolor.euclid_bundle import build_joint_bundle
from qso_pcolor.gaussmix import GaussianMixture
import sys; sys.path.insert(0, str(Path(__file__).parent))
from fit_stage2 import euclid_colour_map  # noqa: E402

MODELS = Path('models/multisurvey_psf')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--bundle', required=True); a = ap.parse_args()
    st1 = MODELS/f'euclid_stage1_{a.bundle}'; fits = MODELS/'work/euclid'/f'stage2_{a.bundle}'/'fits'
    lat1 = json.loads((st1/'latent.json').read_text()); lay = lat1['layout']; qc = lay.get('qso_coordinates')
    qso = []
    for j in range(len(lat1['qso'])):
        m = GaussianMixture.from_dict(json.loads((fits/f'qso_{j:02d}.json').read_text())['mixture'])
        if qc is not None:                    # Euclid rows fitted as colours vs the magnitude coordinate: back to the
            Ui = np.linalg.inv(euclid_colour_map(m.n_dim, qc['index']))   # stage-1 convention (Euclid magnitudes)
            covs = Ui @ m.covs @ Ui.T; labels = tuple(GaussianMixture.from_dict(lat1['qso'][j]).labels)
            m = GaussianMixture(m.weights, m.means @ Ui.T, 0.5*(covs + covs.swapaxes(1, 2)), labels)
        qso.append(m)
    tasks = {t['name']: t for t in json.loads((fits.parent/'tasks.json').read_text())}
    b1 = GaussianMixture.from_dict(lat1['background']); W = np.zeros(b1.n_components); M = b1.means.copy(); C = b1.covs.copy()
    for j in range(8):
        t = tasks[f'bkg_{j}']; m = GaussianMixture.from_dict(json.loads((fits/f'bkg_{j}.json').read_text())['mixture'])
        comps = np.asarray(t['components']); W[comps] = t['share']*m.weights; M[comps] = m.means; C[comps] = m.covs
    bkg = GaussianMixture(W/W.sum(), M, C, b1.labels)
    # Thinning and c_Q recomputed for the refitted mixtures (PLAN 2.5): scripts/euclid/selection.py run on the
    # stage-2 legacy_psf bundle -> work/euclid/selection_stage2_<bundle>; built first (legacy_psf needs no selection).
    sel_path = MODELS/'work/euclid'/f'selection_stage2_{a.bundle}'/'selection.json'
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else None
    variants = [('legacy_psf', None, None)]
    if sel is None:
        print('no', sel_path, '- run selection.py on the legacy_psf bundle, then rebuild for euclid_point')
    else:
        variants.append(('euclid_point', dict(name='euclid_point', thinning=sel['thinning'], source=str(sel_path)), sel['c_q']))
    for name, selection, cq in variants:
        out = build_joint_bundle(st1, MODELS/f'euclid_stage2_{name}_{a.bundle}', qso_latent=qso, bkg_latent=bkg,
                                 selection=selection, c_q=cq, meta=dict(fits=str(fits)))
        print('built', out)
