#!/usr/bin/env python
"""Artificial colour-grid scores (F11, F15) for the bundles the note is drawn from.

Same grids as the release checks (validate_unified_release.py): 15 x 15 Legacy (g-r, r-z) grid from
-6 to 8 mag at fixed r (18.5, 21), (l, b) = (180, 45) deg, configs/full_sample_release.json grid
settings. Writes <out>/grid_<hemi>_<mag>_raw.npz (bundle scorer, no support cut) and _unified.npz
(with support percentiles) for each model name.
Usage: python scripts/method_unified/grid_scores.py
"""
import json
from pathlib import Path
import sys

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor import PSFMultiSurveyBaseline, Photometry
from qso_pcolor.unified import UnifiedPSFModel
from validate_full_sample_release import run_scores
from validate_psf_catchalls import grid_data
from validate_unified_pilot import run_unified
from common import POINTERS

OUT = Path('models/multisurvey_psf/work/method_unified/grids_note_ood')   # promoted scorer with the calibrated outside-both test


def main():
    rcfg = json.loads(Path('configs/full_sample_release.json').read_text()); rcfg['batch_size'] = 64
    for name, path in POINTERS.items():
        out = OUT/name; out.mkdir(parents=True, exist_ok=True)
        base = PSFMultiSurveyBaseline.load(path); unified = UnifiedPSFModel.load(path)
        for hemi in ('south', 'north'):
            for mag in rcfg['grid']['reference_magnitudes'][:2]:
                g = grid_data(base.model, hemi, mag, {'validation': rcfg['grid']}); phot = Photometry(g['flux'], g['variance'], g['bands'])
                sky = SkyCoord(l=g['l']*u.deg, b=g['b']*u.deg, frame='galactic').icrs
                raw = run_scores(base, g, rcfg, out/f'grid_{hemi}_{mag}_raw.npz')
                uni = run_unified(unified, phot, sky.ra.deg, sky.dec.deg, g['zprimary'], rcfg, out/f'grid_{hemi}_{mag}_unified.npz')
                print(name, hemi, mag, 'scorable', int(raw['eligible'].sum()), 'p>0.5', int((raw['eligible'] & (raw['p_quasar'] > .5)).sum()), flush=True)


if __name__ == '__main__':
    main()
