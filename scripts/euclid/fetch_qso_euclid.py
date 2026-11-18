#!/usr/bin/env python
"""Euclid DR1 MER match (nearest within 1 arcsec, both wide tables) for the frozen 1,657,444 QSO targets.

Output: ~/data/qso_p_color/catalogues/euclid_dr1/<hash>/match.npz aligned with QROOT targets.npz.
"""
import argparse
from pathlib import Path
import numpy as np
from qso_pcolor.euclid_data import match

QROOT = Path('models/multisurvey_psf/work/full_sample_photometry/e2615aaa39862aff')
CACHE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--radius', type=float, default=1.0)
    a = p.parse_args()
    t = np.load(QROOT/'targets.npz')
    r = match(t['ra'], t['dec'], CACHE/'qso_targets', radius_arcsec=a.radius, log=lambda s: print(s, flush=True))
    print('DONE', int(r['matched'].sum()), 'of', len(r['matched']), flush=True)
