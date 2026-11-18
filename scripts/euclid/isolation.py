#!/usr/bin/env python
"""Isolation-neighbour counts (docs/euclid/PLAN.md 2.1) for every matched QSO target and background row.

Neighbour: another MER source within 1.5" with vis_det = 1, spurious_flag = 0 and VIS template flux > 10^-1 x the
object's (Delta VIS < 2.5 mag). Writes neighbours.npz next to each match file (aligned with it; -1 = unmatched).
"""
import glob
from pathlib import Path
import time
import numpy as np
from qso_pcolor.euclid_data import isolation_neighbours

BASE = Path.home()/'data/qso_p_color/catalogues/euclid_dr1'

if __name__ == '__main__':
    for path in sorted(glob.glob(str(BASE/'qso_targets/*/match.npz')) + glob.glob(str(BASE/'background/*/*/match.npz'))):
        out = Path(path).with_name('neighbours.npz')
        if out.exists():
            continue
        m = np.load(path); ok = m['matched']; t = time.monotonic()
        n = np.full(len(ok), -1)
        n[ok] = isolation_neighbours(m['table'][ok], m['object_id'][ok], None, None)
        np.savez_compressed(out, neighbours=n, radius_arcsec=1.5, dmag=2.5)
        print(path, int(ok.sum()), 'isolated', int((n[ok] == 0).sum()), f'{time.monotonic()-t:.0f} s', flush=True)
