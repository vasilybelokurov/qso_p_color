#!/usr/bin/env python
"""Fraction of the EDF-North deep-MER area observed spectroscopically by DESI DR1.

For each EDF-N coverage probe with deep MER rows (~/data/.../deep_coverage/probes.json, 0.5-deg grid), count DESI DR1
spectroscopic targets (rows of desi_dr1.photometry, which lists observed targets; no zpix join, for speed) within PROBE_R deg. A probe
with zero redshifts is outside DR1 coverage. DESI DR1 density where covered is >> 1 per PROBE_R disc, so the test is
binary in practice (the count distribution is reported to check that).

Output: ~/data/qso_p_color/catalogues/euclid_dr1/deep_coverage/edfn_desi_probes.json and a line on stdout.

    python scripts/score_all/edfn_desi_coverage.py
"""
import json
from pathlib import Path
import numpy as np
import sqlutilpy as sqlutil

D = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/deep_coverage'
PROBE_R = 0.1


def main():
    pr = json.loads((D/'probes.json').read_text())['N']
    pts = [p for p, n in zip(pr['pts'], pr['n']) if n > 0]
    vals = ','.join(f'({i},{a},{b})' for i, (a, b) in enumerate(pts))
    r = sqlutil.get(f"""SELECT v.i, (SELECT count(*) FROM desi_dr1.photometry p
        WHERE q3c_radial_query(p.ra, p.dec, v.ra, v.dec, {PROBE_R})) FROM (VALUES {vals}) v(i, ra, dec)""",
        preamb="SET statement_timeout='1800s'")
    n = np.zeros(len(pts), int); n[r[0]] = r[1]
    out = dict(probe_r_deg=PROBE_R, pts=pts, n_desi=n.tolist(), frac_covered=float((n > 0).mean()),
               count_quantiles_covered=np.percentile(n[n > 0], [5, 25, 50, 75, 95]).tolist() if (n > 0).any() else None)
    (D/'edfn_desi_probes.json').write_text(json.dumps(out))
    print(json.dumps({k: out[k] for k in ('frac_covered', 'count_quantiles_covered')}))


if __name__ == '__main__':
    main()
