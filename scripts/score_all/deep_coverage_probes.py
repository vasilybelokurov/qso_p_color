"""Coverage of euclid_dr1.mer_catalogue_deep_survey around the three Euclid Deep Fields: row counts in r = 0.03 deg
probes on a 0.5-deg grid within 4 deg of each centre (EDF-N 269.73 +66.02, EDF-F 52.93 -28.09, EDF-S 61.24 -48.42;
centres UNVERIFIED to the arcminute). Area estimate = covered probes x 0.25 deg^2 (coarse).

    python scripts/score_all/deep_coverage_probes.py
"""
import json
from pathlib import Path
import numpy as np, sqlutilpy as sqlutil
OUT = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/deep_coverage/probes.json'
C = {'N': (269.73, 66.02), 'F': (52.93, -28.09), 'S': (61.24, -48.42)}
out = {}
for k, (ra0, dec0) in C.items():
    dd = np.arange(-4, 4.01, 0.5); pts = []
    for dy in dd:
        for dx in dd:
            if dx*dx+dy*dy <= 16: pts.append((ra0 + dx/np.cos(np.deg2rad(dec0+dy)), dec0+dy))
    vals = ','.join(f'({i},{a},{b})' for i, (a, b) in enumerate(pts))
    r = sqlutil.get(f"""SELECT v.i, (SELECT count(*) FROM euclid_dr1.mer_catalogue_deep_survey m
        WHERE q3c_radial_query(m.right_ascension, m.declination, v.ra, v.dec, 0.03)) FROM (VALUES {vals}) v(i, ra, dec)""",
        preamb="SET statement_timeout='900s'")
    n = np.zeros(len(pts), int); n[r[0]] = r[1]
    out[k] = dict(pts=pts, n=n.tolist())
    a = np.pi*0.03**2
    cov = n > 0
    print(k, 'grid pts', len(pts), 'covered', cov.sum(), '-> approx area deg2', cov.sum()*0.25,
          'median density/deg2 (covered)', int(np.median(n[cov])/a) if cov.any() else 0, flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out))
