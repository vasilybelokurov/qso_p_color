#!/usr/bin/env python
"""Quasar labels for the score-all evaluation (PI decision D3, 19 Nov 2026): DESI DR1 QSOs in Euclid Deep Field North.

DESI DR1 spectroscopic QSOs (spectype QSO, zwarn = 0, zcat_primary) within 4 deg of EDF-N (RA 269.73, Dec +66.02;
centre UNVERIFIED to the arcminute, the radius covers the field), with their Legacy DR9 photometry from
desi_dr1.photometry, matched to the nearest Euclid DR1 deep-field MER source within 1" (qso_pcolor.euclid_data.match,
table euclid_dr1.mer_catalogue_deep_survey, same columns as the wide tables). These objects were never used by this
project. Deep-field photometry is deeper than wide; degrading it to wide depth happens at evaluation time.

Output: ~/data/qso_p_color/catalogues/euclid_dr1/edfn_qso/edfn_qso.npz (DESI columns prefixed desi_, MER columns
prefixed mer_), and a summary on stdout.

    python scripts/score_all/fetch_edfn_qsos.py
"""
import json
from pathlib import Path

import numpy as np
import sqlutilpy as sqlutil

from qso_pcolor.euclid_data import match

OUT = Path.home()/'data/qso_p_color/catalogues/euclid_dr1/edfn_qso'
CENTRE, RADIUS = (269.73, 66.02), 4.0
DEEP = ('euclid_dr1.mer_catalogue_deep_survey',)
COLS = ('p.targetid, p.ra, p.dec, z.z, z.zerr, p.release, p.maskbits, p.fracflux_r, p.ebv, '
        + ', '.join(f'p.flux_{b}, p.flux_ivar_{b}, p.mw_transmission_{b}' for b in ('g', 'r', 'z', 'w1', 'w2')))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q = sqlutil.get(f"""WITH p AS MATERIALIZED (SELECT * FROM desi_dr1.photometry
                          WHERE q3c_radial_query(ra, dec, {CENTRE[0]}, {CENTRE[1]}, {RADIUS}))
        SELECT {COLS} FROM p JOIN desi_dr1.zpix z ON z.targetid = p.targetid
        WHERE z.spectype = 'QSO' AND z.zwarn = 0 AND z.zcat_primary ORDER BY p.targetid""",
                    asDict=True, preamb="SET statement_timeout = '1800s'")
    q = {k: np.asarray(v) for k, v in q.items()}
    m = match(q['ra'], q['dec'], OUT/'match', radius_arcsec=1.0, tables=DEEP, log=lambda s: print(s, flush=True))
    out = {f'desi_{k}': v for k, v in q.items()}; out.update({f'mer_{k}': np.asarray(v) for k, v in m.items()})
    np.savez_compressed(OUT/'edfn_qso.npz', **out)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = 22.5 - 2.5*np.log10(q['flux_r'])
    mt = m['matched']
    summary = dict(n_desi_qso=int(len(mt)), n_matched_deep=int(mt.sum()), n_matched_r_gt_22_5=int((mt & (r > 22.5)).sum()),
                   z_quartiles=[float(x) for x in np.nanpercentile(q['z'][mt], [25, 50, 75])])
    (OUT/'summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
