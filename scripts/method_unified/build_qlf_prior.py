#!/usr/bin/env python
"""Bundles whose quasar abundance prior is the eBOSS quasar luminosity function (Legacy r reference bands).

The promoted prior is spectroscopic DESI + SDSS counts times one completeness constant (2.37); against the
completeness-corrected eBOSS QLF it is 1.4-3x too high at r < 20 and 2-50x too low at r > 22.3
(scripts/method_unified/qso_prior_vs_qlf.py). Here Sigma_Q(z, r) for decals_dr9_south:r and decals_dr9_north:r
(the reference band of every scored object) is replaced by the PLE+LEDE prediction of Palanque-Delabrouille et
al. 2016 (A&A 587, A41), Table 7 (counts per 10,000 deg^2 per 0.5 mag per Delta z = 1):
  z      per magnitude row, the cumulative counts at z = 0, 1, ..., 5 are interpolated with a monotone cubic
         (PCHIP) and each fine z cell gets the difference across its edges, so bin totals are preserved;
  mag    log density interpolated linearly between row centres (15.75 ... 23.75), the last slope extrapolated.
The fit covers 0.68 < z < 4 and g_dered < 22.5 (r ~ 22.3); outside that the table is the paper's own model
extrapolation. Colour models, background densities and every other file are unchanged; other reference bands
keep their old priors (never used: Legacy r is always the reference band under the faint limit).

Writes models/multisurvey_psf/<bundle>_qlf (new bundle_id, manifest hashes updated). Promoted pointers are untouched.
Usage: python scripts/method_unified/build_qlf_prior.py
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qso_prior_vs_qlf import LEDE, R

SOURCES = {'20261003_magdep': '20261005_magdep_qlf', '20261003_magindep': '20261005_magindep_qlf'}
BANDS = ('decals_dr9_south:r', 'decals_dr9_north:r')
Z_EDGES_T = np.arange(0., 5.01, 1.)          # Table 7 redshift bins (z = 5-6 column omitted, < 0.1% of counts)


def qlf_density(z_edges, mag_centres):
    """Sigma_Q [deg^-2 mag^-1 per unit z] averaged over each z cell, at the given magnitude centres."""
    per_row = LEDE/10000/.5                    # deg^-2 mag^-1 per Delta z = 1 bin
    cells = np.zeros((len(R), len(z_edges) - 1))
    for i in range(len(R)):
        cum = PchipInterpolator(Z_EDGES_T, np.r_[0, np.cumsum(per_row[i])])
        cells[i] = np.diff(cum(np.clip(z_edges, 0, 5)))/np.diff(z_edges)
    logc = np.log10(np.maximum(cells, 1e-30))
    out = np.zeros((len(z_edges) - 1, len(mag_centres)))
    for j in range(len(z_edges) - 1):
        lo = np.interp(mag_centres, R, logc[:, j])
        slope = logc[-1, j] - logc[-2, j]                       # per 0.5 mag
        beyond = mag_centres > R[-1]
        lo[beyond] = logc[-1, j] + slope*(mag_centres[beyond] - R[-1])/.5
        below = mag_centres < R[0]
        lo[below] = logc[0, j] + (logc[1, j] - logc[0, j])*(mag_centres[below] - R[0])/.5
        out[j] = 10**lo
    return out


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('models/multisurvey_psf')
    for src, dst in SOURCES.items():
        s, d = root/src, root/dst
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)
        pri = json.loads((d/'priors.json').read_text())
        for band in BANDS:
            q = pri['anchors'][band]['qso_prior']
            ze, mc = np.array(q['z_edges']), np.array(q['mag_centres'])
            old = np.array(q['sigma']); new = qlf_density(ze, mc)
            q['sigma'] = new.tolist()
            q['meta'] = dict(q['meta'], abundance_policy='eBOSS QLF (Palanque-Delabrouille et al. 2016, A&A 587, A41, Table 7, '
                             'PLE+LEDE); PCHIP in z preserving Delta z = 1 bin totals; log-linear in magnitude',
                             abundance_source='scripts/method_unified/build_qlf_prior.py', previous_abundance_policy=q['meta'].get('abundance_policy'),
                             qlf_fit_range='0.68 < z < 4, g_dered < 22.5 (r ~ 22.3); outside: model extrapolation')
            dz = np.diff(ze)[:, None]; dm = np.diff(q['mag_edges'])[None]
            print(dst, band, 'total deg^-2: old %.1f new %.1f' % ((old*dz*dm).sum(), (new*dz*dm).sum()))
        (d/'priors.json').write_text(json.dumps(pri))
        man = json.loads((d/'manifest.json').read_text())
        man['files']['priors.json'] = sha(d/'priors.json'); man['bundle_id'] = dst
        man['status'] = f'candidate 5 October 2026: {src} with the eBOSS QLF abundance prior for Legacy r (not promoted)'
        man['source_bundle'] = src
        (d/'manifest.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
