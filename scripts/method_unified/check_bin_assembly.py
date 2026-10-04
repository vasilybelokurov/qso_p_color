#!/usr/bin/env python
"""Check the assembled (concatenated) binned background against hard per-bin routing.

Held-out evaluation rows of a binned design run (role 3, test cones, faint limit, flagged removed,
never fitted): ln p(other bands | Legacy r) under (a) the concatenated 160-component mixture (bin weights
x bin share, as in the bundles) and (b) the single bin containing the object's Legacy r. All-sky
weights. Paired differences reported overall, by hemisphere of the Legacy r band and by distance
to the nearest interior bin edge. Note: (b) uses Legacy r, which is also the conditioning band, so
both are proper conditional densities.

Usage: python scripts/method_unified/check_bin_assembly.py [designs_dir]
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_background_designs as T
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.projected_xd import native_mixture

D = Path(sys.argv[1] if len(sys.argv) > 1 else 'models/multisurvey_psf/work/stellar_binned/20261003')


def main():
    cfg, layout, d, r, anchor, op, k, cur, snr = T.setup()
    share = np.array([c['pool'] for c in json.loads((D/'prepare.json').read_text())['counts']], float); share /= share.sum()
    ms = [GaussianMixture.from_dict(json.loads((D/f'binned_{j}.json').read_text())['mixture_u']) for j in range(len(share))]
    cat = native_mixture(GaussianMixture(np.concatenate([s*m.weights for s, m in zip(share, ms)]),
                                         np.concatenate([m.means for m in ms]), np.concatenate([m.covs for m in ms])), *op)
    bins = [native_mixture(m, *op) for m in ms]
    ev = np.load(D/'rows.npz')['eval']
    sc = T.heldout(lambda rr: [(np.ones(len(rr), bool), cat)], ev, d, anchor, r)
    sr = T.heldout(lambda rr: [(T.bin_of(rr) == j, bins[j]) for j in range(len(bins))], ev, d, anchor, r)
    dd = sc - sr; south = anchor[ev] == layout['native_labels'].index(T.S)
    inner = np.array(T.EDGES[1:-1]); dist = np.min(np.abs(r[ev][:, None] - inner[None]), axis=1)
    se = lambda x: x.std()/np.sqrt(len(x))
    out = dict(n=int(len(ev)), all=[float(dd.mean()), float(se(dd))],
               south=[float(dd[south].mean()), float(se(dd[south]))], north=[float(dd[~south].mean()), float(se(dd[~south]))],
               within_0p1_of_edge=[float(dd[dist < .1].mean()), float(se(dd[dist < .1]))],
               away_gt_0p3=[float(dd[dist > .3].mean()), float(se(dd[dist > .3]))],
               by_bin=[[float(dd[T.bin_of(r[ev]) == j].mean()), float(se(dd[T.bin_of(r[ev]) == j]))] for j in range(len(bins))])
    Path('docs/method_unified/bin_assembly_check.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
