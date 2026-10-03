#!/usr/bin/env python
"""Test bundle: a promoted bundle with the binned background and the faint limit (step 3).

Copies a promoted bundle and changes only the non-QSO side:
  background  the 8 Legacy-r bin mixtures of test_background_designs.py concatenated into one
              160-component joint mixture, bin weights scaled by each bin's share of eligible
              training objects; converted from (Legacy r, colours) back to latent bands and
              projected to native bands as the completion does.
  spatial     all-sky weights only (no HEALPix cells): sky dependence is tested separately.
  faint limit model meta faint_limit (Legacy r South then North, S/N >= 10; also the reference).
  catch-all   Student-t moments from the new background; fractions refitted on calibration
              (role 2) rows that pass the faint limit, capped at --catchall-rows.
QSO models, priors (no background dependence) and the QSO support threshold are kept.
latent.json is copied unchanged (its background entry is the old one; only QSO entries are used).
The source bundle is not modified.

Usage::

    python scripts/method_unified/build_binned_test_bundle.py --source models/multisurvey_psf/13866e45ef794059 \\
        --designs models/multisurvey_psf/work/background_designs/test2 --out models/multisurvey_psf/work/binned_bundle/independent
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.joint_spatial import JointSpatialWeights
from qso_pcolor.multisurvey import MultiSurveyModel, MultiSurveyOutlier
from qso_pcolor.multisurvey_data import Photometry
from qso_pcolor.outlier import mixture_moments
from qso_pcolor.unified import native_view
from compare_psf_catchalls import evaluate_densities, fit_fractions
from complete_unified_pilot import fit_arrays
from run_unified_pilot import magnitude_colour_matrix

RUN = Path('models/multisurvey_psf/work/unified_full/20261001/13866e45ef794059')   # prepared data, layout, config
FAINT = dict(bands=['decals_dr9_south:r', 'decals_dr9_north:r'], min_snr=10.)


def binned_background(designs, layout):
    """Concatenated latent mixture (latent band coordinates) of the per-bin fits."""
    share = np.array([c['pool'] for c in json.loads((designs/'prepare.json').read_text())['counts']], float); share /= share.sum()
    mixes = [GaussianMixture.from_dict(json.loads((designs/f'binned_{j}.json').read_text())['mixture_u']) for j in range(len(share))]
    tinv = np.linalg.inv(magnitude_colour_matrix(layout['latent_labels'], 'legacy:r'))
    means = np.concatenate([m.means for m in mixes]) @ tinv.T
    covs = tinv @ np.concatenate([m.covs for m in mixes]) @ tinv.T
    return GaussianMixture(np.concatenate([s*m.weights for s, m in zip(share, mixes)]), means, .5*(covs + covs.swapaxes(-1, -2)),
                           tuple(layout['latent_labels']))


_state = {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--designs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--catchall-rows', type=int, default=40000)
    p.add_argument('--workers', type=int, default=8); a = p.parse_args()
    cfg = json.loads((RUN/'config.json').read_text()); layout = json.loads((RUN/'layout.json').read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    old = MultiSurveyModel.load(a.source/'model.json')
    latent_b = binned_background(a.designs, layout); bmix = native_view(latent_b, layout, 'stars')
    spatial = JointSpatialWeights(old.spatial_background.nside, old.spatial_background.nside_parent, old.spatial_background.n0,
                                  bmix.weights, meta=dict(scope='all-sky weights only (binned background test)'))
    meta = dict(old.meta, faint_limit=FAINT, background_population=old.meta.get('background_population', '') +
                f'; binned in Legacy r ({len(latent_b.weights)} components) from {a.designs}')
    model = dataclasses.replace(old, background=GaussianMixture(bmix.weights, bmix.means, bmix.covs, old.background.labels),
                                spatial_background=spatial, meta=meta)
    model.save(a.out/'model.json')
    # Catch-all: same construction as complete_unified_pilot.py, fractions on faint-limited calibration rows.
    catch = cfg['catchall']; mean, cov = mixture_moments([model.background])
    outlier = MultiSurveyOutlier(mean, catch['kappa']**2*cov, catch['kappa'], 0., model.transform.bands, model.transform_id,
        {'*': ([0., 1.], [catch['global_prior_mean']])}, meta=dict(population='psf', model_run_id=model.meta['run_id']),
        family=catch['family'], nu=catch['nu'], noise=catch['noise'])
    data = fit_arrays(RUN, 'stars'); rr = np.flatnonzero(data['role'] == 2)
    has, bright, _ = model.faint_limit_status(Photometry(data['flux'][rr], data['variance'][rr], model.transform.bands))
    rr = rr[has & bright]; rr = np.sort(np.random.default_rng(20261003).choice(rr, min(a.catchall_rows, len(rr)), replace=False))
    outlier.save(a.out/'outlier_unfitted.json')
    chunks = [rr[i:i+2000] for i in range(0, len(rr), 2000)]
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init,
                             initargs=(str(a.out/'model.json'), str(a.out/'outlier_unfitted.json'))) as pool:
        parts = list(pool.map(_chunk, chunks))
    logs = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    outlier.fractions = fit_fractions(logs['background'], logs['outlier'], logs['anchor'], logs['magnitude'], model.transform.bands,
                                      n_bins=catch['magnitude_bins'], strength=catch['pooling_strength'], config=catch)
    outlier.meta.update(roles=['calib'], rows=int(len(rr)), faint_limit=FAINT); outlier.save(a.out/'outlier.json')
    (a.out/'outlier_unfitted.json').unlink()
    shutil.copy(a.source/'priors.json', a.out/'priors.json')
    for name in ('latent.json', 'support.json'):
        shutil.copy(a.source/name, a.out/name)
    manifest = json.loads((a.source/'manifest.json').read_text())
    manifest.update(files={n: file_hash(a.out/n) for n in ('model.json', 'priors.json', 'outlier.json')},
                    bundle_id=a.out.name+'_binned_test', status='TEST ONLY: binned background, all-sky weights, faint limit; not for release',
                    unified_files={n: hashlib.sha256((a.out/n).read_bytes()).hexdigest() for n in ('latent.json', 'support.json')})
    write_json(a.out/'manifest.json', manifest)
    write_json(a.out/'build.json', dict(source=str(a.source), designs=str(a.designs), components=int(len(bmix.weights)),
                                        catchall_rows=int(len(rr)), faint_limit=FAINT))
    print('BUNDLE', a.out, flush=True)


def _init(model_path, outlier_path):
    _state['model'] = MultiSurveyModel.load(model_path); _state['outlier'] = MultiSurveyOutlier.load(outlier_path)
    _state['data'] = fit_arrays(RUN, 'stars')


def _chunk(rows):
    return evaluate_densities(_state['model'], _state['data'], rows, {'outlier': _state['outlier']}, chunk=64)


if __name__ == '__main__':
    main()
