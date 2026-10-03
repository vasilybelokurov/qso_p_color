#!/usr/bin/env python
"""Bundle with the binned background and the faint limit, from a promoted bundle's QSO side.

Copies a promoted bundle and changes only the non-QSO side:
  background  the 8 Legacy-r bin mixtures of test_background_designs.py concatenated into one
              160-component joint mixture, bin weights scaled by each bin's share of eligible
              training objects; converted from (Legacy r, colours) back to latent bands and
              projected to native bands as the completion does.
  spatial     --spatial global: one weight vector (as in the tests); --spatial gate: softmax gate in
              (l, b) (joint_spatial.fit_softmax_gate) fitted on up to --gate-rows training rows
              (fit/select, faint limit, QSO flag removed), tabulated at nside 16 (step 4 choice).
  faint limit model meta faint_limit (Legacy r South then North, S/N >= 10; also the reference).
  catch-all   Student-t moments from the new background; fractions refitted on calibration
              (role 2) rows that pass the faint limit (and, with --drop-flagged, are not flagged
              likely QSOs), capped at --catchall-rows.
QSO models, priors (no background dependence) and the QSO support threshold are kept.
latent.json keeps the QSO entries and records the new latent background.
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
    p.add_argument('--workers', type=int, default=8); p.add_argument('--spatial', choices=('global', 'gate'), default='global')
    p.add_argument('--gate-rows', type=int, default=300000); p.add_argument('--drop-flagged', action='store_true')
    p.add_argument('--status', default='TEST ONLY: binned background, faint limit; not for release')
    p.add_argument('--bundle-id'); a = p.parse_args()
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
    data = fit_arrays(RUN, 'stars'); flag = np.load(RUN/'stars'/'qso_flag.npy') if a.drop_flagged else np.zeros(len(data['role']), bool)
    gate_info = None
    if a.spatial == 'gate':
        from qso_pcolor.joint_spatial import fit_softmax_gate, gate_to_spatial_weights
        tr = np.flatnonzero(np.isin(data['role'], cfg['fit_roles']) & ~flag)
        has, bright, _ = model.faint_limit_status(Photometry(data['flux'][tr], data['variance'][tr], model.transform.bands))
        tr = tr[has & bright]; tr = np.sort(np.random.default_rng(20261004).choice(tr, min(a.gate_rows, len(tr)), replace=False))
        with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_init_lp,
                                 initargs=(str(a.out/'model.json'),)) as pool:
            lp = np.concatenate(list(pool.map(_lp_chunk, [tr[i:i+1000] for i in range(0, len(tr), 1000)])))
        gate = fit_softmax_gate(lp, data['l'][tr], data['b'][tr], bmix.weights, ridge=1., max_iter=3000)
        gate_info = dict(rows=int(len(tr)), n_iter=gate['n_iter'], converged=gate['converged'], message=gate['message'])
        print('GATE', gate_info, flush=True)
        spatial = gate_to_spatial_weights(gate, nside=16, min_abs_b_deg=cfg.get('min_abs_b_deg', 25.),
                                          meta=dict(roles=['fit', 'select'], rows=int(len(tr)), qso_flag_removed=bool(a.drop_flagged)))
        model = dataclasses.replace(model, spatial_background=spatial); model.save(a.out/'model.json')
    # Catch-all: same construction as complete_unified_pilot.py, fractions on faint-limited calibration rows.
    catch = cfg['catchall']; mean, cov = mixture_moments([model.background])
    outlier = MultiSurveyOutlier(mean, catch['kappa']**2*cov, catch['kappa'], 0., model.transform.bands, model.transform_id,
        {'*': ([0., 1.], [catch['global_prior_mean']])}, meta=dict(population='psf', model_run_id=model.meta['run_id']),
        family=catch['family'], nu=catch['nu'], noise=catch['noise'])
    rr = np.flatnonzero((data['role'] == 2) & ~flag)
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
    outlier.meta.update(roles=['calib'], rows=int(len(rr)), faint_limit=FAINT, qso_flag_removed=bool(a.drop_flagged)); outlier.save(a.out/'outlier.json')
    (a.out/'outlier_unfitted.json').unlink()
    shutil.copy(a.source/'priors.json', a.out/'priors.json')
    shutil.copy(a.source/'support.json', a.out/'support.json')
    latent = json.loads((a.source/'latent.json').read_text()); latent['background'] = latent_b.to_dict()
    latent['background_design'] = dict(kind='binned in Legacy r, concatenated', designs=str(a.designs)); write_json(a.out/'latent.json', latent)
    manifest = json.loads((a.source/'manifest.json').read_text())
    manifest.update(files={n: file_hash(a.out/n) for n in ('model.json', 'priors.json', 'outlier.json')},
                    bundle_id=a.bundle_id or a.out.name+'_binned_test', status=a.status,
                    unified_files={n: hashlib.sha256((a.out/n).read_bytes()).hexdigest() for n in ('latent.json', 'support.json')})
    write_json(a.out/'manifest.json', manifest)
    write_json(a.out/'build.json', dict(source=str(a.source), designs=str(a.designs), components=int(len(bmix.weights)),
                                        catchall_rows=int(len(rr)), faint_limit=FAINT, spatial=a.spatial, gate=gate_info,
                                        qso_flag_removed=bool(a.drop_flagged)))
    print('BUNDLE', a.out, flush=True)


def _init(model_path, outlier_path):
    _state['model'] = MultiSurveyModel.load(model_path); _state['outlier'] = MultiSurveyOutlier.load(outlier_path)
    _state['data'] = fit_arrays(RUN, 'stars')


def _init_lp(model_path):
    _state['model'] = MultiSurveyModel.load(model_path); _state['data'] = fit_arrays(RUN, 'stars')


def _lp_chunk(rows):
    from qso_pcolor.spatial import component_log_prob
    m, d = _state['model'], _state['data']; f = m.transform(Photometry(d['flux'][rows], d['variance'][rows], m.transform.bands))
    return component_log_prob(m.background, f.x, f.cov, f.observed)


def _chunk(rows):
    return evaluate_densities(_state['model'], _state['data'], rows, {'outlier': _state['outlier']}, chunk=64)


if __name__ == '__main__':
    main()
