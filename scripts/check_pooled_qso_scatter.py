#!/usr/bin/env python
"""One bounded QSO scatter intervention and optional QSO-only prototype refit.

Paired QSO excess scatter is not identified as instrumental scatter. Compare
it with the star-calibrated instrumental residual, keeping the QSO offset.
The optional refit reuses all stellar and separate-system fits by explicit
provenance links; it never retrains those or changes the active model.
"""
import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from qso_pcolor.full_sample import file_hash, write_json
from qso_pcolor.multisurvey import conditional_log_prob
from qso_pcolor.projected_xd import native_mixture
from run_pooled_optical_prototype import load_arrays, diagonal, operators_for, fit_task
from validate_pooled_optical_prototype import read_fit, summary


def sensitivity(root, cfg, calibration):
    data = load_arrays(root/'qso'); tasks = json.loads((root/'tasks.json').read_text())
    zs = np.array([t['z'] for t in tasks if t['kind'] == 'qso' and t['mode'] == 'pooled'])
    saved = dict(np.load(root/'predictive_qso_north_separate.npz')); rows = saved['rows']
    nearest = np.argmin(abs(data['zspec'][rows, None]-zs), axis=1)
    h, offset, extra = operators_for(calibration, 'qso', 'pooled')[1]
    result = {}
    for label, t in [('paired_qso_excess', extra), ('stellar_excess', np.array(calibration['populations']['stars']['extra_covariance']))]:
        lp = np.empty(len(rows))
        for j in np.unique(nearest):
            local = np.flatnonzero(nearest == j)
            mix = native_mixture(read_fit(root, 'qso', int(j), 'pooled'), h, offset, t)
            for start in range(0, len(local), cfg['batch_size']):
                ii = local[start:start+cfg['batch_size']]; rr = rows[ii]
                obs = data['observed'][rr]
                anchors = np.array([1, 0, 2])[np.argmax(obs[:, [1, 0, 2]], axis=1)]
                for a in np.unique(anchors):
                    use = anchors == a; sel = rr[use]
                    lp[ii[use]] = conditional_log_prob(mix, data['y'][sel], diagonal(data['variance'][sel]), data['observed'][sel], int(a))
        informative = data['observed'][rows].sum(axis=1) >= 2
        result[label] = summary((lp-saved['logp'])[informative])
        np.save(root/('sensitivity_'+label+'.npy'), lp)
    write_json(root/'qso_scatter_sensitivity.json', dict(results=result,
        scope='Prediction-only intervention on the original pooled QSO fit; not a new trained candidate.',
        code=file_hash(Path(__file__))))
    print('SENSITIVITY', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--fit', action='store_true'); args = parser.parse_args()
    cfg = json.loads(Path('configs/pooled_optical_prototype.json').read_text())
    parent = Path(json.loads((Path(cfg['output'])/'current.json').read_text())['directory'])
    calibration = json.loads((parent/'calibration.json').read_text())
    sensitivity(parent, cfg, calibration)
    if not args.fit: return
    revised = deepcopy(calibration)
    qso = revised['populations']['qso']
    qso['paired_excess_covariance_not_added'] = qso['extra_covariance']
    qso['extra_covariance'] = revised['populations']['stars']['extra_covariance']
    qso['extra_covariance_source'] = 'Stellar overlap instrumental residual; QSO variability remains in the population density, not a second added paired-epoch variance.'
    cfg = dict(cfg, workers=2, qso_extra_covariance_source='stellar_instrumental_residual',
        parent_experiment=str(parent), policy=cfg['policy']+' One targeted QSO-only refit after the paired-scatter intervention; other fits are reused explicitly.')
    identity = dict(parent=str(parent), parent_prepared=file_hash(parent/'prepared.json'),
        calibration=revised, config=cfg, code=file_hash(Path(__file__)),
        trainer=file_hash(Path('scripts/run_pooled_optical_prototype.py')),
        implementation=file_hash(Path('src/qso_pcolor/projected_xd.py')))
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = parent/('instrumental_scatter_'+tag); root.mkdir(exist_ok=True)
    for folder in ('fits', 'progress'): (root/folder).mkdir(exist_ok=True)
    tasks = json.loads((parent/'tasks.json').read_text())
    reused = {}
    for name in ('qso', 'stars', 'prepared.json'):
        dest = root/name
        if not dest.is_symlink(): dest.symlink_to((parent/name).resolve())
    for task in tasks:
        if task['kind'] == 'qso' and task['mode'] == 'pooled': continue
        name = task['name']+'.json'; target = (parent/'fits'/name).resolve()
        dest = root/'fits'/name
        if not dest.is_symlink(): dest.symlink_to(target)
        reused[name] = str(target)
    write_json(root/'calibration.json', revised); write_json(root/'effective_config.json', cfg)
    write_json(root/'tasks.json', tasks); write_json(root/'identity.json', identity)
    write_json(root/'reused_fits.json', reused)
    write_json(parent/'scatter_variant.json', dict(directory=str(root)))
    pending = [t for t in tasks if t['kind'] == 'qso' and t['mode'] == 'pooled' and not (root/'fits'/(t['name']+'.json')).exists()]
    from qso_pcolor.sky_acquisition import acquisition_lock
    with acquisition_lock(root/'fit.lock'):
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            futures = {pool.submit(fit_task, t, cfg, str(root)): t['name'] for t in pending}
            while futures:
                done, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    print('VARIANT FIT COMPLETE', future.result(), flush=True); del futures[future]
                selected = [t for t in tasks if t['kind'] == 'qso' and t['mode'] == 'pooled']
                work = 0; total = sum(t['n']*t['k']*cfg['max_iter'] for t in selected)
                for t in selected:
                    p = root/'progress'/(t['name']+'.json'); d = json.loads(p.read_text()) if p.exists() else dict(iteration=0, complete=False)
                    work += t['n']*t['k']*(cfg['max_iter'] if (root/'fits'/(t['name']+'.json')).exists() else d['iteration'])
                write_json(root/'progress.json', dict(remaining_fits=len(futures), total_fits=len(selected), budget_progress_fraction=work/total))
        write_json(root/'qso_training_complete.json', dict(identity=identity, active_model_changed=False))
        while not (parent/'training_complete.json').exists():
            print('QSO VARIANT COMPLETE; waiting for the original stellar fits', flush=True); time.sleep(30)
        hashes = {name: file_hash(Path(path)) for name, path in reused.items()}
        write_json(root/'training_complete.json', dict(identity=identity, reused_hashes=hashes, active_model_changed=False))
    print('VARIANT COMPLETE', root, flush=True)


if __name__ == '__main__': main()
