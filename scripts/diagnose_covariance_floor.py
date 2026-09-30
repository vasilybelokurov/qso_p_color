#!/usr/bin/env python
"""Evaluate one constrained update for held slices; save diagnostics, not models."""
import json
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

from qso_pcolor.covariance_diagnostic import constrained_step
from qso_pcolor.convergence_review import checked_rows, verify_parent_identity
from qso_pcolor.full_sample import TrainingRows, write_json
from qso_pcolor.full_training import load_inputs, make_source, slice_ranges
from qso_pcolor.gaussmix import GaussianMixture


def main():
    # One short serial diagnostic at low priority; no additional fitting pool.
    os.nice(10)
    options=json.loads(Path('configs/convergence_continuation.json').read_text())
    cfg=json.loads(Path(options['training_config']).read_text())
    parent=Path(options['parent']);out=Path(options['output'])
    root,_,source,identity=load_inputs(cfg)
    verify_parent_identity(json.loads((parent/'identity.json').read_text()),identity)
    audit=json.loads((out/'audit.json').read_text())
    data=TrainingRows(root/'qso',source.transform.bands)
    ranges=slice_ranges(source,cfg['z_step'])
    results={}
    for name,r in audit['fits'].items():
        if not name.startswith('qso_') or r['action']!='hold_for_diagnosis':
            continue
        record=json.loads((parent/(name+'.json')).read_text())
        rows=checked_rows(data,ranges[int(name.split('_')[1])],cfg,record)
        _,result=constrained_step(make_source(data,rows,source,cfg),
            GaussianMixture.from_dict(record['mixture']),floor=cfg['regularization'],expected_rows=len(rows))
        results[name]=result
        write_json(out/'covariance_floor_diagnostic.json',dict(training_identity=identity,
            floor=cfg['regularization'],results=results,complete=False))
        print(name,result,flush=True)
    write_json(out/'covariance_floor_diagnostic.json',dict(training_identity=identity,
        floor=cfg['regularization'],results=results,complete=True))


if __name__=='__main__':
    main()
