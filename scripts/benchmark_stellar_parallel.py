#!/usr/bin/env python
"""Compare serial and four-worker E steps on identical diagnostic rows; no fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from qso_pcolor.full_training import load_inputs
from qso_pcolor.full_sample import TrainingRows, write_json
from qso_pcolor.gaussmix import GaussianMixture
from qso_pcolor.parallel_em import TrainingBatchFactory, ParallelAccumulator
from qso_pcolor.streaming_xd import accumulate_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.parent/'identity.json').read_text())['config']
    root, _, source, identity = load_inputs(cfg)
    data = TrainingRows(root/'stars', source.transform.bands)
    full_rows = data.select(tuple(cfg['final_shape_roles']))
    # Timing windows only; the actual fitting population is never capped.
    width = 8192
    starts = np.linspace(0, len(full_rows)-width, 16).astype(int)
    rows = np.concatenate([full_rows[start:start+width] for start in starts])
    assert len(np.unique(rows)) == len(rows)
    factory = TrainingBatchFactory(root/'stars', rows, source.transform, cfg['fit_batch_size'])
    checkpoint_text = (args.parent/'background.final.checkpoint.json').read_text()
    checkpoint = json.loads(checkpoint_text)
    mix = GaussianMixture.from_dict(checkpoint['mixture'])
    source_batches = lambda: factory(0, len(rows))
    start = time.perf_counter(); serial = accumulate_batches(source_batches, mix); serial_seconds = time.perf_counter()-start
    print('Serial diagnostic E step:', serial_seconds, 'seconds', flush=True)
    with ParallelAccumulator(factory, len(rows), workers=4, task_rows=8192) as accumulator:
        start = time.perf_counter(); parallel = accumulator(source_batches, mix); parallel_seconds = time.perf_counter()-start
        start = time.perf_counter(); repeat = accumulator(source_batches, mix); steady_seconds = time.perf_counter()-start
    errors = []
    for a, b, c in zip(serial, parallel, repeat):
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(c, b, rtol=0, atol=0)
        errors.append(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    result = dict(rows=len(rows), full_fitting_rows=len(full_rows), k=mix.n_components,
                  workers=4, task_rows=8192, reference_iteration=checkpoint['iteration'],
                  serial_seconds=serial_seconds, parallel_first_seconds=parallel_seconds,
                  parallel_steady_seconds=steady_seconds, steady_speedup=serial_seconds/steady_seconds,
                  maximum_absolute_statistic_differences=errors, repeated_parallel_bitwise_identical=True,
                  input_manifest_sha256=identity['inputs_sha256'],
                  checkpoint_sha256=hashlib.sha256(checkpoint_text.encode()).hexdigest(),
                  parallel_engine_sha256=hashlib.sha256(Path('src/qso_pcolor/parallel_em.py').read_bytes()).hexdigest(),
                  no_parameter_updates=True, data_scope='16 distributed diagnostic windows only; no production sample cap')
    write_json(args.out, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
