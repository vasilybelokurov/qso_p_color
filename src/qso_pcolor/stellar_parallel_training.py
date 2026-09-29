"""Continue a saved stellar fit with parallel E steps and one global M step."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

from . import full_training as training
from .full_sample import TrainingRows, write_json
from .parallel_em import ParallelAccumulator, TrainingBatchFactory
from .sky_acquisition import acquisition_lock
from .streaming_xd import fit_xd_batches

# Exact original serial engine whose unchanged update was factored into an
# injectable accumulation pass. This is checkpoint provenance, not a fit setting.
SERIAL_STREAM_SHA256 = '2bc3d22ac13a0f60ea2dfb873a9c0a75f5c75e5d0b74be07b1ba9d3556b785e4'


def prepare_resume(parent: Path, out: Path, identity: dict) -> None:
    """Copy verified checkpoints to a new implementation identity; retain the parent."""
    old = json.loads((parent/'identity.json').read_text())
    if {k:v for k,v in old.items() if k != 'implementation'} != {k:v for k,v in identity.items() if k != 'implementation'}:
        raise ValueError('checkpoint inputs, configuration or source model changed')
    for name, digest in identity['implementation'].items():
        allowed = {digest, SERIAL_STREAM_SHA256} if name == 'streaming_xd.py' else {digest}
        if old['implementation'].get(name) not in allowed:
            raise ValueError(f'unsupported checkpoint implementation change: {name}')
    if set(old['implementation']) != set(identity['implementation']):
        raise ValueError('checkpoint implementation manifest changed')
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text()) != identity:
            raise ValueError('destination identity mismatch')
        if not (out/'checkpoint_lineage.json').exists():
            raise ValueError('destination lacks checkpoint lineage')
        return
    if out.resolve() == parent.resolve():
        raise ValueError('parallel continuation must preserve a separate parent snapshot')
    copied = {}
    for p in sorted(parent.glob('*.json')):
        if not (p.name.startswith('qso') or p.name.startswith('background')):
            continue
        digest = training.file_hash(p)
        dest = out/p.name
        shutil.copy2(p, dest)
        if training.file_hash(dest) != digest or training.file_hash(p) != digest:
            raise ValueError('checkpoint changed during transfer')
        copied[p.name] = digest
    write_json(out/'checkpoint_lineage.json', dict(parent=str(parent.resolve()),
        parent_identity=old, copied_sha256=copied,
        reason='Equivalent full-data E-step reduction; unchanged M step and saved-history continuation.'))
    write_json(out/'identity.json', identity)


def train_stellar_parallel(cfg: dict, *, resume_from: Path, workers: int, task_rows: int) -> Path:
    """Reuse all QSO fits and continue the stellar checkpoint; never promote."""
    root, _, source, identity = training.load_inputs(cfg)
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = Path(cfg['output_root'])/version
    out.mkdir(parents=True, exist_ok=True)
    with acquisition_lock(resume_from/'worker.lock'), acquisition_lock(out/'worker.lock'):
        prepare_resume(resume_from, out, identity)
        for i, _ in enumerate(training.slice_ranges(source, cfg['z_step'])):
            if not (out/f'qso_{i:02d}.json').exists():
                raise ValueError('all QSO slices must be saved before stellar-only continuation')
        if not (out/'background.final.checkpoint.json').exists():
            raise ValueError('stellar final checkpoint required')
        data = TrainingRows(root/'stars', source.transform.bands)
        rows = data.select(tuple(cfg['final_shape_roles']))
        factory = TrainingBatchFactory(root/'stars', rows, source.transform, cfg['fit_batch_size'])
        engine_files = ('parallel_em.py', 'stellar_parallel_training.py', 'streaming_xd.py')
        engine_hashes = {name:training.file_hash(Path('src/qso_pcolor')/name) for name in engine_files}
        record_path = out/'stellar_parallel_execution.json'
        if record_path.exists() and json.loads(record_path.read_text())['engine_hashes'] != engine_hashes:
            raise ValueError('parallel execution code changed within a checkpoint run')
        checkpoint = json.loads((out/'background.final.checkpoint.json').read_text())
        record = dict(coordinator_pid=os.getpid(), workers=workers, worker_threads=1,
                      task_rows=task_rows, expected_rows=len(rows), engine_hashes=engine_hashes,
                      resumed_iteration=checkpoint['iteration'], state='running')
        write_json(record_path, record)

        def parallel_fit(batch_source, **kwargs):
            if kwargs['expected_rows'] != len(rows):
                raise ValueError('unexpected population in stellar-only continuation')
            with ParallelAccumulator(factory, len(rows), workers=workers, task_rows=task_rows,
                                     status_directory=out) as accumulator:
                return fit_xd_batches(batch_source, accumulator=accumulator, **kwargs)

        try:
            with patch.object(training, 'fit_xd_batches', parallel_fit):
                result = training._train_candidate(cfg, root, source, identity, version, out)
        except BaseException as error:
            record.update(state='failed', error=repr(error)); write_json(record_path, record)
            raise
        record['state'] = 'completed'; write_json(record_path, record)
        return result
