#!/usr/bin/env python
"""Assemble all locally available training measurements; never query or fit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from qso_pcolor.full_sample import assemble_qso, assemble_stars, file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/full_sample_training.json'))
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    q, s = Path(cfg['qso_root']), Path(cfg['stellar_root'])
    sources = [q/'targets.npz', q/'provenance.json', s/'qso_roles.npz', s/'spatial_roles.json',
               s/'cleaning_report.json', s/'photometry_progress.json', s/'coverage_report.json']
    sources += [q/f'progress_{survey}.json' for survey in cfg['acquisition_complete_surveys']+cfg['reuse_only_surveys']]
    for survey in cfg['acquisition_complete_surveys']:
        if not json.loads((q/f'progress_{survey}.json').read_text())['complete']:
            raise ValueError(f'{survey} acquisition incomplete')
    identity = dict(config=cfg, source_hashes={str(p): file_hash(p) for p in sources},
                    implementation=file_hash(Path(__file__)),
                    module=file_hash(Path('src/qso_pcolor/full_sample.py')))
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(cfg['input_root'])/version
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('assembly identity changed')
        for path, digest in manifest['files'].items():
            if file_hash(root/path) != digest:
                raise ValueError(f'assembled file changed: {path}')
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root/'incomplete.json', identity)
        report = dict(qso=assemble_qso(q, s, root/'qso', batch_size=cfg['assembly_batch_size']),
                      stars=assemble_stars(s, root/'stars', batch_size=cfg['assembly_batch_size']))
        write_json(root/'report.json', report)
        files = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.glob('*/*.npy'))}
        files['report.json'] = file_hash(root/'report.json')
        manifest = dict(identity=identity, version=version, files=files, complete=True,
                        no_database_queries=True, model_fitted=False)
        write_json(manifest_path, manifest)
    write_json(Path(cfg['input_root'])/'current.json', dict(directory=str(root), manifest_sha256=file_hash(manifest_path)))
    print('Verified training inputs:', root, flush=True)


if __name__ == '__main__':
    main()
